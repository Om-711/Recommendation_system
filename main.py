from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from bson import ObjectId
from decimal import Decimal
import datetime

from recommendation import (
    hybrid_recommendation_system,
    rating_based_recommendation_system,
    get_closest_match,
    train_als_model,
    get_als_recommendations,
    making_data,
    content_based_recommendations_improved,
    collaborative_filtering_recommendations
)

app = FastAPI()

# Allow cross-origin requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:5173', 'http://localhost:5174', 'https://apnabzaar.netlify.app'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global ALS model (loaded once at startup for speed)
als_model = None
als_user_encoder = None
als_item_encoder = None
als_interactions = None

@app.on_event("startup")
async def startup_event():
    """Load ALS model when server starts"""
    global als_model, als_user_encoder, als_item_encoder, als_interactions
    try:
        _, df_user = making_data()
        als_model, als_user_encoder, als_item_encoder, als_interactions = train_als_model(df_user)
        print("ALS model initialized" if als_model else "ALS model unavailable")
    except Exception as e:
        print(f"ALS init error: {e}")

class RecommendRequest(BaseModel):
    item_name: str
    user_id: str | None = None
    top_n: int = 20

class UserRecommendRequest(BaseModel):
    user_id: str
    top_n: int = 20

@app.get("/")
async def index():
    return RedirectResponse(url="/docs")

# Convert pandas/numpy objects to JSON-friendly format
def to_json(obj):
    if isinstance(obj, pd.DataFrame):
        return to_json(obj.fillna(value=np.nan).replace([np.nan], [None]).to_dict(orient="records"))
    if isinstance(obj, pd.Series):
        return to_json(obj.tolist())
    if isinstance(obj, ObjectId):
        return str(obj)
    if isinstance(obj, (datetime.datetime, datetime.date, datetime.time, pd.Timestamp)):
        try:
            return obj.isoformat()
        except:
            return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    if isinstance(obj, np.ndarray):
        return to_json(obj.tolist())
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: to_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json(v) for v in obj]
    return obj

@app.post("/main")
async def main_page(request: Request):
    """Get top rated products"""
    df_product, _ = making_data()
    top_products = rating_based_recommendation_system(df_product)
    recs = list(top_products.to_dict(orient="records") if hasattr(top_products, "to_dict") else top_products)
    return JSONResponse(content={"Top_rated_products": recs})

@app.post("/als-recommend")
async def als_recommend(user_id: str, top_n: int = 10):
    """Get recommendations using ALS matrix factorization"""
    try:
        df_product, df_user = making_data()
        
        if als_model is None or user_id not in df_user['user_id'].unique():
            top_products = df_product.sort_values('rating', ascending=False).head(top_n)
            return JSONResponse(content={
                "recommendations": to_json(top_products.to_dict(orient="records")),
                "method": "fallback_popular"
            })
        
        product_ids = get_als_recommendations(user_id, als_model, als_user_encoder, als_item_encoder, als_interactions, N=top_n)
        
        if not product_ids:
            top_products = df_product.sort_values('rating', ascending=False).head(top_n)
            return JSONResponse(content={
                "recommendations": to_json(top_products.to_dict(orient="records")),
                "method": "fallback_popular"
            })
        
        products = df_product[df_product['productID'].isin(product_ids)]
        return JSONResponse(content={
            "recommendations": to_json(products.to_dict(orient="records")),
            "count": len(products),
            "method": "als"
        })
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/user-recommend")
async def user_recommend(req: UserRecommendRequest):
    """Get personalized recommendations based on user's purchase history"""
    try:
        df_products, df_user = making_data()
        
        if req.user_id not in df_user['user_id'].unique():
            return JSONResponse(
                content={"error": "User not found", "user_id": req.user_id},
                status_code=404
            )
        
        # Use collaborative filtering with category preference
        recommendations = collaborative_filtering_recommendations(
            df_user, df_products, req.user_id, top_n=req.top_n, category_boost=True
        )
        
        if recommendations.empty:
            user_products = df_user[df_user['user_id'] == req.user_id]['productID'].unique()
            available = df_products[~df_products['productID'].isin(user_products)]
            fallback = available[available['stock'] > 0].sort_values(
                by=['rating', 'reviews'], ascending=[False, False]
            ).head(req.top_n)
            
            if fallback.empty:
                fallback = df_products.sort_values(
                    by=['rating', 'reviews'], ascending=[False, False]
                ).head(req.top_n)
            
            return JSONResponse(content={
                "recommendations": to_json(fallback.to_dict(orient="records")),
                "count": len(fallback),
                "method": "fallback_popular"
            })
        
        return JSONResponse(content={
            "recommendations": to_json(recommendations.to_dict(orient="records")),
            "count": len(recommendations),
            "method": "collaborative_filtering"
        })
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/recommend")
async def recommend(req: RecommendRequest):
    """Get recommendations for a product (hybrid: content + collaborative)"""
    try:
        df_products, df_user = making_data()
        item_name = get_closest_match(req.item_name, df_products['name'].tolist())
        
        if not item_name:
            return JSONResponse(content={"error": "Product not found"}, status_code=404)
        
        # Use hybrid if user logged in, else content-based only
        if req.user_id and req.user_id in df_user['user_id'].unique():
            recs = hybrid_recommendation_system(df_products, df_user, req.user_id, item_name, top_n=20)
        else:
            recs = content_based_recommendations_improved(df_products, item_name, top_n=10)
        
        if isinstance(recs, pd.DataFrame):
            recs = recs.to_dict(orient="records")
        
        return JSONResponse(content={"recommendations": to_json(recs)})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
