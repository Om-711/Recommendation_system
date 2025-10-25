from fastapi import FastAPI
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from bson import ObjectId
from decimal import Decimal
import datetime
from datetime import datetime as dt, timedelta

from recommendation import (
    hybrid_recommendation_system,
    rating_based_recommendation_system,
    get_closest_match,
    train_als_model,
    get_als_recommendations,
    making_data,
    content_based_recommendations_improved,
    collaborative_filtering_recommendations,
    get_frequently_bought_together
)

app = FastAPI()

# CORS setup for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:5173', 'http://localhost:5174', 'https://apnabzaar.netlify.app'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global cache for data (refreshed periodically)
cached_products = None
cached_users = None
cache_timestamp = None
CACHE_DURATION = timedelta(minutes=5)  # Refresh every 5 minutes

# Global ALS model (loaded at startup)
als_model = None
als_user_encoder = None
als_item_encoder = None
als_interactions = None

def get_cached_data():
    """Get cached data or refresh if expired"""
    global cached_products, cached_users, cache_timestamp
    
    # Check if cache needs refresh
    if (cached_products is None or cached_users is None or 
        cache_timestamp is None or 
        dt.now() - cache_timestamp > CACHE_DURATION):
        
        print("Refreshing data cache...")
        cached_products, cached_users = making_data()
        cache_timestamp = dt.now()
        print(f"Cache updated: {len(cached_products)} products, {len(cached_users)} interactions")
    
    return cached_products.copy(), cached_users.copy()

@app.on_event("startup")
async def startup_event():
    """Initialize cache and ALS model when server starts"""
    global als_model, als_user_encoder, als_item_encoder, als_interactions
    try:
        # Load initial cache
        print("Loading initial data cache...")
        df_products, df_user = get_cached_data()
        
        # Train ALS model
        als_model, als_user_encoder, als_item_encoder, als_interactions = train_als_model(df_user)
        print("ALS model loaded" if als_model else "✗ ALS model unavailable")
        print(" Server ready!")
    except Exception as e:
        print(f"Startup error: {e}")

# Request models
class RecommendRequest(BaseModel):
    item_name: str
    user_id: str | None = None
    top_n: int = 20

class UserRecommendRequest(BaseModel):
    user_id: str
    top_n: int = 20

class FrequentlyBoughtTogetherRequest(BaseModel):
    product_id: str
    top_n: int = 6

class ALSRecommendRequest(BaseModel):
    user_id: str
    top_n: int = 10

@app.get("/")
async def index():
    """Redirect to API docs"""
    return RedirectResponse(url="/docs")

# JSON converter for pandas/numpy types
def to_json(obj):
    """Convert DataFrames and numpy types to JSON-safe format"""
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
async def main_page():
    """Get top-rated products for homepage"""
    df_product, _ = get_cached_data()
    top_products = rating_based_recommendation_system(df_product)
    return JSONResponse(content={
        "Top_rated_products": to_json(top_products.to_dict(orient="records"))
    })

@app.post("/als-recommend")
async def als_recommend(req: ALSRecommendRequest):
    """Get ALS matrix factorization recommendations"""
    try:
        df_product, df_user = get_cached_data()
        
        # Check if ALS can be used
        if als_model is None or req.user_id not in df_user['user_id'].unique():
            fallback = df_product.sort_values('rating', ascending=False).head(req.top_n)
            return JSONResponse(content={
                "recommendations": to_json(fallback.to_dict(orient="records")),
                "method": "fallback_popular"
            })
        
        # Get ALS recommendations
        product_ids = get_als_recommendations(
            req.user_id, als_model, als_user_encoder, 
            als_item_encoder, als_interactions, N=req.top_n
        )
        
        if not product_ids:
            fallback = df_product.sort_values('rating', ascending=False).head(req.top_n)
            return JSONResponse(content={
                "recommendations": to_json(fallback.to_dict(orient="records")),
                "method": "fallback_popular"
            })
        
        # Get product details
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
    """Get personalized recommendations based on user history"""
    try:
        df_products, df_user = get_cached_data()
        
        # Check if user exists
        if req.user_id not in df_user['user_id'].unique():
            return JSONResponse(
                content={"error": "User not found", "user_id": req.user_id},
                status_code=404
            )
        
        # Get collaborative filtering recommendations
        recommendations = collaborative_filtering_recommendations(
            df_user, df_products, req.user_id, top_n=req.top_n, category_boost=True
        )
        
        if recommendations.empty:
            # Fallback: popular products user hasn't bought
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
    """Get product recommendations (hybrid or content-based)"""
    try:
        df_products, df_user = get_cached_data()
        
        # Find closest matching product
        item_name = get_closest_match(req.item_name, df_products['name'].tolist())
        
        if not item_name:
            return JSONResponse(
                content={"error": "Product not found"}, 
                status_code=404
            )
        
        # Use hybrid if user is logged in, otherwise content-based
        if req.user_id and req.user_id in df_user['user_id'].unique():
            recs = hybrid_recommendation_system(
                df_products, df_user, req.user_id, item_name, top_n=req.top_n
            )
        else:
            recs = content_based_recommendations_improved(
                df_products, item_name, top_n=min(req.top_n, 10)
            )
        
        if isinstance(recs, pd.DataFrame):
            recs = recs.to_dict(orient="records")
        
        return JSONResponse(content={"recommendations": to_json(recs)})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/frequently-bought-together")
async def frequently_bought_together(req: FrequentlyBoughtTogetherRequest):
    """Get products frequently bought together (association rules)"""
    try:
        df_products, _ = get_cached_data()
        
        # Check if product exists
        if req.product_id not in df_products['productID'].values:
            return JSONResponse(
                content={"error": "Product not found", "product_id": req.product_id},
                status_code=404
            )
        
        # Get recommendations using association rules or co-occurrence
        recommendations = get_frequently_bought_together(
            req.product_id, df_products, top_n=req.top_n
        )
        
        if recommendations.empty:
            return JSONResponse(content={
                "recommendations": [],
                "count": 0,
                "message": "No frequently bought together items found",
                "method": "none"
            })
        
        # Check which method was used
        method = "association_rules" if "confidence" in recommendations.columns else "co_occurrence"
        
        return JSONResponse(content={
            "recommendations": to_json(recommendations.to_dict(orient="records")),
            "count": len(recommendations),
            "method": method
        })
    
    except Exception as e:
        return JSONResponse(
            content={"error": str(e), "product_id": req.product_id}, 
            status_code=500
        )
