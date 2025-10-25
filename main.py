from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
from starlette.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from bson import ObjectId
import numpy as np
import pandas as pd
import datetime
from decimal import Decimal


from recommendation import (
    hybrid_recommendation_system,
    rating_based_recommendation_system,
    get_closest_match,
    train_als_model,
    get_als_recommendations,
    making_data,
    content_based_recommendations_improved,
    collaborative_filtering_recommendations,
    collaborative_filtering_category_aware
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        'http://localhost:5173',
        'http://localhost:5174',
        'https://apnabzaar.netlify.app'
    ], 

    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for pretrained ALS model
als_model = None
als_user_encoder = None
als_item_encoder = None
als_interactions = None

@app.on_event("startup")
async def startup_event():
    """Train ALS model once at startup"""
    global als_model, als_user_encoder, als_item_encoder, als_interactions
    print("\n" + "="*80)
    print("INITIALIZING ALS MODEL AT STARTUP")
    print("="*80)
    try:
        _, df_user = making_data()
        als_model, als_user_encoder, als_item_encoder, als_interactions = train_als_model(df_user)
        if als_model is not None:
            print("✓ ALS model loaded and ready")
        else:
            print("⚠ ALS model could not be trained (insufficient data)")
    except Exception as e:
        print(f"⚠ Error training ALS model: {e}")
        import traceback
        traceback.print_exc()
    print("="*80 + "\n")

class RecommendRequest(BaseModel):
    item_name: str
    user_id: str | None = None
    top_n: int = 20  # Number of recommendations to return (default: 20)

class UserRecommendRequest(BaseModel):
    user_id: str
    top_n: int = 20  # Number of recommendations to return (default: 20)


# templates me html code
templates = Jinja2Templates(directory="templates")


@app.get("/", tags=["default"])
async def index():
    return RedirectResponse(url="/docs")

def making_data_endpoint():
    df_product,df_user = making_data()
    return df_product, df_user


def make_serializable(obj):
    """Recursively convert obj into JSON-serializable Python primitives."""
    if isinstance(obj, pd.DataFrame):
        # Replace NaN values with None before converting
        return make_serializable(obj.fillna(value=np.nan).replace([np.nan], [None]).to_dict(orient="records"))
    if isinstance(obj, pd.Series):
        return make_serializable(obj.tolist())

    if isinstance(obj, ObjectId):
        return str(obj)

    if isinstance(obj, (datetime.datetime, datetime.date, datetime.time, pd.Timestamp)):
        try:
            return obj.isoformat()
        except Exception:
            return str(obj)

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        # Handle NaN and inf values
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        return make_serializable(obj.tolist())

    if isinstance(obj, Decimal):
        return float(obj)
    
    # Handle Python float NaN/inf
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj

    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]

    return obj


@app.post("/main", response_class=JSONResponse)
async def main_page(request: Request):
    df = making_data_endpoint()
    top_products = rating_based_recommendation_system(df)

    if hasattr(top_products, "to_dict"):
        recs = top_products.to_dict(orient="records")
    else:
        recs = top_products  

    recs = list(recs)
    
    return JSONResponse(content={"Top_rated_products": recs})

@app.post("/als-recommend", response_class=JSONResponse)
async def als_recommend(user_id: str, top_n: int = 10):  
    """
    Get ALS-based recommendations for a user.
    Uses pretrained model loaded at startup for fast responses.
    """
    try:
        print(f"\nALS Recommendation request for user: {user_id}, top_n: {top_n}")
        
        # Check if model is available
        if als_model is None:
            print("⚠ ALS model not available, falling back to popular items")
            df_product, df_user = making_data_endpoint()
            # Return top rated products as fallback
            top_products = df_product.sort_values('rating', ascending=False).head(top_n)
            recs = make_serializable(top_products.to_dict(orient="records"))
            return JSONResponse(content={"recommendations": recs, "method": "fallback_popular"})
        
        df_product, df_user = making_data_endpoint()

        # Validate user exists
        if user_id not in df_user['user_id'].unique():
            return JSONResponse(
                content={"error": "User not found", "user_id": user_id}, 
                status_code=404
            )
        
        # Get ALS recommendations using pretrained model
        recommended_product_ids = get_als_recommendations(
            user_id, 
            als_model, 
            als_user_encoder, 
            als_item_encoder, 
            als_interactions, 
            N=top_n
        )
        
        print(f"✓ ALS returned {len(recommended_product_ids)} product IDs")

        if not recommended_product_ids:
            print("⚠ No recommendations from ALS, returning popular items")
            # Fallback to popular items
            top_products = df_product.sort_values('rating', ascending=False).head(top_n)
            recs = make_serializable(top_products.to_dict(orient="records"))
            return JSONResponse(content={"recommendations": recs, "method": "fallback_popular"})

        # Get full product details (not just categories!)
        recommended_products = df_product[df_product['productID'].isin(recommended_product_ids)]
        
        print(f"✓ Found {len(recommended_products)} product details")

        # Convert to JSON-serializable format
        recs = make_serializable(recommended_products.to_dict(orient="records"))

        return JSONResponse(content={
            "recommendations": recs,
            "count": len(recs),
            "method": "als"
        })
    
    except Exception as e:
        print(f"❌ Error in ALS recommendation: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            content={"error": f"An error occurred: {str(e)}"}, 
            status_code=500
        )

@app.post("/user-recommend")
async def user_recommend(req: UserRecommendRequest):
    """
    Get personalized recommendations for a user based on collaborative filtering.
    Uses user behavior patterns and similar users to recommend products.
    """
    try:
        user_id = req.user_id
        top_n = req.top_n
        
        print(f"\n{'='*80}")
        print(f"USER-BASED COLLABORATIVE FILTERING RECOMMENDATION")
        print(f"{'='*80}")
        print(f"User ID: {user_id}")
        print(f"Requested: {top_n} recommendations")
        
        # Load data
        df_products, df_user = making_data_endpoint()
        
        # Validate user exists
        if user_id not in df_user['user_id'].unique():
            print(f"❌ User {user_id} not found in database")
            return JSONResponse(
                content={
                    "error": "User not found",
                    "user_id": user_id,
                    "message": "This user has no interaction history in our system."
                }, 
                status_code=404
            )
        
        # Get user's interaction stats
        user_interactions = df_user[df_user['user_id'] == user_id]
        print(f"✓ User has {len(user_interactions)} interactions")
        print(f"  Event breakdown: {user_interactions['event'].value_counts().to_dict()}")
        
        # Get collaborative filtering recommendations (category-aware by default)
        recommendations = collaborative_filtering_recommendations(
            df_user, 
            df_products, 
            user_id, 
            top_n=top_n,
            category_boost=True  # Prioritize same categories as user's history
        )
        
        print(f"\n{'='*80}")
        print(f"RESULT")
        print(f"{'='*80}")
        
        # Handle empty recommendations
        if recommendations.empty or len(recommendations) == 0:
            print("⚠ No collaborative recommendations available")
            print("  Falling back to popular/top-rated products")
            
            # Fallback: Return top-rated products that user hasn't interacted with
            user_product_ids = df_user[df_user['user_id'] == user_id]['productID'].unique()
            available_products = df_products[~df_products['productID'].isin(user_product_ids)]
            
            # Sort by rating and stock availability
            fallback_recs = available_products[available_products['stock'] > 0].sort_values(
                by=['rating', 'reviews'], 
                ascending=[False, False]
            ).head(top_n)
            
            if len(fallback_recs) == 0:
                # Last resort: just return top products
                fallback_recs = df_products.sort_values(
                    by=['rating', 'reviews'], 
                    ascending=[False, False]
                ).head(top_n)
            
            recs = make_serializable(fallback_recs.to_dict(orient="records"))
            print(f"✓ Returned {len(recs)} fallback recommendations")
            
            return JSONResponse(content={
                "recommendations": recs,
                "count": len(recs),
                "method": "fallback_popular",
                "message": "Showing popular products (insufficient user similarity data)"
            })
        
        # Convert recommendations to JSON-serializable format
        recs = make_serializable(recommendations.to_dict(orient="records"))
        
        print(f"✓ Successfully generated {len(recs)} recommendations")
        print(f"{'='*80}\n")
        
        return JSONResponse(content={
            "recommendations": recs,
            "count": len(recs),
            "method": "collaborative_filtering",
            "message": f"Personalized recommendations based on users with similar preferences"
        })
    
    except Exception as e:
        print(f"\n❌ Error in user recommendation: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"{'='*80}\n")
        
        return JSONResponse(
            content={
                "error": "Internal server error",
                "message": f"An error occurred while generating recommendations: {str(e)}"
            }, 
            status_code=500
        )


@app.post("/recommend")
async def recommend(req: RecommendRequest):
    try:
        item_name = req.item_name
        user_id = req.user_id
        top_n = 10
        print(f"Received item_name: {item_name}, user_id: {user_id}, top_n: {top_n}")
        df_products, df_user = making_data_endpoint()
        df = df_products
        
        if user_id:  
            corrected_item_name = get_closest_match(item_name, df['name'].tolist())
            print(f"Corrected item name: {corrected_item_name}")
            
            # Check if user exists in df_user
            if user_id not in df_user['user_id'].unique():
                print(f"Warning: User {user_id} not found in user data. Using content-based recommendations instead.")
                recommendations = content_based_recommendations_improved(df, corrected_item_name, top_n=top_n)
            else:
                recommendations = hybrid_recommendation_system(df_products, df_user, user_id, corrected_item_name, top_n=20)
        else:
            print(f"No user ID provided. Using content-based recommendations.")
            corrected_item_name = get_closest_match(item_name, df['name'].tolist())
            recommendations = content_based_recommendations_improved(df, corrected_item_name, top_n=top_n)
            print(recommendations)

        if isinstance(recommendations, pd.DataFrame):
            recommendations = recommendations.to_dict(orient="records")

        recs_json_serializable = make_serializable(recommendations)

        return JSONResponse(content={"recommendations": recs_json_serializable})
    
    except Exception as e:
        print(f"Error in /recommend endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            content={"error": f"An error occurred: {str(e)}"}, 
            status_code=500
        )




    
