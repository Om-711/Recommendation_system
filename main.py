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
    content_based_recommendations,
    get_closest_match,
    als_recommendation,
    get_als_recommendations,
    making_data,
    content_based_recommendations_improved
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

class RecommendRequest(BaseModel):
    item_name: str
    user_id: str | None = None


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
        return make_serializable(obj.to_dict(orient="records"))
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
        return float(obj)
    if isinstance(obj, np.ndarray):
        return make_serializable(obj.tolist())


    if isinstance(obj, Decimal):
        return float(obj)


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
async def als_recommend(user_id: str):  

    # df1 = pd.read_csv("D:\College\SEM 5\LAB\SE\dataset\data.csv", nrows=10000)
    df_product, df_user = making_data_endpoint()

    if user_id not in df_user['user_id'].unique():
        return JSONResponse(content={"Error": "User not Found"}, status_code=404)
    
    model, user_encoder, item_encoder, interactions = als_recommendation(user_id)

    recom = get_als_recommendations(user_id, model, user_encoder, item_encoder, interactions)

    recommended_products = df_product[df_product['productID'].isin(recom)]['category'].unique()

    if isinstance(recommended_products, pd.DataFrame):
        recommended_products = recommended_products.to_dict(orient="records")

    recs_json_serializable = make_serializable(recommended_products)

    return JSONResponse(content={"recommendations": recs_json_serializable})



@app.post("/recommend")
async def recommend(req: RecommendRequest):
    item_name = req.item_name
    user_id = req.user_id
    print(f"Received item_name: {item_name}, user_id: {user_id}")
    df_products, df_user = making_data_endpoint()
    df = df_products
    
    # if not user_id:  
    #     corrected_item_name = get_closest_match(item_name, df['name'].tolist())
    #     recommendations = hybrid_recommendation_system(df, user_id, corrected_item_name, top_n=10)
    #     # recommendations = content_based_recommendations(df, corrected_item_name, top_n=10)
    # else:
    corrected_item_name = get_closest_match(item_name, df['name'].tolist())
    recommendations = content_based_recommendations_improved(df, corrected_item_name, top_n=10)
    print(recommendations)

    if isinstance(recommendations, pd.DataFrame):
        recommendations = recommendations.to_dict(orient="records")

    recs_json_serializable = make_serializable(recommendations)

    return JSONResponse(content={"recommendations": recs_json_serializable})




    
