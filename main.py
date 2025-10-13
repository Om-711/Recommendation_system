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
    ], 

    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# templates me html code
templates = Jinja2Templates(directory="templates")


@app.get("/", tags=["default"])
async def index():
    return RedirectResponse(url="/docs")

def making_data_endpoint():
    df,_ = making_data()
    return df

def making_user_data_endopoints():
    _,df = making_data()
    return df

def make_serializable(obj):
    """Recursively convert obj into JSON-serializable Python primitives."""
    # pandas DataFrame / Series
    if isinstance(obj, pd.DataFrame):
        return make_serializable(obj.to_dict(orient="records"))
    if isinstance(obj, pd.Series):
        return make_serializable(obj.tolist())

    # BSON ObjectId
    if isinstance(obj, ObjectId):
        return str(obj)

    # datetimes and dates
    if isinstance(obj, (datetime.datetime, datetime.date, datetime.time, pd.Timestamp)):
        # ensure timezone-aware datetimes are preserved when possible
        try:
            return obj.isoformat()
        except Exception:
            return str(obj)

    # numpy scalars / arrays
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return make_serializable(obj.tolist())

    # Decimal
    if isinstance(obj, Decimal):
        return float(obj)

    # basic containers
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

# @app.post("/als-recommend", response_class=JSONResponse)
# async def als_recommend(user_id: int):  
    
#     # df1 = pd.read_csv("D:\College\SEM 5\LAB\SE\dataset\data.csv", nrows=10000)
#     df_product, df_user = making_data_endpoint()

#     if user_id not in df_user['user_id'].unique():
#         return JSONResponse(content={"Error": "User not Found"}, status_code=404)
    
#     model, user_encoder, item_encoder, interactions = als_recommendation(user_id)

#     recom = get_als_recommendations(user_id, model, user_encoder, item_encoder, interactions)

#     recommended_products = df_product[df_product['productID'].isin(recom)]['category'].unique()

#     if isinstance(recommended_products, pd.DataFrame):
#         recommendations = recommended_products.to_dict(orient="records")

#     recs_json_serializable = make_serializable(recommendations)

#     return JSONResponse(content={"recommendations": recs_json_serializable})


class RecommendRequest(BaseModel):
    item_name: str
    user_id: int | None = None

@app.post("/recommend")
async def recommend(req: RecommendRequest):
    item_name = req.item_name
    user_id = req.user_id
    print(f"Received item_name: {item_name}, user_id: {user_id}")
    df = making_data_endpoint()
    
    corrected_item_name = get_closest_match(item_name, df['name'].tolist())
    
    if not user_id:  
        recommendations = content_based_recommendations_improved(df, corrected_item_name, top_n=10)
        # recommendations = content_based_recommendations(df, corrected_item_name, top_n=10)
    else:
        recommendations = hybrid_recommendation_system(df, user_id, item_name, top_n=10)
    print(recommendations)

    if isinstance(recommendations, pd.DataFrame):
        recommendations = recommendations.to_dict(orient="records")

    recs_json_serializable = make_serializable(recommendations)

    return JSONResponse(content={"recommendations": recs_json_serializable})




    
