from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
from starlette.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from fastapi.responses import JSONResponse
from pydantic import BaseModel


from recommendation import (
    hybrid_recommendation_system,
    rating_based_recommendation_system,
    content_based_recommendations,
    get_closest_match,
    als_recommendation,
    get_als_recommendations,
    making_data
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

# @app.post("/making_data", response_class=JSONResponse)
def making_data_endpoint():
    df = making_data()
    return df

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
    
#     df1 = pd.read_csv("D:\College\SEM 5\LAB\SE\dataset\data.csv", nrows=10000)
#     df = making_data_endpoint()

#     if user_id not in df1['user_id'].unique():
#         return JSONResponse(content={"Error": "User not Found"}, status_code=404)
    
#     model, user_encoder, item_encoder, interactions = als_recommendation(user_id)

#     recom = get_als_recommendations(user_id, model, user_encoder, item_encoder, interactions)

#     recommended_products = df[df['product_id'].isin(recom)]['category_code'].unique()

#     recom_json = recommended_products.to_dict(orient="records")

#     return JSONResponse(content={"recommendations": recom_json})
class RecommendRequest(BaseModel):
    item_name: str
    user_id: int | None = None

@app.post("/recommend")
async def recommend(req: RecommendRequest):
    item_name = req.item_name
    user_id = req.user_id
    print(f"Received item_name: {item_name}, user_id: {user_id}")
    df = making_data_endpoint()
    
    if not user_id:  
        corrected_item_name = get_closest_match(item_name, df['Name'].tolist())
        recommendations = content_based_recommendations(df, corrected_item_name, top_n=10)
    else:
        recommendations = hybrid_recommendation_system(df, user_id, item_name, top_n=10)
    print(recommendations)

    recs_json = recommendations.to_dict(orient="records")

    return JSONResponse(content={"recommendations": recs_json})




    
