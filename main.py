from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
from starlette.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from fastapi.responses import JSONResponse


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
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# templates me html code
templates = Jinja2Templates(directory="templates")


@app.get("/", tags=["default"])
async def index():
    return RedirectResponse(url="/docs")

@app.post("/making_data", response_class=JSONResponse)
def making_data_endpoint():
    df = making_data()
    return df

@app.post("/main", response_class=JSONResponse)
async def main_page(request: Request):
    top_products = rating_based_recommendation_system(df)

    if hasattr(top_products, "to_dict"):
        recs = top_products.to_dict(orient="records")
    else:
        recs = top_products  

    recs = list(recs)
    
    return JSONResponse(content={"Top_rated_products": recs})

@app.post("/als-recommend", response_class=JSONResponse)
async def als_recommend(user_id: int):  
    
    df1 = pd.read_csv("D:\College\SEM 5\LAB\SE\dataset\data.csv", nrows=10000)
    df = making_data_endpoint()

    if user_id not in df1['user_id'].unique():
        return JSONResponse(content={"Error": "User not Found"}, status_code=404)
    
    model, user_encoder, item_encoder, interactions = als_recommendation(user_id)

    recom = get_als_recommendations(user_id, model, user_encoder, item_encoder, interactions)

    recommended_products = df[df['product_id'].isin(recom)]['category_code'].unique()

    recom_json = recommended_products.to_dict(orient="records")

    return JSONResponse(content={"recommendations": recom_json})

@app.post("/recommend")
async def recommend(item_name: str = Form(...), user_id: int = Form(None)):

    df = making_data_endpoint()
    
    if user_id in (None, "", "None"):
        user_id = None
    else:
        user_id = int(user_id)

    if user_id is not None:
        recommendations = hybrid_recommendation_system(df, user_id, item_name, top_n=10)
    else:
        recommendations = content_based_recommendations(df, item_name, top_n=10)

    recs_json = recommendations.to_dict(orient="records")

    return JSONResponse(content={"recommendations": recs_json})




    
