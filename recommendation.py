import os
import pymysql
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares
from sklearn.preprocessing import LabelEncoder
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import pymongo

def making_data():
        
    mongo_url = "mongodb+srv://arshadmansuri1825:u1AYlNbjuA5FpHbb@cluster1.2majmfd.mongodb.net/ECommerce"  # e.g. mongodb+srv://username:password@cluster0.mongodb.net/myDatabase?retryWrites=true&w=majority

    client = MongoClient(mongo_url)

    db = client["ECommerce"]

    product_collection = db["products"]
    user_data_collection = db["users"]

    products = list(product_collection.find())
    users  = list(user_data_collection.find())

    product_data = []
    for p in products:
        if(p.get("isActive")):
            product_data.append({
                "productID": str(p["_id"]),
                "name": p["name"],
                "price": p["price"],
                "category": p["category"],
                "description": p.get("description", ""),
                "images": p.get("images", "Not Found"),
                "stock" : p.get("stock", "0"),
                "rating" : p.get("rating", "0"),
                "reviews" : p.get("reviews", "0"),
                "createdAt": p.get("createdAt", ""),
                "updatedAt": p.get("updatedAt", ""),
                "isActive": p.get("isActive", True)
            })

    user_data = []
    for u in users:
        for history in u.get("history", []):
            user_data.append({
                "user_id": str(u["_id"]),
                "ProdID": str(history.get("productId", "")),
                "Event": history.get("event", ""),
                "Timestamp": history.get("timestamp", "")
            })

    df_products = pd.DataFrame(product_data)
    print(df_products.head())

    return df_products


def content_based_recommendations(df, item_name, top_n=10):
    if item_name not in df['name'].values:
        print(f"Item '{item_name}' not found in the training data.")
        return pd.DataFrame()

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix_content = tfidf_vectorizer.fit_transform(df['description'].fillna(''))

    cosine_similarities_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)
    item_index = df[df['name'] == item_name].index[0]

    similar_items = list(enumerate(cosine_similarities_content[item_index]))
    similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)

    top_similar_items = similar_items[1:top_n+1]
    recommended_item_indices = [x[0] for x in top_similar_items]

    recommended_items_details = df.iloc[recommended_item_indices][['productID', 'name', 'price', 'category', 'description', 'images', 'rating', 'reviews']]
    
    return recommended_items_details

def collaborative_filtering_recommendations(df, target_user_id, top_n=10):
    user_item_matrix = df.pivot_table(index='productID', columns='ProdID', values='Rating', aggfunc='mean').fillna(0)
    user_similarity = cosine_similarity(user_item_matrix)
    target_user_index = user_item_matrix.index.get_loc(target_user_id)
    user_similarities = user_similarity[target_user_index]
    similar_users_indices = user_similarities.argsort()[::-1][1:]
    recommended_items = []

    for user_index in similar_users_indices:
        rated_by_similar_user = user_item_matrix.iloc[user_index]
        not_rated_by_target_user = (rated_by_similar_user == 0) & (user_item_matrix.iloc[target_user_index] == 0)
        recommended_items.extend(user_item_matrix.columns[not_rated_by_target_user][:top_n])

    recommended_items_details = df[df['productID'].isin(recommended_items)][['name', 'reviews', 'category', 'images', 'rating']]
    return recommended_items_details.head(10)

def hybrid_recommendation_system(df, target_user_id, item_name, top_n=10):
    content_recommendations = content_based_recommendations(df, item_name, top_n)
    collaborative_recommendations = collaborative_filtering_recommendations(df, target_user_id)
    combined_recommendations = pd.concat([content_recommendations, collaborative_recommendations]).drop_duplicates()
    return combined_recommendations

def rating_based_recommendation_system(df):
    group_columns = [col for col in df.columns if col != 'rating']
    grouped_df = df.groupby(group_columns)['rating'].mean().reset_index()
    sorted_df = grouped_df.sort_values(by='rating', ascending=False)
    top_10 = sorted_df.head(10)
    return top_10

def get_closest_match(user_input, all_product_names):
    match, score = process.extractOne(user_input, all_product_names)
    return match if score > 60 else None  # adjust threshold as needed

def als_recommendation(user_id, user_history=None):
    # df = pd.read_csv(r"D:\College\SEM 5\LAB\SE\dataset\data.csv", nrows=10000)
    _, df = making_data() 

    event_weights = {
        'view': 1.0,
        'cart': 3.0,
        'add_to_cart': 3.0,
        'purchase': 5.0
    }

    df['weight'] = df['event'].map(event_weights).fillna(0.0).astype(float)
    user_id = 5

    if user_history is not None:
        user_history['event'] = user_history['event'].astype(str).str.lower()
        user_history['weight'] = user_history['event'].map(event_weights).fillna(0.0).astype(float)

    if user_id not in df['user_id'].unique():
        df = pd.concat([df, user_history], ignore_index=True)
        df['event'] = df['event'].astype(str).str.lower()
        df['weight'] = df['event'].map(event_weights).fillna(0.0).astype(float)

    agg = df.groupby(['user_id', 'productID'])['weight'].sum().reset_index()
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    user_encoder.fit(agg['user_id'])
    item_encoder.fit(agg['productID'])
    agg['user_idx'] = user_encoder.transform(agg['user_id'])
    agg['item_idx'] = item_encoder.transform(agg['productID'])

    interactions = coo_matrix(
        (agg['weight'].astype(float), (agg['user_idx'], agg['item_idx'])),
        shape=(len(user_encoder.classes_), len(item_encoder.classes_))
    )

    model = AlternatingLeastSquares(factors=20, regularization=0.1, iterations=30)
    model.fit(interactions.T)

    return model, user_encoder, item_encoder, interactions

def get_top_popular_purchases(user_id, df, N=5):
    purchases_df = df[df['event'] == 'purchase']
    purchase_counts = purchases_df['productID'].value_counts()
    user_df = df[df['user_id'] == user_id]
    user_product_popularity = user_df.groupby('productID')['weight'].sum().sort_values(ascending=False)
    return user_product_popularity.index[:N].tolist()

def get_als_recommendations(user_id, model, user_encoder, item_encoder, interactions, N=5):
    if user_id not in user_encoder.classes_:
        print(f"User {user_id} not found in training data.")
        return []
    user_idx = int(user_encoder.transform([user_id])[0])
    user_items = interactions.tocsr()[user_idx]
    recommended = model.recommend(user_idx, user_items, N=N, filter_already_liked_items=True)

    if isinstance(recommended, tuple) and len(recommended) == 2:
        items = np.asarray(recommended[0])
    elif isinstance(recommended, (list, np.ndarray)) and len(recommended) > 0 and isinstance(recommended[0], (list, tuple, np.ndarray)):
        items = np.array([r[0] for r in recommended])
    else:
        items = np.array([])

    if items.size == 0:
        return []
    return item_encoder.inverse_transform(items.astype(int))

def combined_recommendations(user_id, model, user_encoder, item_encoder, interactions, df, N=10):
    half = N // 2
    als_recs = get_als_recommendations(user_id, model, user_encoder, item_encoder, interactions, N=half)
    popular_recs = get_top_popular_purchases(user_id, df, N=E)
    combined = list(als_recs) + [item for item in popular_recs if item not in als_recs]
    return combined[:N]

