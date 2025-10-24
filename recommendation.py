import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize, LabelEncoder
from fuzzywuzzy import process
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares
from pymongo import MongoClient

MONGO_URL = "mongodb+srv://arshadmansuri1825:u1AYlNbjuA5FpHbb@cluster1.2majmfd.mongodb.net/ECommerce"

def making_data():
    client = MongoClient(MONGO_URL)
    db = client["ECommerce"]
    
    products = list(db["products"].find())
    users = list(db["users"].find())
    orders = list(db["orders"].find())
    
    # Extract products
    product_data = []
    for p in products:
        if p.get("isActive"):
            product_data.append({
                "productID": str(p["_id"]),
                "name": p["name"],
                "price": p["price"],
                "category": p["category"],
                "description": p.get("description", ""),
                "images": p.get("images", ""),
                "stock": p.get("stock", "0"),
                "rating": p.get("rating", "0"),
                "reviews": p.get("reviews", "0"),
                "createdAt": p.get("createdAt", ""),
                "updatedAt": p.get("updatedAt", ""),
                "isActive": p.get("isActive", True)
            })
    
    # Extract user interactions
    user_data = []
    for u in users:
        user_id = str(u["_id"])
        
        # Get browsing history
        for h in u.get("history", []):
            event = h.get("event", {})
            event_type = event.get("type") if isinstance(event, dict) else str(event)
            
            user_data.append({
                "user_id": user_id,
                "productID": str(h.get("productID", "")),
                "event": event_type or "view",
                "Timestamp": h.get("time", ""),
                "duration": h.get("duration", 0) / 1000
            })
        
        # Get purchases from orders
        user_orders = [str(oid) for oid in u.get("orders", [])]
        for order in orders:
            if str(order.get("_id")) in user_orders:
                for item in order.get("items", []):
                    product_id = str(item.get("product", ""))
                    if product_id:
                        user_data.append({
                            "user_id": user_id,
                            "productID": product_id,
                            "event": "purchase",
                            "Timestamp": order.get("createdAt", ""),
                            "duration": 0
                        })
    
    return pd.DataFrame(product_data), pd.DataFrame(user_data)


def get_complementary_keywords(item_name):
    item_lower = item_name.lower()
    
    complements = {
        'butter': ['bread', 'toast', 'jam', 'cheese', 'milk'],
        'bread': ['butter', 'jam', 'cheese', 'peanut'],
        'milk': ['cereal', 'coffee', 'tea', 'cookies'],
        'coffee': ['milk', 'sugar', 'cream', 'biscuit'],
        'tea': ['milk', 'sugar', 'biscuit', 'honey'],
        'rice': ['dal', 'lentil', 'curry', 'oil'],
        'pasta': ['sauce', 'cheese', 'oil', 'garlic'],
        'egg': ['bread', 'butter', 'cheese', 'milk'],
        'cheese': ['bread', 'butter', 'cracker'],
        'yogurt': ['honey', 'granola', 'fruit'],
        'potato': ['onion', 'oil', 'butter'],
        'tomato': ['onion', 'garlic', 'pasta'],
        'oil': ['spice', 'garlic', 'onion'],
        'sugar': ['flour', 'butter', 'milk'],
        'flour': ['sugar', 'butter', 'egg']
    }
    
    for key, items in complements.items():
        if key in item_lower:
            return items
    return []


def content_based_recommendations_improved(df, item_name, top_n=10, weights=None):
    if weights is None:
        weights = {'name': 0.3, 'desc': 0.15, 'category': 0.25, 'price': 0.1, 'complementary': 0.2}
    
    if item_name not in df['name'].values:
        return pd.DataFrame()
    
    df = df.copy()
    df['name_text'] = df['name'].fillna('').astype(str)
    df['desc_text'] = df['description'].fillna('').astype(str)
    df['category_text'] = df['category'].fillna('').astype(str)
    
    # TF-IDF
    name_vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 6)).fit_transform(df['name_text'])
    desc_vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2)).fit_transform(df['desc_text'])
    cat_vec = TfidfVectorizer().fit_transform(df['category_text'])
    
    idx = int(df[df['name'] == item_name].index[0])
    
    # Calculate similarities
    name_sim = cosine_similarity(name_vec[idx], name_vec).ravel()
    desc_sim = cosine_similarity(desc_vec[idx], desc_vec).ravel()
    cat_sim = cosine_similarity(cat_vec[idx], cat_vec).ravel()
    
    # Price similarity
    prices = df['price'].fillna(0).astype(float).values
    price_range = prices.max() - prices.min() + 1e-9
    price_sim = 1.0 - np.clip(np.abs(prices - prices[idx]) / price_range, 0, 1)
    
    # Complementary boost
    keywords = get_complementary_keywords(item_name)
    comp_sim = np.array([
        sum(1.5 if kw in name.lower() else 0 for kw in keywords)
        for name in df['name_text']
    ])
    if comp_sim.max() > 0:
        comp_sim = comp_sim / comp_sim.max()
    
    # Combine scores
    total_w = sum(weights.values())
    w = {k: v / total_w for k, v in weights.items()}
    
    final_score = (
        w['name'] * name_sim +
        w['desc'] * desc_sim +
        w['category'] * cat_sim +
        w['price'] * price_sim +
        w['complementary'] * comp_sim
    )
    
    df['score'] = final_score
    return df[df['name'] != item_name].sort_values('score', ascending=False).head(top_n)


def collaborative_filtering_recommendations(df_user, df_product, user_id, top_n=10, category_boost=True):
    if category_boost:
        return collaborative_filtering_category_aware(df_user, df_product, user_id, top_n)
    return collaborative_filtering_basic(df_user, df_product, user_id, top_n)


def collaborative_filtering_category_aware(df_user, df_product, user_id, top_n=10):
    weights = {'purchase': 10.0, 'add_to_cart': 3.0, 'cart': 3.0, 'view': 1.0}
    
    if df_user.empty or user_id not in df_user['user_id'].unique():
        return pd.DataFrame()
    
    try:
        # Get user's preferred categories
        user_history = df_user[df_user['user_id'] == user_id].copy()
        user_history['weight'] = user_history['event'].map(weights).fillna(0.5)
        
        user_products = df_product[df_product['productID'].isin(user_history['productID'])]
        if user_products.empty:
            return pd.DataFrame()
        
        user_with_cat = user_history.merge(
            df_product[['productID', 'category']], on='productID', how='left'
        )
        
        cat_scores = user_with_cat.groupby('category')['weight'].sum().sort_values(ascending=False)
        preferred_cats = set(cat_scores.index[:3])
        
        # Get basic collaborative recs
        basic_recs = collaborative_filtering_basic(df_user, df_product, user_id, top_n * 3)
        
        if basic_recs.empty:
            return pd.DataFrame()
        
        # Split by category
        same_cat = basic_recs[basic_recs['category'].isin(preferred_cats)]
        other_cat = basic_recs[~basic_recs['category'].isin(preferred_cats)]
        
        # Combine 70/30
        same_count = int(top_n * 0.7)
        other_count = top_n - same_count
        
        result = pd.concat([
            same_cat.head(same_count),
            other_cat.head(other_count)
        ]).drop_duplicates(subset=['productID']).head(top_n)
        
        return result.reset_index(drop=True)
    
    except Exception as e:
        print(f"Collaborative filtering error: {e}")
        return pd.DataFrame()


def collaborative_filtering_basic(df_user, df_product, user_id, top_n=10):
    weights = {'purchase': 10.0, 'add_to_cart': 3.0, 'cart': 3.0, 'view': 1.0}
    
    if df_user.empty or user_id not in df_user['user_id'].unique():
        return pd.DataFrame()
    
    try:
        df_clean = df_user[df_user['productID'].notna() & (df_user['productID'] != '')].copy()
        df_clean['weight'] = df_clean['event'].map(weights).fillna(0.5)
        
        user_item = df_clean.pivot_table(
            index='user_id', columns='productID', values='weight', aggfunc='sum'
        ).fillna(0)
        
        if len(user_item) < 2:
            return pd.DataFrame()
        
        matrix_norm = pd.DataFrame(
            normalize(user_item, axis=1, norm='l2'),
            index=user_item.index, columns=user_item.columns
        )
        
        similarity = cosine_similarity(matrix_norm)
        target_idx = matrix_norm.index.get_loc(user_id)
        similar_users = similarity[target_idx].argsort()[::-1][1:11]
        
        target_items = set(user_item.columns[user_item.iloc[target_idx] > 0])
        scores = {}
        
        for user_idx in similar_users:
            sim = similarity[target_idx][user_idx]
            items = user_item.iloc[user_idx]
            new_items = set(user_item.columns[(items > 0) & (user_item.iloc[target_idx] == 0)])
            
            for item in new_items:
                scores[item] = scores.get(item, 0) + sim * items[item]
        
        if not scores:
            return pd.DataFrame()
        
        top_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        result = []
        for product_id, _ in top_items:
            product = df_product[df_product['productID'] == product_id]
            if not product.empty:
                result.append(product.iloc[0])
        
        return pd.DataFrame(result).reset_index(drop=True) if result else pd.DataFrame()
    
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()


def hybrid_recommendation_system(df_product, df_user, user_id, item_name, top_n=20):
    try:
        content_recs = content_based_recommendations_improved(df_product, item_name, top_n * 2)
        
        if len(content_recs) >= top_n:
            return content_recs.head(top_n)
        
        collab_recs = collaborative_filtering_basic(df_user, df_product, user_id, top_n)
        
        if content_recs.empty:
            return collab_recs.head(top_n)
        if collab_recs.empty:
            return content_recs.head(top_n)
        
        content_count = int(top_n * 0.7)
        collab_count = top_n - content_count
        
        combined = pd.concat([
            content_recs.head(content_count),
            collab_recs[~collab_recs['productID'].isin(content_recs['productID'])].head(collab_count)
        ]).drop_duplicates(subset=['productID']).head(top_n)
        
        return combined
    
    except Exception as e:
        print(f"Hybrid error: {e}")
        return content_based_recommendations_improved(df_product, item_name, top_n)


def rating_based_recommendation_system(df):
    group_cols = [col for col in df.columns if col != 'rating']
    grouped = df.groupby(group_cols)['rating'].mean().reset_index()
    return grouped.sort_values(by='rating', ascending=False).head(10)


def get_closest_match(user_input, all_names):
    match, score = process.extractOne(user_input, all_names)
    return match if score > 60 else None


def train_als_model(df_user=None):
    if df_user is None:
        _, df_user = making_data()
    
    df = df_user[
        (df_user['productID'].notna()) & 
        (df_user['productID'] != '') & 
        (df_user['productID'].astype(str).str.strip() != '')
    ].copy()
    
    if df.empty:
        return None, None, None, None
    
    weights = {'view': 1.0, 'cart': 3.0, 'add_to_cart': 3.0, 'purchase': 10.0}
    df['weight'] = df['event'].map(weights).fillna(0.0)
    
    agg = df.groupby(['user_id', 'productID'])['weight'].sum().reset_index()
    agg['user_id'] = agg['user_id'].astype(str)
    agg['productID'] = agg['productID'].astype(str)
    
    user_enc = LabelEncoder()
    item_enc = LabelEncoder()
    agg['user_idx'] = user_enc.fit_transform(agg['user_id'])
    agg['item_idx'] = item_enc.fit_transform(agg['productID'])
    
    user_item_csr = coo_matrix(
        (agg['weight'].astype(float), (agg['user_idx'], agg['item_idx'])),
        shape=(len(user_enc.classes_), len(item_enc.classes_))
    ).tocsr()
    
    item_user_csr = user_item_csr.T.tocsr()
    model = AlternatingLeastSquares(factors=20, regularization=0.1, iterations=30, random_state=42)
    model.fit(item_user_csr)
    
    return model, user_enc, item_enc, item_user_csr


def get_als_recommendations(user_id, model, user_enc, item_enc, interactions, N=10):
    if model is None or user_enc is None or user_id not in user_enc.classes_:
        return []
    
    try:
        user_idx = int(user_enc.transform([user_id])[0])
        user_items = interactions[:, user_idx]
        
        if user_items.nnz >= len(item_enc.classes_):
            return []
        
        recommended = model.recommend(
            userid=user_idx,
            user_items=user_items,
            N=min(N, len(item_enc.classes_) - user_items.nnz),
            filter_already_liked_items=True
        )
        
        if isinstance(recommended, tuple) and len(recommended) == 2:
            item_indices = np.asarray(recommended[0])
        elif isinstance(recommended, (list, np.ndarray)):
            item_indices = np.asarray(recommended)
        else:
            return []
        
        if item_indices.size == 0:
            return []
        
        valid_mask = (item_indices >= 0) & (item_indices < len(item_enc.classes_))
        valid_indices = item_indices[valid_mask]
        
        if len(valid_indices) == 0:
            return []
        
        return item_enc.inverse_transform(valid_indices.astype(int)).tolist()
    
    except Exception as e:
        print(f"ALS error: {e}")
        return []
