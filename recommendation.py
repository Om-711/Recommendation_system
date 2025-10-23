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
                "productID": str(history.get("productID", "")),  
                "event": history.get("event", {}).get("type","Not Found"),
                "Timestamp": history.get("time", ""),
                "duration":history.get("duration", 0)/1000 # Convert milli-second to second betwa
            })

    df_products = pd.DataFrame(product_data)
    print(df_products.head())

    df_user = pd.DataFrame(user_data)
    print(df_user.head())

    return df_products, df_user



def get_complementary_keywords(item_name):
    """Map products to complementary items for better recommendations"""
    item_lower = item_name.lower()
    
    complementary_map = {
        'butter': ['bread', 'toast', 'jam', 'margarine', 'cheese', 'milk'],
        'bread': ['butter', 'jam', 'cheese', 'peanut', 'sandwich', 'spread'],
        'milk': ['cereal', 'coffee', 'tea', 'cookies', 'chocolate', 'bread'],
        'coffee': ['milk', 'sugar', 'cream', 'tea', 'biscuit', 'cookie'],
        'tea': ['milk', 'sugar', 'biscuit', 'cookie', 'honey', 'lemon'],
        'rice': ['dal', 'lentil', 'curry', 'oil', 'spice', 'beans'],
        'pasta': ['sauce', 'tomato', 'cheese', 'oil', 'garlic', 'basil'],
        'chicken': ['rice', 'spice', 'oil', 'sauce', 'bread', 'vegetables'],
        'egg': ['bread', 'butter', 'cheese', 'milk', 'bacon', 'toast'],
        'cheese': ['bread', 'butter', 'milk', 'cracker', 'wine', 'pasta'],
        'yogurt': ['honey', 'granola', 'fruit', 'cereal', 'berries'],
        'potato': ['onion', 'oil', 'spice', 'butter', 'cheese', 'cream'],
        'tomato': ['onion', 'garlic', 'pasta', 'sauce', 'basil', 'oil'],
        'oil': ['spice', 'garlic', 'onion', 'rice', 'pasta', 'cooking'],
        'sugar': ['flour', 'butter', 'milk', 'egg', 'vanilla', 'baking'],
        'flour': ['sugar', 'butter', 'egg', 'milk', 'yeast', 'baking']
    }
    
    for key, complements in complementary_map.items():
        if key in item_lower:
            return complements
    return []

def content_based_recommendations_improved(df, item_name, top_n=10, weights=None, verbose=False):
    if weights is None:
        weights = {'name': 0.3, 'desc': 0.15, 'category': 0.25, 'price': 0.1, 'complementary': 0.2}

    if item_name not in df['name'].values:
        if verbose:
            print(f"Item '{item_name}' not found.")
        return pd.DataFrame()

    # Ensure required columns exist
    for col in ['description', 'category', 'price', 'rating']:
        if col not in df.columns:
            df[col] = '' if col != 'price' else 0.0

    df = df.copy()
    df['name_text'] = df['name'].fillna('').astype(str)
    df['desc_text'] = df['description'].fillna('').astype(str)
    df['category_text'] = df['category'].fillna('').astype(str)

    # TF-IDF vectorization
    name_tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 6)).fit_transform(df['name_text'])
    desc_tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1,2)).fit_transform(df['desc_text'])
    cat_tfidf = TfidfVectorizer().fit_transform(df['category_text'])

    item_idx = int(df[df['name'] == item_name].index[0])

    # Calculate similarities
    name_sim = cosine_similarity(name_tfidf[item_idx], name_tfidf).ravel()
    desc_sim = cosine_similarity(desc_tfidf[item_idx], desc_tfidf).ravel()
    cat_sim = cosine_similarity(cat_tfidf[item_idx], cat_tfidf).ravel()

    # Price similarity
    prices = df['price'].fillna(0).astype(float).values
    price_range = prices.max() - prices.min() + 1e-9
    price_sim = 1.0 - np.clip(np.abs(prices - prices[item_idx]) / price_range, 0, 1)

    # Complementary product boost
    complementary_keywords = get_complementary_keywords(item_name)
    complementary_sim = np.zeros(len(df))
    for i, name in enumerate(df['name_text']):
        name_lower = name.lower()
        # Give high score if product name contains complementary keywords
        complementary_sim[i] = sum(1.5 if keyword in name_lower else 0 for keyword in complementary_keywords)
    # Normalize
    if complementary_sim.max() > 0:
        complementary_sim = complementary_sim / complementary_sim.max()

    # Combine scores
    total_weight = sum(weights.values())
    w = {k: v / total_weight for k, v in weights.items()}
    
    final_score = (
        w['name'] * name_sim +
        w['desc'] * desc_sim +
        w['category'] * cat_sim +
        w['price'] * price_sim +
        w['complementary'] * complementary_sim
    )

    df['score'] = final_score
    results = df[df['name'] != item_name].sort_values('score', ascending=False).head(top_n)
    
    return results


def collaborative_filtering_recommendations(df_user, df_product, target_user_id, top_n=10):
    """Simplified collaborative filtering using event weights"""
    return collaborative_filtering_recommendations_event_weighted(df_user, df_product, target_user_id, top_n)

def collaborative_filtering_recommendations_event_weighted(df_user, df_product, target_user_id, top_n=10):
    """Collaborative filtering using event weighting - optimized for sparse data"""
    event_weights = {'purchase': 5.0, 'add_to_cart': 3.0, 'cart': 3.0, 'view': 1.0, 'Not Found': 0.5}
    
    if df_user.empty or target_user_id not in df_user['user_id'].unique():
        return pd.DataFrame()
    
    try:
        # Clean and weight data
        df_clean = df_user[df_user['productID'].notna() & (df_user['productID'] != '')].copy()
        df_clean['weight'] = df_clean['event'].map(event_weights).fillna(0.5)
        
        # Create normalized user-item matrix
        user_item_matrix = df_clean.pivot_table(
            index='user_id', columns='productID', values='weight', aggfunc='sum'
        ).fillna(0)
        
        if len(user_item_matrix) < 2:
            return pd.DataFrame()
        
        from sklearn.preprocessing import normalize
        matrix_norm = pd.DataFrame(
            normalize(user_item_matrix, axis=1, norm='l2'),
            index=user_item_matrix.index, columns=user_item_matrix.columns
        )
        
        # Find similar users and get recommendations
        user_similarity = cosine_similarity(matrix_norm)
        target_idx = matrix_norm.index.get_loc(target_user_id)
        similar_users = user_similarity[target_idx].argsort()[::-1][1:11]  # Top 10 similar users
        
        target_items = set(user_item_matrix.columns[user_item_matrix.iloc[target_idx] > 0])
        recommended_scores = {}
        
        for user_idx in similar_users:
            similarity = user_similarity[target_idx][user_idx]
            user_items = user_item_matrix.iloc[user_idx]
            new_items = set(user_item_matrix.columns[(user_items > 0) & (user_item_matrix.iloc[target_idx] == 0)])
            
            for item in new_items:
                score = similarity * user_items[item]
                recommended_scores[item] = recommended_scores.get(item, 0) + score
        
        # Get top N items and product details
        top_items = [item for item, _ in sorted(recommended_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]]
        return df_product[df_product['productID'].isin(top_items)].drop_duplicates(subset=['productID']).head(top_n)
        
    except Exception as e:
        print(f"Collaborative filtering error: {e}")
        return pd.DataFrame()

def hybrid_recommendation_system(df_product, df_user, target_user_id, item_name, top_n=20):
    """
    Hybrid recommendation combining content-based (with complementary products) 
    and collaborative filtering. Prioritizes content-based for better relevance.
    """
    try:
        # Get content-based recommendations (includes complementary products boost)
        content_recs = content_based_recommendations_improved(df_product, item_name, top_n * 2)
        
        # If we have enough content recommendations, prioritize them
        if len(content_recs) >= top_n:
            return content_recs.head(top_n)
        
        # Otherwise, supplement with collaborative filtering
        collab_recs = collaborative_filtering_recommendations_event_weighted(
            df_user, df_product, target_user_id, top_n=top_n
        )
        
        if content_recs.empty:
            return collab_recs.head(top_n)
        if collab_recs.empty:
            return content_recs.head(top_n)
        
        # Merge: 70% content-based (complementary products), 30% collaborative
        content_count = int(top_n * 0.7)
        collab_count = top_n - content_count
        
        combined = pd.concat([
            content_recs.head(content_count),
            collab_recs[~collab_recs['productID'].isin(content_recs['productID'])].head(collab_count)
        ]).drop_duplicates(subset=['productID']).head(top_n)
        print(f"Combined recommendations: {len(combined)}")
        
        return combined
    
    except Exception as e:
        print(f"Hybrid recommendation error: {e}")
        return content_based_recommendations_improved(df_product, item_name, top_n)

def rating_based_recommendation_system(df):
    group_columns = [col for col in df.columns if col != 'rating']
    grouped_df = df.groupby(group_columns)['rating'].mean().reset_index()
    sorted_df = grouped_df.sort_values(by='rating', ascending=False)
    top_10 = sorted_df.head(10)
    return top_10

def get_closest_match(user_input, all_product_names):
    match, score = process.extractOne(user_input, all_product_names)
    return match if score > 60 else None  # adjust threshold as needed

def train_als_model(df_user=None):
    """
    Train ALS model once and return model + encoders.
    This should be called at startup, not on every request.
    """
    if df_user is None:
        _, df_user = making_data()
    
    # Clean data - remove empty productIDs
    empty_mask = (df_user['productID'].isna()) | (df_user['productID'] == '') | (df_user['productID'].astype(str).str.strip() == '')
    df = df_user[~empty_mask].copy()
    
    if df.empty:
        print(" No valid user interaction data for ALS training")
        return None, None, None, None

    event_weights = {
        'view': 1.0,
        'cart': 3.0,
        'add_to_cart': 3.0,
        'purchase': 5.0
    }

    df['weight'] = df['event'].map(event_weights).fillna(0.0).astype(float)

    agg = df.groupby(['user_id', 'productID'])['weight'].sum().reset_index()
    agg['user_id'] = agg['user_id'].astype(str)
    agg['productID'] = agg['productID'].astype(str)

    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    user_encoder.fit(agg['user_id'])
    item_encoder.fit(agg['productID'])
    agg['user_idx'] = user_encoder.transform(agg['user_id'])
    agg['item_idx'] = item_encoder.transform(agg['productID'])

    # Create user-item interaction matrix (users x items)
    user_item_matrix = coo_matrix(
        (agg['weight'].astype(float), (agg['user_idx'], agg['item_idx'])),
        shape=(len(user_encoder.classes_), len(item_encoder.classes_))
    )

    print(f"Training ALS model with {len(user_encoder.classes_)} users and {len(item_encoder.classes_)} items...")
    print(f"User-item matrix shape: {user_item_matrix.shape} (users x items)")
    
    # implicit library's ALS.fit() expects item-user matrix (items x users)
    # Convert to CSR for efficient operations
    user_item_csr = user_item_matrix.tocsr()
    item_user_csr = user_item_csr.T.tocsr()  # Transpose to items x users
    
    print(f"Item-user matrix shape: {item_user_csr.shape} (items x users)")
    
    model = AlternatingLeastSquares(factors=20, regularization=0.1, iterations=30)
    model.fit(item_user_csr)
    print("✓ ALS model trained successfully")
    print(f"  User factors shape: {model.user_factors.shape}")
    print(f"  Item factors shape: {model.item_factors.shape}")

    # Return both matrices for proper recommendation lookup
    return model, user_encoder, item_encoder, user_item_csr

def als_recommendation(user_id, user_history=None):
    """
    DEPRECATED: This function retrains the model every time.
    Use pretrained model from main.py instead.
    Kept for backward compatibility.
    """
    _, df = making_data() 
    return train_als_model(df)

def get_top_popular_purchases(user_id, df, N=5):
    purchases_df = df[df['event'] == 'purchase']
    purchase_counts = purchases_df['productID'].value_counts()
    user_df = df[df['user_id'] == user_id]
    user_product_popularity = user_df.groupby('productID')['weight'].sum().sort_values(ascending=False)
    return user_product_popularity.index[:N].tolist()

def get_als_recommendations(user_id, model, user_encoder, item_encoder, user_item_interactions, N=10):
    """
    Get ALS-based recommendations for a user.
    
    Args:
        user_id: The user ID to get recommendations for
        model: Trained ALS model
        user_encoder: LabelEncoder for users
        item_encoder: LabelEncoder for items
        user_item_interactions: CSR matrix of shape (users x items)
        N: Number of recommendations to return
        
    Returns:
        List of product IDs (strings)
    """
    if model is None or user_encoder is None:
        print("⚠ ALS model not available")
        return []
        
    if user_id not in user_encoder.classes_:
        print(f"⚠ User {user_id} not found in ALS training data.")
        return []
        
    try:
        # Get user index
        user_idx = int(user_encoder.transform([user_id])[0])
        
        # Get user's interaction vector from the user-item matrix
        # user_item_interactions is already in CSR format and has shape (users x items)
        user_items = user_item_interactions[user_idx]
        
        print(f"Getting ALS recommendations for user {user_id} (index: {user_idx})")
        print(f"  User has {user_items.nnz} interactions out of {len(item_encoder.classes_)} total items")
        
        # Calculate how many items user has already interacted with
        user_interacted_items = user_items.nnz
        total_items = len(item_encoder.classes_)
        max_recommendations = total_items - user_interacted_items
        
        # Limit N to available recommendations
        actual_N = min(N, max_recommendations)
        
        if actual_N <= 0:
            print(f"⚠ User {user_id} has already interacted with all items")
            return []
        
        # Get recommendations from the model
        # model.recommend expects:
        #   - userid: int index
        #   - user_items: sparse vector of user's interactions (length = num_items)
        #   - N: number of recommendations
        #   - filter_already_liked_items: whether to exclude items user has interacted with
        recommended = model.recommend(
            userid=user_idx, 
            user_items=user_items, 
            N=actual_N, 
            filter_already_liked_items=True
        )
        
        print(f"  Model returned: {type(recommended)}")
        
        # Handle different return formats from implicit library
        # Newer versions return (indices, scores), older versions might return just indices
        if isinstance(recommended, tuple) and len(recommended) == 2:
            item_indices = np.asarray(recommended[0])
            scores = np.asarray(recommended[1])
            print(f"  Got {len(item_indices)} recommendations with scores")
        elif isinstance(recommended, (list, np.ndarray)):
            item_indices = np.asarray(recommended)
            scores = None
            print(f"  Got {len(item_indices)} recommendations without scores")
        else:
            print(f"⚠ Unexpected recommendation format: {type(recommended)}")
            return []

        if item_indices.size == 0:
            print("⚠ No recommendations returned from model")
            return []
        
        # Ensure indices are within bounds
        valid_mask = (item_indices >= 0) & (item_indices < len(item_encoder.classes_))
        valid_indices = item_indices[valid_mask]
        
        if len(valid_indices) == 0:
            print("⚠ No valid item indices after filtering")
            return []
        
        # Convert indices back to product IDs
        product_ids = item_encoder.inverse_transform(valid_indices.astype(int)).tolist()
        print(f"✓ Returning {len(product_ids)} product recommendations")
        
        return product_ids
        
    except Exception as e:
        print(f"❌ Error in ALS recommendations: {e}")
        import traceback
        traceback.print_exc()
        return []

def combined_recommendations(user_id, model, user_encoder, item_encoder, interactions, df, N=10):

    half = max(1, N // 2)
    als_recs = get_als_recommendations(user_id, model, user_encoder, item_encoder, interactions, N=half)

    popular_recs = get_top_popular_purchases(user_id, df, N=N)

    combined = list(als_recs) + [item for item in popular_recs if item not in als_recs]

    return combined[:N]
