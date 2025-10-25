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
    order_collection = db["orders"]  # Add orders collection

    products = list(product_collection.find())
    users  = list(user_data_collection.find())
    orders = list(order_collection.find())  # Fetch orders

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
    event_types_found = set()
    users_purchase_count = 0
    
    # Extract data from users collection
    for u in users:
        user_id = str(u["_id"])
        
        # Extract browsing history (views, add_to_cart)
        for history in u.get("history", []):
            event_obj = history.get("event", {})
            if isinstance(event_obj, dict):
                event_type = event_obj.get("type", "Not Found")
            else:
                # If event is stored as a string directly
                event_type = str(event_obj) if event_obj else "Not Found"
            
            event_types_found.add(event_type)
            
            user_data.append({
                "user_id": user_id,
                "productID": str(history.get("productID", "")),  
                "event": event_type,
                "Timestamp": history.get("time", ""),
                "duration":history.get("duration", 0)/1000 # Convert milli-second to second betwa
            })
        
        # Extract purchases from user's orders
        # users.orders contains order IDs (references to orders collection)
        user_order_ids = u.get("orders", [])
        if user_order_ids:
            # Convert ObjectIds to strings for lookup
            user_order_ids_str = [str(oid) for oid in user_order_ids]
            
            # Find matching orders in orders collection
            for order in orders:
                order_id_str = str(order.get("_id"))
                if order_id_str in user_order_ids_str:
                    # Extract items from this order
                    order_items = order.get("items", [])
                    order_date = order.get("createdAt", "")
                    
                    for item in order_items:
                        # item has 'product' field (not productId!)
                        product_id = str(item.get("product", ""))
                        if product_id and product_id != "":
                            user_data.append({
                                "user_id": user_id,
                                "productID": product_id,
                                "event": "purchase",  # This is a purchase!
                                "Timestamp": order_date,
                                "duration": 0
                            })
                            users_purchase_count += 1
                            event_types_found.add("purchase")
    
    print(f"\n[DATA EXTRACTION] Extracted {users_purchase_count} purchases from users' orders")

    df_products = pd.DataFrame(product_data)
    print(f"\n[DATA EXTRACTION] Products loaded: {len(df_products)}")
    print(df_products.head())

    df_user = pd.DataFrame(user_data)
    print(f"\n[DATA EXTRACTION] User interactions loaded: {len(df_user)}")
    print(f"[DATA EXTRACTION] Unique users: {df_user['user_id'].nunique()}")
    print(f"[DATA EXTRACTION] Event types found in database: {event_types_found}")
    print(f"[DATA EXTRACTION] Event type counts: {df_user['event'].value_counts().to_dict()}")
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


def collaborative_filtering_recommendations(df_user, df_product, target_user_id, top_n=10, category_boost=True):
    """
    Simplified collaborative filtering using event weights
    """
    print(f"User data: {df_user.head()}")
    print(f"Product data: {df_product.head()}")
    print(f"Target user ID: {target_user_id}")
    print(f"Top N: {top_n}")
    
    if category_boost:
        return collaborative_filtering_category_aware(df_user, df_product, target_user_id, top_n)
    else:
        return collaborative_filtering_recommendations_event_weighted(df_user, df_product, target_user_id, top_n)

def collaborative_filtering_category_aware(df_user, df_product, target_user_id, top_n=10, category_weight=0.7):
    """
    Category-aware collaborative filtering that prioritizes recommendations from 
    the same categories as user's purchase history.
    
   
    """
    # Purchases from orders collection have highest weight
    event_weights = {'purchase': 10.0, 'add_to_cart': 3.0, 'cart': 3.0, 'view': 1.0, 'Not Found': 0.5}
    
    print(f"\n{'='*10}")
    print(f"CATEGORY-AWARE COLLABORATIVE FILTERING")
    print(f"{'='*10}")
    print(f"Target User: {target_user_id}")
    
    if df_user.empty or target_user_id not in df_user['user_id'].unique():
        print("ERROR: User not found!")
        return pd.DataFrame()
    
    try:
        # Get target user's history and identify their preferred categories
        target_user_history = df_user[df_user['user_id'] == target_user_id]
        user_product_ids = target_user_history['productID'].unique()
        user_products = df_product[df_product['productID'].isin(user_product_ids)]
        
        # Find user's preferred categories (weighted by purchase events)
        target_history_weighted = target_user_history.copy()
        target_history_weighted['weight'] = target_history_weighted['event'].map(event_weights).fillna(0.5)
        
        # Merge with product categories
        target_with_categories = target_history_weighted.merge(
            df_product[['productID', 'category']], 
            on='productID', 
            how='left'
        )
        
        # Calculate category preferences (higher weight for purchases vs views)
        category_scores = target_with_categories.groupby('category')['weight'].sum().sort_values(ascending=False)
        preferred_categories = set(category_scores.index[:3])  # Top 3 categories
        
        print(f"\nUser's Preferred Categories (based on {len(target_user_history)} interactions):")
        for cat, score in category_scores.head(3).items():
            print(f"  - {cat}: {score:.1f} (weighted interaction score)")
        
        # Get standard collaborative filtering recommendations
        collab_recs = collaborative_filtering_recommendations_event_weighted(
            df_user, df_product, target_user_id, top_n=top_n * 3  # Get more candidates
        )
        
        if collab_recs.empty:
            print("\nNo collaborative recommendations available")
            return pd.DataFrame()
        
        # Boost scores for products in preferred categories
        print(f"\nApplying category boost...")
        same_category_recs = collab_recs[collab_recs['category'].isin(preferred_categories)].copy()
        other_category_recs = collab_recs[~collab_recs['category'].isin(preferred_categories)].copy()
        
        print(f"  Same category recommendations: {len(same_category_recs)}")
        print(f"  Other category recommendations: {len(other_category_recs)}")
        
        # Calculate how many from each category
        same_category_count = int(top_n * category_weight)
        other_category_count = top_n - same_category_count
        
        # Combine: prioritize same category, then add others
        final_recs = pd.concat([
            same_category_recs.head(same_category_count),
            other_category_recs.head(other_category_count)
        ]).drop_duplicates(subset=['productID']).head(top_n)
        
        print(f"\nFinal Recommendations ({len(final_recs)} products):")
        print(f"  From preferred categories: {len(final_recs[final_recs['category'].isin(preferred_categories)])}")
        print(f"  From other categories: {len(final_recs[~final_recs['category'].isin(preferred_categories)])}")
        
        if not final_recs.empty:
            print(f"\nTop 5 recommendations:")
            for i, row in final_recs.head(5).iterrows():
                category_marker = "[*]" if row['category'] in preferred_categories else "[ ]"
                print(f"  {category_marker} {row['name']} ({row['category']}) - ${row['price']}")
        
        print(f"{'='*80}\n")
        
        return final_recs.reset_index(drop=True)
        
    except Exception as e:
        print(f"\nERROR in category-aware collaborative filtering: {e}")
        import traceback
        traceback.print_exc()
        print(f"{'='*80}\n")
        return pd.DataFrame()

def collaborative_filtering_recommendations_event_weighted(df_user, df_product, target_user_id, top_n=10):
    """Collaborative filtering using event weighting - optimized for sparse data"""
    # Purchases from orders collection have highest weight
    event_weights = {'purchase': 10.0, 'add_to_cart': 3.0, 'cart': 3.0, 'view': 1.0, 'Not Found': 0.5}
    
    print(f"\n{'='*80}")
    print(f"COLLABORATIVE FILTERING DEBUG")
    print(f"{'='*80}")
    print(f"Target User: {target_user_id}")
    print(f"Total users in data: {df_user['user_id'].nunique()}")
    print(f"Total products in data: {df_product['productID'].nunique()}")
    
    if df_user.empty:
        print("ERROR: User dataframe is empty!")
        return pd.DataFrame()
    
    if target_user_id not in df_user['user_id'].unique():
        print(f"ERROR: User {target_user_id} not found in user data!")
        print(f"Available users: {df_user['user_id'].unique()[:5]}...")
        return pd.DataFrame()
    
    try:
        # Get target user's history
        target_user_history = df_user[df_user['user_id'] == target_user_id]
        print(f"\nTarget user history ({len(target_user_history)} interactions):")
        print(f"  Events: {target_user_history['event'].value_counts().to_dict()}")
        print(f"  Products: {target_user_history['productID'].tolist()[:10]}")
        
        # Show product categories for user's history
        user_product_ids = target_user_history['productID'].unique()
        user_products = df_product[df_product['productID'].isin(user_product_ids)]
        if not user_products.empty:
            print(f"  Categories in history: {user_products['category'].value_counts().to_dict()}")
        
        # Clean and weight data
        df_clean = df_user[df_user['productID'].notna() & (df_user['productID'] != '')].copy()
        print(f"\nAfter cleaning: {len(df_clean)} valid interactions")
        
        df_clean['weight'] = df_clean['event'].map(event_weights).fillna(0.5)
        print(f"Event types found: {df_clean['event'].unique()}")
        print(f"Weights applied: {df_clean.groupby('event')['weight'].first().to_dict()}")
        
        # Create normalized user-item matrix
        user_item_matrix = df_clean.pivot_table(
            index='user_id', columns='productID', values='weight', aggfunc='sum'
        ).fillna(0)
        
        print(f"\nUser-Item Matrix shape: {user_item_matrix.shape}")
        print(f"  Users: {len(user_item_matrix)}")
        print(f"  Products: {len(user_item_matrix.columns)}")
        
        if len(user_item_matrix) < 2:
            print("ERROR: Not enough users for collaborative filtering (need at least 2)")
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
        
        print(f"\nSimilar users found: {len(similar_users)}")
        for i, user_idx in enumerate(similar_users[:5]):
            similar_user_id = user_item_matrix.index[user_idx]
            similarity_score = user_similarity[target_idx][user_idx]
            print(f"  {i+1}. User {similar_user_id}: similarity = {similarity_score:.4f}")
        
        target_items = set(user_item_matrix.columns[user_item_matrix.iloc[target_idx] > 0])
        print(f"\nTarget user has interacted with {len(target_items)} products")
        
        recommended_scores = {}
        
        for user_idx in similar_users:
            similarity = user_similarity[target_idx][user_idx]
            user_items = user_item_matrix.iloc[user_idx]
            new_items = set(user_item_matrix.columns[(user_items > 0) & (user_item_matrix.iloc[target_idx] == 0)])
            
            for item in new_items:
                score = similarity * user_items[item]
                recommended_scores[item] = recommended_scores.get(item, 0) + score
        
        print(f"\nRecommendation candidates: {len(recommended_scores)}")
        
        if not recommended_scores:
            print("WARNING: No recommendations generated (all similar users have same products as target)")
            return pd.DataFrame()
        
        # Get top N items sorted by score and preserve order
        top_items = sorted(recommended_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        print(f"\nTop {len(top_items)} recommendations (by score):")
        for i, (product_id, score) in enumerate(top_items[:5]):
            product_info = df_product[df_product['productID'] == product_id]
            if not product_info.empty:
                product_name = product_info.iloc[0]['name']
                product_category = product_info.iloc[0]['category']
                print(f"  {i+1}. {product_name} ({product_category}) - Score: {score:.4f}")
        
        # Get product details in the correct order
        result_products = []
        for product_id, score in top_items:
            product = df_product[df_product['productID'] == product_id]
            if not product.empty:
                result_products.append(product.iloc[0])
        
        print(f"\nFinal result: {len(result_products)} products returned")
        print(f"{'='*80}\n")
        
        if result_products:
            return pd.DataFrame(result_products).reset_index(drop=True)
        return pd.DataFrame()
        
    except Exception as e:
        print(f"\nERROR in Collaborative filtering: {e}")
        import traceback
        traceback.print_exc()
        print(f"{'='*80}\n")
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
    """Train ALS model and return model + encoders"""
    if df_user is None:
        _, df_user = making_data()
    
    # Clean data
    empty_mask = (df_user['productID'].isna()) | (df_user['productID'] == '') | (df_user['productID'].astype(str).str.strip() == '')
    df = df_user[~empty_mask].copy()
    
    if df.empty:
        print(" No valid user interaction data for ALS training")
        return None, None, None, None

    # Event weights
    event_weights = {'view': 1.0, 'cart': 3.0, 'add_to_cart': 3.0, 'purchase': 50.0}
    df['weight'] = df['event'].map(event_weights).fillna(0.0).astype(float)

    # Aggregate and encode
    agg = df.groupby(['user_id', 'productID'])['weight'].sum().reset_index()
    agg['user_id'] = agg['user_id'].astype(str)
    agg['productID'] = agg['productID'].astype(str)

    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    agg['user_idx'] = user_encoder.fit_transform(agg['user_id'])
    agg['item_idx'] = item_encoder.fit_transform(agg['productID'])

    # Create user-item matrix
    user_item_csr = coo_matrix(
        (agg['weight'].astype(float), (agg['user_idx'], agg['item_idx'])),
        shape=(len(user_encoder.classes_), len(item_encoder.classes_))
    ).tocsr()

    print(f"✓ Training ALS: {len(user_encoder.classes_)} users, {len(item_encoder.classes_)} items")
    
    # Train model with item-user matrix (implicit library convention)
    item_user_csr = user_item_csr.T.tocsr()
    model = AlternatingLeastSquares(factors=20, regularization=0.1, iterations=30, random_state=42)
    model.fit(item_user_csr)
    
    print(f" ALS trained - User factors: {model.user_factors.shape}, Item factors: {model.item_factors.shape}")

    return model, user_encoder, item_encoder, item_user_csr

def als_recommendation(user_id, user_history=None):
    _, df = making_data() 
    return train_als_model(df)

def get_top_popular_purchases(user_id, df, N=5):
    purchases_df = df[df['event'] == 'purchase']
    purchase_counts = purchases_df['productID'].value_counts()
    user_df = df[df['user_id'] == user_id]
    user_product_popularity = user_df.groupby('productID')['weight'].sum().sort_values(ascending=False)
    return user_product_popularity.index[:N].tolist()

def get_als_recommendations(user_id, model, user_encoder, item_encoder, item_user_interactions, N=10):
    """
    Get ALS-based recommendations for a user.
    """
    if model is None or user_encoder is None:
        print(" ALS model not available")
        return []
        
    if user_id not in user_encoder.classes_:
        print(f" User {user_id} not found in ALS training data")
        return []
        
    try:
        user_idx = int(user_encoder.transform([user_id])[0])
        
        # Get user's column from item-user matrix (items x users)
        # This represents the user's interaction vector in the same format as training
        user_items = item_user_interactions[:, user_idx]
        
        print(f" Getting ALS recommendations for user {user_id} (idx: {user_idx})")
        print(f"  User interactions: {user_items.nnz}/{len(item_encoder.classes_)}")
        
        if user_items.nnz >= len(item_encoder.classes_):
            print("⚠ User has interacted with all items")
            return []
        
        # Get recommendations
        recommended = model.recommend(
            userid=user_idx, 
            user_items=user_items, 
            N=min(N, len(item_encoder.classes_) - user_items.nnz), 
            filter_already_liked_items=True
        )
        
        # Parse recommendation results
        if isinstance(recommended, tuple) and len(recommended) == 2:
            item_indices, scores = np.asarray(recommended[0]), np.asarray(recommended[1])
        elif isinstance(recommended, (list, np.ndarray)):
            item_indices = np.asarray(recommended)
        else:
            print(f" Unexpected format: {type(recommended)}")
            return []

        if item_indices.size == 0:
            return []
        
        # Validate and convert indices to product IDs
        valid_mask = (item_indices >= 0) & (item_indices < len(item_encoder.classes_))
        valid_indices = item_indices[valid_mask]
        
        if len(valid_indices) == 0:
            return []
        
        product_ids = item_encoder.inverse_transform(valid_indices.astype(int)).tolist()
        print(f" Returning {len(product_ids)} ALS recommendations")
        
        return product_ids
        
    except Exception as e:
        print(f" ALS error: {e}")
        import traceback
        traceback.print_exc()
        return []

def combined_recommendations(user_id, model, user_encoder, item_encoder, interactions, df, N=10):

    half = max(1, N // 2)
    als_recs = get_als_recommendations(user_id, model, user_encoder, item_encoder, interactions, N=half)

    popular_recs = get_top_popular_purchases(user_id, df, N=N)

    combined = list(als_recs) + [item for item in popular_recs if item not in als_recs]

    return combined[:N]
