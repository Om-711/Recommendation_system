import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize, LabelEncoder
from fuzzywuzzy import process
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares
from pymongo import MongoClient
from collections import defaultdict, Counter

# Try importing association rule mining library
try:
    from mlxtend.frequent_patterns import apriori, association_rules
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False

MONGO_URL = "mongodb+srv://arshadmansuri1825:u1AYlNbjuA5FpHbb@cluster1.2majmfd.mongodb.net/ECommerce"

# ============ DATA LOADING ============

def making_data():
    """Load products and user interactions from MongoDB"""
    client = MongoClient(MONGO_URL)
    db = client["ECommerce"]
    
    # Get active products only
    products = []
    for p in db["products"].find({"isActive": True}):
        # Handle reviews field (can be int or list)
        reviews = p.get("reviews", 0)
        review_count = len(reviews) if isinstance(reviews, list) else int(reviews) if reviews else 0
        
        products.append({
            "productID": str(p["_id"]),
            "name": p.get("name", ""),
            "price": float(p.get("price", 0)),
            "category": p.get("category", ""),
            "description": p.get("description", ""),
            "images": p.get("images", ""),
            "stock": int(p.get("stock", 0)),
            "rating": float(p.get("rating", 0)),
            "reviews": review_count,
            "createdAt": p.get("createdAt", ""),
            "updatedAt": p.get("updatedAt", ""),
            "isActive": True
        })
    
    # Get user interactions
    users = list(db["users"].find())
    orders = list(db["orders"].find())
    
    interactions = []
    for user in users:
        user_id = str(user["_id"])
        
        # Browsing history (views, cart)
        for h in user.get("history", []):
            event_data = h.get("event", {})
            event_type = event_data.get("type") if isinstance(event_data, dict) else str(event_data)
            
            interactions.append({
                "user_id": user_id,
                "productID": str(h.get("productID", "")),
                "event": event_type or "view",
                "Timestamp": h.get("time", ""),
                "duration": h.get("duration", 0) / 1000
            })
        
        # Actual purchases from orders
        user_order_ids = [str(oid) for oid in user.get("orders", [])]
        for order in orders:
            if str(order.get("_id")) in user_order_ids:
                for item in order.get("items", []):
                    pid = str(item.get("product", ""))
                    if pid:
                        interactions.append({
                            "user_id": user_id,
                            "productID": pid,
                            "event": "purchase",
                            "Timestamp": order.get("createdAt", ""),
                            "duration": 0
                        })
    
    return pd.DataFrame(products), pd.DataFrame(interactions)


# ============ HELPER FUNCTIONS ============

def get_complementary_keywords(item_name):
    """Get items that go well together (like bread and butter)"""
    item_lower = item_name.lower()
    
    pairs = {
        'butter': ['bread', 'toast', 'jam'], 'bread': ['butter', 'jam', 'cheese'],
        'milk': ['cereal', 'coffee', 'cookies'], 'coffee': ['milk', 'sugar', 'cream'],
        'tea': ['milk', 'sugar', 'biscuit'], 'rice': ['dal', 'curry', 'oil'],
        'pasta': ['sauce', 'cheese', 'oil'], 'egg': ['bread', 'butter', 'cheese'],
        'yogurt': ['honey', 'granola'], 'potato': ['onion', 'oil']
    }
    
    for key, items in pairs.items():
        if key in item_lower:
            return items
    return []


def get_closest_match(user_input, all_names):
    """Find closest product name match (handles typos)"""
    match, score = process.extractOne(user_input, all_names)
    return match if score > 60 else None


# ============ CONTENT-BASED RECOMMENDATIONS ============

def content_based_recommendations_improved(df, item_name, top_n=10):
    """Recommend similar products based on name, category, and price"""
    if item_name not in df['name'].values:
        return pd.DataFrame()
    
    df = df.copy()
    
    # Prepare text fields
    df['name_text'] = df['name'].fillna('').astype(str)
    df['desc_text'] = df['description'].fillna('').astype(str)
    df['cat_text'] = df['category'].fillna('').astype(str)
    
    # Calculate text similarities
    name_vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 6)).fit_transform(df['name_text'])
    desc_vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2)).fit_transform(df['desc_text'])
    cat_vec = TfidfVectorizer().fit_transform(df['cat_text'])
    
    idx = df[df['name'] == item_name].index[0]
    
    name_sim = cosine_similarity(name_vec[idx], name_vec).flatten()
    desc_sim = cosine_similarity(desc_vec[idx], desc_vec).flatten()
    cat_sim = cosine_similarity(cat_vec[idx], cat_vec).flatten()
    
    # Price similarity (prefer similar price range)
    prices = df['price'].values
    price_diff = np.abs(prices - prices[idx])
    price_sim = 1.0 - np.clip(price_diff / (prices.max() + 1), 0, 1)
    
    # Boost complementary items
    keywords = get_complementary_keywords(item_name)
    comp_boost = np.array([
        sum(1.5 if kw in name.lower() else 0 for kw in keywords)
        for name in df['name_text']
    ])
    comp_boost = comp_boost / (comp_boost.max() + 1e-9)
    
    # Final score (weighted combination)
    score = (0.3 * name_sim + 0.15 * desc_sim + 0.25 * cat_sim + 
             0.1 * price_sim + 0.2 * comp_boost)
    
    df['score'] = score
    return df[df['name'] != item_name].sort_values('score', ascending=False).head(top_n)


# ============ COLLABORATIVE FILTERING ============

def collaborative_filtering_recommendations(df_user, df_product, user_id, top_n=10, category_boost=True):
    """Recommend based on what similar users purchased"""
    
    # Event weights (purchases matter most)
    weights = {'purchase': 10.0, 'add_to_cart': 3.0, 'cart': 3.0, 'view': 1.0}
    
    if df_user.empty or user_id not in df_user['user_id'].unique():
        return pd.DataFrame()
    
    try:
        # Clean data
        df_clean = df_user[df_user['productID'].notna() & (df_user['productID'] != '')].copy()
        df_clean['weight'] = df_clean['event'].map(weights).fillna(0.5)
        
        # Create user-product matrix
        user_item = df_clean.pivot_table(
            index='user_id', columns='productID', values='weight', aggfunc='sum'
        ).fillna(0)
        
        if len(user_item) < 2:
            return pd.DataFrame()
        
        # Normalize and find similar users
        matrix_norm = pd.DataFrame(
            normalize(user_item, axis=1, norm='l2'),
            index=user_item.index, columns=user_item.columns
        )
        
        similarity = cosine_similarity(matrix_norm)
        user_idx = matrix_norm.index.get_loc(user_id)
        similar_users = similarity[user_idx].argsort()[::-1][1:11]  # Top 10 similar users
        
        # Get items the target user hasn't tried
        target_items = set(user_item.columns[user_item.iloc[user_idx] > 0])
        scores = {}
        
        for sim_user_idx in similar_users:
            sim_score = similarity[user_idx][sim_user_idx]
            user_items = user_item.iloc[sim_user_idx]
            
            # Find new items this similar user liked
            new_items = user_items[(user_items > 0) & (user_item.iloc[user_idx] == 0)]
            
            for item, weight in new_items.items():
                scores[item] = scores.get(item, 0) + sim_score * weight
        
        if not scores:
            return pd.DataFrame()
        
        # Category-aware filtering if enabled
        if category_boost:
            # Find user's preferred categories
            user_history = df_user[df_user['user_id'] == user_id]
            user_with_cat = user_history.merge(
                df_product[['productID', 'category']], on='productID', how='left'
            )
            user_with_cat['weight'] = user_with_cat['event'].map(weights).fillna(0.5)
            
            cat_scores = user_with_cat.groupby('category')['weight'].sum()
            top_categories = set(cat_scores.nlargest(3).index)
            
            # Boost scores for preferred categories
            for product_id in scores:
                product = df_product[df_product['productID'] == product_id]
                if not product.empty and product.iloc[0]['category'] in top_categories:
                    scores[product_id] *= 1.5
        
        # Get top recommendations
        top_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        results = []
        for product_id, _ in top_items:
            product = df_product[df_product['productID'] == product_id]
            if not product.empty:
                results.append(product.iloc[0])
        
        return pd.DataFrame(results) if results else pd.DataFrame()
    
    except Exception as e:
        print(f"Collaborative filtering error: {e}")
        return pd.DataFrame()


# ============ HYBRID SYSTEM ============

def hybrid_recommendation_system(df_product, df_user, user_id, item_name, top_n=20):
    """Mix content-based and collaborative (70-30 split)"""
    try:
        # Get both types of recommendations
        content_recs = content_based_recommendations_improved(df_product, item_name, top_n * 2)
        
        if len(content_recs) >= top_n:
            return content_recs.head(top_n)
        
        collab_recs = collaborative_filtering_recommendations(df_user, df_product, user_id, top_n)
        
        # Fallback to whichever works
        if content_recs.empty:
            return collab_recs.head(top_n)
        if collab_recs.empty:
            return content_recs.head(top_n)
        
        # Combine 70% content + 30% collaborative
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


# ============ RATING-BASED ============

def rating_based_recommendation_system(df):
    """Get top-rated products"""
    try:
        # Simple sort by rating and reviews
        return df.sort_values(by=['rating', 'reviews'], ascending=False).head(10)
    except Exception as e:
        print(f"Rating error: {e}")
        return df.head(10)


# ============ ALS (MATRIX FACTORIZATION) ============

def train_als_model(df_user=None):
    """Train ALS model for personalized recommendations"""
    if df_user is None:
        _, df_user = making_data()
    
    # Clean data
    df = df_user[
        (df_user['productID'].notna()) & 
        (df_user['productID'] != '') & 
        (df_user['productID'].astype(str).str.strip() != '')
    ].copy()
    
    if df.empty:
        return None, None, None, None
    
    # Weight different events
    weights = {'view': 1.0, 'cart': 3.0, 'add_to_cart': 3.0, 'purchase': 10.0}
    df['weight'] = df['event'].map(weights).fillna(0.0)
    
    # Aggregate weights
    agg = df.groupby(['user_id', 'productID'])['weight'].sum().reset_index()
    agg['user_id'] = agg['user_id'].astype(str)
    agg['productID'] = agg['productID'].astype(str)
    
    # Encode to numeric indices
    user_enc = LabelEncoder()
    item_enc = LabelEncoder()
    agg['user_idx'] = user_enc.fit_transform(agg['user_id'])
    agg['item_idx'] = item_enc.fit_transform(agg['productID'])
    
    # Create sparse matrix
    user_item_csr = coo_matrix(
        (agg['weight'].astype(float), (agg['user_idx'], agg['item_idx'])),
        shape=(len(user_enc.classes_), len(item_enc.classes_))
    ).tocsr()
    
    # Train model
    model = AlternatingLeastSquares(factors=20, regularization=0.1, iterations=30, random_state=42)
    model.fit(user_item_csr.T.tocsr())
    
    return model, user_enc, item_enc, user_item_csr


def get_als_recommendations(user_id, model, user_enc, item_enc, interactions, N=10):
    """Get personalized recommendations using trained ALS model"""
    if model is None or user_enc is None or user_id not in user_enc.classes_:
        return []
    
    try:
        user_idx = int(user_enc.transform([user_id])[0])
        
        # Find items user already interacted with
        user_row = interactions[user_idx].toarray().flatten()
        already_seen = set(np.where(user_row > 0)[0])
        
        if len(already_seen) >= len(item_enc.classes_):
            return []
        
        # Calculate scores for all items
        user_factor = model.item_factors[user_idx]
        item_factors = model.user_factors
        scores = item_factors.dot(user_factor)
        
        # Get top N new items
        item_scores = [(idx, score) for idx, score in enumerate(scores) if idx not in already_seen]
        item_scores.sort(key=lambda x: x[1], reverse=True)
        top_items = item_scores[:N]
        
        # Convert back to product IDs
        item_indices = [idx for idx, _ in top_items]
        return item_enc.inverse_transform(item_indices).tolist()
    
    except Exception as e:
        print(f"ALS error: {e}")
        return []


# ============ FREQUENTLY BOUGHT TOGETHER ============

def build_frequent_itemsets(min_support=0.01, min_confidence=0.3):
    """Build association rules using Apriori algorithm"""
    if not MLXTEND_AVAILABLE:
        return pd.DataFrame(), {}
    
    try:
        client = MongoClient(MONGO_URL)
        db = client["ECommerce"]
        orders = list(db["orders"].find())
        
        if not orders:
            return pd.DataFrame(), {}
        
        # Extract transactions (orders with 2+ items)
        transactions = []
        for order in orders:
            items = [str(item.get("product", "")) for item in order.get("items", [])]
            items = [pid for pid in items if pid and pid != ""]
            if len(items) >= 2:
                transactions.append(items)
        
        if len(transactions) < 10:
            return pd.DataFrame(), {}
        
        # Create one-hot encoded basket
        all_products = set(pid for trans in transactions for pid in trans)
        basket = pd.DataFrame(0, index=range(len(transactions)), columns=list(all_products))
        
        for idx, trans in enumerate(transactions):
            for product in trans:
                basket.at[idx, product] = 1
        
        # Run Apriori
        frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)
        
        if frequent_itemsets.empty:
            frequent_itemsets = apriori(basket, min_support=0.005, use_colnames=True)
        
        if frequent_itemsets.empty:
            return pd.DataFrame(), {}
        
        # Generate association rules
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        
        if rules.empty:
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
        
        # Build lookup dictionary
        product_pairs = defaultdict(list)
        
        for _, rule in rules.iterrows():
            antecedents = list(rule['antecedents'])
            consequents = list(rule['consequents'])
            
            if len(antecedents) == 1:
                product_id = antecedents[0]
                for consequent in consequents:
                    product_pairs[product_id].append({
                        'product_id': consequent,
                        'confidence': rule['confidence'],
                        'lift': rule['lift']
                    })
        
        # Sort by score
        for product_id in product_pairs:
            product_pairs[product_id] = sorted(
                product_pairs[product_id],
                key=lambda x: x['confidence'] * x['lift'],
                reverse=True
            )
        
        return rules, dict(product_pairs)
    
    except Exception as e:
        print(f"Apriori error: {e}")
        return pd.DataFrame(), {}


def get_co_purchased_products_fallback(product_id, df_product, top_n=6):
    """Fallback: Find products in same orders (simple co-occurrence)"""
    try:
        client = MongoClient(MONGO_URL)
        db = client["ECommerce"]
        orders = list(db["orders"].find())
        
        co_occurrence = Counter()
        
        for order in orders:
            items = [str(item.get("product", "")) for item in order.get("items", []) if item.get("product")]
            
            if product_id in items:
                for pid in items:
                    if pid != product_id:
                        co_occurrence[pid] += 1
        
        if not co_occurrence:
            return pd.DataFrame()
        
        # Get top N
        top_products = co_occurrence.most_common(top_n)
        
        results = []
        for pid, count in top_products:
            product_info = df_product[df_product['productID'] == pid]
            if not product_info.empty:
                product_dict = product_info.iloc[0].to_dict()
                product_dict['co_occurrence_count'] = count
                results.append(product_dict)
        
        return pd.DataFrame(results) if results else pd.DataFrame()
    
    except Exception as e:
        print(f"Fallback error: {e}")
        return pd.DataFrame()


def get_frequently_bought_together(product_id, df_product, top_n=6, min_support=0.01):
    """Get products frequently bought together (with fallback)"""
    try:
        # Try association rules first
        rules, product_pairs = build_frequent_itemsets(min_support=min_support, min_confidence=0.2)
        
        if product_id in product_pairs:
            recommendations = product_pairs[product_id][:top_n]
            
            results = []
            for rec in recommendations:
                rec_product_id = rec['product_id']
                product_info = df_product[df_product['productID'] == rec_product_id]
                
                if not product_info.empty:
                    product_dict = product_info.iloc[0].to_dict()
                    product_dict['confidence'] = rec['confidence']
                    product_dict['lift'] = rec['lift']
                    results.append(product_dict)
            
            if results:
                return pd.DataFrame(results)
        
        # Fallback to co-occurrence
        return get_co_purchased_products_fallback(product_id, df_product, top_n)
    
    except Exception as e:
        print(f"Frequently bought together error: {e}")
        return get_co_purchased_products_fallback(product_id, df_product, top_n)
