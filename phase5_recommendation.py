"""
================================================
PHASE 5: RECOMMENDATION SYSTEM
Tourism Experience Analytics Project
================================================
Goal: Suggest top attractions a user will enjoy

Approaches:
  1. Collaborative Filtering
     — User-item matrix + cosine similarity
     — "Users like you also liked..."
  2. Content-Based Filtering
     — Attraction feature similarity
     — "Because you visited X, try Y..."

Output: Ranked list of attraction recommendations
================================================
Run AFTER phase1_data_preparation.py
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
print("=" * 55)
print("  PHASE 5: RECOMMENDATION SYSTEM")
print("=" * 55)

df = pd.read_csv('data/master_cleaned.csv')
print(f"\nLoaded: {df.shape[0]} rows × {df.shape[1]} columns")

os.makedirs('models', exist_ok=True)

# ─────────────────────────────────────────────
# ─── METHOD 1: COLLABORATIVE FILTERING ───────
# ─────────────────────────────────────────────
print("\n[1/4] Building collaborative filtering model...")

# Build user-item matrix: rows = users, cols = attractions, values = ratings
# Filter to users/attractions with enough interactions for meaningful similarity
min_user_ratings    = 3   # user must have rated at least 3 attractions
min_attr_ratings    = 5   # attraction must have at least 5 ratings

user_counts = df['UserId'].value_counts()
attr_counts = df['AttractionId'].value_counts()

active_users = user_counts[user_counts >= min_user_ratings].index
active_attrs = attr_counts[attr_counts >= min_attr_ratings].index

df_filtered = df[df['UserId'].isin(active_users) & df['AttractionId'].isin(active_attrs)]
print(f"  After filtering: {df_filtered['UserId'].nunique():,} users, {df_filtered['AttractionId'].nunique():,} attractions")

# Pivot to user-item matrix
user_item = df_filtered.pivot_table(
    index='UserId', columns='AttractionId', values='Rating', aggfunc='mean'
).fillna(0)

print(f"  User-item matrix shape: {user_item.shape}")

# Compute user-user cosine similarity
# Using sparse matrix for efficiency
sparse_matrix = csr_matrix(user_item.values)
user_similarity = cosine_similarity(sparse_matrix)
user_sim_df = pd.DataFrame(user_similarity, index=user_item.index, columns=user_item.index)

print("  User similarity matrix computed.")

def collaborative_recommend(user_id, top_n=10, n_similar_users=20):
    """
    Recommend attractions for a user based on similar users' ratings.

    Parameters:
        user_id     : int  — the user to recommend for
        top_n       : int  — number of attractions to return
        n_similar_users : int — how many similar users to consider

    Returns:
        DataFrame with AttractionId and predicted score, sorted descending
    """
    if user_id not in user_sim_df.index:
        # Cold-start: return globally popular attractions
        popular = (df_filtered.groupby('AttractionId')['Rating']
                   .mean()
                   .sort_values(ascending=False)
                   .head(top_n))
        return pd.DataFrame({'AttractionId': popular.index, 'Score': popular.values})

    # Attractions this user has already rated
    rated_by_user = user_item.loc[user_id]
    already_visited = rated_by_user[rated_by_user > 0].index.tolist()

    # Find top N similar users (excluding the user themselves)
    similar_users = (user_sim_df[user_id]
                     .drop(index=user_id)
                     .sort_values(ascending=False)
                     .head(n_similar_users)
                     .index)

    # Weighted average of similar users' ratings
    similar_ratings = user_item.loc[similar_users]
    similarity_scores = user_sim_df.loc[similar_users, user_id].values.reshape(-1, 1)

    weighted_ratings = (similar_ratings.values * similarity_scores).sum(axis=0)
    sum_similarities = np.abs(similarity_scores).sum()

    if sum_similarities > 0:
        predicted_scores = weighted_ratings / sum_similarities
    else:
        predicted_scores = weighted_ratings

    scores = pd.Series(predicted_scores, index=user_item.columns)

    # Remove already-visited
    scores = scores.drop(index=[a for a in already_visited if a in scores.index])

    return (scores.sort_values(ascending=False)
                  .head(top_n)
                  .reset_index()
                  .rename(columns={'AttractionId': 'AttractionId', 0: 'Score'}))


# ─────────────────────────────────────────────
# ─── METHOD 2: CONTENT-BASED FILTERING ───────
# ─────────────────────────────────────────────
print("\n[2/4] Building content-based filtering model...")

# Create attraction feature profile
attr_features = df.drop_duplicates('AttractionId')[
    ['AttractionId', 'Attraction', 'AttractionType', 'AttractionVisitCount', 'AttractionAvgRating']
].copy()

# Encode attraction type
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
le = LabelEncoder()
attr_features['AttractionType_enc'] = le.fit_transform(attr_features['AttractionType'].astype(str))

# Normalize numeric features
scaler = MinMaxScaler()
attr_features[['AttractionVisitCount_norm', 'AttractionAvgRating_norm']] = scaler.fit_transform(
    attr_features[['AttractionVisitCount', 'AttractionAvgRating']].fillna(0)
)

# Feature matrix for similarity
feature_cols = ['AttractionType_enc', 'AttractionVisitCount_norm', 'AttractionAvgRating_norm']
attr_feature_matrix = attr_features[feature_cols].values

# Compute attraction-attraction cosine similarity
attr_similarity = cosine_similarity(attr_feature_matrix)
attr_ids = attr_features['AttractionId'].values
attr_sim_df = pd.DataFrame(attr_similarity, index=attr_ids, columns=attr_ids)

print(f"  Attraction similarity matrix: {attr_sim_df.shape}")


def content_based_recommend(user_id, top_n=10):
    """
    Recommend attractions similar to what the user has visited,
    based on attraction features (type, popularity, avg rating).

    Returns:
        DataFrame with AttractionId and Score
    """
    user_history = df[df['UserId'] == user_id][['AttractionId', 'Rating']]

    if user_history.empty:
        # Cold-start: return top-rated attractions
        top = attr_features.sort_values('AttractionAvgRating', ascending=False).head(top_n)
        return top[['AttractionId', 'AttractionAvgRating']].rename(
            columns={'AttractionAvgRating': 'Score'})

    already_visited = user_history['AttractionId'].tolist()

    # Weighted similarity: attractions the user rated highly drive more recommendations
    scores = pd.Series(dtype=float)
    for _, row in user_history.iterrows():
        attr_id = row['AttractionId']
        user_rating = row['Rating']
        if attr_id in attr_sim_df.index:
            sim_scores = attr_sim_df[attr_id] * (user_rating / 5.0)
            scores = scores.add(sim_scores, fill_value=0)

    # Remove visited
    scores = scores.drop(index=[a for a in already_visited if a in scores.index], errors='ignore')

    return (scores.sort_values(ascending=False)
                  .head(top_n)
                  .reset_index()
                  .rename(columns={'index': 'AttractionId', 0: 'Score'}))


# ─────────────────────────────────────────────
# ─── METHOD 3: HYBRID RECOMMENDER ────────────
# ─────────────────────────────────────────────
def hybrid_recommend(user_id, top_n=10, collab_weight=0.6, content_weight=0.4):
    """
    Combine collaborative and content-based scores.
    collab_weight + content_weight should sum to 1.0.
    """
    collab = collaborative_recommend(user_id, top_n=top_n * 2)
    content = content_based_recommend(user_id, top_n=top_n * 2)

    # Normalize scores to 0–1
    if len(collab) > 0 and collab['Score'].max() > 0:
        collab['Score'] = collab['Score'] / collab['Score'].max()
    if len(content) > 0 and content['Score'].max() > 0:
        content['Score'] = content['Score'] / content['Score'].max()

    collab.columns   = ['AttractionId', 'CollabScore']
    content.columns  = ['AttractionId', 'ContentScore']

    merged = pd.merge(collab, content, on='AttractionId', how='outer').fillna(0)
    merged['HybridScore'] = (collab_weight  * merged['CollabScore'] +
                             content_weight * merged['ContentScore'])

    top = merged.sort_values('HybridScore', ascending=False).head(top_n)

    # Attach attraction names
    top = top.merge(
        attr_features[['AttractionId', 'Attraction', 'AttractionType', 'AttractionAvgRating']],
        on='AttractionId', how='left'
    )
    return top


# ─────────────────────────────────────────────
# STEP 3: TEST THE RECOMMENDER
# ─────────────────────────────────────────────
print("\n[3/4] Testing recommender on a sample user...")

sample_user = df['UserId'].value_counts().idxmax()  # most active user
print(f"  Sample user ID: {sample_user}")

print("\n  --- Collaborative Filtering Recommendations ---")
collab_recs = collaborative_recommend(sample_user, top_n=5)
collab_recs = collab_recs.merge(
    attr_features[['AttractionId', 'Attraction', 'AttractionType']],
    on='AttractionId', how='left'
)
print(collab_recs[['AttractionId', 'Attraction', 'AttractionType']].to_string(index=False))

print("\n  --- Content-Based Recommendations ---")
content_recs = content_based_recommend(sample_user, top_n=5)
content_recs = content_recs.merge(
    attr_features[['AttractionId', 'Attraction', 'AttractionType']],
    on='AttractionId', how='left'
)
print(content_recs[['AttractionId', 'Attraction', 'AttractionType']].to_string(index=False))

print("\n  --- Hybrid Recommendations ---")
hybrid_recs = hybrid_recommend(sample_user, top_n=5)
print(hybrid_recs[['Attraction', 'AttractionType', 'AttractionAvgRating', 'HybridScore']].to_string(index=False))

# ─────────────────────────────────────────────
# STEP 4: SAVE MODELS & DATA
# ─────────────────────────────────────────────
print("\n[4/4] Saving recommendation models...")

recommender_data = {
    'user_item_matrix':  user_item,
    'user_sim_df':       user_sim_df,
    'attr_sim_df':       attr_sim_df,
    'attr_features':     attr_features,
    'active_users':      list(active_users),
    'active_attrs':      list(active_attrs),
}
with open('models/recommender.pkl', 'wb') as f:
    pickle.dump(recommender_data, f)
print("  Saved: models/recommender.pkl")

# Save helper functions as a module (imported by Streamlit)
print("  Recommendation functions ready for Streamlit import.")

print(f"\n{'='*55}")
print("  PHASE 5 COMPLETE ✓")
print(f"{'='*55}")
print("\n  Three recommendation modes available:")
print("    1. Collaborative filtering — user similarity")
print("    2. Content-based filtering — attraction features")
print("    3. Hybrid (default) — best of both")
