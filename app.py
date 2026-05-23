import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder


st.set_page_config(
    page_title="Tourism Analytics",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px; border-radius: 12px; color: white;
    text-align: center; margin: 5px;
}
.metric-value { font-size: 2rem; font-weight: bold; }
.metric-label { font-size: 0.85rem; opacity: 0.85; margin-top: 4px; }
.section-header {
    background: linear-gradient(90deg, #4C72B0, #764ba2);
    color: white; padding: 10px 20px; border-radius: 8px;
    font-size: 1.1rem; font-weight: bold; margin-bottom: 15px;
}
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #4C72B0, #764ba2);
    color: white; border: none; border-radius: 8px;
    padding: 12px; font-size: 1rem; font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    path = "data/master_cleaned.csv"
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_resource
def load_models():
    """
    Load all .pkl model files safely.
    Uses protocol-safe loading to handle NumPy version differences
    between Colab (where models were saved) and local Python.
    """
    m = {}
    paths = {
        'regression':     'models/regression_model.pkl',
        'classification': 'models/classification_model.pkl',
        'recommender':    'models/recommender.pkl',
        'visit_mode_enc': 'models/visit_mode_encoder.pkl',
        'label_encoders': 'models/label_encoders.pkl',
    }
    load_errors = {}
    for key, path in paths.items():
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    m[key] = pickle.load(f)
            except Exception as e:
                load_errors[key] = str(e)
    # Store any errors so we can show them in the UI
    m['_errors'] = load_errors
    return m

df     = load_data()
models = load_models()

# ── Check for data file ──
if df is None:
    st.error("❌  data/master_cleaned.csv not found!")
    st.info("""
    **How to fix:**
    1. Open the **FIXED Colab notebook** (`tourism_Analytics_FIXED.ipynb`)
    2. Run all cells from top to bottom
    3. Run the last **Download** cell — it creates `tourism_app_files.zip`
    4. Extract the zip — you will get `models/` and `data/` folders
    5. Copy both folders into the **same folder as this app.py**
    6. Re-run:  `streamlit run app.py`
    """)
    st.stop()

# ── Check for model loading errors (NumPy version mismatch etc.) ──
load_errors = models.get('_errors', {})
if load_errors:
    st.error("❌  Some models failed to load. This is usually a **NumPy version mismatch**.")
    st.markdown("### Root cause")
    st.code(
        "\n".join([f"{k}: {v}" for k, v in load_errors.items()]),
        language="text"
    )
    st.markdown("### Fix — two options:")
    st.info("""
    **Option A — Recommended (upgrade NumPy on your PC):**
    Open a terminal in VS Code and run:
    ```
    pip install numpy --upgrade
    pip install scikit-learn --upgrade
    streamlit run app.py
    ```

    **Option B — Re-save models with Colab (if Option A fails):**
    1. Open `tourism_Analytics_FIXED.ipynb` in Colab
    2. Run ALL cells again (the fixed notebook saves with protocol=2)
    3. Download the new `tourism_app_files.zip`
    4. Replace your `models/` folder with the new one
    5. Re-run `streamlit run app.py`
    """)
    st.stop()


SEASON_MAP  = {1:'Winter',2:'Winter',3:'Spring',4:'Spring',5:'Spring',
               6:'Summer',7:'Summer',8:'Summer',9:'Autumn',10:'Autumn',
               11:'Autumn',12:'Winter'}
MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']
COLORS      = ['#4C72B0','#DD8452','#55A868','#C44E52','#8172B2','#937860']


with st.sidebar:
    st.markdown("## 🌍 Tourism Analytics")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Dashboard",
         "🔮 Predict Visit Mode",
         "⭐ Predict Rating",
         "🗺️ Recommendations"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Dataset Overview**")
    st.metric("Total Visits",   f"{len(df):,}")
    st.metric("Unique Users",   f"{df['UserId'].nunique():,}")
    st.metric("Attractions",    f"{df['AttractionId'].nunique():,}")
    st.metric("Avg Rating",     f"{df['Rating'].mean():.2f} ⭐")
    st.markdown("---")
    st.markdown("**Models Loaded**")
    for key, label in [('regression','Regression'),
                       ('classification','Classification'),
                       ('recommender','Recommender')]:
        icon = "✅" if key in models else "❌"
        st.caption(f"{icon} {label}")


def encode(val, col):
    le = LabelEncoder()
    le.fit(df[col].astype(str).unique())
    return int(le.transform([val])[0]) if val in le.classes_ else 0

def build_row(visit_mode, continent, region, attr_type,
              visit_year, visit_month, season,
              attr_avg, attr_cnt, user_avg, user_cnt, rating=3):
    is_peak = 1 if visit_month in [6, 7, 8, 12] else 0
    return pd.DataFrame([{
        'Rating':               rating,
        'VisitYear':            visit_year,
        'VisitMonth':           visit_month,
        'IsPeakSeason':         is_peak,
        'Season_enc':           encode(season,      'Season'),
        'UserContinent_enc':    encode(continent,   'UserContinent'),
        'UserRegion_enc':       encode(region,      'UserRegion'),
        'AttractionType_enc':   encode(attr_type,   'AttractionType'),
        'AttractionAvgRating':  attr_avg,
        'AttractionVisitCount': attr_cnt,
        'UserAvgRating':        user_avg,
        'UserVisitCount':       user_cnt,
        'VisitModeName_enc':    encode(visit_mode,  'VisitModeName'),
    }])


if page == "🏠 Dashboard":
    st.title("🌍 Tourism Experience Analytics Dashboard")
    st.markdown("Explore patterns, predict visit modes, estimate ratings, and get attraction recommendations.")
    st.markdown("---")

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in [
        (c1, f"{len(df):,}", "Total Visits"),
        (c2, f"{df['UserId'].nunique():,}", "Unique Users"),
        (c3, f"{df['AttractionId'].nunique():,}", "Attractions"),
        (c4, f"{df['Rating'].mean():.2f} ⭐", "Average Rating"),
    ]:
        col.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-value'>{val}</div>"
            f"<div class='metric-label'>{lbl}</div></div>",
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1: Rating + Visit Mode
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-header'>⭐ Rating Distribution</div>", unsafe_allow_html=True)
        rc   = df['Rating'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 3))
        bars = ax.bar(rc.index, rc.values,
                      color=['#d62728','#ff7f0e','#ffd700','#2ca02c','#1f77b4'],
                      edgecolor='white')
        ax.set_xlabel('Rating'); ax.set_ylabel('Count'); ax.set_xticks([1,2,3,4,5])
        for bar, v in zip(bars, rc.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+200,
                    f'{v:,}', ha='center', fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown("<div class='section-header'>🚌 Visit Mode Share</div>", unsafe_allow_html=True)
        vm = df['VisitModeName'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.pie(vm.values, labels=vm.index, autopct='%1.1f%%',
               startangle=140, colors=COLORS)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Row 2: Geography + Monthly
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-header'>🌐 Visits by Continent</div>", unsafe_allow_html=True)
        cont = df['UserContinent'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh(cont.index[::-1], cont.values[::-1], color='#4C72B0', edgecolor='white')
        ax.set_xlabel('Number of Visits')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown("<div class='section-header'>📅 Monthly Visit Trend</div>", unsafe_allow_html=True)
        monthly = df.groupby('VisitMonth').size()
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(range(1,13), monthly.values, marker='o', color='#55A868', linewidth=2)
        ax.fill_between(range(1,13), monthly.values, alpha=0.15, color='#55A868')
        ax.set_xticks(range(1,13))
        ax.set_xticklabels(MONTH_NAMES, rotation=30, fontsize=7)
        ax.set_ylabel('Visits')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Row 3: Attraction type + Top table
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-header'>🏛️ Top Attraction Types</div>", unsafe_allow_html=True)
        tv = df['AttractionType'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(tv.index[::-1], tv.values[::-1], color='#DD8452', edgecolor='white')
        ax.set_xlabel('Visits')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown("<div class='section-header'>🏆 Top 10 Attractions (min 50 visits)</div>", unsafe_allow_html=True)
        top_a = (df.groupby('Attraction')
                   .agg(AvgRating=('Rating','mean'), Visits=('TransactionId','count'))
                   .query('Visits >= 50')
                   .sort_values('AvgRating', ascending=False)
                   .head(10).reset_index())
        top_a['AvgRating'] = top_a['AvgRating'].round(2)
        st.dataframe(top_a[['Attraction','AvgRating','Visits']],
                     use_container_width=True, hide_index=True)


elif page == "🔮 Predict Visit Mode":
    st.title("🔮 Predict Visit Mode")
    st.markdown("Predict whether the trip is **Business, Couples, Family, Friends, or Solo**.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-header'>👤 User Details</div>", unsafe_allow_html=True)
        continent   = st.selectbox("Continent", sorted(df['UserContinent'].dropna().unique()))
        region_opts = sorted(df[df['UserContinent']==continent]['UserRegion'].dropna().unique())
        region      = st.selectbox("Region", region_opts)
        visit_year  = st.number_input("Visit Year", min_value=2015, max_value=2025, value=2023)
        visit_month = st.select_slider("Visit Month", options=list(range(1,13)),
                                        format_func=lambda x: MONTH_NAMES[x-1], value=6)
        user_avg    = st.slider("Your Avg Past Rating",   1.0, 5.0, 4.0, 0.1)
        user_cnt    = st.number_input("Your Total Past Visits", min_value=1, value=5)

    with col2:
        st.markdown("<div class='section-header'>🏛️ Attraction Details</div>", unsafe_allow_html=True)
        attr_type = st.selectbox("Attraction Type", sorted(df['AttractionType'].dropna().unique()))
        attr_avg  = st.slider("Attraction Avg Rating",   1.0, 5.0, 4.0, 0.1)
        attr_cnt  = st.number_input("Attraction Total Visits", min_value=1, value=100)

    season = SEASON_MAP[visit_month]
    st.info(f"Month: **{MONTH_NAMES[visit_month-1]}**  |  Season: **{season}**  |  "
            f"Peak Season: **{'Yes' if visit_month in [6,7,8,12] else 'No'}**")
    st.markdown("---")

    if st.button("🔮 Predict Visit Mode"):
        if 'classification' not in models:
            st.error("Classification model not found. Please run the Colab notebook first.")
        else:
            clf     = models['classification']
            model   = clf['model']
            feats   = clf['features']
            classes = clf['class_names']

            row  = build_row(classes[0], continent, region, attr_type,
                             visit_year, visit_month, season,
                             attr_avg, attr_cnt, user_avg, user_cnt)
            pred   = model.predict(row[feats].fillna(0))[0]
            probas = model.predict_proba(row[feats].fillna(0))[0]
            label  = classes[pred]

            emoji_map = {'Business':'💼','Couples':'💑','Family':'👨‍👩‍👧',
                         'Friends':'👫','Solo':'🧍'}
            emoji = emoji_map.get(label, '🧳')

            st.success(f"### {emoji} Predicted Visit Mode: **{label}**")
            st.markdown("#### Confidence per Mode")

            prob_df = pd.DataFrame({'Mode': classes, 'Probability': probas})\
                        .sort_values('Probability', ascending=False)
            fig, ax = plt.subplots(figsize=(8, 3))
            clrs = ['#4C72B0' if c==label else '#aec7e8' for c in prob_df['Mode']]
            ax.barh(prob_df['Mode'], prob_df['Probability'], color=clrs, edgecolor='white')
            ax.set_xlabel('Probability'); ax.set_xlim(0, 1.1)
            for i, v in enumerate(prob_df['Probability']):
                ax.text(v+0.01, i, f'{v:.1%}', va='center', fontsize=9)
            plt.tight_layout(); st.pyplot(fig); plt.close()


elif page == "⭐ Predict Rating":
    st.title("⭐ Predict Attraction Rating")
    st.markdown("Estimate what rating a user will give to an attraction **before** they visit.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-header'>👤 User Details</div>", unsafe_allow_html=True)
        continent   = st.selectbox("Continent", sorted(df['UserContinent'].dropna().unique()))
        region_opts = sorted(df[df['UserContinent']==continent]['UserRegion'].dropna().unique())
        region      = st.selectbox("Region", region_opts)
        visit_mode  = st.selectbox("Visit Mode", sorted(df['VisitModeName'].dropna().unique()))
        visit_year  = st.number_input("Visit Year", min_value=2015, max_value=2025, value=2023)
        visit_month = st.select_slider("Visit Month", options=list(range(1,13)),
                                        format_func=lambda x: MONTH_NAMES[x-1], value=6)
        user_avg    = st.slider("Your Avg Past Rating",   1.0, 5.0, 4.0, 0.1)
        user_cnt    = st.number_input("Your Total Past Visits", min_value=1, value=5)

    with col2:
        st.markdown("<div class='section-header'>🏛️ Attraction Details</div>", unsafe_allow_html=True)
        attr_type = st.selectbox("Attraction Type", sorted(df['AttractionType'].dropna().unique()))
        attr_avg  = st.slider("Attraction Avg Rating",   1.0, 5.0, 4.0, 0.1)
        attr_cnt  = st.number_input("Attraction Total Visits", min_value=1, value=100)

    season = SEASON_MAP[visit_month]
    st.markdown("---")

    if st.button("⭐ Predict Rating"):
        if 'regression' not in models:
            st.error("Regression model not found. Please run the Colab notebook first.")
        else:
            reg   = models['regression']
            model = reg['model']
            feats = reg['features']
            row   = build_row(visit_mode, continent, region, attr_type,
                              visit_year, visit_month, season,
                              attr_avg, attr_cnt, user_avg, user_cnt)
            pred  = float(np.clip(model.predict(row[feats].fillna(0))[0], 1, 5))

            ca, cb, cc = st.columns([1, 2, 1])
            with cb:
                st.markdown(
                    f"<div style='text-align:center; background:#f0f4ff; "
                    f"border-left:5px solid #4C72B0; padding:20px; border-radius:8px;'>"
                    f"<div style='font-size:2.5rem; font-weight:bold; color:#4C72B0'>"
                    f"{pred:.2f} / 5.0</div>"
                    f"<div style='font-size:1.8rem'>{'⭐' * int(round(pred))}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                st.markdown("<br>", unsafe_allow_html=True)

                # Gauge
                fig, ax = plt.subplots(figsize=(6, 1.2))
                ax.barh([''], [5], color='#e0e0e0', height=0.5)
                bar_color = '#4C72B0' if pred >= 3.5 else '#DD8452' if pred >= 2.5 else '#d62728'
                ax.barh([''], [pred], color=bar_color, height=0.5)
                ax.set_xlim(0, 5); ax.set_xticks([1,2,3,4,5])
                ax.set_title(f'Predicted Rating: {pred:.2f}', fontsize=11)
                plt.tight_layout(); st.pyplot(fig); plt.close()

                if pred >= 4.5:   st.success("Excellent match! User will love this attraction.")
                elif pred >= 3.5: st.info("Good match. User should enjoy this.")
                elif pred >= 2.5: st.warning("Average match. Consider a better option.")
                else:             st.error("Poor match. This attraction may not suit this user.")


elif page == "🗺️ Recommendations":
    st.title("🗺️ Personalized Attraction Recommendations")
    st.markdown("AI-powered recommendations using collaborative and content-based filtering.")
    st.markdown("---")

    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        user_id = st.number_input(
            "Enter User ID",
            min_value=int(df['UserId'].min()),
            max_value=int(df['UserId'].max()),
            value=int(df['UserId'].value_counts().idxmax())
        )
    with col2:
        top_n = st.slider("Top N", 5, 20, 10)
    with col3:
        method = st.radio("Method",
                          ["Hybrid (Best)", "Collaborative Filtering", "Content-Based"],
                          horizontal=True)

    if st.button("🗺️ Get Recommendations"):
        if 'recommender' not in models:
            st.error("Recommender not found. Please run the Colab notebook first.")
        else:
            rec           = models['recommender']
            user_item     = rec['user_item_matrix']
            user_sim_df   = rec['user_sim_df']
            attr_sim_df   = rec['attr_sim_df']
            attr_features = rec['attr_features']

            # Show history
            history = df[df['UserId']==user_id][['Attraction','AttractionType','Rating']].drop_duplicates()
            if not history.empty:
                with st.expander(f"User {user_id} visited {len(history)} attraction(s) — click to see"):
                    st.dataframe(history.head(10), use_container_width=True, hide_index=True)

            # Collaborative filtering function
            def collab(uid, n):
                if uid not in user_sim_df.index:
                    pop = df.groupby('AttractionId')['Rating'].mean()\
                              .sort_values(ascending=False).head(n)
                    return pd.DataFrame({'AttractionId': pop.index, 'Score': pop.values})
                rated    = user_item.loc[uid]
                visited  = rated[rated > 0].index.tolist()
                top_sims = [u for u in user_sim_df[uid].drop(index=uid, errors='ignore')
                              .sort_values(ascending=False).head(20).index
                              if u in user_item.index]
                sim_r    = user_item.loc[top_sims]
                sim_w    = user_sim_df.loc[top_sims, uid].values.reshape(-1, 1)
                scores   = pd.Series(
                    (sim_r.values * sim_w).sum(0) / (np.abs(sim_w).sum() + 1e-9),
                    index=user_item.columns
                )
                scores = scores.drop(index=[a for a in visited if a in scores.index],
                                     errors='ignore')
                return (scores.sort_values(ascending=False)
                              .head(n).reset_index()
                              .rename(columns={0: 'Score'}))

            # Content-based filtering function
            def content(uid, n):
                hist = df[df['UserId']==uid][['AttractionId','Rating']]
                if hist.empty:
                    top = attr_features.sort_values('AttractionAvgRating', ascending=False).head(n)
                    return top[['AttractionId','AttractionAvgRating']]\
                               .rename(columns={'AttractionAvgRating':'Score'})
                visited = hist['AttractionId'].tolist()
                scores  = pd.Series(dtype=float)
                for _, row in hist.iterrows():
                    aid = row['AttractionId']
                    if aid in attr_sim_df.index:
                        scores = scores.add(
                            attr_sim_df[aid] * (row['Rating'] / 5.0), fill_value=0
                        )
                scores = scores.drop(
                    index=[a for a in visited if a in scores.index], errors='ignore'
                )
                return (scores.sort_values(ascending=False)
                              .head(n).reset_index()
                              .rename(columns={'index': 'AttractionId', 0: 'Score'}))

            # Build recommendations
            with st.spinner("Finding the best attractions for you..."):
                if "Collaborative" in method:
                    recs = collab(user_id, top_n)
                elif "Content" in method:
                    recs = content(user_id, top_n)
                else:
                    c1 = collab(user_id, top_n*2).rename(columns={'Score': 'CS'})
                    c2 = content(user_id, top_n*2).rename(columns={'Score': 'CB'})
                    mg = pd.merge(c1, c2, on='AttractionId', how='outer').fillna(0)
                    if mg['CS'].max() > 0: mg['CS'] /= mg['CS'].max()
                    if mg['CB'].max() > 0: mg['CB'] /= mg['CB'].max()
                    mg['Score'] = 0.6 * mg['CS'] + 0.4 * mg['CB']
                    recs = (mg.sort_values('Score', ascending=False)
                              .head(top_n)[['AttractionId','Score']])

                recs = recs.merge(
                    attr_features[['AttractionId','Attraction',
                                   'AttractionType','AttractionAvgRating']],
                    on='AttractionId', how='left'
                ).dropna(subset=['Attraction'])

                recs['Rank']  = range(1, len(recs)+1)
                recs['Score'] = recs['Score'].round(4)
                recs['AttractionAvgRating'] = recs['AttractionAvgRating'].round(2)

            st.success(f"Found {len(recs)} recommendations for User {user_id}")

            col1, col2 = st.columns([3, 2])
            with col1:
                st.markdown("#### Recommended Attractions")
                st.dataframe(
                    recs[['Rank','Attraction','AttractionType',
                           'AttractionAvgRating','Score']],
                    use_container_width=True, hide_index=True
                )
            with col2:
                st.markdown("#### Score Chart")
                names  = [a[:25]+'…' if len(str(a))>25 else str(a) for a in recs['Attraction']]
                scores = recs['Score'].values
                fig, ax = plt.subplots(figsize=(6, max(3.5, len(recs)*0.38)))
                bars = ax.barh(names[::-1], scores[::-1],
                               color='#4C72B0', edgecolor='white', height=0.6)
                ax.set_xlabel('Recommendation Score')
                for bar, v in zip(bars, scores[::-1]):
                    ax.text(v+0.005, bar.get_y()+bar.get_height()/2,
                            f'{v:.3f}', va='center', fontsize=7)
                plt.tight_layout(); st.pyplot(fig); plt.close()


st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#888; font-size:12px;'>"
    "Tourism Experience Analytics | Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)
