"""
================================================
PHASE 2: EXPLORATORY DATA ANALYSIS (EDA)
Tourism Experience Analytics Project
================================================
Steps:
  1. Rating distribution
  2. Visit mode distribution
  3. User geography (continent / country)
  4. Popular attraction types
  5. Monthly & seasonal trends
  6. Correlation heatmap
  7. Top attractions by rating
================================================
Run AFTER phase1_data_preparation.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# LOAD CLEANED DATA
# ─────────────────────────────────────────────
print("=" * 55)
print("  PHASE 2: EXPLORATORY DATA ANALYSIS")
print("=" * 55)

df = pd.read_csv('data/master_cleaned.csv')
print(f"\nLoaded: {df.shape[0]} rows × {df.shape[1]} columns")

os.makedirs('outputs/eda', exist_ok=True)

# ─────────────────────────────────────────────
# STYLE SETUP
# ─────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2', '#937860']

# ─────────────────────────────────────────────
# PLOT 1: RATING DISTRIBUTION
# ─────────────────────────────────────────────
print("\n[1/7] Plotting rating distribution...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Rating Distribution', fontsize=15, fontweight='bold')

# Count plot
rating_counts = df['Rating'].value_counts().sort_index()
axes[0].bar(rating_counts.index, rating_counts.values, color=COLORS, edgecolor='white', linewidth=0.8)
axes[0].set_title('Rating Frequency')
axes[0].set_xlabel('Rating (1–5)')
axes[0].set_ylabel('Number of Reviews')
for i, v in enumerate(rating_counts.values):
    axes[0].text(i + 1, v + 200, f'{v:,}', ha='center', fontsize=9)

# Pie chart
axes[1].pie(rating_counts.values, labels=[f'Rating {r}' for r in rating_counts.index],
            autopct='%1.1f%%', colors=COLORS, startangle=140)
axes[1].set_title('Rating Share (%)')

plt.tight_layout()
plt.savefig('outputs/eda/01_rating_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: outputs/eda/01_rating_distribution.png")

# ─────────────────────────────────────────────
# PLOT 2: VISIT MODE DISTRIBUTION
# ─────────────────────────────────────────────
print("[2/7] Plotting visit mode distribution...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Visit Mode Analysis', fontsize=15, fontweight='bold')

vm_counts = df['VisitModeName'].value_counts()
axes[0].barh(vm_counts.index, vm_counts.values, color=COLORS)
axes[0].set_title('Visit Mode Frequency')
axes[0].set_xlabel('Number of Visits')
for i, v in enumerate(vm_counts.values):
    axes[0].text(v + 100, i, f'{v:,}', va='center', fontsize=9)

# Rating by visit mode (boxplot)
df.boxplot(column='Rating', by='VisitModeName', ax=axes[1], patch_artist=True)
axes[1].set_title('Rating Distribution by Visit Mode')
axes[1].set_xlabel('Visit Mode')
axes[1].set_ylabel('Rating')
plt.suptitle('')

plt.tight_layout()
plt.savefig('outputs/eda/02_visit_mode.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: outputs/eda/02_visit_mode.png")

# ─────────────────────────────────────────────
# PLOT 3: USER GEOGRAPHY
# ─────────────────────────────────────────────
print("[3/7] Plotting user geography...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('User Geography', fontsize=15, fontweight='bold')

# Continent
cont_counts = df['UserContinent'].value_counts()
axes[0].bar(cont_counts.index, cont_counts.values, color=COLORS, edgecolor='white')
axes[0].set_title('Users by Continent')
axes[0].set_ylabel('Number of Visits')
axes[0].tick_params(axis='x', rotation=20)

# Top 15 countries
country_counts = df['UserCountry'].value_counts().head(15)
axes[1].barh(country_counts.index[::-1], country_counts.values[::-1], color='#4C72B0')
axes[1].set_title('Top 15 User Countries')
axes[1].set_xlabel('Number of Visits')

plt.tight_layout()
plt.savefig('outputs/eda/03_user_geography.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: outputs/eda/03_user_geography.png")

# ─────────────────────────────────────────────
# PLOT 4: ATTRACTION TYPES
# ─────────────────────────────────────────────
print("[4/7] Plotting attraction type analysis...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Attraction Type Analysis', fontsize=15, fontweight='bold')

# Visit count by type
type_visits = df['AttractionType'].value_counts().head(12)
axes[0].barh(type_visits.index[::-1], type_visits.values[::-1], color='#55A868', edgecolor='white')
axes[0].set_title('Most Visited Attraction Types')
axes[0].set_xlabel('Number of Visits')

# Avg rating by type
type_rating = df.groupby('AttractionType')['Rating'].mean().sort_values(ascending=False).head(12)
axes[1].barh(type_rating.index[::-1], type_rating.values[::-1], color='#DD8452', edgecolor='white')
axes[1].set_title('Avg Rating by Attraction Type')
axes[1].set_xlabel('Average Rating')
axes[1].set_xlim(0, 5)
for i, v in enumerate(type_rating.values[::-1]):
    axes[1].text(v + 0.05, i, f'{v:.2f}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig('outputs/eda/04_attraction_types.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: outputs/eda/04_attraction_types.png")

# ─────────────────────────────────────────────
# PLOT 5: MONTHLY & SEASONAL TRENDS
# ─────────────────────────────────────────────
print("[5/7] Plotting seasonal trends...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Temporal Trends', fontsize=15, fontweight='bold')

# Monthly visit volume
monthly = df.groupby('VisitMonth').size()
month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
axes[0].plot(range(1,13), monthly.values, marker='o', color='#4C72B0', linewidth=2)
axes[0].set_xticks(range(1,13))
axes[0].set_xticklabels(month_names)
axes[0].set_title('Visit Volume by Month')
axes[0].set_ylabel('Number of Visits')
axes[0].fill_between(range(1,13), monthly.values, alpha=0.2, color='#4C72B0')

# Seasonal rating
season_order = ['Spring', 'Summer', 'Autumn', 'Winter']
season_rating = df.groupby('Season')['Rating'].mean().reindex(season_order)
axes[1].bar(season_rating.index, season_rating.values, color=COLORS, edgecolor='white')
axes[1].set_title('Average Rating by Season')
axes[1].set_ylabel('Average Rating')
axes[1].set_ylim(0, 5)
for i, v in enumerate(season_rating.values):
    axes[1].text(i, v + 0.05, f'{v:.2f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('outputs/eda/05_seasonal_trends.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: outputs/eda/05_seasonal_trends.png")

# ─────────────────────────────────────────────
# PLOT 6: CORRELATION HEATMAP
# ─────────────────────────────────────────────
print("[6/7] Plotting correlation heatmap...")

numeric_cols = ['Rating', 'VisitYear', 'VisitMonth', 'IsPeakSeason',
                'AttractionAvgRating', 'AttractionVisitCount',
                'UserAvgRating', 'UserVisitCount']
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5, ax=ax)
ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/eda/06_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: outputs/eda/06_correlation_heatmap.png")

# ─────────────────────────────────────────────
# PLOT 7: TOP ATTRACTIONS
# ─────────────────────────────────────────────
print("[7/7] Plotting top attractions...")

# Top 15 attractions with at least 50 visits, sorted by avg rating
top_attr = (df.groupby('Attraction')
              .agg(AvgRating=('Rating', 'mean'), VisitCount=('TransactionId', 'count'))
              .query('VisitCount >= 50')
              .sort_values('AvgRating', ascending=False)
              .head(15))

fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.barh(top_attr.index[::-1], top_attr['AvgRating'][::-1], color='#C44E52', edgecolor='white')
ax.set_title('Top 15 Attractions by Average Rating\n(min. 50 visits)', fontsize=13, fontweight='bold')
ax.set_xlabel('Average Rating')
ax.set_xlim(0, 5.5)
for i, (rating, count) in enumerate(zip(top_attr['AvgRating'][::-1], top_attr['VisitCount'][::-1])):
    ax.text(rating + 0.05, i, f'{rating:.2f}  ({count} visits)', va='center', fontsize=8)
plt.tight_layout()
plt.savefig('outputs/eda/07_top_attractions.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: outputs/eda/07_top_attractions.png")

# ─────────────────────────────────────────────
# SUMMARY STATS
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("  KEY INSIGHTS FROM EDA")
print("=" * 55)
print(f"  Total transactions  : {len(df):,}")
print(f"  Unique users        : {df['UserId'].nunique():,}")
print(f"  Unique attractions  : {df['AttractionId'].nunique():,}")
print(f"  Average rating      : {df['Rating'].mean():.2f}")
print(f"  Most common mode    : {df['VisitModeName'].mode()[0]}")
print(f"  Most visited type   : {df['AttractionType'].value_counts().idxmax()}")
print(f"  Top user continent  : {df['UserContinent'].value_counts().idxmax()}")
print(f"\n  All EDA plots saved to: outputs/eda/")
print(f"\n{'='*55}")
print("  PHASE 2 COMPLETE ✓")
print(f"{'='*55}")
