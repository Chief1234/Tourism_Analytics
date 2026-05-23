"""
================================================
PHASE 1: DATA PREPARATION
Tourism Experience Analytics Project
================================================
Steps:
  1. Load all 9 Excel files
  2. Merge into one master DataFrame
  3. Clean missing values & outliers
  4. Feature Engineering
  5. Encode categorical variables
  6. Save cleaned dataset
================================================
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# STEP 1: LOAD ALL DATA FILES
# ─────────────────────────────────────────────
# Update DATA_PATH to the folder where your Excel files are located
DATA_PATH = "Tourism Dataset/"

print("=" * 55)
print("  PHASE 1: DATA PREPARATION")
print("=" * 55)

print("\n[1/6] Loading all Excel files...")

transaction  = pd.read_excel(os.path.join(DATA_PATH, "Transaction.xlsx"))
user         = pd.read_excel(os.path.join(DATA_PATH, "User.xlsx"))
item         = pd.read_excel(os.path.join(DATA_PATH, "Item.xlsx"))
city         = pd.read_excel(os.path.join(DATA_PATH, "City.xlsx"))
country      = pd.read_excel(os.path.join(DATA_PATH, "Country.xlsx"))
region       = pd.read_excel(os.path.join(DATA_PATH, "Region.xlsx"))
continent    = pd.read_excel(os.path.join(DATA_PATH, "Continent.xlsx"))
visit_mode   = pd.read_excel(os.path.join(DATA_PATH, "Mode.xlsx"))
attr_type    = pd.read_excel(os.path.join(DATA_PATH, "Type.xlsx"))

print(f"  Transaction : {transaction.shape}")
print(f"  User        : {user.shape}")
print(f"  Item        : {item.shape}")
print(f"  City        : {city.shape}")
print(f"  Country     : {country.shape}")
print(f"  Region      : {region.shape}")
print(f"  Continent   : {continent.shape}")
print(f"  VisitMode   : {visit_mode.shape}")
print(f"  AttrType    : {attr_type.shape}")

# ─────────────────────────────────────────────
# STEP 2: MERGE INTO MASTER DATAFRAME
# ─────────────────────────────────────────────
print("\n[2/6] Merging all tables into master DataFrame...")

# Remove placeholder rows (ID = 0 means unknown/missing)
city      = city[city['CityId'] != 0]
country   = country[country['CountryId'] != 0]
region    = region[region['RegionId'] != 0]
continent = continent[continent['ContinentId'] != 0]
visit_mode = visit_mode[visit_mode['VisitModeId'] != 0]

# Step 2a: Transaction + User
df = transaction.merge(user, on='UserId', how='left')

# Step 2b: Decode VisitMode number → label
df = df.merge(
    visit_mode.rename(columns={'VisitModeId': 'VisitMode', 'VisitMode': 'VisitModeName'}),
    on='VisitMode', how='left'
)

# Step 2c: Add Attraction details (Item table)
df = df.merge(item, on='AttractionId', how='left')

# Step 2d: Add Attraction Type label
df = df.merge(attr_type, on='AttractionTypeId', how='left')

# Step 2e: Add User's City, Country, Region, Continent names
df = df.merge(city.rename(columns={'CityId': 'CityId', 'CityName': 'UserCity', 'CountryId': 'UserCountryId'}),
              on='CityId', how='left')

df = df.merge(country.rename(columns={'CountryId': 'UserCountryId', 'Country': 'UserCountry', 'RegionId': 'UserRegionId'}),
              on='UserCountryId', how='left')

df = df.merge(region.rename(columns={'RegionId': 'UserRegionId', 'Region': 'UserRegion', 'ContinentId': 'UserContinentId'}),
              on='UserRegionId', how='left')

df = df.merge(continent.rename(columns={'ContinentId': 'UserContinentId', 'Continent': 'UserContinent'}),
              on='UserContinentId', how='left')

print(f"  Master DataFrame shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")

# ─────────────────────────────────────────────
# STEP 3: DATA CLEANING
# ─────────────────────────────────────────────
print("\n[3/6] Cleaning data...")

print(f"  Missing values before cleaning:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# Drop rows where key target columns are missing
df.dropna(subset=['Rating', 'VisitModeName'], inplace=True)

# Fill missing text columns with 'Unknown'
text_cols = ['UserCity', 'UserCountry', 'UserRegion', 'UserContinent',
             'Attraction', 'AttractionType', 'AttractionAddress']
for col in text_cols:
    if col in df.columns:
        df[col] = df[col].fillna('Unknown')

# Fill missing numeric IDs with 0
id_cols = ['CityId', 'CountryId', 'RegionId', 'ContinentId',
           'UserCountryId', 'UserRegionId', 'UserContinentId', 'AttractionTypeId']
for col in id_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0).astype(int)

# Remove invalid ratings (must be 1–5)
df = df[df['Rating'].between(1, 5)]

# Remove invalid year/month
df = df[(df['VisitYear'] >= 2000) & (df['VisitYear'] <= 2025)]
df = df[(df['VisitMonth'] >= 1) & (df['VisitMonth'] <= 12)]

print(f"  Missing values after cleaning:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"  Cleaned shape: {df.shape}")

# ─────────────────────────────────────────────
# STEP 4: FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n[4/6] Engineering new features...")

# Season from month
def get_season(month):
    if month in [12, 1, 2]:   return 'Winter'
    elif month in [3, 4, 5]:  return 'Spring'
    elif month in [6, 7, 8]:  return 'Summer'
    else:                      return 'Autumn'

df['Season'] = df['VisitMonth'].apply(get_season)

# Is weekend travel indicator (approximated — no day info, so we use month patterns)
# We add a feature: "peak season" (summer = most tourists)
df['IsPeakSeason'] = df['VisitMonth'].isin([6, 7, 8, 12]).astype(int)

# Average rating per attraction (historical popularity)
attr_avg_rating = df.groupby('AttractionId')['Rating'].mean().rename('AttractionAvgRating')
df = df.merge(attr_avg_rating, on='AttractionId', how='left')

# Number of visits per attraction (popularity count)
attr_visit_count = df.groupby('AttractionId')['TransactionId'].count().rename('AttractionVisitCount')
df = df.merge(attr_visit_count, on='AttractionId', how='left')

# Average rating per user (user tendency to rate high/low)
user_avg_rating = df.groupby('UserId')['Rating'].mean().rename('UserAvgRating')
df = df.merge(user_avg_rating, on='UserId', how='left')

# Number of visits per user (user activity level)
user_visit_count = df.groupby('UserId')['TransactionId'].count().rename('UserVisitCount')
df = df.merge(user_visit_count, on='UserId', how='left')

print(f"  New features added: Season, IsPeakSeason, AttractionAvgRating,")
print(f"  AttractionVisitCount, UserAvgRating, UserVisitCount")

# ─────────────────────────────────────────────
# STEP 5: ENCODE CATEGORICAL VARIABLES
# ─────────────────────────────────────────────
print("\n[5/6] Encoding categorical variables...")

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
categorical_cols = ['VisitModeName', 'UserContinent', 'UserRegion', 'Season', 'AttractionType']

for col in categorical_cols:
    if col in df.columns:
        df[col + '_Encoded'] = le.fit_transform(df[col].astype(str))

# Save the label encoder mappings for use in the Streamlit app
import pickle
encoders = {}
for col in categorical_cols:
    if col in df.columns:
        enc = LabelEncoder()
        enc.fit(df[col].astype(str))
        encoders[col] = enc

os.makedirs('models', exist_ok=True)
with open('models/label_encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

print(f"  Encoded columns: {[c + '_Encoded' for c in categorical_cols if c in df.columns]}")
print(f"  VisitMode class distribution:\n{df['VisitModeName'].value_counts()}")

# ─────────────────────────────────────────────
# STEP 6: SAVE CLEANED DATASET
# ─────────────────────────────────────────────
print("\n[6/6] Saving cleaned dataset...")

os.makedirs('data', exist_ok=True)
df.to_csv('data/master_cleaned.csv', index=False)

print(f"  Saved: data/master_cleaned.csv  ({df.shape[0]} rows × {df.shape[1]} cols)")
print(f"\n{'='*55}")
print("  PHASE 1 COMPLETE ✓")
print(f"{'='*55}")
print(f"\nSample of final dataset:")
print(df[['UserId', 'AttractionId', 'Rating', 'VisitModeName',
          'UserContinent', 'AttractionType', 'Season',
          'AttractionAvgRating', 'UserAvgRating']].head(3).to_string())
