🌍 Tourism Experience Analytics

An end-to-end Machine Learning + Analytics + Recommendation System project built using Python, Pandas, Scikit-Learn, and Streamlit.

This interactive analytics platform explores global tourism behavior, predicts visitor patterns, and recommends attractions using Hybrid Collaborative Filtering.

🚀 Features
📊 1. Tourism Analytics Dashboard

A.Rating distribution

B.Visit mode analysis

C.Monthly visit trends

D.User geography (continents/regions)

E.Attraction type insights

F.Top attractions by ratings

🔮 2. Visit Mode Prediction

Predict whether a user is visiting as:
Business, Couples, Family, Friends, or Solo
(using Gradient Boosting Classifier)

⭐ 3. Attraction Rating Prediction

Predict the expected rating (1–5) before a user visits an attraction
(using Gradient Boosting Regressor)

🗺️ 4. Personalized Recommendations

Hybrid recommender using:

* Collaborative Filtering (User Similarity)
  
* Content-Based Filtering (Attraction Similarity)
  
* Weighted Hybrid Scores
  
📁 Project Structure
tourism_project/
│── app.py                     ← Streamlit Web App
│── phase1_data_preparation.py ← Clean & merge datasets
│── phase2_eda.py              ← Visualizations
│── phase3_regression.py       ← Rating prediction
│── phase4_classification.py   ← Visit mode classification
│── phase5_recommendation.py   ← Recommender system
│── run_all.py                 ← Automates all phases
│── requirements.txt
│
├── Tourism Dataset/           ← Raw 9 Excel files here
├── data/
│     └── master_cleaned.csv   ← Prepared dataset
├── models/
│     ├── regression_model.pkl
│     ├── classification_model.pkl
│     ├── recommender.pkl
│     ├── visit_mode_encoder.pkl
│     └── label_encoders.pkl
└── outputs/
      ├── eda/
      ├── regression/
      └── classification/

 📂 Dataset Overview (9 Files)
 | File             | Description                                       |
| ---------------- | ------------------------------------------------- |
| Transaction.xlsx | User visits, ratings, month/year, visit mode      |
| User.xlsx        | User geography (continent, region, country, city) |
| Item.xlsx        | Attractions metadata                              |
| City.xlsx        | City dictionary                                   |
| Country.xlsx     | Country dictionary                                |
| Region.xlsx      | Region dictionary                                 |
| Continent.xlsx   | 6 continents                                      |
| Mode.xlsx        | Visit modes                                       |
| Type.xlsx        | Attraction types                                  |

⚙️ Installation & Running

1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Place dataset
Put all 9 Excel files into Tourism Dataset/ folder.

3️⃣ Run all steps automatically
python run_all.py

4️⃣ Launch Streamlit App
streamlit run app.py

🧠 Machine Learning Models

1.Regression (Rating Prediction)
   a. Linear Regression
   
   b. Random Forest Regressor
   
   c. Gradient Boosting Regressor
   
2.Classification (Visit Mode Prediction)
    a. Logistic Regression
    
    b. Random Forest Classifier
    
    c. Gradient Boosting Classifier
    
3.Recommendation System
    a. User-User Collaborative Filtering
    
    b. Attraction-Feature Content Based Filtering
    
    c. Hybrid Recommender (60% CF + 40% CB)

🏆 Highlights

1. Fully automated ML pipeline
   
2.Custom engineered features (Season, Peak Season, Attraction Scores)

Beautiful Streamlit UI

4.Supports deployment on Streamlit Cloud / AWS / Render

📜 License

MIT License.




