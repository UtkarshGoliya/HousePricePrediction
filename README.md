# Mumbai House Price Prediction

An end-to-end **Machine Learning project** to predict residential property prices in Mumbai using real estate listing data, feature engineering, and a deployed web application.


## 📌 Overview

This project estimates the **data-driven fair market value** of a residential flat based on size, configuration, location signals, amenities, and building characteristics.

The model predicts statistical market value, not broker-quoted prices.


## Highlights

- 12,000+ property listings cleaned and processed  
- 145 raw features → **15 high-impact engineered features**  
- Multiple models trained and compared  
- **Random Forest Regressor** selected as final model  
- **R² ≈ 0.933**  
- Deployed using **Streamlit**


## Model Performance

| Model | R² |
|------|----|
| Linear Regression | 0.807 |
| Gradient Boosting | 0.926 |
| **Random Forest (Final)** | **0.933** |


## Web Application

An interactive **Streamlit app** allows users to:
- Enter property details
- Adjust market context
- Instantly predict house price


## ⚠️ Notes & Limitations

- Predictions represent fair value, not broker quotes  
- Premium properties may be under-estimated due to mean-encoding  


##  Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Joblib


