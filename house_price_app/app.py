import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Mumbai House Price Predictor",
    layout="centered"
)

st.title("🏙️ Mumbai House Price Predictor")
st.write("Predict house prices using a trained ML model")

@st.cache_resource
def load_model():
    return joblib.load("../house_price_rf_model.pkl")

model = load_model()

st.subheader("Property Details")

usable_area_sqft = st.number_input(
    "Usable Area (sqft)", min_value=200, value=800
)

bedroom = st.number_input(
    "Bedrooms", min_value=1, max_value=10, value=2
)

bathroom = st.number_input(
    "Bathrooms", min_value=1, max_value=10, value=2
)

balconies = st.number_input(
    "Balconies", min_value=0, max_value=5, value=1
)

floor_no = st.number_input(
    "Floor Number", min_value=0, max_value=100, value=3
)

floors = st.number_input(
    "Total Floors in Building", min_value=1, max_value=100, value=10
)

maintenance = st.number_input(
    "Maintenance Charges (₹/month)", min_value=0, value=3000
)

facing = st.selectbox(
    "Facing Direction",
    ["East", "West", "North", "South", "North-East", "North-West"]
)

is_prime = st.selectbox(
    "Prime Location Property?",
    ["Yes", "No"]
)

st.subheader("Location Context (Market Signals)")

area_price_mean = st.number_input(
    "Average Area Price (₹)",
    min_value=1_000_000,
    value=15_000_000,
    step=500_000
)

city_price_mean = st.number_input(
    "Average City Price (₹)",
    min_value=1_000_000,
    value=18_000_000,
    step=500_000
)

amenity_count = st.slider(
    "Number of Amenities",
    min_value=0,
    max_value=20,
    value=5
)

bathroom_ratio = bathroom / bedroom
floor_ratio = floor_no / floors
luxury_score = amenity_count * 2 + balconies * 1.5
is_prime_value = 1 if is_prime == "Yes" else 0


if st.button("🔮 Predict Price"):

    input_df = pd.DataFrame([{
        "usable_area_sqft": usable_area_sqft,
        "bedroom": bedroom,
        "Bathroom": bathroom,
        "bathroom_ratio": bathroom_ratio,
        "balconies": balconies,
        "Floor No": floor_no,
        "floors": floors,
        "floor_ratio": floor_ratio,
        "area_price_mean": area_price_mean,
        "city_price_mean": city_price_mean,
        "amenity_count": amenity_count,
        "luxury_score": luxury_score,
        "Facing": facing,
        "Maintenance Charges": maintenance,
        "isPrimeLocationProperty_Y": is_prime_value
    }])

    log_price = model.predict(input_df)[0]
    predicted_price = np.expm1(log_price)

    st.success(
        f"💰 Estimated Property Price: ₹ {predicted_price:,.0f}"
    )
