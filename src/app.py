import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Property Price Prediction", page_icon="🏡", layout="centered")

st.title("🏡 Property Price Prediction App")
st.write("Enter the property details below to estimate its price using trained ML models.")

# Load the saved models
lr_path = os.path.join("../artifacts", "linear_regression.joblib")
rf_path = os.path.join("../artifacts", "random_forest.joblib")

if not os.path.exists(lr_path) or not os.path.exists(rf_path):
    st.error("Model files not found! Please train models first using train.py")
    st.stop()

lr_model = joblib.load(lr_path)
rf_model = joblib.load(rf_path)

st.sidebar.header("Enter Property Details")
area = st.sidebar.number_input("Living Area (GrLivArea, sqft)", min_value=300, max_value=10000, value=1500)
bedrooms = st.sidebar.number_input("Bedrooms Above Ground", min_value=1, max_value=10, value=3)
bathrooms = st.sidebar.number_input("Full Bathrooms", min_value=1, max_value=5, value=2)
year_built = st.sidebar.number_input("Year Built", min_value=1900, max_value=2025, value=2010)
neighborhood = st.sidebar.text_input("Neighborhood", "CollgCr")

input_data = pd.DataFrame({
    'GrLivArea': [area],
    'BedroomAbvGr': [bedrooms],
    'FullBath': [bathrooms],
    'YearBuilt': [year_built],
    'Neighborhood': [neighborhood]
})

st.subheader("Entered Property Data")
st.write(input_data)

# Predict button
if st.button("Predict Price"):
    try:
        lr_pred = lr_model.predict(input_data)[0]
        rf_pred = rf_model.predict(input_data)[0]

        st.success(f"Linear Regression Estimate: ₹{lr_pred:,.0f}")
        st.success(f"Random Forest Estimate: ₹{rf_pred:,.0f}")

        avg_price = (lr_pred + rf_pred) / 2
        st.metric("Average Estimated Price", f"₹{avg_price:,.0f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")