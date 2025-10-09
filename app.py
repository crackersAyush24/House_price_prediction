import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("🏠 House Price Prediction")

model = pickle.load(open("model_xgb.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))
columns = pickle.load(open("model_columns.pkl", "rb"))

data = pd.read_csv("train.csv")
amenities = [c for c in columns if c not in ['Area', 'No. of Bedrooms', 'Location_encoded']]

locations = data['Location'].unique()
location = st.selectbox("Select Location", locations)
area = st.number_input("Area (in sq ft)", min_value=100)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10)

st.subheader("Select Amenities")
selected_amenities = {}
for amenity in amenities:
    selected_amenities[amenity] = st.checkbox(amenity)

if st.button("Predict Price"):
    loc_encoded = le.transform([location])[0]
    features = pd.DataFrame([{**{"Area": area, "No. of Bedrooms": bedrooms, "Location_encoded": loc_encoded},
                              **{a: int(selected_amenities[a]) for a in amenities}}])
    features = features[columns]
    predicted_log_price = model.predict(features)[0]
    predicted_price = np.exp(predicted_log_price)
    st.success(f"🏷️ Predicted House Price: ₹{predicted_price:,.0f}")
