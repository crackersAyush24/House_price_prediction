# streamlit_app.py
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

# -------------------------
# Load data and model
# -------------------------
df = pd.read_csv("train.csv")  # your CSV with 'Location' column
le = LabelEncoder()
df['Location_encoded'] = le.fit_transform(df['Location'])
locations = list(le.classes_)

# Load your trained model
model = pickle.load(open("model.pkl", "rb"))  # replace with your model file path

# -------------------------
# Streamlit UI
# -------------------------
st.title("House Price Prediction")

# Input fields
area = st.number_input("Area (in sq ft)", min_value=100, max_value=10000, value=1000)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=2)
location_name = st.selectbox("Select Location", locations)

# Predict button
if st.button("Predict Price"):
    location_encoded = le.transform([location_name])[0]
    features = [[area, bedrooms, location_encoded]]  # add other features if needed
    predicted_price = model.predict(features)[0]
    
    st.success(f"Predicted House Price: ₹{predicted_price:,.0f}")
