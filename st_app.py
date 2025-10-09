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

# Basic inputs
area = st.number_input("Area (in sq ft)", min_value=100, max_value=10000, value=1000)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=2)
location_name = st.selectbox("Select Location", locations)

# Additional features
st.subheader("Amenities")
gymnasium = st.checkbox("Gymnasium")
lift = st.checkbox("Lift Available")
car_parking = st.checkbox("Car Parking")
maintenance_staff = st.checkbox("Maintenance Staff")
security_24x7 = st.checkbox("24x7 Security")
childrens_play_area = st.checkbox("Children's Play Area")
clubhouse = st.checkbox("Clubhouse")
intercom = st.checkbox("Intercom")
landscaped_gardens = st.checkbox("Landscaped Gardens")
indoor_games = st.checkbox("Indoor Games")
gas_connection = st.checkbox("Gas Connection")
jogging_track = st.checkbox("Jogging Track")
swimming_pool = st.checkbox("Swimming Pool")
new_resale = st.checkbox("New/Resale")  # 1 for New, 0 for Resale

# Predict button
if st.button("Predict Price"):
    location_encoded = le.transform([location_name])[0]
    features = [[
        area,
        bedrooms,
        location_encoded,
        int(new_resale),
        int(gymnasium),
        int(lift),
        int(car_parking),
        int(maintenance_staff),
        int(security_24x7),
        int(childrens_play_area),
        int(clubhouse),
        int(intercom),
        int(landscaped_gardens),
        int(indoor_games),
        int(gas_connection),
        int(jogging_track),
        int(swimming_pool)
    ]]
    
    predicted_price = model.predict(features)[0]
    st.success(f"Predicted House Price: ₹{predicted_price:,.0f}")
