import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# -------------------------
# Load data and model
# -------------------------
df = pd.read_csv("train.csv")  # make sure this is your original training CSV

# Encode locations
le = LabelEncoder()
df['Location_encoded'] = le.fit_transform(df['Location'])
locations = list(le.classes_)  # e.g., ['Kharghar', 'Powai', ...]

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# -------------------------
# Streamlit App
# -------------------------
st.title("🏠 House Price Prediction")

st.write("Fill in the details below to predict the house price:")

# User Inputs
area = st.number_input("Area (in sq ft)", min_value=100, max_value=5000, value=1000)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=2)
location_name = st.selectbox("Select Location", locations)

# Amenities / features expected by model (8 features)
new_resale = st.checkbox("New / Resale (New = checked)")  # 1 = new, 0 = resale
gymnasium = st.checkbox("Gymnasium")
lift = st.checkbox("Lift Available")
car_parking = st.checkbox("Car Parking")
maintenance_staff = st.checkbox("Maintenance Staff")

# Predict button
if st.button("Predict Price"):
    # Encode location
    try:
        location_encoded = le.transform([location_name])[0]
    except:
        st.error("Location not recognized!")
        st.stop()

    # Prepare features in order
    features = [[
        area,
        bedrooms,
        location_encoded,
        int(new_resale),
        int(gymnasium),
        int(lift),
        int(car_parking),
        int(maintenance_staff)
    ]]

    # Debug: Show features
    st.write("Features sent to model:", features)

    # Predict price
    predicted_price = model.predict(features)[0]

    # Display result in lakhs if too big
    if predicted_price > 1_00_000:
        st.success(f"🏷️ Predicted House Price: ₹{predicted_price:,.0f}")
    else:
        st.success(f"🏷️ Predicted House Price: ₹{predicted_price:,.0f}")

