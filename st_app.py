import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Load data and model
data = pd.read_csv('train.csv')
model = pickle.load(open('model.pkl', 'rb'))

# Encode locations
le = LabelEncoder()
data['Location'] = le.fit_transform(data['Location'])
locations = sorted(data['Location'].unique())

# Streamlit UI
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

st.title("🏡 House Price Prediction App")
st.write("Enter property details below to estimate the house price.")

# Input fields
location_name = st.selectbox("Select Location", sorted(data['Location'].unique()))
area = st.number_input("Enter Area (in sq ft)", min_value=200.0, max_value=10000.0, value=1000.0)
bhk = st.number_input("Number of Bedrooms (BHK)", min_value=1, max_value=10, step=1)
new_resale = st.checkbox("New Property (Check if New)")
gym = st.checkbox("Gymnasium")
car = st.checkbox("Car Parking")
indoor = st.checkbox("Indoor Games")
jog = st.checkbox("Jogging Track")

# Convert categorical inputs
new = 1 if new_resale else 0
gymn = 1 if gym else 0
carp = 1 if car else 0
indo = 1 if indoor else 0
jogg = 1 if jog else 0

# Find encoded location value
n = 0
for i in range(len(data)):
    if data['Location'][i] == location_name:
        n = data['Location'][i]
        break

# Prediction button
if st.button("Predict Price 💰"):
    input_data = pd.DataFrame([[area, n, bhk, new, gymn, carp, indo, jogg]], 
                              columns=['Area', 'Location', 'No. of Bedrooms', 
                                       'New/Resale', 'Gymnasium', 'Car Parking', 
                                       'Indoor Games', 'Jogging Track'])
    pred = model.predict(input_data)[0] * 1e6
    st.success(f"🏠 Estimated Price: ₹ {np.round(pred, 2):,}")

st.markdown("---")
st.caption("Developed with ❤️ using Streamlit and Random Forest Regressor")
