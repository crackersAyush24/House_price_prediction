from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

app = Flask(__name__)

# -------------------------
# Load data and model
# -------------------------
df = pd.read_csv("train.csv")

# Encode locations
le = LabelEncoder()
df['Location_encoded'] = le.fit_transform(df['Location'])
locations = list(le.classes_)  # ['Kharghar', 'Powai', ...]

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# -------------------------
# Routes
# -------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get form data
        area = float(request.form['Area'])
        bedrooms = int(request.form['Bedrooms'])
        location_name = request.form['Location']

        # Convert location name to encoded number
        location_encoded = le.transform([location_name])[0]

        # Get amenities/features
        new_resale = int(request.form.get('New_Resale', 0))
        gymnasium = int(request.form.get('Gymnasium', 0))
        lift = int(request.form.get('Lift', 0))
        car_parking = int(request.form.get('Car_Parking', 0))
        maintenance_staff = int(request.form.get('Maintenance_Staff', 0))
        security_24x7 = int(request.form.get('Security_24x7', 0))
        childrens_play_area = int(request.form.get('Childrens_Play_Area', 0))
        clubhouse = int(request.form.get('Clubhouse', 0))
        intercom = int(request.form.get('Intercom', 0))
        landscaped_gardens = int(request.form.get('Landscaped_Gardens', 0))
        indoor_games = int(request.form.get('Indoor_Games', 0))
        gas_connection = int(request.form.get('Gas_Connection', 0))
        jogging_track = int(request.form.get('Jogging_Track', 0))
        swimming_pool = int(request.form.get('Swimming_Pool', 0))

        # Prepare features in correct order
        features = [[
            area,
            bedrooms,
            location_encoded,
            new_resale,
            gymnasium,
            lift,
            car_parking,
            maintenance_staff,
            security_24x7,
            childrens_play_area,
            clubhouse,
            intercom,
            landscaped_gardens,
            indoor_games,
            gas_connection,
            jogging_track,
            swimming_pool
        ]]

        # Predict price
        predicted_price = model.predict(features)[0]

        return render_template("index.html", locations=locations, prediction=predicted_price)

    return render_template("index.html", locations=locations, prediction=None)

# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
