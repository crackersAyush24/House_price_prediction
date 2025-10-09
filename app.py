from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

app = Flask(__name__)

# -------------------------
# Load data and model
# -------------------------
df = pd.read_csv("train.csv")  # Your CSV file with all features

# Encode locations
le = LabelEncoder()
df['Location_encoded'] = le.fit_transform(df['Location'])
locations = list(le.classes_)  # ['Kharghar', 'Powai', ...]

# Load your trained model
# Load your trained model
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

        # Optional: binary features
        gym = int(request.form.get('Gymnasium', 0))
        lift = int(request.form.get('Lift', 0))
        car_parking = int(request.form.get('Car_Parking', 0))
        maintenance = int(request.form.get('Maintenance_Staff', 0))
        security = int(request.form.get('Security_24x7', 0))
        children_play = int(request.form.get('Childrens_Play_Area', 0))
        clubhouse = int(request.form.get('Clubhouse', 0))
        intercom = int(request.form.get('Intercom', 0))
        gardens = int(request.form.get('Landscaped_Gardens', 0))
        indoor_games = int(request.form.get('Indoor_Games', 0))
        gas = int(request.form.get('Gas_Connection', 0))
        jogging = int(request.form.get('Jogging_Track', 0))
        swimming = int(request.form.get('Swimming_Pool', 0))
        new_resale = int(request.form.get('New_Resale', 0))

        # Convert location name to encoded number
        location_encoded = le.transform([location_name])[0]

        # Prepare features in correct order (adjust according to your model)
        features = [[
            area,
            bedrooms,
            new_resale,
            gym,
            lift,
            car_parking,
            maintenance,
            security,
            children_play,
            clubhouse,
            intercom,
            gardens,
            indoor_games,
            gas,
            jogging,
            swimming,
            location_encoded
        ]]

        # Predict price
        predicted_price = model.predict(features)[0]

        return render_template("index.html", locations=locations, prediction=predicted_price)

    return render_template("index.html", locations=locations, prediction=None)

# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
