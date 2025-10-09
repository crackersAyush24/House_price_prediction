import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import pickle

data = pd.read_csv("train.csv")

le = LabelEncoder()
data["Location_encoded"] = le.fit_transform(data["Location"])

data["Price_log"] = np.log(data["Price"])

x = data[[
    "Area", "No. of Bedrooms", "Location_encoded",
    "New/Resale", "Gymnasium", "Lift Available",
    "Car Parking", "Maintenance Staff", "24x7 Security",
    "Children's Play Area", "Clubhouse", "Intercom",
    "Landscaped Gardens", "Indoor Games", "Gas Connection",
    "Jogging Track", "Swimming Pool"
]]

y = data["Price_log"]

x["Area"] = np.clip(x["Area"], x["Area"].quantile(0.05), x["Area"].quantile(0.95))
y = np.clip(y, y.quantile(0.05), y.quantile(0.95))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=42)

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print("R2 score:", r2_score(y_test, y_pred))

pickle.dump(model, open("model_xgb.pkl", "wb"))
pickle.dump(le, open("label_encoder.pkl", "wb"))
import pickle

# Save model columns for Streamlit app
model_columns = list(x.columns)  # x = your features DataFrame
with open("model_columns.pkl", "wb") as f:
    pickle.dump(model_columns, f)
