
````markdown
# 🏠 House Price Prediction

A machine learning project to predict house prices based on property features like area, location, and amenities. It includes **Random Forest** and **XGBoost** models and a **Streamlit web app** for interactive predictions.

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Features](#features)  
- [Technologies](#technologies)  
- [Setup & Installation](#setup--installation)  
- [Usage](#usage)  
- [Project Structure](#project-structure)  
- [Future Improvements](#future-improvements)  

---

## Project Overview

This project predicts house prices using a dataset containing multiple features. Data preprocessing handles outliers and categorical variables, and trained models are saved for reuse. The Streamlit web app allows users to input property details and get instant predictions.

---

## Features

- Handles **numerical and categorical features**.  
- Removes outliers using percentile clipping for better predictions.  
- Predicts prices with **Random Forest** and **XGBoost** models.  
- Interactive **Streamlit** app for easy user input.  
- Saves models, label encoders, and column info as `.pkl` files for reuse.  

---
## Technologies

- Python 3  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  
- Streamlit  
- Pickle  

---

## Setup & Installation

1. **Clone the repository**:

```bash
git clone <repository-url>
cd House-Price-Prediction-main
````

2. **Create a virtual environment**:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
# If requirements.txt not available, install manually:
pip install pandas numpy scikit-learn xgboost streamlit
```

---

## Usage

1. **Train the models**:

```bash
python Main.py
```

This will generate:

* `model_xgb.pkl` – XGBoost model
* `model.pkl` – Random Forest model
* `label_encoder.pkl` – LabelEncoder for categorical features
* `model_columns.pkl` – Columns used for prediction

2. **Run the Streamlit app**:

```bash
streamlit run app.py
```

* Open [http://localhost:8501](http://localhost:8501) in your browser.
* Enter house details and get predicted price.

---

## Project Structure



```
House-Price-Prediction-main/
├── app.py                  # Streamlit app
├── Main.py                 # Model training script
├── train.csv               # Dataset
├── model_xgb.pkl           # XGBoost model
├── model.pkl               # Random Forest model
├── label_encoder.pkl       # Label encoder
├── model_columns.pkl       # Columns used in model
├── images/                 # Optional images
├── templates/              # Optional templates
├── README.md
└── venv/                   # Virtual environment
```

---

## Future Improvements

* Add **more features**: nearby schools, hospitals, transportation.
* Apply **hyperparameter tuning** for higher accuracy.
* Deploy the app on **cloud platforms** like Heroku, AWS, or Streamlit Cloud.
* Include **explainable AI** features to show which factors affect price predictions.

---

```




