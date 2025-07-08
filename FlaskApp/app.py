import skops.io as sio
from joblib import load
import shap
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify, url_for, session, redirect
from flask_cors import CORS
import matplotlib.pyplot as plt
import base64
import io
from io import StringIO
import os
import warnings
from sklearn.exceptions import InconsistentVersionWarning

app = Flask(__name__)
app.secret_key = '1234'
CORS(app)

# Explicitly specify trusted types
trusted_types = [
'sklearn.model_selection.train_test_split',
'sklearn.impute.SimpleImputer',
'sklearn.preprocessing.OneHotEncoder',
'sklearn.preprocessing.StandardScaler',
'sklearn.feature_selection.mutual_info_regression',
'sklearn.feature_selection.RFE',
'sklearn.feature_selection.SelectFromModel',
'sklearn.ensemble.IsolationForest',
'sklearn.ensemble.RandomForestRegressor',
'sklearn.ensemble.GradientBoostingRegressor',
'sklearn.ensemble.AdaBoostRegressor',
'sklearn.linear_model.LinearRegression',
'sklearn.svm.LinearSVR',
'sklearn.tree.DecisionTreeRegressor',
'sklearn.pipeline.Pipeline',
'sklearn.metrics.mean_squared_error',
'sklearn.metrics.mean_absolute_error',
'numpy.dtype',
'sklearn._loss.link.IdentityLink',
'sklearn._loss.link.Interval',
'sklearn._loss.loss.HalfSquaredError',
]

# Load the trained model
model = sio.load('./Ames_Sale_Price_Model.skops', trusted=trusted_types)

# Load the scaler
scaler = load('./scaler.joblib')

# Define feature names (must match training data order)
feature_names = [
    "Overall Qual",       # Overall Quality Rating
    "Gr Liv Area",        # Living Area Sq Ft
    "Garage Area",        # Garage Area Sq Ft
    "Year Built",         # Year Built
    "BsmtFin SF 1",       # Basement Finished Sq Ft
    "Garage Cars",        # Garage Cars Capacity
    "Full Bath"           # Full Bathrooms
]

@app.route('/')
def home():
    image_url = url_for("static", filename="images/Ames_image.jpeg")
    return render_template("index.html", image_url=image_url)

@app.route('/predict', methods=['POST'])
def predict():
    print("Received a POST request to /predict")
    data = request.get_json()
    print("Input JSON:", data)

    # Parse features from input
    input_data = [float(data.get(feat, 0)) for feat in feature_names]
    input_df = pd.DataFrame([input_data], columns=feature_names)

    # Scale for prediction
    input_scaled = scaler.transform(input_df)

    # Convert input_scaled to DataFrame
    input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)

    # Predict sale price
    prediction = model.predict(input_scaled_df)[0]

    # Store raw input in session for SHAP use
    session['last_input'] = input_scaled_df.to_json()

    print("Prediction complete. Sending JSON response.")

    return jsonify({
        'prediction': round(prediction, 2),
        'shap_link': url_for('shap_page')
    })

@app.route('/shap')
def shap_page():
    if 'last_input' not in session:
        return render_template("shap.html", shap_plot=None)

    input_df = pd.read_json(StringIO(session['last_input']))

    # SHAP logic
    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(input_df)

        fig, ax = plt.subplots(figsize=(8, 4))
        shap.plots.waterfall(shap_values[0], show=False)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        return render_template("shap.html", shap_plot=image_base64)
    except Exception as e:
        print(f"SHAP plot error: {e}")
        return render_template("shap.html", shap_plot=None)


if __name__ == '__main__':
    app.run(debug=True)
