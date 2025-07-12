# DSA-Paul_Kehinde_Adenigba
DSA AI/ML Final Project: Design and Deployment of an AI-Powered Predictive System.

# 🏠 Ames House Sale Price Estimator
**A Machine Learning Web Application with SHAP Explainability**

---

 ### 🔍 Overview
 The Ames House Price Estimator is a full-stack machine learning web application that predicts house sale prices based on key structural features. It leverages:
 * **Gradient Boosting Regression**
 * **Feature Engineering & Selection**
 * **Model explainability using SHAP**
 * **Flask for deployment**
 * **Interactive frontend with SHAP visualizations**

---

### 📊 Problem Statement
Users of the real estate platform struggle to estimate an appropriate sale price budget without manually comparing multiple listings. This inefficiency results in poor user experience. A solution is needed to quickly estimate sale prices based on known home characteristics and help users understand what influences the predicted value.

### 🧠 Project Question
How can we design a machine learning-powered budget estimator that predicts house sale prices based on user-defined features and explains feature contributions using SHAP plots to improve customer experience and decision-making on the real estate aggregation platform?

**Objective:** Predict house sale prices in Ames, Iowa based on structured features.

The model should:
* Accept user input through a web form
* Predict sale price using a trained regression model
* Show interpretable explanations using SHAP (Shapeley Additive Explanations)

---

### 🧪 Data Preprocessing and Modeling (Jupyter Notebook)

### ✅ Feature Selection and Scaling
* Selected 7 high-impact features based on feature importance:
 * `Overall Qual`, `Gr Liv Area`, `Garage Area`, `Year Built`, `BsmtFin SF 1`, `Garage Cars`, `Full Bath`
 * Standardized input features using StandardScaler.

### ✅ Model Training
* Algorithm: `GradientBoosting Regressor`
* Hyperparameters optimized via `optuna`
* Final Model: `optimal_gbr0`
### ✅ Feature Importance Visualization
`plot_feature_importance_type1(optimal_gbr_mod, X_train.columns, top_n=7)`

This produced a barplot of the top 7 important features.
### ✅ Model Explainability (SHAP)
```
explainer = shap.Explainer(optimal_gbr_mod)
shap_values = explainer(X_test)
shap.plots.waterfall(shap_values[0])
```

---

### 📀 Model Serialization
```
sio.dump(optimal_gbr_mod, './Ames_Sale_Price_Model.skops')
joblib.dump(scaler_7_features, './scaler.joblib')
```
* Model saved using skops for secure deserialization
* Scaler saved via joblib

---

### 🌐 Flask Web Application (Deployed Locally via PyCharm)

### 🧠 Backend Logic (app.py)
* Framework: Flask
* Model Loading: `skops.io.load()` with trusted types
* Prediction Endpoint: /predict
 * Receives JSON payload
 * Scales input
 * Predicts price
 * Stores the last input in session for SHAP
* SHAP Plot Endpoint: /shap
 * Generates a SHAP waterfall plot for the most recent prediction

### 🔐 Security
* Uses Flask session with `secret_key` for storing SHAP input securely
* Trusted scikit-learn types specified explicitly to avoid deserialization risks

---

### Frontend Design
`index.html`
* Clean form UI for user input
* Background image set with `{{ image_url }}`
* Responsive design with loading spinner
* Asynchronous form submission via `fetch()`
* Displays predicted price with a SHAP explanation button

`shap.html`
* Displays a base64-encoded SHAP waterfall plot
* Includes a fallback if session input is missing
* “Back to Home” button for smooth navigation

---

### User Workflow
1. **User visits home page** → Fills out house feature form
2. **Clicks “Predict”** → Flask predicts using the trained model
3. **Prediction displayed** → Click “View SHAP Explanation”
4. **SHAP page** shows a visual breakdown of the model prediction

---

### 🎯 Example Prediction
<img src="https://github.com/Pauladen/DSA-Paul_Kehinde_Adenigba/raw/main/Sreenshots/prediction.png" width="100%" height="auto"/>

---

### 📈 Example SHAP Plot
Visual representation of how each feature contributed to the final sale price prediction.
`<img src="data:image/png;base64,{{ shap_plot }}" alt="SHAP Plot">`
(actual image embedded in app)

<img src="https://github.com/Pauladen/DSA-Paul_Kehinde_Adenigba/raw/main/Sreenshots/explanations.png" width="100%" height="auto"/>

### 📌 Key Components 
* **🎯f(x) (Predicted Output)**: The final predicted value, which is $159,913.227 in this plot.
* **📊E[f(x)] (Expected Base Value)**: The starting point of the waterfall, representing the average prediction for the dataset, which is $184,521.424 in this plot.

### 🔍 Interpretation
In this example:

**🟥'Overall Quality Rating'** (input=6) and **'Year Built'** (input=1960) contributed negatively, decreasing the predicted price.

**🟩'Living Area Sq Ft'** (input=1656 Sq Ft) and **'Basement Finished Sq Ft'** (input=639 Sq Ft) contributed positively, increasing the predicted price above the base value.

### 🧱 Features Definitions

* **🧱Overall Qual-** Rates the overall material and finish of the house
* **📐Gr Liv Area-** Above grade (ground) living area square feet
* **🚗Garage Area-** Size of garage in square feet
* **🏚Year Built-** Original construction date
* **🛠BsmtFin SF 1-** Type 1 finished basement square feet
* **🚙Garage Cars-** Size of garage in car capacity
* **🚿Full Bath-** Full bathrooms above grad

---

### 💡 Key Highlights
* End-to-end ML workflow (data → model → deployment)
* Secure model serialization with skops
* Transparent model predictions using SHAP
* Full-stack integration with Flask + JS frontend

---

### 📁 File Structure
├── requirements.txt

├── app.py

├── Ames_Sale_Price_Model.skops

├── scaler.joblib

├── templates/

│   ├── index.html

│   └── shap.html

├── static/

│   └── images/

│       └── Ames_image.jpeg

---

### 🚀 How to Run Locally
-- Clone repository

```
git clone https://github.com/yourusername/ames-price-estimator.git
cd ames-price-estimator
```

-- Install requirements

`pip install -r requirements.txt`

-- Run Flask app

```
python app.py
Open browser at: http://127.0.0.1:5000
```

---

### 🧠 Future Enhancements
* Deploy on Hugging Face Spaces / Render / AWS
* Expand to include categorical and engineered features
* Enable CSV upload and batch predictions
* Add XAI dashboards (e.g., SHAP summary or force plots)

---

### 📚 Technologies Used
Tool                 |     Purpose
---------------------|-------------
Python               |	Core ML and logic
Scikit-learn         |	Model training and scaling
SHAP                 |	Explainability
Flask                |	Web framework
HTML/CSS/JS          |	Frontend UI
Matplotlib/Seaborn   |	Visualizations
skops + joblib       |	Secure model persistence


### <img width="64" height="64" alt="image" src="https://github.com/user-attachments/assets/f294387a-cd39-42c3-b5d2-06c188a1eb79" /> Author

**[Adenigba, Paul Kehinde]**

Data Scientist & Machine Learning Engineer
[LinkedIn](https://www.linkedin.com/in/paul-kehinde-adenigba-a4b182304) | [GitHub](https://github.com/Pauladen/DSA-Paul_Kehinde_Adenigba.git) 







