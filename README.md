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
```explainer = shap.Explainer(optimal_gbr_mod)
shap_values = explainer(X_test)
shap.plots.waterfall(shap_values[0])```















## Prediction
<img src="https://github.com/Pauladen/DSA-Paul_Kehinde_Adenigba/raw/main/Sreenshots/prediction.png" width="100%" height="auto"/>

## Impacts of Features on the Prediction
<img src="https://github.com/Pauladen/DSA-Paul_Kehinde_Adenigba/raw/main/Sreenshots/explanations.png" width="100%" height="auto"/>

The **SHAP (SHapeley Addition explanation)** waterfall plot visually represents how individual features contribute to a prediction.

### Key Components 
* **f(x) (Predicted Output)**: The final predicted value, which is $159,913.227 in this plot.
* **E[f(x)] (Expected Base Value)**: The starting point of the waterfall, representing the average prediction for the dataset, which is $184,521.424 in this plot.

### Interpretation
In this example, **'Overall Quality Rating'** (6 in the input form) and **'Year Built'** (1960 in the input form) significantly decreased the predicted house sale price from baseline ($184,521.424), while **'Living Area Sq Ft'** (1656 Sq Ft in the input form) and **'Basement Finished Sq Ft'** (639 Sq Ft in the input form) significantly increased it.

### Definition of Features

* **Overall Qual**: Rates the overall material and finish of the house
* **Gr Liv Area**: Above grade (ground) living area square feet
* **Garage Area**: Size of garage in square feet
* **Year Built**: Original construction date
* **BsmtFin SF 1**: Type 1 finished basement square feet
* **Garage Cars**: Size of garage in car capacity
* **Full Bath**: Full bathrooms above grad
