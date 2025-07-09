# DSA-Paul_Kehinde_Adenigba
DSA AI/ML Final Project: Design and Deployment of an AI-Powered Predictive System.

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

