# SYSTEMTRON-Breast-Cancer-Detection-GenAI

## Project Overview
This project uses Generative AI techniques combined with Machine Learning to predict whether a tumor is Malignant or Benign.
The system processes input features, applies preprocessing, and uses a trained Random Forest model for classification.
The entire preprocessing and model pipeline has been saved so predictions can be made instantly without retraining.

## Technologies Used
Python 3

Pandas, NumPy – Data handling

Scikit-learn – Model training and preprocessing

Joblib – Saving/loading models

Jupyter Notebook – Development environment

## Workflow
Data preprocessing (handling missing values, scaling features)

Training multiple models and selecting the best (Random Forest)

Saving the preprocessing + model pipeline as a .pkl file

Making predictions using the saved pipeline

## Model Performance
Best Model: Random Forest Classifier

Accuracy: ~98% on test data

## Example Prediction
Input: Tumor measurement values
Output: Malignant or Benign

Example:
Prediction: Malignant

## How to Run
Open the Jupyter Notebook file in your environment.

Install required libraries:


pip install pandas numpy scikit-learn joblib


## Load the saved pipeline and run predictions:


import joblib
pipeline = joblib.load("breast_cancer_pipeline.pkl")
prediction = pipeline.predict([[<your_features_here>]])
print("Prediction:", "Malignant" if prediction[0] == 0 else "Benign")

## 👨‍💻 Author

Developed by
**Syeda Alia Samia**  
GitHub:[Syeda Alia Samia](https://github.com/your-github-username)
`


