
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# --- Load Model and Scaler ---
# Assuming the model and scaler are in the same directory as the Streamlit app
model_filename = 'best_logistic_regression_model.pkl'
scaler_filename = 'scaler.pkl'

try:
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)
    with open(scaler_filename, 'rb') as file:
        scaler = pickle.load(file)
    st.success("Model and Scaler loaded successfully!")
except Exception as e:
    st.error(f"Error loading model or scaler: {e}. Make sure '{model_filename}' and '{scaler_filename}' are in the same directory.")
    st.stop() # Stop the app if loading fails

# --- Streamlit UI ---
st.set_page_config(page_title="Breast Cancer Diagnosis App", layout="centered")
st.title("Breast Cancer Diagnosis Predictor")
st.write("Enter the patient's feature values to predict whether the tumor is Malignant or Benign.")

# --- Feature Input ---
# The features the model expects (from your X_train.columns)
feature_columns = ['Mean Radius', 'Mean Texture', 'Mean Perimeter', 'Mean Area',
       'Mean Smoothness', 'Mean Compactness', 'Mean Concavity',
       'Mean Concave Points', 'Mean Symmetry', 'Mean Fractal Dimension',
       'Radius SE', 'Texture SE', 'Perimeter SE', 'Area SE', 'Smoothness SE',
       'Compactness SE', 'Concavity SE', 'Concave Points SE', 'Symmetry SE',
       'Fractal Dimension SE', 'Worst Radius', 'Worst Texture',
       'Worst Perimeter', 'Worst Area', 'Worst Smoothness',
       'Worst Compactness', 'Worst Concavity', 'Worst Concave Points',
       'Worst Symmetry', 'Worst Fractal Dimension']

input_data = {}
# Create input fields for each feature
st.subheader("Tumor Feature Inputs")
for col in feature_columns:
    # Using a generic float input; you might want to customize min/max/step based on feature ranges
    input_data[col] = st.number_input(f"Enter {col}", value=0.0, format="%.4f", key=col)

# --- Prediction Button ---
if st.button("Predict Diagnosis"):
    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])

        # Scale the input data using the loaded scaler
        scaled_input = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(scaled_input)
        prediction_proba = model.predict_proba(scaled_input)

        st.subheader("Prediction Results:")
        if prediction[0] == 1:
            st.error(f"The model predicts a **Malignant** tumor (Probability: {prediction_proba[0][1]:.2f})")
        else:
            st.success(f"The model predicts a **Benign** tumor (Probability: {prediction_proba[0][0]:.2f})")

        st.write("--- Debug Information ---")
        st.write("Input Data (Unscaled):", input_df)
        st.write("Input Data (Scaled):", scaled_input)
        st.write("Prediction Probability:", prediction_proba)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

