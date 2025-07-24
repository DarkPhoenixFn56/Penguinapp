import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("Penguin Species Predictor")

# Load the model
try:
    with open('penguins_clf.pkl', 'rb') as f:
        clf = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Load the cleaned data for feature setup
@st.cache_data
def load_data():
    return pd.read_csv("penguins_cleaned.csv")

df = load_data()

# Set up user inputs
island = st.selectbox("Island", df["island"].unique())
sex = st.selectbox("Sex", df["sex"].unique())
bill_length_mm = st.slider("Bill Length (mm)", float(df["bill_length_mm"].min()), float(df["bill_length_mm"].max()))
bill_depth_mm = st.slider("Bill Depth (mm)", float(df["bill_depth_mm"].min()), float(df["bill_depth_mm"].max()))
flipper_length_mm = st.slider("Flipper Length (mm)", float(df["flipper_length_mm"].min()), float(df["flipper_length_mm"].max()))
body_mass_g = st.slider("Body Mass (g)", float(df["body_mass_g"].min()), float(df["body_mass_g"].max()))

# Prepare input data
input_dict = {
    'island': island,
    'bill_length_mm': bill_length_mm,
    'bill_depth_mm': bill_depth_mm,
    'flipper_length_mm': flipper_length_mm,
    'body_mass_g': body_mass_g,
    'sex': sex
}

input_df = pd.DataFrame([input_dict])

# One-hot encoding for categorical variables
full_df = pd.get_dummies(df.drop('species', axis=1))
input_df_encoded = pd.get_dummies(input_df)

# Ensure same columns as training data
missing_cols = set(full_df.columns) - set(input_df_encoded.columns)
for col in missing_cols:
    input_df_encoded[col] = 0

# Align column order
input_df_encoded = input_df_encoded[full_df.columns]

# Make prediction
try:
    prediction = clf.predict(input_df_encoded)
    st.subheader(f"Predicted Penguin Species: 🐧 {prediction[0]}")
except Exception as e:
    st.error(f"Prediction failed: {e}")
