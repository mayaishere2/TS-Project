"""
app.py
Streamlit interface to use the trained Random Forest pipeline (FFT + PCA).
"""

import streamlit as st
import pandas as pd
import joblib
import numpy as np
from scipy.fft import fft
from sklearn.base import BaseEstimator, TransformerMixin

st.title("Sleep Stage Classification (FFT + PCA + Random Forest)")
st.write("This app uses a trained Random Forest model to classify sleep stages from time-series data using FFT and PCA features.")
st.write("We had to find the fitting model to handle this case as it was unlabeled data and the steps we went through from data analysis to final model choosing are thouroughly documented in the notebook: https://colab.research.google.com/drive/16hfWjG8bzLXIEstCrhjPEklaF_nMSC9D#scrollTo=tZj63qR9_W_M")
st.write("Upload a CSV file containing raw time-series data (no FFT applied), or download our **training dataset** for testing.")

# ---- Custom FFT Transformer ----
class FFTTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.n_features_ = X.shape[1]
        return self

    def transform(self, X):
        return np.abs(fft(X))[:, :self.n_features_ // 2]

# ---- Load trained pipeline ----
@st.cache_resource
def load_pipeline():
    return joblib.load("rf_fft_pca_pipeline.joblib")

model = load_pipeline()

# ---- Provide sample data download (local file) ----
try:
    sample_df = pd.read_csv("SleepTrain5000.csv")
    st.download_button(
        label="Download SleepTrain5000.csv (Training Data)",
        data=sample_df.to_csv(index=False),
        file_name="SleepTrain5000.csv",
        mime="text/csv"
    )
except FileNotFoundError:
    st.warning("SleepTrain5000.csv not found in the app directory.")

# ---- Upload user file ----
uploaded_file = st.file_uploader("Upload a CSV of raw time-series (no FFT)", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:", df.head())

    if st.button("Predict"):
        X_input = df.values
        preds = model.predict(X_input)
        st.write(f"Predictions (first 20): {preds[:20]}")
        pred_df = pd.DataFrame({'Predicted_Class': preds})
        st.download_button("Download Predictions", pred_df.to_csv(index=False), "predictions.csv")
