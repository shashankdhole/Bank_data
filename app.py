import streamlit as st
import pandas as pd
import joblib
import numpy as np

scaler = joblib.load("scaler.pkl")
kmeans = joblib.load("kmeans_model.pkl")

st.set_page_config(page_title="Bank Customer Segmentation")
st.title("🏦 Bank Customer Segmentation App")

age = st.number_input("Age", 18, 100, 30)
balance = st.number_input("Account Balance", value=1000)
duration = st.number_input("Last Call Duration (seconds)", value=100)
campaign = st.number_input("Campaign Contacts", min_value=1, value=1)
previous = st.number_input("Previous Contacts", min_value=0, value=0)

cluster_labels = {
    0: "🟢 High Engagement, High Value Customers",
    1: "🔴 Low Engagement, Low Value Customers",
    2: "🟡 Loyal Customers with Moderate Engagement"
}

if st.button("Predict Customer Segment"):
    # 🔥 NUMPY ARRAY — NO FEATURE NAMES
    input_array = np.array([[age, balance, duration, campaign, previous]])

    scaled = scaler.transform(input_array)
    cluster = kmeans.predict(scaled)[0]

    st.success(f"Customer Segment: {cluster_labels[cluster]}")

