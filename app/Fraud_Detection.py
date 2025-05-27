import streamlit as st
import numpy as np
import joblib

# Load the model and scaler
model = joblib.load('models/random_forest_model.pkl')
scaler = joblib.load('models/scaler.pkl')

st.title("💳 Credit Card Fraud Detector")

# Collect input features from the user
st.header("🔍 Enter Transaction Data")

# There are 7 features
distance_from_home = st.number_input("Distance from home (m)", value=0.0)
distance_from_last_transaction = st.number_input("Distance from last transaction (m)", value=0.0)
ratio_to_median_purchase_price = st.number_input("Ratio of transaction amount to median of all transactions", value=0.0)

repeat_retailer = st.selectbox("Repeat retailer?", ("Yes", "No"))
used_chip = st.selectbox("Does the transaction use chip (credit card)?", ("Yes", "No"))
used_pin_number = st.selectbox("Does the transaction use PIN number?", ("Yes", "No"))
online_order = st.selectbox("Is the transaction an online order?", ("Yes", "No"))

# Convert Yes/No to 1/0
repeat_retailer = 1 if repeat_retailer == "Yes" else 0
used_chip = 1 if used_chip == "Yes" else 0
used_pin_number = 1 if used_pin_number == "Yes" else 0
online_order = 1 if online_order == "Yes" else 0

# Collect into array
input_data = np.array([[distance_from_home, 
                        distance_from_last_transaction, 
                        ratio_to_median_purchase_price, 
                        repeat_retailer, 
                        used_chip,
                        used_pin_number,
                        online_order]])

# Predict button
if st.button("🔎 Predict Fraud"):
  # Scale input
  input_scaled = scaler.transform(input_data)

  # Predict
  prediction = model.predict(input_scaled)[0]
  proba = model.predict_proba(input_scaled)[0][1]
  
  if prediction == 1:
    st.error(f"🚨 Fraud Detected! (Probability: {proba:.2f})")
  else:
    st.success(f"✅ Not Fraud (Probability of fraud: {proba:.2f})")