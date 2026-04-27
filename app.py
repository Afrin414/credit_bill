import streamlit as st
import pandas as pd
import pickle

# Load model & scaler
import os
import pickle

model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')

with open(model_path, 'rb') as f:
    model = pickle.load(f)
import os
import pickle

scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.pkl')

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

st.title("💳 Credit Card Default Prediction")

st.write("Enter customer details:")

# ✅ Inputs (same as training features)
limit_bal = st.number_input("Credit Limit (LIMIT_BAL)", 10000, 1000000, 50000)
age = st.number_input("Age (AGE)", 18, 100, 30)
bill_amt1 = st.number_input("Last Bill Amount (BILL_AMT1)", 0, 1000000, 20000)
pay_amt1 = st.number_input("Last Payment Amount (PAY_AMT1)", 0, 1000000, 5000)

# Create dataframe
input_df = pd.DataFrame({
    'LIMIT_BAL': [limit_bal],
    'AGE': [age],
    'BILL_AMT1': [bill_amt1],
    'PAY_AMT1': [pay_amt1]
})

# Scale input
input_scaled = scaler.transform(input_df)

# Predict
if st.button("Predict"):

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    st.subheader("📊 Result:")

    if prediction[0] == 1:
        st.error("⚠️ High Risk: Likely to Default")
    else:
        st.success("✅ Low Risk: Safe Customer")

    # Show probability
    st.write(f"🔢 Default Probability: {probability[0][1]*100:.2f}%")