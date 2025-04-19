import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved models and transformers
model = joblib.load('hybrid_model.pkl')
xgb = joblib.load('xgb_model.pkl')
ohe = joblib.load('xgb_encoder.pkl')
preprocessor = joblib.load('data_preprocessor.pkl')

st.title("🛍️ eCommerce Customer Churn Predictor")

st.write("Fill in customer details to predict whether they will churn or not.")

# Input fields
user_input = {
    "Tenure": st.slider("Tenure", 0, 100, 12),
    "PreferredLoginDevice": st.selectbox("Preferred Login Device", ["Phone", "Mobile Phone"]),
    "CityTier": st.selectbox("City Tier", [1, 2, 3]),
    "WarehouseToHome": st.slider("Distance Warehouse to Home (km)", 1, 50, 5),
    "PreferredPaymentMode": st.selectbox("Payment Mode", ["Credit Card", "Debit Card", "UPI", "CC", "COD"]),
    "Gender": st.selectbox("Gender", ["Male", "Female"]),
    "HourSpendOnApp": st.slider("Hours Spent on App", 0.0, 10.0, 3.0),
    "NumberOfDeviceRegistered": st.slider("Devices Registered", 1, 10, 2),
    "PreferedOrderCat": st.selectbox("Preferred Order Category", ["Mobile", "Laptop & Accessory", "Others"]),
    "SatisfactionScore": st.slider("Satisfaction Score", 1, 5, 3),
    "MaritalStatus": st.selectbox("Marital Status", ["Single", "Married"]),
    "NumberOfAddress": st.slider("Number of Addresses", 1, 10, 2),
    "Complain": st.selectbox("Complain Registered", [0, 1]),
    "OrderAmountHikeFromlastYear": st.slider("Order Hike % from Last Year", 0, 100, 10),
    "CouponUsed": st.selectbox("Coupon Used", [0, 1]),
    "OrderCount": st.slider("Order Count", 0, 20, 3),
    "DaySinceLastOrder": st.slider("Days Since Last Order", 0, 100, 10),
    "CashbackAmount": st.slider("Cashback Amount", 0.0, 500.0, 100.0)
}

# Create DataFrame
input_df = pd.DataFrame([user_input])

if st.button("Predict Churn"):
    # Apply preprocessing
    processed = preprocessor.transform(input_df)
    
    # XGBoost tree encoding
    xgb_leaves = xgb.apply(processed)
    xgb_encoded = ohe.transform(xgb_leaves)

    # Reshape for LSTM/CNN input
    combined_input = np.concatenate([np.expand_dims(processed, axis=1), np.expand_dims(xgb_encoded, axis=1)], axis=2)

    # Predict
    prediction = model.predict(combined_input)
    pred_class = int(prediction.flatten()[0] > 0.5)

    st.success(f"Prediction: {'🔴 Will Churn' if pred_class == 1 else '🟢 Will Stay'}")
    st.metric("Churn Probability", f"{float(prediction.flatten()[0]):.2f}")
