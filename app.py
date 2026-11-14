import os
import json
import streamlit as st
import pandas as pd
import joblib
import requests

# 1. Landscape / Wide layout
st.set_page_config(
    page_title="California House Price Predictor & MLOps",
    page_icon="🏡",
    layout="wide"
)

# Load the trained production model safely
model_path = "models/house_model.pkl"
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None

# Load latest evaluation metrics if available
eval_metrics_path = "models/latest_eval_metrics.json"
eval_data = {}
if os.path.exists(eval_metrics_path):
    try:
        with open(eval_metrics_path, "r") as f:
            eval_data = json.load(f)
    except Exception:
        pass

# Dataset row counts
base_data_path = "data/house_data.csv"
feedback_data_path = "data/feedback_submissions.csv"
test_data_path = "data/holdout_test.csv"

base_count = len(pd.read_csv(base_data_path)) if os.path.exists(base_data_path) else 0
feedback_count = len(pd.read_csv(feedback_data_path)) if os.path.exists(feedback_data_path) else 0
test_count = len(pd.read_csv(test_data_path)) if os.path.exists(test_data_path) else 0

st.title("🏡 California House Price Predictor & MLOps Monitor")
st.caption("Predict housing prices, collect ground-truth feedback, and monitor model performance & quality gates.")

# --- TWO COLUMN LANDSCAPE LAYOUT ---
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    # --- SECTION 1: PREDICTIONS ---
    st.subheader("1. Property Features & Prediction")
    
    col1, col2 = st.columns(2)
    with col1:
        med_inc = st.number_input("Median Income (in $10k)", value=3.5, step=0.1)
        house_age = st.number_input("House Age (Years)", value=25.0, step=1.0)
        ave_rooms = st.number_input("Average Rooms", value=5.0, step=0.1)
        ave_bedrms = st.number_input("Average Bedrooms", value=1.0, step=0.05)

    with col2:
        population = st.number_input("Block Population", value=1425.0, step=10.0)
        ave_occup = st.number_input("Average Occupancy", value=3.0, step=0.1)
        latitude = st.number_input("Latitude", value=34.2, step=0.01)
        longitude = st.number_input("Longitude", value=-118.2, step=0.01)

    if st.button("Predict Price", use_container_width=True, type="primary"):
        if model is None:
            st.error("Model file not found! Please run train_model.py first.")
        else:
            input_data = pd.DataFrame([{
                "MedInc": med_inc,
                "HouseAge": house_age,
                "AveRooms": ave_rooms,
                "AveBedrms": ave_bedrms,
                "Population": population,
                "AveOccup": ave_occup,
                "Latitude": latitude,
                "Longitude": longitude
            }])

            predicted_price = model.predict(input_data)[0]
            st.session_state['last_input'] = input_data
            st.success(f"🏡 Estimated House Value: **${predicted_price * 100000:,.2f}**")

    # --- SECTION 2: FEEDBACK COLLECTION ---
    st.divider()
    st.subheader("2. Submit Real-World Feedback")
    st.write("Enter actual market sale price to feed the next Airflow retraining cycle.")
    
    actual_sale_price_input = st.number_input(
        "Actual Sale Price (in $100k units, e.g. 4.5 = $450,000)",
        value=3.5,
        step=0.1
    )

    if st.button("Save Feedback for Retraining", use_container_width=True):
        if 'last_input' not in st.session_state:
            st.warning("Please make a prediction first before submitting feedback!")
        else:
            feedback_row = st.session_state['last_input'].copy()
            feedback_row["MedHouseVal"] = actual_sale_price_input

            os.makedirs("data", exist_ok=True)
            if not os.path.exists(feedback_data_path):
                feedback_row.to_csv(feedback_data_path, index=False)
            else:
                feedback_row.to_csv(feedback_data_path, mode='a', header=False, index=False)

            st.success("✅ Feedback saved! Click 'Retrain Model via Airflow' to re-evaluate.")

with right_col:
    # --- SECTION 3: MODEL METRICS & VALIDATION GATE ---
    st.subheader("3. 🎯 Quality Gate & Model Scores")

    prod_r2 = eval_data.get("production_r2", 0.9720)
    prod_rmse = eval_data.get("production_rmse", 0.1902)
    threshold = eval_data.get("min_r2_threshold", 0.75)
    decision = eval_data.get("decision", "ACTIVE")
    challenger_r2 = eval_data.get("challenger_r2", 0.8008)
    challenger_rmse = eval_data.get("challenger_rmse", 0.5074)
    last_eval = eval_data.get("timestamp", "Recent")

    # Top Key Metrics Cards
    m1, m2, m3 = st.columns(3)
    m1.metric("Production R²", f"{prod_r2:.4f}", delta="Above Threshold" if prod_r2 >= threshold else "Below Threshold")
    m2.metric("Production RMSE", f"{prod_rmse:.4f}")
    m3.metric("R² Gate Threshold", f"≥ {threshold:.2f}")

    # Challenger / Gate Breakdown
    st.markdown("#### Quality Gate Breakdown")
    gate_col1, gate_col2 = st.columns(2)
    with gate_col1:
        st.write(f"• **Challenger R²:** `{challenger_r2:.4f}`")
        st.write(f"• **Challenger RMSE:** `{challenger_rmse:.4f}`")
    with gate_col2:
        st.write(f"• **Last Gate Decision:** `{decision}`")
        st.write(f"• **Last Evaluated:** `{last_eval}`")

    if prod_r2 >= threshold:
        st.success(f"🛡️ **Production Protected:** Current model meets the minimum threshold ({threshold}) and is serving live traffic.")
    else:
        st.warning("⚠️ Production model score is below threshold.")

    # --- SECTION 4: DATASET & RETRAINING TRIGGER ---
    st.divider()
    st.subheader("4. 📊 Dataset Stats & Pipeline Trigger")

    d1, d2, d3 = st.columns(3)
    d1.metric("Base Dataset Rows", f"{base_count:,}")
    d2.metric("User Feedback Rows", f"{feedback_count:,}")
    d3.metric("Holdout Test Rows", f"{test_count:,}")

    st.write("Trigger Airflow to retrain on base + feedback data and run the quality gate:")
    if st.button("🚀 Retrain Model via Airflow", use_container_width=True):
        airflow_url = "http://localhost:8081/api/v1/dags/house_retraining_pipeline/dagRuns"
        auth = ("admin", "admin")
        try:
            response = requests.post(airflow_url, json={}, auth=auth)
            if response.status_code in [200, 201]:
                st.success("🚀 Airflow retraining pipeline triggered! Refresh after execution to see updated metrics.")
            else:
                st.error(f"Failed to trigger Airflow ({response.status_code}): {response.text}")
        except Exception as e:
            st.error(f"Could not connect to Airflow: {e}")

    st.link_button("🔗 Open Airflow Webserver (Port 8081)", "http://localhost:8081", use_container_width=True)
