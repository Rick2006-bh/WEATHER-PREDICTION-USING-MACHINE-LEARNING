import streamlit as st
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="Multi-City India Weather Predictor", layout="wide")

st.title("🌦️ Multi-City India Weather Predictor")
st.markdown("Select a city and adjust weather parameters to predict average temperature and rain chances using Random Forest models.")

# Check if model files exist
models_exist = os.path.exists("temp_model.pkl") and os.path.exists("rain_model.pkl") and os.path.exists("preprocessor.pkl")

if not models_exist:
    st.error("⚠️ Model files missing! Please run 'python train_model.py' in your terminal first to train the models.")
else:
    # Load assets
    with open("temp_model.pkl", "rb") as f:
        temp_model = pickle.load(f)
    with open("rain_model.pkl", "rb") as f:
        rain_model = pickle.load(f)
    with open("preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)

    # Sidebar layout for Inputs
    st.sidebar.header("📍 Input Conditions")
    
    # 1. City Selection (The feature you requested)
    city = st.sidebar.selectbox("Choose City", ["Chennai", "Kolkata", "Mumbai", "Srinagar"])
    
    # 2. Contextual Inputs
    month = st.sidebar.slider("Month of the Year", 1, 12, 6, help="1=Jan, 6=June, 12=Dec")
    sunshine_hours = st.sidebar.slider("Expected Sunshine (Hours)", 0.0, 13.0, 8.0)
    wind_speed = st.sidebar.slider("Max Wind Speed (km/h)", 0.0, 60.0, 15.0)

    # Convert sunshine hours back to seconds for the model processing
    sunshine_seconds = sunshine_hours * 3600

    # Layout Split into two visual cards
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Your Inputs Summary")
        input_data = pd.DataFrame({
            "City": [city],
            "Month": [month],
            "Sunshine_Seconds": [sunshine_seconds],
            "WindSpeed_Max": [wind_speed]
        })
        st.table(input_data.assign(Sunshine_Hours=sunshine_hours).drop(columns=['Sunshine_Seconds']))

    with col2:
        st.subheader("🔮 ML Model Predictions")
        
        if st.button("Predict Weather", type="primary"):
            # Preprocess inputs
            processed_inputs = preprocessor.transform(input_data)
            
            # Run Predictions
            pred_temp = temp_model.predict(processed_inputs)[0]
            pred_rain_class = rain_model.predict(processed_inputs)[0]
            pred_rain_proba = rain_model.predict_proba(processed_inputs)[0][1]

            # Display Temperature Result
            st.metric(label="Predicted Average Temperature", value=f"{pred_temp:.1f} °C")
            
            # Display Rain Result
            if pred_rain_class == 1:
                st.error(f"🌧️ Rain Predicted! (Confidence: {pred_rain_proba * 100:.1f}%)")
            else:
                st.success(f"☀️ No Rain Expected (Dry Confidence: {(1 - pred_rain_proba) * 100:.1f}%)")

    # Historical Reference Data view
    st.markdown("---")
    st.subheader("📂 Reference Historical Insights")
    if os.path.exists("Final_Weather_Dataset.csv"):
        raw_df = pd.read_csv("Final_Weather_Dataset.csv")
        city_summary = raw_df[raw_df['City'] == city][['Temp_Mean', 'Rain_mm', 'WindSpeed_Max']].describe()
        st.write(f"Historical averages for **{city}** based on your uploaded dataset:")
        st.dataframe(city_summary.T)