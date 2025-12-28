import streamlit as st
import pandas as pd
from statsmodels.tsa.arima.model import ARIMAResults
import matplotlib.pyplot as plt
import warnings
import os
from io import StringIO

warnings.filterwarnings('ignore')

# Load pre-trained ARIMA model using statsmodels native load
@st.cache_resource
def load_model():
    model = ARIMAResults.load('arima_model.pkl')
    return model

model = load_model()

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('df_final.csv', parse_dates=['date'])
    df = df.sort_values('date').set_index('date')
    return df

df = load_data()

# AQI Category & Travel Advice
def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good", "Air quality is excellent for travel to Lahore. Enjoy your trip!"
    elif aqi <= 100:
        return "Moderate", "Air quality is acceptable, but sensitive people should take care."
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "Not ideal for travel if you have respiratory issues."
    elif aqi <= 200:
        return "Unhealthy", "Poor air quality — reconsider travel or wear a mask."
    elif aqi <= 300:
        return "Very Unhealthy", "Very poor air quality; avoid travel if possible."
    else:
        return "Hazardous", "Extremely dangerous — strongly advise against traveling."

# Title & UI
st.title("🌫️ Lahore Air Quality Predictor")

st.sidebar.header("Forecast Settings")
forecast_steps = st.sidebar.slider("Forecast Days", 1, 90, 30)

st.write("Generate a forecast to see predicted AQI and travel advice.")

forecast = None
avg_aqi = None
if st.button("Generate Forecast"):
    try:
        forecast = model.forecast(steps=forecast_steps)
        avg_aqi = forecast.mean()
        
        st.subheader(f"{forecast_steps}-Day Forecast")
        st.write(f"**Average Forecasted AQI:** {avg_aqi:.1f}")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(forecast_steps), forecast, marker='o', color='steelblue')
        ax.set_title("Forecasted AQI Over Time")
        ax.set_xlabel("Days Ahead")
        ax.set_ylabel("AQI")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Confidence intervals
        conf_int = model.get_forecast(steps=forecast_steps).conf_int()
        st.subheader("95% Confidence Intervals (First 10 Days)")
        st.dataframe(conf_int.head(10))
        
        # Travel Advice
        category, advice = get_aqi_category(avg_aqi)
        st.subheader("🧳 Travel Advice for Lahore")
        st.markdown(f"**AQI Category:** {category} (Avg AQI: {avg_aqi:.1f})")
        st.info(advice)
        
    except Exception as e:
        st.error(f"Error generating forecast: {e}")

# Recent Data
st.subheader("Recent Historical Data")
st.dataframe(df.tail(10)[['aqi_pm2.5', 'avg_temp_f', 'avg_humidity_percent', 'avg_wind_speed_mph']])

# Feedback Form (only after forecast)
if forecast is not None:
    st.header("📝 Feedback on This Prediction")
    with st.form("feedback_form"):
        usability = st.slider("App Usability (1-5)", 1, 5, 3)
        accuracy = st.slider("Prediction Accuracy (1-5)", 1, 5, 3)
        realistic = st.text_area("Was the prediction realistic?")
        suggestions = st.text_area("General suggestions")
        submitted = st.form_submit_button("Submit Feedback")
        
        if submitted:
            feedback_entry = {
                "Usability": [usability],
                "Accuracy": [accuracy],
                "Realistic_Feedback": [realistic],
                "Suggestions": [suggestions],
                "Avg_Forecast_AQI": [avg_aqi]
            }
            fb_df = pd.DataFrame(feedback_entry)
            file = "user_feedback.csv"
            if os.path.exists(file):
                existing_df = pd.read_csv(file)
                fb_df = pd.concat([existing_df, fb_df], ignore_index=True)
            fb_df.to_csv(file, index=False)
            st.success("Thank you! Your feedback has been saved.")

# Download Feedback CSV Button (admin feature)
st.header("Download Feedback Data")
file_path = 'user_feedback.csv'
if os.path.exists(file_path):
    with open(file_path, 'r') as f:
        csv_data = f.read()
    st.download_button(
        label="Download user_feedback.csv",
        data=csv_data,
        file_name='user_feedback.csv',
        mime='text/csv'
    )
else:
    st.info("No feedback submitted yet. Submit some to generate the CSV.")

st.info("Model: ARIMA saved with statsmodels native method | Assignment 4 Deployment")
st.caption("Data Science | BSCS-F22")
