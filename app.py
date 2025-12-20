%%writefile app.py
import streamlit as st
import pandas as pd
from statsmodels.tsa.arima.model import ARIMAResults
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')

# Load pre-trained ARIMA model using statsmodels native load
@st.cache_resource
def load_model():
    model = ARIMAResults.load('arima_model.pkl')  # Use the new file name
    return model

model = load_model()

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('df_final.csv', parse_dates=['date'])
    df = df.sort_values('date').set_index('date')
    return df

df = load_data()

# AQI Category Function
def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good", "Air quality is excellent for travel to Lahore. Enjoy your trip!"
    elif aqi <= 100:
        return "Moderate", "Air quality is acceptable for travel, but sensitive individuals should take precautions."
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "Not ideal for travel if you have respiratory issues."
    elif aqi <= 200:
        return "Unhealthy", "Poor air quality; reconsider travel or wear masks."
    elif aqi <= 300:
        return "Very Unhealthy", "Very poor air quality; avoid travel if possible."
    else:
        return "Hazardous", "Extremely hazardous; strongly advise against traveling."

# Title & UI
st.title('Lahore Air Quality Predictor')
st.sidebar.header('Forecast Settings')
forecast_steps = st.sidebar.slider('Forecast Days', 1, 90, 30)

st.write("Generate a forecast to see predicted AQI and travel advice.")

forecast = None
if st.button('Generate Forecast'):
    try:
        forecast = model.forecast(steps=forecast_steps)
        
        avg_aqi = forecast.mean()
        st.subheader(f'{forecast_steps}-Day Forecast (Avg AQI: {avg_aqi:.2f})')
        
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(range(forecast_steps), forecast, marker='o', color='blue')
        ax.set_title('Forecasted AQI')
        ax.set_xlabel('Days Ahead')
        ax.set_ylabel('AQI')
        ax.grid(True)
        st.pyplot(fig)
        
        conf_int = model.get_forecast(steps=forecast_steps).conf_int()
        st.subheader('Confidence Intervals')
        st.dataframe(conf_int.head(10))
        
        category, advice = get_aqi_category(avg_aqi)
        st.subheader('Travel Advice')
        st.write(f"**{category}** – {advice}")
        
    except Exception as e:
        st.error(f"Error: {e}")

# Recent Data
st.subheader('Recent Historical Data')
st.dataframe(df.tail(10)[['aqi_pm2.5', 'avg_temp_f', 'avg_humidity_percent', 'avg_wind_speed_mph']])

# Feedback Form (after forecast)
if forecast is not None:
    st.header('Feedback on This Prediction')
    with st.form('feedback_form'):
        usability = st.slider('Usability (1-5)', 1, 5, 3)
        accuracy = st.slider('Prediction Accuracy (1-5)', 1, 5, 3)
        pred_feedback = st.text_area('Was the prediction realistic?')
        suggestions = st.text_area('Suggestions')
        submitted = st.form_submit_button('Submit')
        
        if submitted:
            data = {'Usability': [usability], 'Accuracy': [accuracy],
                    'Prediction_Feedback': [pred_feedback], 'Suggestions': [suggestions],
                    'Avg_AQI': [avg_aqi]}
            df_fb = pd.DataFrame(data)
            file = 'user_feedback.csv'
            if os.path.exists(file):
                df_fb = pd.concat([pd.read_csv(file), df_fb])
            df_fb.to_csv(file, index=False)
            st.success('Feedback saved!')

st.info("Deployed for Assignment 4 | Model uses statsmodels native serialization.")
st.markdown("---")
st.write("BSCS-F22 | Data Science")
