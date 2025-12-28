import streamlit as st
import pandas as pd
from statsmodels.tsa.arima.model import ARIMAResults
import matplotlib.pyplot as plt
import warnings

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

# Function to categorize AQI and provide travel advice
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
        return "Hazardous", "Extremely hazardous; strongly advise against traveling to Lahore."

# App UI
st.title("🌫️ Lahore Air Quality Predictor")

st.sidebar.header("Forecast Settings")
forecast_steps = st.sidebar.slider("Forecast Days", min_value=1, max_value=90, value=30)

st.write("Click below to generate the AQI forecast based on historical data (2019–2023).")

if st.button("Generate Forecast"):
    try:
        # Generate forecast
        forecast = model.forecast(steps=forecast_steps)
        avg_aqi = forecast.mean()
        
        st.subheader(f"{forecast_steps}-Day AQI Forecast")
        st.write(f"**Average Forecasted AQI:** {avg_aqi:.1f}")
        st.write(f"**Min AQI:** {forecast.min():.1f} | **Max AQI:** {forecast.max():.1f}")
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(forecast_steps), forecast, marker='o', color='steelblue', label='Forecasted AQI')
        ax.set_title("Forecasted AQI Over Time")
        ax.set_xlabel("Days Ahead")
        ax.set_ylabel("AQI")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Confidence Intervals
        conf_int = model.get_forecast(steps=forecast_steps).conf_int()
        st.subheader("95% Confidence Intervals (First 10 Days)")
        st.dataframe(conf_int.head(10))
        
        # Travel Advice
        category, advice = get_aqi_category(avg_aqi)
        st.subheader("🧳 Travel Advice for Lahore")
        st.markdown(f"**AQI Category:** {category} (Average AQI: {avg_aqi:.1f})")
        st.info(advice)
        
    except Exception as e:
        st.error(f"Error generating forecast: {e}")

# Recent Historical Data
st.subheader("Recent Historical Data")
st.dataframe(df.tail(10)[['aqi_pm2.5', 'avg_temp_f', 'avg_humidity_percent', 'avg_wind_speed_mph']])

