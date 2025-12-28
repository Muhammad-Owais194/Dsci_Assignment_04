import streamlit as st
import pandas as pd
from statsmodels.tsa.arima.model import ARIMAResults
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Built-in Google Sheets connection (no gspread!)
conn = st.connection("gsheets", type="gsheets")

# Load model and data
@st.cache_resource
def load_model():
    return ARIMAResults.load('arima_model.pkl')

model = load_model()

@st.cache_data
def load_data():
    df = pd.read_csv('df_final.csv', parse_dates=['date']).set_index('date')
    return df

df = load_data()

# AQI category (same as before)
def get_aqi_category(aqi):
    if aqi <= 50: return "Good", "Excellent for travel!"
    elif aqi <= 100: return "Moderate", "Acceptable."
    elif aqi <= 150: return "Unhealthy for Sensitive", "Limit exposure."
    elif aqi <= 200: return "Unhealthy", "Reconsider travel."
    elif aqi <= 300: return "Very Unhealthy", "Avoid travel."
    else: return "Hazardous", "Do not travel."

st.title("🌫️ Lahore Air Quality Predictor")
forecast_steps = st.sidebar.slider("Forecast Days", 1, 90, 30)

forecast = None
avg_aqi = None
if st.button("Generate Forecast"):
    forecast = model.forecast(steps=forecast_steps)
    avg_aqi = forecast.mean()
    st.subheader(f"Avg AQI: {avg_aqi:.1f}")
    fig, ax = plt.subplots()
    ax.plot(range(forecast_steps), forecast, marker='o')
    st.pyplot(fig)
    conf_int = model.get_forecast(steps=forecast_steps).conf_int()
    st.subheader("Confidence Intervals")
    st.dataframe(conf_int.head(10))
    category, advice = get_aqi_category(avg_aqi)
    st.subheader("Travel Advice")
    st.info(f"{category} – {advice}")

st.subheader("Recent Data")
st.dataframe(df.tail(10)[['aqi_pm2.5', 'avg_temp_f', 'avg_humidity_percent', 'avg_wind_speed_mph']])

# Feedback (after forecast)
if forecast is not None:
    st.header("Feedback")
    with st.form("feedback_form"):
        usability = st.slider("Usability (1-5)", 1, 5, 3)
        accuracy = st.slider("Accuracy (1-5)", 1, 5, 3)
        realistic = st.text_area("Realistic?")
        suggestions = st.text_area("Suggestions")
        submitted = st.form_submit_button("Submit")
        
        if submitted:
            data = [{"Usability": usability, "Accuracy": accuracy, "Realistic": realistic, "Suggestions": suggestions, "Avg_AQI": avg_aqi}]
            conn.update(worksheet="Sheet1", data=data)  # Appends
            st.success("Feedback saved!")
            st.rerun()

# Display feedback
st.header("Submitted Feedback")
try:
    feedback_df = conn.read(worksheet="Sheet1")
    if not feedback_df.empty:
        st.dataframe(feedback_df)
        csv = feedback_df.to_csv(index=False)
        st.download_button("Download CSV", csv, "feedback.csv", "text/csv")
    else:
        st.info("No feedback yet.")
except:
    st.info("Submit feedback to see it here.")

st.caption("Assignment 4 | BSCS-F22")
