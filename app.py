import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
from statsmodels.tsa.arima.model import ARIMAResults
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Built-in GSheets connection (uses secrets directly)
conn = st.connection("gsheets", type=GSheetsConnection)

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

# AQI function (same as before)
def get_aqi_category(aqi):
    if aqi <= 50: return "Good", "Excellent for travel!"
    elif aqi <= 100: return "Moderate", "Acceptable, sensitive groups take care."
    elif aqi <= 150: return "Unhealthy for Sensitive", "Limit outdoor time if sensitive."
    elif aqi <= 200: return "Unhealthy", "Reconsider travel or wear mask."
    elif aqi <= 300: return "Very Unhealthy", "Avoid travel if possible."
    else: return "Hazardous", "Strongly advise against traveling."

st.title("🌫️ Lahore Air Quality Predictor")
st.sidebar.header("Forecast Settings")
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
    category, advice = get_aqi_category(avg_aqi)
    st.subheader("🧳 Travel Advice")
    st.markdown(f"**{category}** – {advice}")

# Recent data
st.subheader("Recent Data")
st.dataframe(df.tail(10)[['aqi_pm2.5', 'avg_temp_f', 'avg_humidity_percent', 'avg_wind_speed_mph']])

# Feedback
if forecast is not None:
    st.header("📝 Feedback")
    with st.form("feedback_form"):
        usability = st.slider("Usability (1-5)", 1, 5, 3)
        accuracy = st.slider("Accuracy (1-5)", 1, 5, 3)
        realistic = st.text_area("Was prediction realistic?")
        suggestions = st.text_area("Suggestions")
        submitted = st.form_submit_button("Submit")
        
        if submitted:
            data = [{"Usability": usability, "Accuracy": accuracy, "Realistic": realistic, "Suggestions": suggestions, "Avg_AQI": avg_aqi}]
            conn.update(worksheet="Sheet1", data=data)  # Appends to Sheet1
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
    st.info("Feedback will appear after first submission.")
