import streamlit as st
import pandas as pd
from statsmodels.tsa.arima.model import ARIMAResults
import matplotlib.pyplot as plt
import warnings
import os
import requests
import base64

warnings.filterwarnings('ignore')

# GitHub API Config (add GITHUB_TOKEN as secret in Streamlit Cloud dashboard)
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", "")  # Fallback if not set
GITHUB_OWNER = "Muhammad-Owais194"
GITHUB_REPO = "Dsci_Assignment_04"
FEEDBACK_PATH = "user_feedback.csv"  # Path in repo

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
        st.subheader(f"{forecast_steps}-Day Forecast (Avg AQI: {avg_aqi:.1f})")
        
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
    st.write("Please provide your feedback on the generated prediction.")

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
                "Avg_Forecast_AQI": [forecast.mean() if forecast is not None else None]
            }
            fb_df = pd.DataFrame(feedback_entry)
            
            # Local save first
            file = "user_feedback.csv"
            if os.path.exists(file):
                existing_df = pd.read_csv(file)
                fb_df = pd.concat([existing_df, fb_df], ignore_index=True)
            fb_df.to_csv(file, index=False)
            
            # Commit to GitHub
            try:
                # Get current SHA (if file exists in repo)
                get_url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/contents/{FEEDBACK_PATH}"
                headers = {
                    "Accept": "application/vnd.github+json",
                    "Authorization": f"Bearer {GITHUB_TOKEN}",
                    "X-GitHub-Api-Version": "2022-11-28"
                }
                response = requests.get(get_url, headers=headers)
                sha = None
                if response.status_code == 200:
                    sha = response.json().get("sha")
                
                # Read and base64 encode the local CSV
                with open(file, "rb") as f:
                    content = f.read()
                    encoded_content = base64.b64encode(content).decode("utf-8")
                
                # Prepare payload
                payload = {
                    "message": "Update user_feedback.csv with new feedback",
                    "content": encoded_content,
                    "branch": "main"
                }
                if sha:
                    payload["sha"] = sha
                
                # PUT request to update/create file
                update_url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/contents/{FEEDBACK_PATH}"
                update_response = requests.put(update_url, json=payload, headers=headers)
                
                if update_response.status_code in (200, 201):
                    st.success("Thank you! Your feedback has been saved and committed to GitHub.")
                else:
                    st.warning("Feedback saved locally but GitHub commit failed: " + str(update_response.status_code))
                    st.write(update_response.json())
                
            except Exception as e:
                st.warning(f"Feedback saved locally but GitHub commit error: {e}")
            st.experimental_rerun()  # Refresh to show updated table

# Display Submitted Feedback (always try to load)
st.header("Submitted Feedback Data")
file_path = 'user_feedback.csv'
if os.path.exists(file_path):
    feedback_df = pd.read_csv(file_path)
    st.dataframe(feedback_df)  # Display the table of all feedback
    
    # Download button
    csv = feedback_df.to_csv(index=False)
    st.download_button(
        label="Download user_feedback.csv",
        data=csv,
        file_name='user_feedback.csv',
        mime='text/csv'
    )
else:
    st.info("No feedback submitted yet. Submit some to see the data here.")



