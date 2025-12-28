import streamlit as st
import pandas as pd
from statsmodels.tsa.arima.model import ARIMAResults
import matplotlib.pyplot as plt
import warnings
import gspread
from google.oauth2.service_account import Credentials
import json

warnings.filterwarnings('ignore')

# Google Sheets Setup
SHEET_ID = st.secrets["SHEET_ID"]
GOOGLE_SERVICE_ACCOUNT = json.loads(st.secrets["GOOGLE_SERVICE_ACCOUNT"])

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
creds = Credentials.from_service_account_info(GOOGLE_SERVICE_ACCOUNT, scopes=scope)
client = gspread.authorize(creds)
sheet = client.open_by_key(SHEET_ID).sheet1  # Or specify worksheet name

# Load model and data (same as before)
@st.cache_resource
def load_model():
    return ARIMAResults.load('arima_model.pkl')

model = load_model()

@st.cache_data
def load_data():
    df = pd.read_csv('df_final.csv', parse_dates=['date']).set_index('date')
    return df

df = load_data()

# ... (keep your AQI category function, title, forecast code, travel advice — same as previous)

# Feedback Form
if forecast is not None:
    st.header("📝 Feedback on This Prediction")
    with st.form("feedback_form"):
        usability = st.slider("App Usability (1-5)", 1, 5, 3)
        accuracy = st.slider("Prediction Accuracy (1-5)", 1, 5, 3)
        realistic = st.text_area("Was the prediction realistic?")
        suggestions = st.text_area("General suggestions")
        submitted = st.form_submit_button("Submit Feedback")
        
        if submitted:
            row = [usability, accuracy, realistic, suggestions, avg_aqi]
            sheet.append_row(row)
            st.success("Feedback saved to Google Sheets!")
            st.experimental_rerun()

# Display Feedback
st.header("Submitted Feedback")
try:
    records = sheet.get_all_records()
    if records:
        feedback_df = pd.DataFrame(records)
        st.dataframe(feedback_df)
        csv = feedback_df.to_csv(index=False)
        st.download_button("Download CSV", csv, "user_feedback.csv", "text/csv")
    else:
        st.info("No feedback yet.")
except:
    st.info("Feedback will appear here after first submission.")
