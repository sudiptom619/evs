import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from meteostat import Point, Hourly
from streamlit_autorefresh import st_autorefresh


# APP CONFIGURATION

st.set_page_config(
page_title="Realtime Rainfall Predictor",
page_icon="🌧️",
layout="centered"
)

st.title("🌦️ Realtime Rainfall Prediction Dashboard")
st.markdown("#### Predict rainfall probability from live weather data.")



# LOAD MODEL + PIPELINE



@st.cache_resource
def load_model():
  model = joblib.load("best_model.pkl")
  pipeline = joblib.load("pipeline.pkl")
  return model, pipeline

model, pipeline = load_model()



# REFRESH AUTOMATICALLY EVERY 10 MINUTES



st_autorefresh(interval=10 * 60 * 1000, key="data_refresh")



# LOCATION SELECTION



st.sidebar.header("📍 Select Location")
locations = {
"Ranchi": (23.3441, 85.3096),
"Kolkata": (22.5726, 88.3639),
"Delhi": (28.6139, 77.2090),
"Mumbai": (19.0760, 72.8777),
"Chennai": (13.0827, 80.2707),
"Bangalore": (12.9716, 77.5946)
}
choice = st.sidebar.selectbox("Choose a city", list(locations.keys()))
lat, lon = locations[choice]



# FETCH LIVE WEATHER DATA



with st.spinner("Fetching latest weather data..."):
  try:
    location = Point(lat, lon)
    start = datetime.now() - timedelta(hours=1)
    end = datetime.now()
    data = Hourly(location, start, end)
    data = data.fetch()

    
    if data.empty:
        st.error("⚠️ No data available for this location right now.")
    else:
        # Reset index so 'time' becomes a column if it was the index
      latest = data.tail(1).reset_index()

      # Get timestamp safely (some Meteostat DataFrames have 'time' as index)
      if 'time' in latest.columns:
          fetch_time = latest.iloc[0]['time']
      else:
          fetch_time = latest.index[0]  # fallback

      st.success(f"✅ Live data fetched for {choice} at {fetch_time:%H:%M %p}")
      st.dataframe(latest)


        # -------------------------------
        #  FEATURE EXTRACTION
        # -------------------------------
        # Map Meteostat columns to your model features
      feature_map = {
            "temp": "temparature",
            "pres": "pressure",
            "rhum": "humidity",
            "dwpt": "dewpoint",
            "wspd": "windspeed"
        }

      for meteostat_col, model_col in feature_map.items():
            if meteostat_col in latest.columns:
                latest.rename(columns={meteostat_col: model_col}, inplace=True)

        # Fill any missing columns with zeros
      model_features = pipeline.feature_names_in_
      for col in model_features:
            if col not in latest.columns:
                latest[col] = 0

      X_live = latest[model_features]


        #  PREDICTION

      X_scaled = pipeline.transform(X_live)
      prediction = model.predict(X_scaled)[0]

      st.markdown("### 🌤️ Rainfall Prediction:")
      if prediction == 1:
            st.success("☔ **Rain likely today. Keep an umbrella handy!**")
      else:
            st.info("🌞 **No rain expected. Enjoy your day!**")

  except Exception as e:
    st.error(f"Error fetching data: {e}")



# FOOTER



st.markdown("---")
st.caption("Built with ❤️ by BCS2A — powered by Meteostat & Streamlit.")
