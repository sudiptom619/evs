
import ssl, certifi, warnings
warnings.filterwarnings("ignore")

ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from meteostat import Point, Hourly
from streamlit_autorefresh import st_autorefresh
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from streamlit_folium import st_folium
import folium
import asyncio
import aiohttp



# Page Setup



st.set_page_config(page_title="Realtime Rainfall Predictor", page_icon="🌧️", layout="centered")
st.title("🌦️ Realtime Rainfall Prediction Dashboard")
st.caption("Fetch live weather data from any location and predict rainfall in real time.")


# Cached Loaders


@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = joblib.load("best_model.pkl")
    pipeline = joblib.load("pipeline.pkl")
    return model, pipeline

@st.cache_resource(show_spinner=False)
def get_geolocator():
    geolocator = Nominatim(user_agent="rainfall_app")
    return RateLimiter(geolocator.geocode, min_delay_seconds=1, swallow_exceptions=True)

# Load model/pipeline

try:
    model, pipeline = load_artifacts()
except Exception as e:
    st.error(f"❌ Could not load model/pipeline: {e}")
    st.stop()


# Auto Refresh every 10 minutes


st_autorefresh(interval=10 * 60 * 1000, key="auto_refresh")


# Sidebar: Location Selection

st.sidebar.header("📍 Choose Location Input Mode")

mode = st.sidebar.radio(
"Select method:",
["🗺️ Map (click anywhere)", "🏙️ Type a city name", "📐 Manual coordinates"],
index=0
)

if "lat" not in st.session_state:
    st.session_state.lat = None
if "lon" not in st.session_state:
    st.session_state.lon = None

lat = st.session_state.lat
lon = st.session_state.lon

# Mode 1 — Map Click

map_data = None

if mode.startswith("🗺️"):
    st.sidebar.write("Click anywhere on the map to select a location.")
    default_lat, default_lon = 22.5726, 88.3639
    m = folium.Map(location=[default_lat, default_lon], zoom_start=5)
    folium.LatLngPopup().add_to(m)
    map_data = st_folium(m, width=700, height=500)

new_lat, new_lon = st.session_state.lat, st.session_state.lon

if map_data and map_data.get("last_clicked"):
    clicked = map_data["last_clicked"]
    if clicked:
        new_lat, new_lon = clicked["lat"], clicked["lng"]

if new_lat is not None and new_lon is not None:
    if new_lat != st.session_state.lat or new_lon != st.session_state.lon:
        st.session_state.lat, st.session_state.lon = new_lat, new_lon
        st.rerun()



# Mode 2 — City Name Geocoding (FIXED + VISIBLE SEARCH BOX)

elif mode.startswith("🏙️"):
    st.sidebar.write("Type a city name below to search and select from suggestions.")


    geocode = get_geolocator()

    if "city_name" not in st.session_state:
        st.session_state.city_name = ""
    if "suggestions" not in st.session_state:
        st.session_state.suggestions = []

    city_input = st.sidebar.text_input(
        "Enter city name:",
        value=st.session_state.city_name,
        key="city_input_box",
        placeholder="e.g., Kolkata, Delhi, Mumbai..."
    )

    if len(city_input.strip()) > 2 and city_input != st.session_state.city_name:
        try:
            with st.spinner("🔍 Searching for matches..."):
                nom = Nominatim(user_agent="rainfall_app_suggestions")
                results = nom.geocode(city_input, exactly_one=False, limit=5, addressdetails=True)
                if results:
                    st.session_state.suggestions = [res.address for res in results]
                else:
                    st.session_state.suggestions = []
        except Exception as e:
            st.sidebar.warning(f"Geocoding error: {e}")
            st.session_state.suggestions = []

    if st.session_state.suggestions:
        st.sidebar.write("Suggestions:")
        selected = st.sidebar.radio(
            "Choose a location:",
            st.session_state.suggestions,
            key="suggestion_choice"
        )

        if selected:
            loc = geocode(selected)
            if loc:
                st.session_state.city_name = selected
                st.session_state.lat = float(loc.latitude)
                st.session_state.lon = float(loc.longitude)
                st.sidebar.success(f"📍 {selected}: ({loc.latitude:.4f}, {loc.longitude:.4f})")
                st.success(f"✅ Selected: {selected}")
                st.map(pd.DataFrame([[loc.latitude, loc.longitude]], columns=["lat", "lon"]))
            else:
                st.sidebar.warning("⚠️ Could not geocode the selected city.")
    else:
        st.sidebar.info("Type at least 3 letters to see suggestions.")


# Mode 3 — Manual Coordinates

if mode.startswith("📐"):
    lat = st.sidebar.number_input("Latitude:", value=22.5726, format="%.6f")
    lon = st.sidebar.number_input("Longitude:", value=88.3639, format="%.6f")
if st.sidebar.button("Set Location"):
    st.session_state.lat, st.session_state.lon = lat, lon
    st.rerun()


# Fetch Live Weather + Predict


lat = st.session_state.lat
lon = st.session_state.lon

if lat is not None and lon is not None:
    try:
        location = Point(lat, lon)
        end = datetime.utcnow()
        start = end - timedelta(hours=6)
        data = Hourly(location, start, end).fetch()


        if data.empty:
            st.warning("⚠️ No live data available for this location right now.")
        else:
            latest = data.tail(1).reset_index()
            fetch_time = latest["time"].iloc[0] if "time" in latest.columns else latest.index[0]

            st.success(f"✅ Live data fetched for ({lat:.4f}, {lon:.4f}) at {fetch_time:%Y-%m-%d %H:%M UTC}")

            if not latest.empty:
                row = latest.iloc[0]
                temp = row.get("temp", np.nan)
                dewpt = row.get("dwpt", np.nan)
                rh = row.get("rhum", np.nan)
                pressure = row.get("pres", np.nan)
                wspd = row.get("wspd", np.nan)
                prcp = row.get("prcp", np.nan)

                st.markdown("""
                    <style>
                    .metric-card {
                        background: linear-gradient(135deg, rgba(255,255,255,0.85), rgba(240,248,255,0.95));
                        border-radius: 18px;
                        padding: 20px;
                        margin: 10px;
                        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                        text-align: center;
                        transition: all 0.3s ease;
                    }
                    .metric-card:hover {
                        transform: translateY(-5px);
                        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
                    }
                    .metric-label {
                        font-size: 1.1rem;
                        color: #333;
                        font-weight: 600;
                    }
                    .metric-value {
                        font-size: 1.8rem;
                        font-weight: 700;
                        color: #0077b6;
                        margin-top: 5px;
                    }
                    </style>
                """, unsafe_allow_html=True)

                col1, col2, col3 = st.columns(3)
                col4, col5, col6 = st.columns(3)

                col1.markdown(f"<div class='metric-card'><div class='metric-label'>🌡️ Temperature</div><div class='metric-value'>{temp:.1f} °C</div></div>", unsafe_allow_html=True)
                col2.markdown(f"<div class='metric-card'><div class='metric-label'>💧 Dew Point</div><div class='metric-value'>{dewpt:.1f} °C</div></div>", unsafe_allow_html=True)
                col3.markdown(f"<div class='metric-card'><div class='metric-label'>🌫️ Humidity</div><div class='metric-value'>{rh:.0f}%</div></div>", unsafe_allow_html=True)
                col4.markdown(f"<div class='metric-card'><div class='metric-label'>🌬️ Wind Speed</div><div class='metric-value'>{wspd:.1f} km/h</div></div>", unsafe_allow_html=True)
                col5.markdown(f"<div class='metric-card'><div class='metric-label'>📊 Pressure</div><div class='metric-value'>{pressure:.1f} hPa</div></div>", unsafe_allow_html=True)
                col6.markdown(f"<div class='metric-card'><div class='metric-label'>🌧️ Precipitation</div><div class='metric-value'>{prcp:.2f} mm</div></div>", unsafe_allow_html=True)
            else:
                st.info("No recent weather data found.")

# 🌍 AIR QUALITY SECTION (ASYNC FIXED)

            st.subheader("🌍 Air Quality Overview")

            async def fetch_air_quality(lat, lon):
                url = (
                    f"https://air-quality-api.open-meteo.com/v1/air-quality?"
                    f"latitude={lat}&longitude={lon}"
                    "&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,european_aqi"
                )
                timeout = aiohttp.ClientTimeout(total=10)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            return await response.json()
                        return None

            async def get_air_quality_data(lat, lon):
                aq_data = await fetch_air_quality(lat, lon)
                if not aq_data or "hourly" not in aq_data or not aq_data["hourly"]:
                    await asyncio.sleep(2)
                    aq_data = await fetch_air_quality(lat, lon)
                return aq_data

            async def main():
                try:
                    aq_data = await get_air_quality_data(lat, lon)
                    if not aq_data or "hourly" not in aq_data:
                        st.warning("⚠️ Could not fetch air quality data right now.")
                        return

                    air = pd.DataFrame(aq_data["hourly"])
                    latest_air = air.tail(1).reset_index(drop=True)

                    pm10 = latest_air["pm10"][0]
                    pm25 = latest_air["pm2_5"][0]
                    spm = 1.3 * pm10

                    c8 = st.columns(1)[0]
                    c8.markdown(
                        f"<div class='aq-card'><div class='aq-label'>🌪️ SPM</div><div class='aq-value'>{spm:.1f} µg/m³</div></div>",
                        unsafe_allow_html=True,
                    )

                    co = latest_air["carbon_monoxide"][0]
                    no2 = latest_air["nitrogen_dioxide"][0]
                    so2 = latest_air["sulphur_dioxide"][0]
                    o3 = latest_air["ozone"][0]
                    aqi = latest_air["european_aqi"][0]

                    st.markdown("""
                        <style>
                        .aq-card {
                            background: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(245,248,255,0.9));
                            border-radius: 18px;
                            padding: 20px;
                            margin: 10px;
                            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                            text-align: center;
                            transition: all 0.3s ease;
                        }
                        .aq-card:hover {
                            transform: translateY(-4px);
                            box-shadow: 0 8px 18px rgba(0,0,0,0.15);
                        }
                        .aq-label {
                            font-size: 1.1rem;
                            color: #333;
                            font-weight: 600;
                        }
                        .aq-value {
                            font-size: 1.8rem;
                            font-weight: 700;
                            color: #0077b6;
                            margin-top: 5px;
                        }
                        </style>
                    """, unsafe_allow_html=True)

                    c1, c2, c3 = st.columns(3)
                    c4, c5, c6, c7 = st.columns(4)

                    c1.markdown(f"<div class='aq-card'><div class='aq-label'>🌫️ PM₂․₅</div><div class='aq-value'>{pm25:.1f} µg/m³</div></div>", unsafe_allow_html=True)
                    c2.markdown(f"<div class='aq-card'><div class='aq-label'>🌫️ PM₁₀</div><div class='aq-value'>{pm10:.1f} µg/m³</div></div>", unsafe_allow_html=True)
                    c3.markdown(f"<div class='aq-card'><div class='aq-label'>🧪 CO</div><div class='aq-value'>{co:.1f} µg/m³</div></div>", unsafe_allow_html=True)
                    c4.markdown(f"<div class='aq-card'><div class='aq-label'>💨 NO₂</div><div class='aq-value'>{no2:.1f} µg/m³</div></div>", unsafe_allow_html=True)
                    c5.markdown(f"<div class='aq-card'><div class='aq-label'>🔥 SO₂</div><div class='aq-value'>{so2:.1f} µg/m³</div></div>", unsafe_allow_html=True)
                    c6.markdown(f"<div class='aq-card'><div class='aq-label'>☀️ O₃</div><div class='aq-value'>{o3:.1f} µg/m³</div></div>", unsafe_allow_html=True)
                    c7.markdown(f"<div class='aq-card'><div class='aq-label'>🌍 AQI</div><div class='aq-value'>{aqi:.0f}</div></div>", unsafe_allow_html=True)

                    if aqi <= 20:
                        verdict, color = "🩵 Air is *Excellent* — breathe freely!", "#00BFA5"
                    elif aqi <= 40:
                        verdict, color = "💚 Air is *Good* — healthy and clear.", "#4CAF50"
                    elif aqi <= 60:
                        verdict, color = "💛 Air is *Moderate* — acceptable for most.", "#FFC107"
                    elif aqi <= 80:
                        verdict, color = "🧡 Air is *Poor* — sensitive people take caution.", "#FF9800"
                    elif aqi <= 100:
                        verdict, color = "❤️ Air is *Very Poor* — limit outdoor activity.", "#F44336"
                    else:
                        verdict, color = "☠️ Air is *Hazardous*! Stay indoors!", "#B71C1C"

                    st.markdown(
                        f"<div style='text-align:center;padding:18px;border-radius:15px;background:{color};color:white;font-weight:bold;font-size:1.2rem;margin-top:20px;'>{verdict}</div>",
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.error(f"Error fetching air quality: {e}")

        asyncio.run(main())

# Optional Chart + Rain Prediction

        if len(data) > 3:
            st.subheader("📈 Temperature & Humidity (Last 24h)")
            chart_cols = [c for c in ["temp", "rhum"] if c in data.columns]
            if chart_cols:
                df_plot = data.copy().reset_index()
                df_plot = df_plot[df_plot["time"] >= datetime.utcnow() - timedelta(hours=24)]
                st.line_chart(df_plot.set_index("time")[chart_cols])

        feature_map = {"temp": "temparature", "pres": "pressure", "rhum": "humidity", "dwpt": "dewpoint", "wspd": "windspeed"}
        for m_col, mdl_col in feature_map.items():
            if m_col in latest.columns:
                latest.rename(columns={m_col: mdl_col}, inplace=True)

        model_features = getattr(pipeline, "feature_names_in_", None)
        if model_features is None:
            st.error("Your pipeline doesn’t expose feature names.")
            st.stop()

        for col in model_features:
            if col not in latest.columns:
                latest[col] = 0

        X_live = latest[model_features]
        X_scaled = pipeline.transform(X_live)
        prediction = model.predict(X_scaled)[0]

        st.subheader("🌤️ Rainfall Prediction")
        if int(prediction) == 1:
            st.success("☔ **Rain likely at this location!**")
        else:
            st.info("🌞 **No rain expected right now.**")

    except Exception as e:
        st.error(f"Error fetching data: {e}")


    else:
        st.info("👆 Select a location using the map, city name, or coordinates to fetch live data.")

# Footer

st.markdown("---")
st.caption("Built by BCS2A • Powered by Meteostat, Streamlit & our trained model.")
