"""
app.py

Professional Streamlit dashboard for the AQI Prediction System.

Features
--------
- Blank landing page before city selection
- User-controlled dashboard generation
- Real-time AQI monitoring
- Current weather monitoring
- Pollutant monitoring
- Health guidance
- 3-day / 72-hour AQI forecasting
- 3-hour forecast intervals
- Interactive Plotly visualizations
- Daily forecast summaries
- Forecast statistics
- Downloadable forecast CSV

Backend
-------
FastAPI endpoints:

    GET  /health
    GET  /cities
    GET  /current/{city}
    POST /forecast
"""

from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


# ============================================================
# API CONFIGURATION
# ============================================================

API_BASE_URL = "http://127.0.0.1:8000"

HEALTH_URL = f"{API_BASE_URL}/health"
CITIES_URL = f"{API_BASE_URL}/cities"
CURRENT_URL = f"{API_BASE_URL}/current"
FORECAST_URL = f"{API_BASE_URL}/forecast"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Air Quality Intelligence",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROFESSIONAL COLORFUL CSS
# ============================================================

st.markdown(
    """
    <style>

    /* =====================================================
       GLOBAL
       ===================================================== */

    .stApp {
        background:
            linear-gradient(
                135deg,
                #F8FAFF 0%,
                #F5FAFF 35%,
                #F9F7FF 70%,
                #FFF9F2 100%
            );
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.4rem;
        padding-bottom: 3rem;
    }

    h1 {
        color: #243B64 !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }

    h2,
    h3 {
        color: #293B5F !important;
        font-weight: 750 !important;
    }


    /* =====================================================
       LIGHT SIDEBAR
       ===================================================== */

    section[data-testid="stSidebar"] {

        background:
            linear-gradient(
                180deg,
                #E7F3FF 0%,
                #EAF8FF 30%,
                #EAFBF6 65%,
                #F2EEFF 100%
            );

        border-right: 1px solid #D5E5F1;
    }


    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {

        color: #19324D !important;
    }


    section[data-testid="stSidebar"] label {

        color: #243B53 !important;

        font-weight: 700 !important;
    }


    section[data-testid="stSidebar"] hr {

        border-color: #C9DCE8;
    }


    /* =====================================================
       CITY SELECT
       ===================================================== */

    section[data-testid="stSidebar"]
    div[data-baseweb="select"] > div {

        background-color: #FFFFFF !important;

        border: 1px solid #AAC7DA !important;

        border-radius: 10px !important;

        min-height: 45px;

        box-shadow:
            0 3px 10px rgba(
                30,
                64,
                100,
                0.07
            );
    }


    /* Force selected city / placeholder text black */

    section[data-testid="stSidebar"]
    div[data-baseweb="select"] {

        color: #111827 !important;
    }


    section[data-testid="stSidebar"]
    div[data-baseweb="select"] span {

        color: #111827 !important;

        font-weight: 650 !important;
    }


    section[data-testid="stSidebar"]
    div[data-baseweb="select"] * {

        color: #111827 !important;
    }


    section[data-testid="stSidebar"]
    div[data-baseweb="select"] svg {

        fill: #334155 !important;

        color: #334155 !important;
    }


    /* =====================================================
       SIDEBAR BUTTON
       ===================================================== */

    section[data-testid="stSidebar"]
    .stButton > button {

        width: 100%;

        min-height: 47px;

        margin-top: 8px;

        border: none;

        border-radius: 10px;

        color: #FFFFFF !important;

        font-weight: 750;

        background:
            linear-gradient(
                90deg,
                #7C3AED 0%,
                #2563EB 48%,
                #06B6D4 100%
            );

        box-shadow:
            0 6px 16px
            rgba(
                37,
                99,
                235,
                0.22
            );
    }


    section[data-testid="stSidebar"]
    .stButton > button:hover {

        color: #FFFFFF !important;

        border: none !important;

        transform: translateY(-1px);

        box-shadow:
            0 8px 20px
            rgba(
                37,
                99,
                235,
                0.28
            );
    }


    /* =====================================================
       METRIC CARDS
       ===================================================== */

    div[data-testid="stMetric"] {

        background:
            linear-gradient(
                135deg,
                rgba(255,255,255,0.99),
                rgba(248,251,255,0.97)
            );

        border: 1px solid #E2E8F0;

        border-radius: 16px;

        padding: 18px 20px;

        box-shadow:
            0 7px 20px
            rgba(
                38,
                54,
                91,
                0.07
            );
    }


    div[data-testid="stMetricLabel"] {

        color: #66768F;

        font-weight: 650;
    }


    div[data-testid="stMetricValue"] {

        color: #26395D;

        font-weight: 800;
    }


    /* =====================================================
       DOWNLOAD BUTTON
       ===================================================== */

    .stDownloadButton > button {

        width: 100%;

        border: none;

        border-radius: 10px;

        color: #FFFFFF;

        font-weight: 700;

        background:
            linear-gradient(
                90deg,
                #0F766E,
                #0891B2,
                #2563EB
            );
    }


    /* =====================================================
       TABS
       ===================================================== */

    button[data-baseweb="tab"] {

        font-weight: 700;
    }


    /* =====================================================
       TABLES
       ===================================================== */

    div[data-testid="stDataFrame"] {

        border:
            1px solid #E2E8F0;

        border-radius: 14px;

        overflow: hidden;
    }


    /* =====================================================
       ALERT BOXES
       ===================================================== */

    div[data-testid="stAlert"] {

        border-radius: 13px;
    }


    footer {

        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# AQI HELPERS
# ============================================================

def get_aqi_category(aqi):
    """
    Convert AQI into category.
    """

    if aqi is None:
        return "Unknown"

    if aqi <= 50:
        return "Good"

    if aqi <= 100:
        return "Moderate"

    if aqi <= 150:
        return "Unhealthy for Sensitive Groups"

    if aqi <= 200:
        return "Unhealthy"

    if aqi <= 300:
        return "Very Unhealthy"

    return "Hazardous"


def get_aqi_color(aqi):
    """
    Return AQI category color.
    """

    if aqi is None:
        return "#64748B"

    if aqi <= 50:
        return "#10B981"

    if aqi <= 100:
        return "#FACC15"

    if aqi <= 150:
        return "#FB923C"

    if aqi <= 200:
        return "#F43F5E"

    if aqi <= 300:
        return "#A855F7"

    return "#881337"


def get_health_advice(aqi):
    """
    Return health advice based on AQI.
    """

    if aqi is None:

        return (
            "Current AQI information is unavailable."
        )

    if aqi <= 50:

        return (
            "Air quality is good. "
            "Normal outdoor activities can continue."
        )

    if aqi <= 100:

        return (
            "Air quality is acceptable for most people. "
            "Very sensitive individuals should monitor "
            "prolonged outdoor exposure."
        )

    if aqi <= 150:

        return (
            "Sensitive groups may experience health effects. "
            "Children, older adults, and individuals with "
            "respiratory conditions should reduce prolonged "
            "outdoor activity."
        )

    if aqi <= 200:

        return (
            "Unhealthy conditions are possible for everyone. "
            "Sensitive groups should avoid prolonged "
            "outdoor activity."
        )

    if aqi <= 300:

        return (
            "Very unhealthy conditions are expected. "
            "Outdoor exposure should be reduced significantly."
        )

    return (
        "Hazardous air quality conditions are expected. "
        "Avoid unnecessary outdoor activity and follow "
        "public-health guidance."
    )


def nearest_forecast_value(
    dataframe,
    target_hours,
):
    """
    Return AQI forecast closest to target hours.
    """

    temp = dataframe.copy()

    temp["distance"] = (
        temp["hours_ahead"]
        - target_hours
    ).abs()

    index = temp[
        "distance"
    ].idxmin()

    return float(
        temp.loc[
            index,
            "predicted_aqi",
        ]
    )


# ============================================================
# API HELPERS
# ============================================================

def api_get(url):
    """
    Execute GET request.
    """

    response = requests.get(
        url,
        timeout=20,
    )

    response.raise_for_status()

    return response.json()


def api_post(
    url,
    payload,
):
    """
    Execute POST request.
    """

    response = requests.post(
        url,
        json=payload,
        timeout=30,
    )

    response.raise_for_status()

    return response.json()


@st.cache_data(ttl=30)
def get_health():
    """
    Get backend health status.
    """

    return api_get(
        HEALTH_URL
    )


@st.cache_data(ttl=60)
def get_cities():
    """
    Get supported city list.
    """

    response = api_get(
        CITIES_URL
    )

    return response.get(
        "cities",
        [],
    )


def get_current_conditions(city):
    """
    Fetch current AQI/weather data.
    """

    return api_get(
        f"{CURRENT_URL}/{city}"
    )


def get_forecast(city):
    """
    Generate 3-day forecast using the production GET endpoint.
    """

    return api_get(
        f"{FORECAST_URL}/{city}"
    )


# ============================================================
# BACKEND STATUS
# ============================================================

try:

    backend_health = get_health()

    backend_online = True

except Exception:

    backend_health = {}

    backend_online = False


# ============================================================
# SESSION STATE
# ============================================================

if "current_data" not in st.session_state:

    st.session_state.current_data = None


if "forecast_data" not in st.session_state:

    st.session_state.forecast_data = None


if "active_city" not in st.session_state:

    st.session_state.active_city = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "🌐 AQI Control Center"
    )

    # --------------------------------------------------------
    # SERVICE STATUS
    # --------------------------------------------------------

    if backend_online:

        st.success(
            "● Services Online"
        )

    else:

        st.error(
            "● Backend Offline"
        )

        st.code(
            "uvicorn src.api.main:app --reload"
        )

        st.stop()

    st.divider()


    # ========================================================
    # LOCATION
    # ========================================================

    st.subheader(
        "📍 Location"
    )


    try:

        city_list = get_cities()

    except Exception as error:

        st.error(
            f"Unable to load cities: {error}"
        )

        st.stop()


    if not city_list:

        st.error(
            "No cities available."
        )

        st.stop()


    # --------------------------------------------------------
    # Add placeholder instead of default Lahore
    # --------------------------------------------------------

    city_options = [
        "Choose a city..."
    ] + city_list


    selected_city = st.selectbox(
        "Select City",
        options=city_options,
        index=0,
        key="city_selector",
    )


    # ========================================================
    # SINGLE GENERATE BUTTON
    # ========================================================

    generate_dashboard = st.button(
        "✨ Generate Air Quality Dashboard",
        use_container_width=True,
    )


    if (
        selected_city
        != "Choose a city..."
    ):

        st.caption(
            "Generate live AQI monitoring and "
            f"the 3-day forecast for {selected_city}."
        )

    else:

        st.caption(
            "Choose a city first, then generate "
            "the complete dashboard."
        )


    st.divider()


    # ========================================================
    # FORECAST SETTINGS
    # ========================================================

    st.subheader(
        "⚙ Forecast Settings"
    )


    setting_col1, setting_col2 = (
        st.columns(2)
    )


    with setting_col1:

        st.caption(
            "Horizon"
        )

        st.write(
            "**3 Days**"
        )


    with setting_col2:

        st.caption(
            "Interval"
        )

        st.write(
            "**3 Hours**"
        )


    st.caption(
        "Forecast Points"
    )

    st.write(
        "**24 predictions**"
    )


    st.divider()


    # ========================================================
    # FEATURES
    # ========================================================

    st.subheader(
        "✨ Capabilities"
    )


    st.caption(
        "✓ Real-time AQI monitoring"
    )

    st.caption(
        "✓ Live pollutant monitoring"
    )

    st.caption(
        "✓ Weather conditions"
    )

    st.caption(
        "✓ 72-hour AQI forecast"
    )

    st.caption(
        "✓ Health guidance"
    )

    st.caption(
        "✓ Risk assessment"
    )


# ============================================================
# GENERATE COMPLETE DASHBOARD
# ============================================================

if generate_dashboard:

    # --------------------------------------------------------
    # Validate selection
    # --------------------------------------------------------

    if (
        selected_city
        == "Choose a city..."
    ):

        st.warning(
            "Please select a city before "
            "generating the dashboard."
        )

    else:

        try:

            with st.spinner(
                "Loading live air-quality data "
                f"and 3-day forecast for {selected_city}..."
            ):

                # --------------------------------------------
                # LIVE DATA
                # --------------------------------------------

                live_response = (
                    get_current_conditions(
                        selected_city
                    )
                )

                # --------------------------------------------
                # FORECAST
                # --------------------------------------------

                forecast_response = (
                    get_forecast(
                        selected_city
                    )
                )


            # -----------------------------------------------
            # Store only after successful requests
            # -----------------------------------------------

            st.session_state.current_data = (
                live_response
            )

            st.session_state.forecast_data = (
                forecast_response
            )

            st.session_state.active_city = (
                selected_city
            )


        except Exception as error:

            st.error(
                "Dashboard generation failed."
            )

            st.caption(
                str(error)
            )


# ============================================================
# LANDING PAGE
# ============================================================

if (
    st.session_state.current_data is None
    or
    st.session_state.forecast_data is None
    or
    st.session_state.active_city is None
):

    # --------------------------------------------------------
    # EMPTY INITIAL SCREEN
    # --------------------------------------------------------

    st.title(
        "🌤️ Air Quality Intelligence"
    )


    st.subheader(
        "Real-Time Monitoring & "
        "AI-Powered Air Quality Forecasting"
    )


    st.write(
        "Monitor current air-quality conditions and "
        "generate a machine-learning powered AQI "
        "forecast for the next three days."
    )


    st.info(
        "👈 Select a city from the sidebar and click "
        "**Generate Air Quality Dashboard** to begin."
    )


    st.write("")


    landing_1, landing_2, landing_3 = (
        st.columns(3)
    )


    with landing_1:

        st.subheader(
            "🌍 Live Monitoring"
        )

        st.write(
            "View current AQI, PM2.5, PM10, "
            "temperature, humidity, wind and "
            "atmospheric conditions."
        )


    with landing_2:

        st.subheader(
            "🔮 3-Day AI Forecast"
        )

        st.write(
            "Generate 24 AQI predictions covering "
            "the next 72 hours at 3-hour intervals."
        )


    with landing_3:

        st.subheader(
            "🩺 Health Intelligence"
        )

        st.write(
            "View AQI classifications, risk alerts "
            "and health recommendations."
        )


    st.write("")


    feature_1, feature_2, feature_3 = (
        st.columns(3)
    )


    with feature_1:

        st.info(
            "🧪 Pollutant Intelligence\n\n"
            "PM2.5 • PM10 • NO₂ • O₃ • SO₂ • CO"
        )


    with feature_2:

        st.info(
            "📈 Forecast Analytics\n\n"
            "Trend analysis • Daily summaries • "
            "Forecast statistics"
        )


    with feature_3:

        st.info(
            "📥 Export Results\n\n"
            "Download the complete 3-day forecast "
            "as a CSV file."
        )


    st.stop()


# ============================================================
# ACTIVE DASHBOARD CITY
# ============================================================

active_city = (
    st.session_state.active_city
)


current = (
    st.session_state.current_data
)


forecast_response = (
    st.session_state.forecast_data
)


# ============================================================
# FORECAST METADATA
# ============================================================

forecast_source = (
    forecast_response.get(
        "data_source",
        "Unknown",
    )
)

forecast_model = (
    forecast_response.get(
        "model",
        "Unknown",
    )
)

forecast_version = (
    forecast_response.get(
        "version",
        "Unknown",
    )
)


# ============================================================
# DASHBOARD HEADER
# ============================================================

st.title(
    "🌤️ Air Quality Intelligence Dashboard"
)


st.caption(
    "Real-Time Monitoring • Pollutant Intelligence • "
    "Machine Learning Powered 3-Day Forecasting"
)


st.divider()


# ============================================================
# CITY TITLE
# ============================================================

st.header(
    f"📍 {active_city}"
)


st.success(
    f"Dashboard generated successfully for {active_city}."
)


# ============================================================
# PRODUCTION FORECAST STATUS
# ============================================================

status_1, status_2, status_3 = (
    st.columns(3)
)

status_1.metric(
    "🤖 Production Model",
    forecast_model,
)

status_2.metric(
    "🏷️ Model Version",
    forecast_version,
)

if forecast_source == "REALTIME":

    status_3.metric(
        "📡 Forecast Source",
        "Real-Time",
    )

    st.success(
        "📡 This forecast is generated from the latest "
        "real-time feature history."
    )

elif forecast_source == "HISTORICAL_FALLBACK":

    status_3.metric(
        "📚 Forecast Source",
        "Historical Fallback",
    )

    st.info(
        "📚 Real-time history is still accumulating. "
        "The forecasting system is currently using the "
        "latest validated historical feature history. "
        "It will switch automatically to real-time forecasting "
        "once sufficient live history is available."
    )

else:

    status_3.metric(
        "🗂️ Forecast Source",
        str(forecast_source),
    )


# ============================================================
# LIVE MONITORING
# ============================================================

st.subheader(
    "🟢 Live Air Quality Monitoring"
)


# ------------------------------------------------------------
# Extract live values
# ------------------------------------------------------------

current_aqi = current.get(
    "current_aqi"
)


current_pm25 = current.get(
    "pm2_5"
)


current_pm10 = current.get(
    "pm10"
)


current_temperature = current.get(
    "temperature"
)


current_humidity = current.get(
    "humidity"
)


current_category = current.get(
    "aqi_category",
    "Unknown",
)


current_alert = current.get(
    "alert_level",
    "Unknown",
)


current_weather = current.get(
    "weather",
    "Unknown",
)


# ============================================================
# CURRENT AQI STATUS
# ============================================================

if current_aqi is not None:

    if current_aqi <= 50:

        st.success(
            f"Current AQI: {current_aqi} — "
            f"{current_category}"
        )

    elif current_aqi <= 100:

        st.info(
            f"Current AQI: {current_aqi} — "
            f"{current_category}"
        )

    elif current_aqi <= 150:

        st.warning(
            f"Current AQI: {current_aqi} — "
            f"{current_category}"
        )

    else:

        st.error(
            f"Current AQI: {current_aqi} — "
            f"{current_category}"
        )


# ============================================================
# PRIMARY LIVE METRICS
# ============================================================

live_1, live_2, live_3, live_4, live_5 = (
    st.columns(5)
)


live_1.metric(
    "🌫️ Current AQI",
    (
        f"{current_aqi}"
        if current_aqi is not None
        else "N/A"
    ),
)


live_2.metric(
    "🫧 PM2.5",
    (
        f"{current_pm25:.2f} µg/m³"
        if current_pm25 is not None
        else "N/A"
    ),
)


live_3.metric(
    "🌁 PM10",
    (
        f"{current_pm10:.2f} µg/m³"
        if current_pm10 is not None
        else "N/A"
    ),
)


live_4.metric(
    "🌡️ Temperature",
    (
        f"{current_temperature:.1f} °C"
        if current_temperature is not None
        else "N/A"
    ),
)


live_5.metric(
    "💧 Humidity",
    (
        f"{current_humidity}%"
        if current_humidity is not None
        else "N/A"
    ),
)


# ============================================================
# WEATHER METRICS
# ============================================================

pressure = current.get(
    "pressure"
)


wind_speed = current.get(
    "wind_speed"
)


cloud_cover = current.get(
    "cloud_cover"
)


visibility = current.get(
    "visibility"
)


weather_1, weather_2, weather_3, weather_4 = (
    st.columns(4)
)


weather_1.metric(
    "🧭 Pressure",
    (
        f"{pressure} hPa"
        if pressure is not None
        else "N/A"
    ),
)


weather_2.metric(
    "💨 Wind Speed",
    (
        f"{wind_speed:.2f} m/s"
        if wind_speed is not None
        else "N/A"
    ),
)


weather_3.metric(
    "☁️ Cloud Cover",
    (
        f"{cloud_cover}%"
        if cloud_cover is not None
        else "N/A"
    ),
)


weather_4.metric(
    "👁️ Visibility",
    (
        f"{visibility / 1000:.1f} km"
        if visibility is not None
        else "N/A"
    ),
)


st.caption(
    f"Weather: {str(current_weather).title()} "
    f"• Alert Level: {current_alert} "
    f"• Last Updated: "
    f"{current.get('timestamp', 'N/A')}"
)


# ============================================================
# HEALTH ADVISORY
# ============================================================

st.subheader(
    "🩺 Current Health Advisory"
)


current_guidance = current.get(
    "health_guidance"
)


if not current_guidance:

    current_guidance = (
        get_health_advice(
            current_aqi
        )
    )


if current_aqi is not None:

    if current_aqi <= 50:

        st.success(
            current_guidance
        )

    elif current_aqi <= 100:

        st.info(
            current_guidance
        )

    elif current_aqi <= 150:

        st.warning(
            current_guidance
        )

    else:

        st.error(
            current_guidance
        )


# ============================================================
# LIVE POLLUTANT PROFILE
# ============================================================

st.subheader(
    "🧪 Live Pollutant Profile"
)


pollutant_names = [
    "PM2.5",
    "PM10",
    "NO₂",
    "O₃",
    "SO₂",
    "CO / 10",
]


pollutant_values = [
    current.get(
        "pm2_5",
        0,
    ) or 0,

    current.get(
        "pm10",
        0,
    ) or 0,

    current.get(
        "nitrogen_dioxide",
        0,
    ) or 0,

    current.get(
        "ozone",
        0,
    ) or 0,

    current.get(
        "sulphur_dioxide",
        0,
    ) or 0,

    (
        (
            current.get(
                "carbon_monoxide",
                0,
            )
            or 0
        )
        / 10
    ),
]


pollutant_colors = [
    "#6366F1",
    "#06B6D4",
    "#F97316",
    "#A855F7",
    "#10B981",
    "#F43F5E",
]


pollutant_chart = go.Figure(
    data=[
        go.Bar(
            x=pollutant_names,
            y=pollutant_values,
            marker_color=(
                pollutant_colors
            ),
            text=[
                f"{value:.1f}"
                for value
                in pollutant_values
            ],
            textposition="outside",
            hovertemplate=(
                "%{x}"
                "<br>Value: %{y:.2f}"
                "<extra></extra>"
            ),
        )
    ]
)


pollutant_chart.update_layout(
    height=390,
    paper_bgcolor="white",
    plot_bgcolor="white",
    margin=dict(
        l=30,
        r=20,
        t=20,
        b=25,
    ),
    showlegend=False,
    yaxis=dict(
        title="Concentration",
        gridcolor="#EEF2F7",
    ),
    xaxis=dict(
        title="",
        showgrid=False,
    ),
)


st.plotly_chart(
    pollutant_chart,
    use_container_width=True,
)


st.caption(
    "CO is divided by 10 only for chart readability. "
    "Underlying live pollutant data is unchanged."
)


# ============================================================
# FORECAST SECTION
# ============================================================

st.divider()


st.subheader(
    "🔮 AI-Powered 3-Day AQI Forecast"
)


forecast_rows = forecast_response.get(
    "forecast",
    [],
)


if not forecast_rows:

    st.error(
        "No forecast records were returned."
    )

    st.stop()


# ============================================================
# PREPARE FORECAST DATA
# ============================================================

forecast_df = pd.DataFrame(
    forecast_rows
)


forecast_df["timestamp"] = pd.to_datetime(
    forecast_df["timestamp"],
    errors="coerce",
)


forecast_df["predicted_aqi"] = pd.to_numeric(
    forecast_df["predicted_aqi"],
    errors="coerce",
)


forecast_df["hours_ahead"] = pd.to_numeric(
    forecast_df["hours_ahead"],
    errors="coerce",
)


forecast_df = (
    forecast_df
    .dropna(
        subset=[
            "timestamp",
            "predicted_aqi",
            "hours_ahead",
        ]
    )
    .sort_values(
        "timestamp"
    )
    .reset_index(
        drop=True
    )
)


if forecast_df.empty:

    st.error(
        "No valid forecast records were returned."
    )

    st.stop()


# ============================================================
# FORECAST VALUES
# ============================================================

nearest_aqi = float(
    forecast_df.iloc[0][
        "predicted_aqi"
    ]
)


day_1_aqi = nearest_forecast_value(
    forecast_df,
    24,
)


day_2_aqi = nearest_forecast_value(
    forecast_df,
    48,
)


day_3_aqi = nearest_forecast_value(
    forecast_df,
    72,
)


average_aqi = float(
    forecast_df[
        "predicted_aqi"
    ].mean()
)


maximum_aqi = float(
    forecast_df[
        "predicted_aqi"
    ].max()
)


minimum_aqi = float(
    forecast_df[
        "predicted_aqi"
    ].min()
)


# ============================================================
# FORECAST METRIC CARDS
# ============================================================

forecast_1, forecast_2, forecast_3, forecast_4 = (
    st.columns(4)
)


forecast_1.metric(
    "⏱️ Next 3 Hours",
    f"{nearest_aqi:.0f}",
    get_aqi_category(
        nearest_aqi
    ),
)


forecast_2.metric(
    "🌅 Next Day",
    f"{day_1_aqi:.0f}",
    get_aqi_category(
        day_1_aqi
    ),
)


forecast_3.metric(
    "🌆 Day +2",
    f"{day_2_aqi:.0f}",
    get_aqi_category(
        day_2_aqi
    ),
)


forecast_4.metric(
    "🌇 Day +3",
    f"{day_3_aqi:.0f}",
    get_aqi_category(
        day_3_aqi
    ),
)


# ============================================================
# FORECAST TABS
# ============================================================

(
    tab_forecast,
    tab_daily,
    tab_health,
    tab_insights,
    tab_table,
) = st.tabs(
    [
        "📈 Forecast Trend",
        "📅 Daily Summary",
        "🩺 Health Guidance",
        "📊 Data Insights",
        "📋 Full Forecast",
    ]
)


# ============================================================
# TAB 1: FORECAST TREND
# ============================================================

with tab_forecast:

    st.subheader(
        "72-Hour AQI Forecast Trend"
    )


    forecast_chart = go.Figure()


    # --------------------------------------------------------
    # AQI BACKGROUND REGIONS
    # --------------------------------------------------------

    forecast_chart.add_hrect(
        y0=0,
        y1=50,
        fillcolor="#10B981",
        opacity=0.12,
        line_width=0,
    )


    forecast_chart.add_hrect(
        y0=50,
        y1=100,
        fillcolor="#FACC15",
        opacity=0.13,
        line_width=0,
    )


    forecast_chart.add_hrect(
        y0=100,
        y1=150,
        fillcolor="#FB923C",
        opacity=0.13,
        line_width=0,
    )


    forecast_chart.add_hrect(
        y0=150,
        y1=200,
        fillcolor="#F43F5E",
        opacity=0.11,
        line_width=0,
    )


    forecast_chart.add_hrect(
        y0=200,
        y1=300,
        fillcolor="#A855F7",
        opacity=0.10,
        line_width=0,
    )


    forecast_chart.add_hrect(
        y0=300,
        y1=500,
        fillcolor="#881337",
        opacity=0.10,
        line_width=0,
    )


    marker_colors = [
        get_aqi_color(
            value
        )
        for value
        in forecast_df[
            "predicted_aqi"
        ]
    ]


    forecast_chart.add_trace(
        go.Scatter(
            x=forecast_df[
                "timestamp"
            ],
            y=forecast_df[
                "predicted_aqi"
            ],
            mode="lines+markers",
            line=dict(
                color="#2563EB",
                width=4,
            ),
            marker=dict(
                size=9,
                color=marker_colors,
                line=dict(
                    color="white",
                    width=1.5,
                ),
            ),
            fill="tozeroy",
            fillcolor=(
                "rgba(37,99,235,0.07)"
            ),
            hovertemplate=(
                "<b>%{x|%d %b %Y %H:%M}</b>"
                "<br>AQI: %{y:.1f}"
                "<extra></extra>"
            ),
        )
    )


    forecast_chart.add_hline(
        y=100,
        line_dash="dot",
        line_color="#D97706",
        opacity=0.7,
    )


    forecast_chart.add_hline(
        y=150,
        line_dash="dot",
        line_color="#E11D48",
        opacity=0.7,
    )


    forecast_chart.update_layout(
        height=520,
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        showlegend=False,
        margin=dict(
            l=30,
            r=20,
            t=20,
            b=30,
        ),
        xaxis=dict(
            title="Forecast Time",
            gridcolor="#EEF2F7",
        ),
        yaxis=dict(
            title="AQI",
            gridcolor="#EEF2F7",
            rangemode="tozero",
        ),
    )


    st.plotly_chart(
        forecast_chart,
        use_container_width=True,
    )


    st.caption(
        "AQI background colors represent "
        "standard health-risk categories."
    )


# ============================================================
# TAB 2: DAILY SUMMARY
# ============================================================

with tab_daily:

    st.subheader(
        "Daily AQI Forecast Summary"
    )


    daily_df = forecast_df.copy()


    daily_df["date"] = (
        daily_df[
            "timestamp"
        ].dt.date
    )


    daily_summary = (
        daily_df
        .groupby(
            "date"
        )
        .agg(
            Average_AQI=(
                "predicted_aqi",
                "mean",
            ),
            Minimum_AQI=(
                "predicted_aqi",
                "min",
            ),
            Maximum_AQI=(
                "predicted_aqi",
                "max",
            ),
        )
        .round(2)
        .reset_index()
    )


    daily_summary[
        "Category"
    ] = daily_summary[
        "Average_AQI"
    ].apply(
        get_aqi_category
    )


    daily_summary.columns = [
        "Date",
        "Average AQI",
        "Minimum AQI",
        "Maximum AQI",
        "Category",
    ]


    st.dataframe(
        daily_summary,
        use_container_width=True,
        hide_index=True,
    )


    # --------------------------------------------------------
    # DAILY BAR CHART
    # --------------------------------------------------------

    daily_chart = go.Figure()


    daily_chart.add_trace(
        go.Bar(
            x=daily_summary[
                "Date"
            ].astype(str),
            y=daily_summary[
                "Average AQI"
            ],
            marker_color=[
                get_aqi_color(
                    value
                )
                for value
                in daily_summary[
                    "Average AQI"
                ]
            ],
            text=daily_summary[
                "Average AQI"
            ].round(1),
            textposition="outside",
        )
    )


    daily_chart.update_layout(
        height=390,
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        xaxis=dict(
            title="Forecast Date",
        ),
        yaxis=dict(
            title="Average AQI",
            gridcolor="#EEF2F7",
        ),
    )


    st.plotly_chart(
        daily_chart,
        use_container_width=True,
    )


# ============================================================
# TAB 3: HEALTH GUIDANCE
# ============================================================

with tab_health:

    st.subheader(
        "Forecast Health Guidance"
    )


    if maximum_aqi <= 50:

        st.success(
            get_health_advice(
                maximum_aqi
            )
        )

    elif maximum_aqi <= 100:

        st.info(
            get_health_advice(
                maximum_aqi
            )
        )

    elif maximum_aqi <= 150:

        st.warning(
            get_health_advice(
                maximum_aqi
            )
        )

    else:

        st.error(
            get_health_advice(
                maximum_aqi
            )
        )


    guidance_df = pd.DataFrame(
        [
            {
                "AQI Range": "0–50",
                "Category": "Good",
                "Recommendation":
                    "Normal outdoor activities.",
            },
            {
                "AQI Range": "51–100",
                "Category": "Moderate",
                "Recommendation":
                    "Sensitive people should monitor exposure.",
            },
            {
                "AQI Range": "101–150",
                "Category":
                    "Unhealthy for Sensitive Groups",
                "Recommendation":
                    "Sensitive groups should reduce prolonged "
                    "outdoor activity.",
            },
            {
                "AQI Range": "151–200",
                "Category": "Unhealthy",
                "Recommendation":
                    "Everyone should reduce prolonged "
                    "outdoor activity.",
            },
            {
                "AQI Range": "201–300",
                "Category": "Very Unhealthy",
                "Recommendation":
                    "Significantly reduce outdoor exposure.",
            },
            {
                "AQI Range": "301+",
                "Category": "Hazardous",
                "Recommendation":
                    "Avoid outdoor activity whenever possible.",
            },
        ]
    )


    st.dataframe(
        guidance_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# TAB 4: DATA INSIGHTS
# ============================================================

with tab_insights:

    st.subheader(
        "Forecast Statistics"
    )


    insight_1, insight_2, insight_3, insight_4 = (
        st.columns(4)
    )


    insight_1.metric(
        "📊 Average AQI",
        f"{average_aqi:.1f}",
    )


    insight_2.metric(
        "🔺 Maximum AQI",
        f"{maximum_aqi:.1f}",
    )


    insight_3.metric(
        "🔻 Minimum AQI",
        f"{minimum_aqi:.1f}",
    )


    insight_4.metric(
        "↔️ AQI Range",
        f"{maximum_aqi - minimum_aqi:.1f}",
    )


    peak_record = forecast_df.loc[
        forecast_df[
            "predicted_aqi"
        ].idxmax()
    ]


    minimum_record = forecast_df.loc[
        forecast_df[
            "predicted_aqi"
        ].idxmin()
    ]


    insight_left, insight_right = (
        st.columns(2)
    )


    with insight_left:

        st.warning(
            "Peak AQI Forecast\n\n"
            f"{maximum_aqi:.1f}\n\n"
            f"{peak_record['timestamp'].strftime('%d %b %Y %H:%M')}"
        )


    with insight_right:

        st.success(
            "Lowest AQI Forecast\n\n"
            f"{minimum_aqi:.1f}\n\n"
            f"{minimum_record['timestamp'].strftime('%d %b %Y %H:%M')}"
        )


    category_distribution = (
        forecast_df[
            "predicted_aqi"
        ]
        .apply(
            get_aqi_category
        )
        .value_counts()
        .rename_axis(
            "AQI Category"
        )
        .reset_index(
            name="Forecast Points"
        )
    )


    st.subheader(
        "AQI Category Distribution"
    )


    st.dataframe(
        category_distribution,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# TAB 5: FULL FORECAST
# ============================================================

with tab_table:

    st.subheader(
        "Complete 72-Hour Forecast"
    )


    display_columns = [
        "timestamp",
        "forecast_step",
        "hours_ahead",
        "predicted_aqi",
        "aqi_category",
        "alert_level",
    ]

    if "health_guidance" in forecast_df.columns:
        display_columns.append(
            "health_guidance"
        )

    display_df = forecast_df[
        display_columns
    ].copy()


    display_df = display_df.rename(
        columns={
            "timestamp": "Forecast Time",
            "forecast_step": "Step",
            "hours_ahead": "Hours Ahead",
            "predicted_aqi": "Predicted AQI",
            "aqi_category": "AQI Category",
            "alert_level": "Alert Level",
            "health_guidance": "Health Guidance",
        }
    )


    display_df[
        "Predicted AQI"
    ] = display_df[
        "Predicted AQI"
    ].round(2)


    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=520,
    )


# ============================================================
# FORECAST RISK ASSESSMENT
# ============================================================

st.subheader(
    "⚠️ 72-Hour Risk Assessment"
)


if maximum_aqi > 300:

    st.error(
        "Hazardous AQI conditions are forecast "
        "during the next 72 hours."
    )

elif maximum_aqi > 200:

    st.error(
        "Very Unhealthy AQI conditions are forecast "
        "during the next 72 hours."
    )

elif maximum_aqi > 150:

    st.warning(
        "Unhealthy AQI conditions are forecast "
        "during the next 72 hours."
    )

elif maximum_aqi > 100:

    st.warning(
        "Forecast AQI may affect sensitive groups."
    )

else:

    st.success(
        "No Unhealthy or Hazardous conditions "
        "are forecast during the next 72 hours."
    )


# ============================================================
# EXPORT
# ============================================================

st.subheader(
    "📥 Export Forecast"
)


export_df = forecast_df.copy()


export_df.insert(
    0,
    "city",
    active_city,
)


csv_data = export_df.to_csv(
    index=False
).encode(
    "utf-8"
)


safe_city = (
    active_city
    .lower()
    .replace(
        " ",
        "_",
    )
)


download_left, download_center, download_right = (
    st.columns(
        [1, 2, 1]
    )
)


with download_center:

    st.download_button(
        label="⬇ Download 3-Day Forecast CSV",
        data=csv_data,
        file_name=(
            f"{safe_city}_"
            "aqi_3day_forecast.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(
    "🌤️ Air Quality Intelligence Platform "
    "• Real-Time Monitoring "
    "• Machine Learning Forecasting "
    f"• Dashboard Session: "
    f"{datetime.now().strftime('%d %b %Y %H:%M')}"
)