from datetime import datetime
from textwrap import dedent
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

from src.model_training.predict import (
    SUPPORTED_CITIES,
    build_hourly_aqi_history,
    forecast_city,
    get_model_feature_columns,
    load_historical_dataset,
    load_production_model,
    load_realtime_dataset,
    prepare_model_input,
    select_city_forecast_source,
    update_aqi_features,
    update_time_features,
)
from src.realtime.realtime_monitor import (
    get_current_conditions as fetch_current_conditions,
)


# ============================================================
# STREAMLIT CLOUD CONFIGURATION
# ============================================================

def configure_runtime_secrets():
    """
    Expose Streamlit secrets as environment variables expected by
    the project's existing data-collection modules.

    Local development continues to work with ordinary environment
    variables or a local .env file.
    """

    for key in (
        "OPENWEATHER_API_KEY",
        "HOPSWORKS_API_KEY",
    ):
        if os.getenv(key):
            continue

        try:
            value = st.secrets.get(key)
        except Exception:
            value = None

        if value:
            os.environ[key] = str(value)


configure_runtime_secrets()


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Air Quality Intelligence",
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

    
    /* =====================================================
       AQI STATUS CARDS
       ===================================================== */

    .aqi-card {
        border-radius: 16px;
        padding: 18px 20px;
        border: 1px solid;
        min-height: 128px;
        box-shadow: 0 7px 20px rgba(38, 54, 91, 0.06);
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .aqi-card .aqi-label {
        font-size: 0.80rem;
        font-weight: 750;
        letter-spacing: 0.045em;
        text-transform: uppercase;
        margin-bottom: 8px;
        opacity: 0.82;
    }

    .aqi-card .aqi-value {
        font-size: 2rem;
        line-height: 1.05;
        font-weight: 850;
        margin-bottom: 8px;
    }

    .aqi-card .aqi-category {
        font-size: 0.92rem;
        font-weight: 720;
    }

    .aqi-good {
        background: #ECFDF5;
        border-color: #86EFAC;
        color: #166534;
    }

    .aqi-moderate {
        background: #FEFCE8;
        border-color: #FDE047;
        color: #854D0E;
    }

    .aqi-sensitive {
        background: #FFF7ED;
        border-color: #FDBA74;
        color: #9A3412;
    }

    .aqi-unhealthy {
        background: #FEF2F2;
        border-color: #FCA5A5;
        color: #991B1B;
    }

    .aqi-very-unhealthy {
        background: #FAF5FF;
        border-color: #D8B4FE;
        color: #6B21A8;
    }

    .aqi-hazardous {
        background: #FFF1F2;
        border-color: #FDA4AF;
        color: #881337;
    }

    .aqi-unknown {
        background: #F8FAFC;
        border-color: #CBD5E1;
        color: #475569;
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



def get_aqi_card_class(aqi):
    """
    Return CSS class for AQI severity card.
    """

    if aqi is None:
        return "aqi-unknown"

    if aqi <= 50:
        return "aqi-good"

    if aqi <= 100:
        return "aqi-moderate"

    if aqi <= 150:
        return "aqi-sensitive"

    if aqi <= 200:
        return "aqi-unhealthy"

    if aqi <= 300:
        return "aqi-very-unhealthy"

    return "aqi-hazardous"


def render_aqi_card(
    label,
    aqi,
    category=None,
):
    """
    Render a professional AQI card with category-specific coloring.
    """

    card_class = get_aqi_card_class(
        aqi
    )

    if aqi is None:
        value_text = "N/A"
    else:
        value_text = f"{float(aqi):.0f}"

    if not category:
        category = get_aqi_category(
            aqi
        )

    st.markdown(
        f"""
        <div class="aqi-card {card_class}">
            <div class="aqi-label">{label}</div>
            <div class="aqi-value">{value_text}</div>
            <div class="aqi-category">{category}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
# PRODUCTION DATA HELPERS
# ============================================================

@st.cache_resource
def load_dashboard_resources():
    """
    Load the production model and forecasting datasets once per
    Streamlit application process.

    Streamlit Community Cloud redeploys when the connected GitHub
    branch changes, so newly committed real-time feature history is
    picked up automatically on redeploy.
    """

    model, production_info = load_production_model()

    expected_features = get_model_feature_columns(
        model
    )

    realtime_df = None
    historical_df = None

    try:
        realtime_df = load_realtime_dataset()
    except Exception as error:
        print(
            "Real-time feature loading failed: "
            f"{error}"
        )

    try:
        historical_df = load_historical_dataset()
    except Exception as error:
        print(
            "Historical fallback loading failed: "
            f"{error}"
        )

    if realtime_df is None and historical_df is None:
        raise RuntimeError(
            "Neither real-time nor historical forecasting "
            "data is available."
        )

    return (
        model,
        production_info,
        expected_features,
        realtime_df,
        historical_df,
    )


def get_health():
    """Return production-resource readiness for the sidebar."""

    (
        model,
        production_info,
        expected_features,
        realtime_df,
        historical_df,
    ) = load_dashboard_resources()

    return {
        "status": "healthy",
        "production_model": production_info[
            "model_name"
        ],
        "production_version": production_info[
            "version"
        ],
        "realtime_rows": (
            len(realtime_df)
            if realtime_df is not None
            else 0
        ),
        "historical_rows": (
            len(historical_df)
            if historical_df is not None
            else 0
        ),
        "cities": len(SUPPORTED_CITIES),
        "forecasting": True,
        "realtime_monitoring": True,
    }


@st.cache_data(ttl=300)
def get_cities():
    """Return supported project cities."""

    return list(SUPPORTED_CITIES)


def get_current_conditions(city):
    """
    Fetch current AQI, pollutants, and weather directly from the
    project's real-time monitoring module.
    """

    return fetch_current_conditions(
        city
    )


def get_forecast(city):
    """
    Generate a production 3-day forecast directly from the
    production registry model and latest available feature history.
    """

    (
        model,
        production_info,
        expected_features,
        realtime_df,
        historical_df,
    ) = load_dashboard_resources()

    (
        city_data,
        source_name,
    ) = select_city_forecast_source(
        city=city,
        realtime_df=realtime_df,
        historical_df=historical_df,
        expected_columns=expected_features,
    )

    forecast_df = forecast_city(
        model=model,
        city_data=city_data,
        city=city,
        expected_columns=expected_features,
        production_info=production_info,
        source_name=source_name,
    )

    forecast_records = forecast_df.copy()

    if "timestamp" in forecast_records.columns:
        forecast_records["timestamp"] = (
            forecast_records["timestamp"]
            .astype(str)
        )

    return {
        "city": city,
        "model": production_info[
            "model_name"
        ],
        "version": production_info[
            "version"
        ],
        "forecast_days": 3,
        "forecast_interval_hours": 3,
        "predictions_count": len(forecast_records),
        "data_source": source_name,
        "forecast": forecast_records.to_dict(
            orient="records"
        ),
        "status": "success",
    }



# ============================================================
# MODEL EXPLAINABILITY
# ============================================================

def _clean_transformed_feature_name(name):
    """
    Convert transformed sklearn feature names into readable labels.
    """

    cleaned = str(name)

    if "__" in cleaned:
        cleaned = cleaned.split("__", 1)[1]

    return cleaned


def _raw_feature_from_transformed_name(
    transformed_name,
    expected_features,
):
    """
    Map one-hot/transformed feature names back to their raw model feature.
    """

    cleaned = _clean_transformed_feature_name(
        transformed_name
    )

    # Exact numeric/raw feature name.
    if cleaned in expected_features:
        return cleaned

    # One-hot names are normally shaped like:
    # city_Lahore, season_Summer, day_of_week_Friday, ...
    for feature in sorted(
        expected_features,
        key=len,
        reverse=True,
    ):
        if cleaned.startswith(
            f"{feature}_"
        ):
            return feature

    return cleaned


def _format_feature_name(name):
    """
    Human-readable feature label for the dashboard.
    """

    special = {
        "pm2_5": "PM2.5",
        "pm10": "PM10",
        "us_aqi": "Current AQI",
        "aqi_lag_1": "AQI Lag 1 Hour",
        "aqi_lag_3": "AQI Lag 3 Hours",
        "aqi_lag_6": "AQI Lag 6 Hours",
        "aqi_lag_24": "AQI Lag 24 Hours",
        "aqi_roll_3": "AQI Rolling 3 Hours",
        "aqi_roll_6": "AQI Rolling 6 Hours",
        "aqi_roll_24": "AQI Rolling 24 Hours",
        "nitrogen_dioxide": "Nitrogen Dioxide",
        "ozone": "Ozone",
        "sulphur_dioxide": "Sulphur Dioxide",
        "carbon_monoxide": "Carbon Monoxide",
        "temperature": "Temperature",
        "humidity": "Humidity",
        "pressure": "Pressure",
        "wind_speed": "Wind Speed",
        "cloud_cover": "Cloud Cover",
        "visibility": "Visibility",
        "day_of_week": "Day of Week",
        "is_weekend": "Weekend Indicator",
        "city": "City",
        "season": "Season",
    }

    if name in special:
        return special[name]

    return (
        str(name)
        .replace("_", " ")
        .title()
    )


def build_next_step_model_input(
    city_data,
    expected_features,
):
    """
    Rebuild the exact first (+3 hour) raw model input used by the
    current production forecast_city() implementation.
    """

    if not isinstance(
        city_data,
        pd.DataFrame,
    ):
        city_data = pd.DataFrame(
            city_data
        )

    if city_data.empty:
        raise ValueError(
            "No city feature history is available for explanation."
        )

    city_data = (
        city_data
        .copy()
    )

    city_data["timestamp"] = pd.to_datetime(
        city_data["timestamp"],
        errors="coerce",
    )

    city_data["us_aqi"] = pd.to_numeric(
        city_data["us_aqi"],
        errors="coerce",
    )

    city_data = (
        city_data
        .dropna(
            subset=[
                "timestamp",
                "us_aqi",
            ]
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if city_data.empty:
        raise ValueError(
            "No valid city AQI history is available for explanation."
        )

    latest_row = (
        city_data
        .iloc[-1]
        .copy()
    )

    latest_timestamp = pd.Timestamp(
        latest_row["timestamp"]
    )

    # IMPORTANT:
    # The current predict.py does NOT use a Python list for AQI history.
    # It builds an hourly pandas Series. update_aqi_features() expects
    # that Series because it calls history.empty and uses history.index.
    history = build_hourly_aqi_history(
        city_data
    )

    forecast_timestamp = (
        latest_timestamp
        + pd.Timedelta(
            hours=3
        )
    )

    previous_aqi = float(
        history.iloc[-1]
    )

    # Match forecast_city(): fill the two intermediate hourly points
    # with persistence before constructing the +3 hour input.
    intermediate_timestamp = (
        latest_timestamp
        + pd.Timedelta(
            hours=1
        )
    )

    while (
        intermediate_timestamp
        < forecast_timestamp
    ):

        history.loc[
            intermediate_timestamp
        ] = previous_aqi

        intermediate_timestamp = (
            intermediate_timestamp
            + pd.Timedelta(
                hours=1
            )
        )

    history = (
        history
        .sort_index()
    )

    future_row = (
        latest_row.copy()
    )

    future_row[
        "timestamp"
    ] = forecast_timestamp

    future_row = update_time_features(
        future_row,
        forecast_timestamp,
    )

    # Exact signature and exact data type used by current predict.py:
    # row, hourly pandas Series, forecast timestamp.
    future_row = update_aqi_features(
        future_row,
        history,
        forecast_timestamp,
    )

    X_future = prepare_model_input(
        future_row,
        expected_features,
    )

    return (
        X_future,
        forecast_timestamp,
    )


@st.cache_data(ttl=300, show_spinner=False)
def calculate_shap_explanation(
    city,
    forecast_source,
):
    """
    Explain the production LightGBM model's next 3-hour prediction.

    Primary method:
        LightGBM's native pred_contrib=True output.

    LightGBM native feature contributions are SHAP values and include
    the expected/base value as the final column. This is more robust
    for the deployed sklearn + LightGBM pipeline than relying only on
    the external SHAP TreeExplainer API.
    """

    (
        model,
        production_info,
        expected_features,
        realtime_df,
        historical_df,
    ) = load_dashboard_resources()

    if not hasattr(model, "named_steps"):
        raise RuntimeError(
            "The production model is not an sklearn Pipeline."
        )

    if (
        "preprocessor" not in model.named_steps
        or "model" not in model.named_steps
    ):
        raise RuntimeError(
            "The production Pipeline must contain "
            "'preprocessor' and 'model' steps."
        )

    (
        city_data,
        source_name,
    ) = select_city_forecast_source(
        city=city,
        realtime_df=realtime_df,
        historical_df=historical_df,
        expected_columns=expected_features,
    )

    (
        X_future,
        forecast_timestamp,
    ) = build_next_step_model_input(
        city_data,
        expected_features,
    )

    preprocessor = (
        model.named_steps["preprocessor"]
    )

    estimator = (
        model.named_steps["model"]
    )

    X_transformed = preprocessor.transform(
        X_future
    )

    if hasattr(X_transformed, "toarray"):
        X_explain = X_transformed.toarray()
    else:
        X_explain = np.asarray(
            X_transformed
        )

    # Ensure numeric dtype. This avoids object-dtype issues that can
    # appear after mixed categorical/numerical preprocessing.
    X_explain = np.asarray(
        X_explain,
        dtype=float,
    )

    # ------------------------------------------------------------
    # FEATURE NAMES AFTER PREPROCESSING
    # ------------------------------------------------------------

    transformed_names = None

    try:
        transformed_names = list(
            preprocessor.get_feature_names_out()
        )
    except Exception:
        transformed_names = None

    if (
        transformed_names is None
        or len(transformed_names) != X_explain.shape[1]
    ):
        # Reconstruct names directly from fitted ColumnTransformer.
        rebuilt_names = []

        for (
            transformer_name,
            transformer,
            columns,
        ) in preprocessor.transformers_:

            if transformer_name == "remainder":
                continue

            column_list = list(columns)

            if transformer_name == "categorical":

                try:
                    names = list(
                        transformer.get_feature_names_out(
                            column_list
                        )
                    )
                except Exception:
                    names = column_list

                rebuilt_names.extend(
                    names
                )

            elif transformer_name == "numerical":

                rebuilt_names.extend(
                    column_list
                )

            else:

                try:
                    names = list(
                        transformer.get_feature_names_out(
                            column_list
                        )
                    )
                except Exception:
                    names = column_list

                rebuilt_names.extend(
                    names
                )

        if len(rebuilt_names) == X_explain.shape[1]:
            transformed_names = rebuilt_names
        else:
            transformed_names = [
                f"feature_{index}"
                for index in range(
                    X_explain.shape[1]
                )
            ]

    # ------------------------------------------------------------
    # EXACT LIGHTGBM SHAP CONTRIBUTIONS
    # ------------------------------------------------------------

    shap_row = None
    base_value = None

    if hasattr(estimator, "booster_"):

        native_contrib = (
            estimator.booster_.predict(
                X_explain,
                pred_contrib=True,
            )
        )

        native_contrib = np.asarray(
            native_contrib,
            dtype=float,
        )

        if native_contrib.ndim == 1:
            native_contrib = (
                native_contrib.reshape(
                    1,
                    -1,
                )
            )

        expected_width = (
            X_explain.shape[1] + 1
        )

        if native_contrib.shape[1] != expected_width:
            raise RuntimeError(
                "Unexpected LightGBM SHAP contribution shape: "
                f"{native_contrib.shape}. Expected "
                f"(1, {expected_width})."
            )

        # Last LightGBM contribution is the expected/base value.
        shap_row = (
            native_contrib[0, :-1]
        )

        base_value = float(
            native_contrib[0, -1]
        )

    else:
        # Fallback for an estimator without a directly available
        # LightGBM booster.
        explainer = shap.TreeExplainer(
            estimator
        )

        explanation = explainer(
            X_explain
        )

        shap_values = np.asarray(
            explanation.values,
            dtype=float,
        )

        if shap_values.ndim == 1:
            shap_row = shap_values
        else:
            shap_row = shap_values[0]

        base_array = np.asarray(
            explanation.base_values
        ).reshape(-1)

        if len(base_array):
            base_value = float(
                base_array[0]
            )

    if shap_row is None:
        raise RuntimeError(
            "SHAP values could not be calculated."
        )

    if len(shap_row) != len(transformed_names):
        raise RuntimeError(
            "SHAP feature count does not match "
            "the transformed feature count."
        )

    # ------------------------------------------------------------
    # AGGREGATE ONE-HOT SHAP VALUES BACK TO RAW FEATURES
    # ------------------------------------------------------------

    contributions = pd.DataFrame(
        {
            "transformed_feature": transformed_names,
            "shap_value": np.asarray(
                shap_row,
                dtype=float,
            ),
        }
    )

    contributions[
        "raw_feature"
    ] = contributions[
        "transformed_feature"
    ].apply(
        lambda name: _raw_feature_from_transformed_name(
            name,
            expected_features,
        )
    )

    aggregated = (
        contributions
        .groupby(
            "raw_feature",
            as_index=False,
        )["shap_value"]
        .sum()
    )

    aggregated["absolute_impact"] = (
        aggregated["shap_value"].abs()
    )

    aggregated["feature"] = (
        aggregated["raw_feature"]
        .apply(
            _format_feature_name
        )
    )

    raw_row = X_future.iloc[0]

    aggregated["feature_value"] = (
        aggregated["raw_feature"]
        .apply(
            lambda feature: raw_row.get(
                feature,
                "N/A",
            )
        )
    )

    aggregated = (
        aggregated
        .sort_values(
            "absolute_impact",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # Use the complete pipeline prediction as the displayed prediction,
    # ensuring it is exactly the same prediction path as the dashboard.
    model_prediction = float(
        model.predict(
            X_future
        )[0]
    )

    # Diagnostic consistency check:
    # prediction ~= baseline + sum(SHAP values)
    reconstructed_prediction = (
        float(base_value)
        + float(
            np.sum(
                shap_row
            )
        )
        if base_value is not None
        else np.nan
    )

    return {
        "city": city,
        "source": source_name,
        "model": production_info.get(
            "model_name",
            "LightGBM",
        ),
        "version": production_info.get(
            "version",
            "Unknown",
        ),
        "forecast_timestamp": str(
            forecast_timestamp
        ),
        "prediction": max(
            0.0,
            model_prediction,
        ),
        "base_value": base_value,
        "reconstructed_prediction": (
            reconstructed_prediction
        ),
        "contributions": aggregated.to_dict(
            orient="records"
        ),
    }


def build_shap_interpretation(
    contribution_df,
):
    """
    Produce a short plain-language explanation from SHAP impacts.
    """

    if contribution_df.empty:
        return (
            "No feature-level explanation is currently available."
        )

    positive = (
        contribution_df[
            contribution_df["shap_value"] > 0
        ]
        .sort_values(
            "absolute_impact",
            ascending=False,
        )
    )

    negative = (
        contribution_df[
            contribution_df["shap_value"] < 0
        ]
        .sort_values(
            "absolute_impact",
            ascending=False,
        )
    )

    pieces = []

    if not positive.empty:
        names = positive[
            "feature"
        ].head(2).tolist()

        pieces.append(
            "The strongest upward pressure on the forecast comes from "
            + " and ".join(names)
            + "."
        )

    if not negative.empty:
        names = negative[
            "feature"
        ].head(2).tolist()

        pieces.append(
            "The main factors reducing the forecast are "
            + " and ".join(names)
            + "."
        )

    if not pieces:
        pieces.append(
            "The model's feature contributions are currently "
            "close to its baseline prediction."
        )

    return " ".join(pieces)



# ============================================================
# SMART ALERTS, ACTIVITY PLANNER, AND CITY COMPARISON
# ============================================================

def get_activity_recommendation(aqi):
    """
    Convert forecast AQI into a simple activity recommendation.
    """

    if aqi is None:
        return {
            "level": "Unknown",
            "outdoor": "Check air-quality data before planning outdoor activity.",
            "exercise": "No recommendation available.",
        }

    if aqi <= 50:
        return {
            "level": "Excellent",
            "outdoor": "Good time for outdoor activities.",
            "exercise": "Suitable for walking, running, cycling, and outdoor exercise.",
        }

    if aqi <= 100:
        return {
            "level": "Acceptable",
            "outdoor": "Outdoor activity is generally acceptable.",
            "exercise": "Sensitive people may prefer lighter or shorter outdoor exercise.",
        }

    if aqi <= 150:
        return {
            "level": "Use Caution",
            "outdoor": "Reduce long outdoor activities if you are sensitive to air pollution.",
            "exercise": "Prefer light exercise and avoid prolonged strenuous activity outdoors.",
        }

    if aqi <= 200:
        return {
            "level": "Limit Outdoor Activity",
            "outdoor": "Limit prolonged outdoor exposure.",
            "exercise": "Prefer indoor exercise, especially for sensitive groups.",
        }

    if aqi <= 300:
        return {
            "level": "Avoid Outdoor Exercise",
            "outdoor": "Outdoor exposure should be kept to a minimum.",
            "exercise": "Move exercise and recreation indoors.",
        }

    return {
        "level": "Stay Indoors",
        "outdoor": "Avoid unnecessary outdoor activity.",
        "exercise": "Avoid outdoor exercise and follow public-health guidance.",
    }


def build_smart_alerts(forecast_df):
    """
    Build practical alerts from the selected city's 72-hour forecast.
    """

    if forecast_df.empty:
        return []

    work = forecast_df.copy()

    work["timestamp"] = pd.to_datetime(
        work["timestamp"]
    )

    work["predicted_aqi"] = pd.to_numeric(
        work["predicted_aqi"],
        errors="coerce",
    )

    work = work.dropna(
        subset=["predicted_aqi"]
    )

    if work.empty:
        return []

    alerts = []

    peak_row = work.loc[
        work["predicted_aqi"].idxmax()
    ]

    minimum_row = work.loc[
        work["predicted_aqi"].idxmin()
    ]

    peak_aqi = float(
        peak_row["predicted_aqi"]
    )

    minimum_aqi = float(
        minimum_row["predicted_aqi"]
    )

    if peak_aqi > 150:
        alerts.append(
            {
                "severity": "danger",
                "title": "High AQI Alert",
                "message": (
                    f"AQI may reach {peak_aqi:.0f} around "
                    f"{peak_row['timestamp'].strftime('%a %d %b, %I:%M %p')}. "
                    "Plan important outdoor activities for a cleaner period."
                ),
            }
        )
    elif peak_aqi > 100:
        alerts.append(
            {
                "severity": "warning",
                "title": "Sensitive Group Alert",
                "message": (
                    f"AQI may reach {peak_aqi:.0f} around "
                    f"{peak_row['timestamp'].strftime('%a %d %b, %I:%M %p')}. "
                    "Sensitive people should reduce prolonged outdoor exposure."
                ),
            }
        )
    else:
        alerts.append(
            {
                "severity": "success",
                "title": "Air Quality Outlook",
                "message": (
                    "No unhealthy AQI period is predicted in the current "
                    "72-hour forecast."
                ),
            }
        )

    alerts.append(
        {
            "severity": "success" if minimum_aqi <= 100 else "warning",
            "title": "Best Outdoor Window",
            "message": (
                f"The cleanest forecast point is "
                f"{minimum_row['timestamp'].strftime('%a %d %b, %I:%M %p')} "
                f"with AQI near {minimum_aqi:.0f} "
                f"({get_aqi_category(minimum_aqi)})."
            ),
        }
    )

    return alerts


def build_activity_windows(forecast_df, limit=8):
    """
    Return the best forecast windows for planning outdoor activity.
    """

    work = forecast_df.copy()

    if work.empty:
        return pd.DataFrame()

    work["timestamp"] = pd.to_datetime(
        work["timestamp"]
    )

    work["predicted_aqi"] = pd.to_numeric(
        work["predicted_aqi"],
        errors="coerce",
    )

    work = work.dropna(
        subset=["predicted_aqi"]
    )

    work["Category"] = work[
        "predicted_aqi"
    ].apply(
        get_aqi_category
    )

    work["Activity Level"] = work[
        "predicted_aqi"
    ].apply(
        lambda value: get_activity_recommendation(
            value
        )["level"]
    )

    work["Time"] = work[
        "timestamp"
    ].dt.strftime(
        "%a %d %b, %I:%M %p"
    )

    work = (
        work
        .sort_values(
            ["predicted_aqi", "timestamp"]
        )
        .head(limit)
    )

    result = work[
        [
            "Time",
            "predicted_aqi",
            "Category",
            "Activity Level",
        ]
    ].copy()

    result.columns = [
        "Recommended Time",
        "Forecast AQI",
        "AQI Category",
        "Activity Advice",
    ]

    result["Forecast AQI"] = (
        result["Forecast AQI"]
        .round(0)
        .astype(int)
    )

    return result


@st.cache_data(ttl=300, show_spinner=False)
def get_city_comparison():
    """
    Compare all supported cities using the production 72-hour forecasts.
    """

    rows = []

    for city in SUPPORTED_CITIES:

        try:
            response = get_forecast(
                city
            )

            city_forecast = pd.DataFrame(
                response["forecast"]
            )

            if city_forecast.empty:
                continue

            city_forecast[
                "predicted_aqi"
            ] = pd.to_numeric(
                city_forecast[
                    "predicted_aqi"
                ],
                errors="coerce",
            )

            city_forecast = (
                city_forecast
                .dropna(
                    subset=["predicted_aqi"]
                )
            )

            if city_forecast.empty:
                continue

            next_aqi = float(
                city_forecast.iloc[0][
                    "predicted_aqi"
                ]
            )

            avg_24h = float(
                city_forecast[
                    city_forecast[
                        "hours_ahead"
                    ].astype(float) <= 24
                ][
                    "predicted_aqi"
                ].mean()
            )

            avg_72h = float(
                city_forecast[
                    "predicted_aqi"
                ].mean()
            )

            peak_72h = float(
                city_forecast[
                    "predicted_aqi"
                ].max()
            )

            rows.append(
                {
                    "City": city,
                    "Next AQI": next_aqi,
                    "24h Average": avg_24h,
                    "72h Average": avg_72h,
                    "72h Peak": peak_72h,
                    "Category": get_aqi_category(
                        next_aqi
                    ),
                    "Data Source": response.get(
                        "data_source",
                        "Unknown",
                    ),
                }
            )

        except Exception as error:
            print(
                f"City comparison failed for {city}: {error}"
            )

    comparison_df = pd.DataFrame(
        rows
    )

    if comparison_df.empty:
        return comparison_df

    comparison_df = (
        comparison_df
        .sort_values(
            "24h Average",
            ascending=True,
        )
        .reset_index(drop=True)
    )

    comparison_df.insert(
        0,
        "Rank",
        range(
            1,
            len(comparison_df) + 1,
        ),
    )

    for column in [
        "Next AQI",
        "24h Average",
        "72h Average",
        "72h Peak",
    ]:
        comparison_df[column] = (
            comparison_df[column]
            .round(1)
        )

    return comparison_df



# ============================================================
# REFERENCE-STYLE DARK DASHBOARD
# ============================================================

st.set_page_config(
    page_title="AQI Predictor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --bg: #06111f;
        --bg2: #081827;
        --sidebar: #0a1a2a;
        --card: #0b1f33;
        --card2: #102a43;
        --border: rgba(130, 164, 200, 0.16);
        --text: #f5f8fc;
        --muted: #9eb0c5;
        --blue: #4f97ff;
        --blue-dark: #2568d8;
        --green: #55e1a5;
        --yellow: #ffd654;
        --orange: #ffa04b;
        --red: #ff5c64;
        --purple: #9f7bff;
    }

    html, body, [class*="css"] {
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 72% 3%, rgba(29,84,142,.16), transparent 28%),
            linear-gradient(135deg, #07131f 0%, #071522 55%, #06111d 100%);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 1550px;
        padding-top: .7rem;
        padding-bottom: 2.5rem;
    }

    section[data-testid="stSidebar"] {
        width: 230px !important;
        min-width: 230px !important;
        background: linear-gradient(180deg, #0a1c2d 0%, #071522 100%);
        border-right: 1px solid rgba(255,255,255,.05);
    }

    section[data-testid="stSidebar"] .block-container {
        padding: 1rem .9rem 1rem .9rem;
    }

    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span {
        color: #e5edf7 !important;
    }

    .side-brand {
        display:flex;
        align-items:center;
        gap:.75rem;
        padding:.2rem .15rem 1rem .15rem;
    }

    .brand-mark {
        width:38px;
        height:38px;
        border-radius:12px 12px 18px 8px;
        background: linear-gradient(145deg,#39d99a,#64f0ba);
        box-shadow:0 10px 24px rgba(73,224,166,.20);
        transform:rotate(-12deg);
        position:relative;
    }

    .brand-mark:after {
        content:"";
        position:absolute;
        width:2px;
        height:23px;
        background:#0b6046;
        left:19px;
        top:8px;
        transform:rotate(24deg);
        border-radius:2px;
    }

    .brand-title {
        font-weight:850;
        font-size:1.2rem;
        letter-spacing:-.02em;
        color:#f8fbff;
    }

    .brand-sub {
        font-size:.78rem;
        color:#93a8be;
        margin-top:.06rem;
    }

    .tagline {
        color:#b5c3d4;
        font-size:.83rem;
        line-height:1.45;
        padding:.1rem .2rem .8rem;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] {
        gap:.18rem;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] > label {
        border-radius:10px;
        padding:.62rem .72rem;
        background: transparent;
        transition:.18s ease;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background: rgba(79,151,255,.09);
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(135deg, #3689ff, #285fd2);
        box-shadow:0 10px 22px rgba(45,112,219,.20);
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p {
        color:white !important;
        font-weight:800;
    }

    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background:#0d2238 !important;
        border:1px solid rgba(142,175,210,.18) !important;
        border-radius:10px !important;
        color:#f8fbff !important;
    }

    section[data-testid="stSidebar"] div[data-baseweb="select"] * {
        color:#f8fbff !important;
    }

    .side-bottom {
        margin-top:1rem;
        background: linear-gradient(180deg, rgba(17,42,67,.9), rgba(10,29,48,.95));
        border:1px solid rgba(127,163,198,.14);
        border-radius:14px;
        padding:.9rem;
    }

    .side-bottom-title {
        color:#f6f9ff;
        font-weight:850;
        line-height:1.25;
        font-size:.92rem;
    }

    .side-bottom-text {
        color:#8fa5bd;
        font-size:.73rem;
        line-height:1.45;
        margin-top:.42rem;
    }

    .topbar {
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap:1rem;
        margin-bottom:.35rem;
    }

    .topbar-city {
        min-width:210px;
    }

    .topbar-meta {
        display:flex;
        align-items:center;
        gap:1rem;
        color:#c6d1de;
        font-size:.85rem;
    }

    .live-pill {
        display:inline-flex;
        align-items:center;
        gap:.42rem;
        padding:.34rem .6rem;
        border-radius:8px;
        color:#93f0c5;
        background:rgba(55,199,143,.10);
        border:1px solid rgba(55,199,143,.18);
        font-weight:800;
    }

    .live-dot {
        width:8px;
        height:8px;
        border-radius:50%;
        background:#49dda2;
        box-shadow:0 0 10px rgba(73,221,162,.75);
    }

    .hero {
        position:relative;
        overflow:hidden;
        min-height:105px;
        border-radius:16px;
        margin:.45rem 0 .8rem;
        display:flex;
        align-items:center;
        justify-content:space-between;
        padding:1.1rem 1.3rem;
        background:
            linear-gradient(90deg, rgba(8,25,42,.98), rgba(7,31,51,.84)),
            radial-gradient(circle at 78% 45%, rgba(87,145,196,.22), transparent 30%);
        border:1px solid rgba(125,160,193,.12);
    }

    .hero:after {
        content:"";
        position:absolute;
        right:0;
        top:0;
        width:42%;
        height:100%;
        background:
          radial-gradient(circle at 65% 25%, rgba(91,151,201,.14), transparent 28%),
          linear-gradient(180deg, rgba(255,255,255,.015), rgba(255,255,255,0));
        opacity:.85;
        pointer-events:none;
    }

    .hero-title {
        font-size:clamp(2rem,3vw,3.15rem);
        font-weight:900;
        letter-spacing:-.04em;
        color:#f8fbff;
        line-height:1.0;
        margin-bottom:.45rem;
    }

    .hero-sub {
        color:#aebfd1;
        font-size:.95rem;
    }

    .hero-location {
        z-index:1;
        text-align:right;
        color:#f4f8ff;
        font-weight:850;
        font-size:1.05rem;
    }

    .hero-location span {
        display:block;
        color:#91a7bf;
        font-size:.75rem;
        font-weight:500;
        margin-top:.1rem;
    }

    .card {
        background: linear-gradient(180deg, rgba(13,32,52,.98), rgba(10,26,44,.98));
        border:1px solid rgba(127,164,199,.15);
        border-radius:14px;
        box-shadow:0 12px 24px rgba(0,0,0,.12);
        height:100%;
    }

    .card-pad {
        padding:.9rem 1rem;
    }

    .card-title {
        color:#f7fbff;
        font-size:.95rem;
        font-weight:850;
        margin-bottom:.75rem;
        display:flex;
        align-items:center;
        justify-content:space-between;
    }

    .muted {
        color:#93a8bf;
    }

    .aqi-card-inner {
        display:grid;
        grid-template-columns: 1.3fr .85fr;
        gap:.8rem;
        align-items:center;
    }

    .aqi-status-title {
        color:#dce6f1;
        font-weight:800;
        font-size:.82rem;
        margin-bottom:.4rem;
    }

    .aqi-number {
        font-size:3.25rem;
        font-weight:900;
        color:#fff;
        line-height:1;
        margin:.1rem 0;
    }

    .aqi-cat {
        font-size:.9rem;
        font-weight:850;
    }


    .icon-heading {display:flex;align-items:center;gap:.55rem;margin-bottom:.6rem;}
    .weather-main {display:flex;align-items:center;gap:.8rem;margin-bottom:.35rem;}
    .forecast-icon {height:38px;display:flex;align-items:center;justify-content:center;margin:.1rem 0 .05rem;}
    .planner-row {display:grid;grid-template-columns:54px 1fr;gap:.75rem;align-items:center;}
    .generate-panel {background:linear-gradient(180deg,rgba(13,32,52,.98),rgba(10,26,44,.98));border:1px solid rgba(127,164,199,.15);border-radius:14px;padding:1.25rem;text-align:center;color:#b8c6d6;margin-top:.8rem;}

    .weather-temp {
        font-size:2.2rem;
        font-weight:900;
        color:#f8fbff;
        line-height:1.05;
    }

    .weather-cond {
        color:#b5c3d4;
        font-size:.82rem;
        margin:.25rem 0 .8rem;
    }

    .weather-list {
        display:grid;
        grid-template-columns:1fr auto;
        gap:.48rem .7rem;
        font-size:.78rem;
    }

    .weather-list .lab {
        color:#9aacc0;
    }

    .weather-list .val {
        color:#f0f4fa;
        font-weight:750;
    }

    .health-text {
        color:#f1f5fa;
        font-size:.93rem;
        font-weight:700;
        line-height:1.45;
        margin:.2rem 0 .8rem;
    }

    .health-bullet {
        color:#a9b9cc;
        font-size:.78rem;
        margin:.38rem 0;
    }

    .pollutant-row {
        display:grid;
        grid-template-columns:repeat(6,minmax(0,1fr));
        gap:.65rem;
    }

    .pollutant {
        background:linear-gradient(180deg,#f9fbfd,#edf3f8);
        color:#102035;
        border-radius:10px;
        padding:.72rem .5rem .64rem;
        text-align:center;
        box-shadow: inset 0 0 0 1px rgba(31,62,89,.10);
    }

    .poll-name {
        font-size:.68rem;
        color:#516178;
        font-weight:850;
    }

    .poll-value {
        font-size:1.12rem;
        font-weight:900;
        color:#0f2036;
        margin:.1rem 0 .38rem;
    }

    .small-chip {
        display:inline-flex;
        align-items:center;
        justify-content:center;
        min-width:78%;
        padding:.24rem .35rem;
        border-radius:7px;
        font-size:.65rem;
        font-weight:850;
    }

    .forecast-row {
        display:grid;
        grid-template-columns:repeat(4,minmax(0,1fr));
        gap:.58rem;
    }

    .forecast-box {
        background:linear-gradient(180deg,#142b45,#102338);
        border:1px solid rgba(140,171,202,.12);
        border-radius:11px;
        padding:.72rem .48rem;
        text-align:center;
    }

    .forecast-label {
        color:#c7d2df;
        font-size:.7rem;
        font-weight:760;
        min-height:1.65rem;
    }

    .forecast-num {
        color:#f9fbff;
        font-size:1.45rem;
        font-weight:900;
        margin:.28rem 0;
    }

    .planner-good {
        background:linear-gradient(135deg,rgba(20,112,77,.45),rgba(11,67,49,.55));
        border:1px solid rgba(63,214,154,.27);
        border-radius:12px;
        padding:.9rem;
        margin-bottom:.65rem;
    }

    .planner-alert {
        background:linear-gradient(135deg,rgba(105,36,45,.38),rgba(59,23,31,.52));
        border:1px solid rgba(255,92,100,.22);
        border-radius:12px;
        padding:.9rem;
    }

    .planner-kicker {
        font-size:.68rem;
        font-weight:850;
        color:#5de2aa;
        margin-bottom:.18rem;
    }

    .planner-title {
        font-size:1.05rem;
        font-weight:900;
        color:#f9fbff;
    }

    .planner-copy {
        color:#b7c5d6;
        font-size:.76rem;
        line-height:1.45;
        margin-top:.24rem;
    }

    div[data-testid="stMetric"] {
        background:linear-gradient(180deg,rgba(13,32,52,.98),rgba(10,26,44,.98));
        border:1px solid rgba(127,164,199,.15);
        border-radius:13px;
        padding:.72rem .8rem;
    }

    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
        color:#91a7be !important;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color:#f8fbff !important;
    }

    [data-testid="stDataFrame"] {
        border:1px solid rgba(127,164,199,.14);
        border-radius:13px;
        overflow:hidden;
    }

    .stButton button {
        background:linear-gradient(135deg,#3284f7,#2866d3);
        color:white;
        border:0;
        border-radius:10px;
        font-weight:800;
    }

    .stButton button:hover {
        color:white;
        border:0;
        filter:brightness(1.05);
    }

    @media (max-width: 1100px) {
        .pollutant-row { grid-template-columns:repeat(3,1fr); }
    }

    @media (max-width: 800px) {
        .pollutant-row { grid-template-columns:repeat(2,1fr); }
        .forecast-row { grid-template-columns:repeat(2,1fr); }
        .aqi-card-inner { grid-template-columns:1fr; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)




def _render_html(html):
    """Render HTML without Markdown interpreting indentation as code."""
    st.markdown(dedent(html).strip(), unsafe_allow_html=True)


def _svg_icon(kind, size=34, color="#f7c843"):
    sun = f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="4" fill="{color}"/><g stroke="{color}" stroke-width="1.8" stroke-linecap="round"><path d="M12 2v2.2M12 19.8V22M4.93 4.93l1.56 1.56M17.51 17.51l1.56 1.56M2 12h2.2M19.8 12H22M4.93 19.07l1.56-1.56M17.51 6.49l1.56-1.56"/></g></svg>'
    cloud = f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M7.5 18h9.2a4.3 4.3 0 0 0 .3-8.59A5.8 5.8 0 0 0 6.15 8.2 4.9 4.9 0 0 0 7.5 18Z" stroke="{color}" stroke-width="1.8" fill="none" stroke-linejoin="round"/></svg>'
    heart = f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M12 21s-7-4.35-9.35-8.5C.5 8.72 2.38 5 6.3 5c2.18 0 3.63 1.18 4.45 2.32C11.57 6.18 13.02 5 15.2 5c3.92 0 5.8 3.72 3.65 7.5C16.5 16.65 12 21 12 21Z" fill="{color}"/></svg>'
    runner = f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="14.5" cy="4.2" r="2" fill="{color}"/><path d="M11 8.2 8.8 12l-3.6 1.6M10.9 8.1l3.4 2.2 3.5.3M9 12.2l2.7 2.4-1.6 4.5M11.7 14.6l3.2 1.3 2.4 3.1" stroke="{color}" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"/></svg>'
    alert = f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M12 3 2.7 20h18.6L12 3Z" fill="{color}"/><rect x="11.1" y="8" width="1.8" height="6.5" rx=".9" fill="#111827"/><circle cx="12" cy="17.2" r="1" fill="#111827"/></svg>'
    return {"sun": sun, "cloud": cloud, "heart": heart, "runner": runner, "alert": alert}.get(kind, cloud)


def _forecast_icon(label):
    if "Day 1" in label or "Day 3" in label:
        return _svg_icon("sun", 32, "#f7c843")
    return _svg_icon("cloud", 32, "#b9d6ff")

# ============================================================
# UI HELPERS
# ============================================================

def _fmt_num(value, digits=1, suffix=""):
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except Exception:
        return str(value)


def _chip(aqi):
    color = get_aqi_color(aqi)
    category = get_aqi_category(aqi)
    return (
        f'<span class="small-chip" style="'
        f'background:{color}22;color:{color};border:1px solid {color}55;">'
        f'{category}</span>'
    )


def _pollutant(name, value):
    return f"""
    <div class="pollutant">
        <div class="poll-name">{name}</div>
        <div class="poll-value">{_fmt_num(value)}</div>
        {_chip(value if name == "AQI" else min(float(value or 0), 300))}
    </div>
    """


def _forecast_box(label, value):
    color = get_aqi_color(value)
    category = get_aqi_category(value)
    return f"""
    <div class="forecast-box">
        <div class="forecast-label">{label}</div>
        <div class="forecast-icon">{_forecast_icon(label)}</div>
        <div class="forecast-num">{float(value):.0f}</div>
        <span class="small-chip" style="background:{color}22;color:{color};border:1px solid {color}55;">{category}</span>
    </div>
    """


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

for key, default in {
    "current_data": None,
    "forecast_data": None,
    "active_city": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="side-brand">
            <div class="brand-mark"></div>
            <div>
                <div class="brand-title">AQI Predictor</div>
                <div class="brand-sub">Pakistan Air Quality</div>
            </div>
        </div>
        <div class="tagline">
            Cleaner Air. Healthier Tomorrow.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not backend_online:
        st.error("Production services are unavailable.")
        st.stop()

    city_list = get_cities()

    if not city_list:
        st.error("No supported cities are available.")
        st.stop()

    default_city = (
        st.session_state.active_city
        if st.session_state.active_city in city_list
        else city_list[0]
    )

    selected_city = st.selectbox(
        "Select City",
        city_list,
        index=city_list.index(default_city),
    )

    generate_dashboard = st.button(
        "Generate Dashboard",
        use_container_width=True,
        type="primary",
    )

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Forecast",
            "City Comparison",
            "Smart Planner",
            "Model Explainability",
            "Data & Insights",
            "About",
        ],
        label_visibility="collapsed",
    )

    st.markdown(
        """
        <div class="side-bottom">
            <div class="side-bottom-title">Better Air<br>A Healthier Pakistan</div>
            <div class="side-bottom-text">
                Data-driven insights for cleaner, healthier cities.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# LOAD DATA
# ============================================================

if generate_dashboard:
    progress = st.progress(0, text=f"Preparing {selected_city} dashboard...")
    try:
        progress.progress(20, text="Loading current AQI and weather...")
        current_data = get_current_conditions(selected_city)
        progress.progress(55, text="Generating 72-hour AQI forecast...")
        forecast_data = get_forecast(selected_city)
        progress.progress(82, text="Preparing insights and explainability views...")
        st.session_state.current_data = current_data
        st.session_state.forecast_data = forecast_data
        st.session_state.active_city = selected_city
        progress.progress(100, text="Dashboard ready")
        progress.empty()
        st.toast(f"{selected_city} dashboard generated successfully.")
    except Exception as error:
        progress.empty()
        st.error("Unable to generate dashboard data.")
        st.code(str(error), language="text")
        st.stop()

if (st.session_state.current_data is None or st.session_state.forecast_data is None or st.session_state.active_city is None):
    _render_html(f"""
    <div class="generate-panel">
        <div style="font-size:1.15rem;font-weight:850;color:#f8fbff;margin-bottom:.35rem;">Select a city and generate the dashboard</div>
        <div>Choose a city from the sidebar, then click <b>Generate Dashboard</b>. Current conditions, forecast, city insights and model explanations will load together.</div>
    </div>
    """)
    st.stop()

active_city = st.session_state.active_city
current = st.session_state.current_data
forecast_response = st.session_state.forecast_data
forecast_source = forecast_response.get("data_source", "Unknown")
forecast_model = forecast_response.get("model", "Unknown")
forecast_version = forecast_response.get("version", "Unknown")
forecast_df = pd.DataFrame(forecast_response.get("forecast", []))
if forecast_df.empty:
    st.error("No forecast data is available.")
    st.stop()
forecast_df["timestamp"] = pd.to_datetime(forecast_df["timestamp"], errors="coerce")
forecast_df["predicted_aqi"] = pd.to_numeric(forecast_df["predicted_aqi"], errors="coerce")
forecast_df["hours_ahead"] = pd.to_numeric(forecast_df["hours_ahead"], errors="coerce")
forecast_df = forecast_df.dropna(subset=["timestamp", "predicted_aqi", "hours_ahead"]).sort_values("timestamp").reset_index(drop=True)
current_aqi = float(current.get("current_aqi") or 0)
current_category = current.get("aqi_category") or get_aqi_category(current_aqi)
nearest_aqi = float(forecast_df.iloc[0]["predicted_aqi"])
day1 = nearest_forecast_value(forecast_df, 24)
day2 = nearest_forecast_value(forecast_df, 48)
day3 = nearest_forecast_value(forecast_df, 72)
last_updated = current.get("timestamp", "")
try:
    display_time = pd.to_datetime(last_updated).strftime("%b %d, %Y  %I:%M %p")
except Exception:
    display_time = str(last_updated)


# ============================================================
# TOP META + HERO
# ============================================================

st.markdown(
    f"""
    <div class="topbar">
        <div></div>
        <div class="topbar-meta">
            <span>{display_time}</span>
            <span class="live-pill"><span class="live-dot"></span>Live Data</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

hero_title = {
    "Dashboard": "Air Quality Overview",
    "Forecast": "72-Hour AQI Forecast",
    "City Comparison": "City Comparison",
    "Smart Planner": "Smart Activity Planner",
    "Model Explainability": "Model Explainability",
    "Data & Insights": "Data & Insights",
    "About": "About AQI Predictor",
}[page]

hero_sub = {
    "Dashboard": "Real-time conditions and AI-powered 3-day forecasts for major Pakistan cities.",
    "Forecast": "Explore the full recursive 72-hour prediction horizon.",
    "City Comparison": "Compare expected air quality across supported cities.",
    "Smart Planner": "Find safer outdoor windows based on predicted AQI.",
    "Model Explainability": "Understand why the production LightGBM model made its prediction.",
    "Data & Insights": "Review forecast statistics, patterns and operational indicators.",
    "About": "Production AQI monitoring, forecasting and explainable machine learning.",
}[page]

st.markdown(
    f"""
    <div class="hero">
        <div>
            <div class="hero-title">{hero_title}</div>
            <div class="hero-sub">{hero_sub}</div>
        </div>
        <div class="hero-location">
            {active_city}
            <span>Pakistan</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DASHBOARD PAGE
# ============================================================

if page == "Dashboard":

    left, middle, right = st.columns([1.65, .82, .88], gap="small")

    with left:
        with st.container(border=True):
            st.markdown("#### Current AQI")
            gauge_col, status_col = st.columns([1.35, .8], gap="small")
            with gauge_col:
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=current_aqi,
                    number={"font": {"size": 52, "color": "#F8FBFF"}},
                    title={"text": current_category, "font": {"size": 15, "color": get_aqi_color(current_aqi)}},
                    gauge={
                        "shape": "angular",
                        "axis": {"range": [0, 300], "tickvals": [0, 50, 100, 150, 200, 300], "tickfont": {"size": 9, "color": "#D4DFEB"}},
                        "bar": {"color": get_aqi_color(current_aqi), "thickness": .13},
                        "bgcolor": "#14283e",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0, 50], "color": "#59dea4"},
                            {"range": [50, 100], "color": "#ffd655"},
                            {"range": [100, 150], "color": "#ffa14c"},
                            {"range": [150, 200], "color": "#ff5c65"},
                            {"range": [200, 300], "color": "#9f7bff"},
                        ],
                    },
                ))
                gauge.update_layout(height=245, margin=dict(l=5, r=5, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", font={"color": "#f8fbff"})
                st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})
            with status_col:
                status_color = get_aqi_color(current_aqi)
                advice = current.get("health_guidance") or get_health_advice(current_aqi)
                _render_html(f"""
                <div style="padding-top:1.25rem;">
                    <div class="aqi-status-title">Air Quality Status</div>
                    <span class="small-chip" style="min-width:130px;background:{status_color}22;color:{status_color};border:1px solid {status_color}55;">{current_category}</span>
                    <div class="muted" style="font-size:.78rem;line-height:1.48;margin-top:.7rem;">{advice}</div>
                    <div class="muted" style="font-size:.68rem;margin-top:.75rem;">Last updated: {display_time}</div>
                </div>
                """)

    with middle:
        temp = current.get("temperature")
        humidity = current.get("humidity")
        wind = current.get("wind_speed")
        pressure = current.get("pressure")
        weather = str(current.get("weather", "Current Conditions")).title()

        _render_html(f"""
        <div class="card card-pad">
            <div class="card-title">Current Weather</div>
            <div class="weather-main">
                <div>{_svg_icon("sun", 46, "#f7c843")}</div>
                <div><div class="weather-temp">{_fmt_num(temp,1,"°C")}</div><div class="weather-cond">{weather}</div></div>
            </div>
            <div class="weather-list">
                <div class="lab">Humidity</div><div class="val">{_fmt_num(humidity,0,"%")}</div>
                <div class="lab">Wind Speed</div><div class="val">{_fmt_num(wind,1," m/s")}</div>
                <div class="lab">Pressure</div><div class="val">{_fmt_num(pressure,0," hPa")}</div>
            </div>
        </div>
        """)

    with right:
        advice = current.get("health_guidance") or get_health_advice(current_aqi)

        _render_html(f"""
        <div class="card card-pad">
            <div class="icon-heading">{_svg_icon("heart", 30, "#ff5d67")}<div class="card-title" style="margin:0;">Health Advice</div></div>
            <div class="health-text">{advice}</div>
            <div class="health-bullet">• Monitor outdoor exposure if you are sensitive.</div>
            <div class="health-bullet">• Check updates before prolonged activity.</div>
            <div class="health-bullet">• Forecast source: {forecast_source}</div>
        </div>
        """)

    st.write("")

    pollutants = [
        ("PM2.5", current.get("pm2_5")),
        ("PM10", current.get("pm10")),
        ("O₃", current.get("ozone")),
        ("NO₂", current.get("nitrogen_dioxide")),
        ("SO₂", current.get("sulphur_dioxide")),
        ("CO", current.get("carbon_monoxide")),
    ]

    pollutant_cards = []

    for name, value in pollutants:
        value_num = 0.0 if value is None else float(value)
        visual_aqi = min(max(value_num, 0.0), 300.0)
        color = get_aqi_color(visual_aqi)
        category = get_aqi_category(visual_aqi)

        pollutant_cards.append(
            f'<div class="pollutant">'
            f'<div class="poll-name">{name}</div>'
            f'<div class="poll-value">{_fmt_num(value)}</div>'
            f'<span class="small-chip" '
            f'style="background:{color}22;color:{color};border:1px solid {color}55;">'
            f'{category}</span>'
            f'</div>'
        )

    pollutant_html = ''.join(pollutant_cards)

    _render_html(
        f'<div class="card card-pad">'
        f'<div class="card-title">Current Pollutants '
        f'<span class="muted">µg/m³</span></div>'
        f'<div class="pollutant-row">{pollutant_html}</div>'
        f'</div>'
    )

    st.write("")

    trend_col, forecast_col = st.columns([1.35, .9], gap="small")

    with trend_col:
        next24 = forecast_df[forecast_df["hours_ahead"] <= 24].copy()

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=next24["timestamp"],
                y=next24["predicted_aqi"],
                mode="lines+markers",
                line={"color": "#4f97ff", "width": 3},
                marker={"size": 6, "color": "#69a9ff"},
                fill="tozeroy",
                fillcolor="rgba(79,151,255,.08)",
                hovertemplate="%{x|%H:%M}<br>AQI %{y:.1f}<extra></extra>",
            )
        )
        fig.update_layout(
            title={"text": "24-Hour AQI Trend", "font": {"size": 16, "color": "#f8fbff"}},
            height=330,
            margin=dict(l=20, r=20, t=48, b=22),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#aebed0"},
            xaxis={"gridcolor": "rgba(255,255,255,.05)", "title": ""},
            yaxis={"gridcolor": "rgba(255,255,255,.05)", "title": "AQI"},
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with forecast_col:
        cards = "".join(
            [
                _forecast_box("Next 3 Hours", nearest_aqi),
                _forecast_box("Day 1", day1),
                _forecast_box("Day 2", day2),
                _forecast_box("Day 3", day3),
            ]
        )
        _render_html(f"""
        <div class="card card-pad">
            <div class="card-title">3-Day Forecast <span class="muted">(Next 72 Hours)</span></div>
            <div class="forecast-row">{cards}</div>
        </div>
        """)

    st.write("")

    factor_col, planner_col = st.columns([1.1, 1], gap="small")

    with factor_col:
        st.markdown('<div class="card-title">Top Factors Influencing Prediction</div>', unsafe_allow_html=True)

        try:
            explanation = calculate_shap_explanation(active_city, forecast_source)
            shap_df = pd.DataFrame(explanation["contributions"]).head(9)
            shap_df = shap_df.sort_values("absolute_impact", ascending=True)

            factor = go.Figure(
                go.Bar(
                    x=shap_df["absolute_impact"],
                    y=shap_df["feature"],
                    orientation="h",
                    marker_color="#4f97ff",
                    hovertemplate="%{y}<br>|SHAP| %{x:.3f}<extra></extra>",
                )
            )
            factor.update_layout(
                height=330,
                margin=dict(l=10, r=10, t=10, b=25),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#aebed0"},
                xaxis={"title": "Relative Feature Importance", "gridcolor": "rgba(255,255,255,.05)"},
                yaxis={"title": ""},
            )
            st.plotly_chart(factor, use_container_width=True, config={"displayModeBar": False})

        except Exception as error:
            st.info("Open Model Explainability for full SHAP details.")
            st.caption(str(error))

    with planner_col:
        st.markdown('<div class="card-title">Smart Activity Planner</div>', unsafe_allow_html=True)

        windows = build_activity_windows(forecast_df, limit=1)

        if not windows.empty:
            best = windows.iloc[0]
            _render_html(f"""
            <div class="planner-good">
                <div class="planner-row">
                    <div>{_svg_icon("runner", 48, "#4ee0a5")}</div>
                    <div>
                        <div class="planner-kicker">Best Time to Go Outside</div>
                        <div class="planner-title">{best["Recommended Time"]}</div>
                        <div class="planner-copy">Forecast AQI: {best["Forecast AQI"]} — {best["AQI Category"]}<br>Outdoor activity: {best["Activity Advice"]}</div>
                    </div>
                </div>
            </div>
            """)

        alerts = build_smart_alerts(forecast_df)

        if alerts:
            alert = alerts[0]
            _render_html(f"""
            <div class="planner-alert">
                <div class="planner-row">
                    <div>{_svg_icon("alert", 46, "#ff5d67")}</div>
                    <div><div class="planner-kicker" style="color:#ff777e;">Air Quality Alert</div><div class="planner-copy">{alert.get("message","")}</div></div>
                </div>
            </div>
            """)

    st.write("")

    try:
        comparison = get_city_comparison()
    except Exception:
        comparison = pd.DataFrame()

    if not comparison.empty:
        comparison = comparison.sort_values("24h Average", ascending=True)

        city_fig = go.Figure(
            go.Bar(
                x=comparison["City"],
                y=comparison["24h Average"],
                marker_color=[
                    get_aqi_color(v)
                    for v in comparison["24h Average"]
                ],
                text=comparison["24h Average"].round(0),
                textposition="outside",
                hovertemplate="%{x}<br>24h Avg AQI %{y:.1f}<extra></extra>",
            )
        )
        city_fig.update_layout(
            title={"text": "City Comparison (24-Hour Average AQI)", "font": {"size": 16, "color": "#f8fbff"}},
            height=330,
            margin=dict(l=20, r=20, t=48, b=22),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#aebed0"},
            xaxis={"gridcolor": "rgba(255,255,255,0)"},
            yaxis={"gridcolor": "rgba(255,255,255,.05)", "title": ""},
            showlegend=False,
        )
        st.plotly_chart(city_fig, use_container_width=True, config={"displayModeBar": False})


# ============================================================
# FORECAST PAGE
# ============================================================

elif page == "Forecast":

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Next 3 Hours", f"{nearest_aqi:.0f} AQI")
    m2.metric("Day 1", f"{day1:.0f} AQI")
    m3.metric("Day 2", f"{day2:.0f} AQI")
    m4.metric("Day 3", f"{day3:.0f} AQI")

    f = go.Figure()
    f.add_trace(
        go.Scatter(
            x=forecast_df["timestamp"],
            y=forecast_df["predicted_aqi"],
            mode="lines+markers",
            line={"color": "#4f97ff", "width": 3},
            marker={"size": 6, "color": "#69a9ff"},
            hovertemplate="%{x|%d %b %H:%M}<br>AQI %{y:.1f}<extra></extra>",
        )
    )
    f.update_layout(
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#aebed0"},
        xaxis={"gridcolor": "rgba(255,255,255,.05)"},
        yaxis={"title": "AQI", "gridcolor": "rgba(255,255,255,.05)"},
    )
    st.plotly_chart(f, use_container_width=True, config={"displayModeBar": False})

    view = forecast_df.copy()
    view["timestamp"] = view["timestamp"].dt.strftime("%a %d %b, %I:%M %p")
    st.dataframe(
        view[["timestamp","hours_ahead","predicted_aqi","aqi_category","alert_level"]],
        use_container_width=True,
        hide_index=True,
    )

    export_df = forecast_df.copy()
    if "city" not in export_df.columns:
        export_df.insert(0, "city", active_city)
    else:
        export_df["city"] = active_city

    st.download_button(
        "Download Forecast CSV",
        export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{active_city.lower()}_aqi_72h_forecast.csv",
        mime="text/csv",
    )


# ============================================================
# CITY COMPARISON PAGE
# ============================================================

elif page == "City Comparison":

    comparison = get_city_comparison()

    if comparison.empty:
        st.warning("City comparison is unavailable.")
    else:
        best = comparison.iloc[0]
        worst = comparison.sort_values("24h Average", ascending=False).iloc[0]

        c1, c2, c3 = st.columns(3)
        c1.metric("Cleanest Forecast", best["City"], f'{best["24h Average"]:.1f} AQI')
        c2.metric("Highest Forecast", worst["City"], f'{worst["24h Average"]:.1f} AQI')
        c3.metric("Cities Compared", len(comparison))

        ordered = comparison.sort_values("24h Average", ascending=True)

        cf = go.Figure(
            go.Bar(
                x=ordered["24h Average"],
                y=ordered["City"],
                orientation="h",
                marker_color=[get_aqi_color(v) for v in ordered["24h Average"]],
                text=ordered["24h Average"].round(1),
                textposition="outside",
            )
        )
        cf.update_layout(
            height=520,
            margin=dict(l=20,r=40,t=20,b=25),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color":"#aebed0"},
            xaxis={"title":"24-Hour Average AQI","gridcolor":"rgba(255,255,255,.05)"},
            yaxis={"title":""},
        )
        st.plotly_chart(cf, use_container_width=True, config={"displayModeBar":False})
        st.dataframe(comparison, use_container_width=True, hide_index=True)


# ============================================================
# SMART PLANNER PAGE
# ============================================================

elif page == "Smart Planner":

    alerts = build_smart_alerts(forecast_df)
    windows = build_activity_windows(forecast_df, limit=8)

    if alerts:
        for alert in alerts:
            st.warning(alert.get("message",""))

    if not windows.empty:
        b1, b2, b3 = st.columns(3)
        best = windows.iloc[0]
        b1.metric("Best Forecast AQI", best["Forecast AQI"])
        b2.metric("Best Time", best["Recommended Time"])
        b3.metric("Activity Advice", best["Activity Advice"])

        st.dataframe(windows, use_container_width=True, hide_index=True)


# ============================================================
# MODEL EXPLAINABILITY PAGE
# ============================================================

elif page == "Model Explainability":

    st.caption(
        "SHAP values explain how each feature moves the next 3-hour AQI prediction "
        "away from the model baseline."
    )

    explanation = calculate_shap_explanation(active_city, forecast_source)

    e1, e2, e3 = st.columns(3)
    e1.metric("Explained Prediction", f'{explanation["prediction"]:.2f} AQI')
    e2.metric("Model Baseline", f'{explanation["base_value"]:.2f} AQI')
    e3.metric("Horizon", "Next 3 Hours")

    contrib = pd.DataFrame(explanation["contributions"])
    top = (
        contrib
        .sort_values("absolute_impact", ascending=False)
        .head(12)
        .sort_values("shap_value", ascending=True)
    )

    sf = go.Figure(
        go.Bar(
            x=top["shap_value"],
            y=top["feature"],
            orientation="h",
            marker_color=[
                "#ff6670" if v > 0 else "#4f97ff"
                for v in top["shap_value"]
            ],
            customdata=top["feature_value"],
            hovertemplate="%{y}<br>SHAP %{x:.3f}<br>Value %{customdata}<extra></extra>",
        )
    )
    sf.add_vline(x=0, line_color="rgba(255,255,255,.35)", line_width=1)
    sf.update_layout(
        height=540,
        margin=dict(l=20,r=20,t=20,b=25),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color":"#aebed0"},
        xaxis={"title":"SHAP Value (AQI impact)","gridcolor":"rgba(255,255,255,.05)"},
        yaxis={"title":""},
    )
    st.plotly_chart(sf, use_container_width=True, config={"displayModeBar":False})

    st.info(build_shap_interpretation(contrib))

    with st.expander("Detailed SHAP values"):
        detail = contrib[
            ["feature","feature_value","shap_value","absolute_impact"]
        ].copy()
        detail.columns = ["Feature","Current Value","SHAP Value","Absolute Impact"]
        st.dataframe(detail, use_container_width=True, hide_index=True)


# ============================================================
# DATA & INSIGHTS PAGE
# ============================================================

elif page == "Data & Insights":

    avg = float(forecast_df["predicted_aqi"].mean())
    peak = float(forecast_df["predicted_aqi"].max())
    low = float(forecast_df["predicted_aqi"].min())

    i1, i2, i3, i4 = st.columns(4)
    i1.metric("72h Average", f"{avg:.1f}")
    i2.metric("72h Peak", f"{peak:.1f}")
    i3.metric("72h Minimum", f"{low:.1f}")
    i4.metric("Forecast Points", len(forecast_df))

    daily = (
        forecast_df
        .assign(Date=forecast_df["timestamp"].dt.date)
        .groupby("Date")
        .agg(
            Average_AQI=("predicted_aqi","mean"),
            Minimum_AQI=("predicted_aqi","min"),
            Maximum_AQI=("predicted_aqi","max"),
        )
        .reset_index()
    )

    st.dataframe(daily.round(1), use_container_width=True, hide_index=True)


# ============================================================
# ABOUT PAGE
# ============================================================

else:

    st.markdown(
        f"""
        <div class="card card-pad">
            <div class="card-title">AQI Predictor</div>
            <div class="muted" style="line-height:1.7;">
                A production air-quality intelligence dashboard for Pakistan.
                It combines current AQI and pollutant observations, weather,
                72-hour recursive forecasting, city comparison, activity planning,
                and SHAP explainability.
                <br><br>
                Model: {forecast_model} {forecast_version}<br>
                Forecast source: {forecast_source}<br>
                Forecast horizon: 72 hours<br>
                Supported cities: {len(SUPPORTED_CITIES)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
