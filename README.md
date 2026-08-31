# AQI Prediction Pipeline

An end-to-end MLOps project for real-time air quality monitoring and
72-hour Air Quality Index (AQI) forecasting across major cities in
Pakistan.

The system combines historical and real-time environmental data, feature
engineering, machine learning, model monitoring, automated workflows,
and a deployed Streamlit dashboard in a single production-oriented
pipeline.

## Live Application

**Live Dashboard:**
https://aqi-prediction-pipeline-b8va3kbak5yemfvgmstjgs.streamlit.app/

The dashboard provides current air-quality conditions, pollutant and
weather information, health guidance, forecast analytics, and 72-hour
AQI predictions.

## Project Overview

The project was developed to demonstrate a complete machine-learning
lifecycle rather than only model training. It includes:

-   Historical air-quality data preparation
-   Real-time environmental data collection
-   AQI calculation and feature engineering
-   Feature storage and management with Hopsworks
-   Machine-learning model training and evaluation
-   Production model registration
-   72-hour recursive AQI forecasting
-   Real-time forecasting with historical fallback
-   FastAPI prediction services
-   Streamlit visualization and monitoring dashboard
-   Data-drift and model-performance monitoring
-   Automated retraining decision logic
-   GitHub Actions workflows
-   Cloud deployment

## Forecasting Scope

The production system supports the following 10 Pakistani cities:

-   Faisalabad
-   Hyderabad
-   Islamabad
-   Karachi
-   Lahore
-   Multan
-   Peshawar
-   Quetta
-   Rawalpindi
-   Sialkot

For each city, the forecasting pipeline generates:

-   Forecast horizon: 72 hours
-   Forecast interval: 3 hours
-   Forecast points: 24 predictions per city

A complete forecasting run therefore produces 240 predictions across all
supported cities.

## System Architecture

``` text
Historical Data                         Real-Time Data
      |                                      |
      v                                      v
Data Cleaning                         OpenWeather Collection
      |                                      |
      v                                      v
Feature Engineering <---------- Real-Time Feature Engineering
      |                                      |
      +------------------+-------------------+
                         |
                         v
                  Feature Management
                     Hopsworks
                         |
                         v
                  Model Training
                         |
                         v
                 Model Evaluation
                         |
                         v
              Production Model Registry
                    LightGBM
                         |
              +----------+----------+
              |                     |
              v                     v
       Forecast Pipeline       Model Monitoring
              |                     |
              v                     v
        FastAPI Services       Drift Detection
              |                     |
              +----------+----------+
                         |
                         v
                Streamlit Dashboard
                         |
                         v
                Cloud Deployment
```

## Dataset

The processed model-training dataset contains:

-   43,170 records
-   33 columns
-   10 cities
-   Historical period from January 2025 to June 2025
-   AQI forecasting target: `target_aqi`

The project also maintains a continuously accumulated real-time feature
dataset used by the production forecasting pipeline when sufficient live
history is available.

## Feature Engineering

The forecasting dataset combines air-pollution, weather, temporal, lag,
and rolling-window information.

The production feature pipeline includes information derived from:

-   AQI
-   PM2.5
-   PM10
-   Carbon monoxide
-   Nitrogen dioxide
-   Ozone
-   Sulphur dioxide
-   Temperature
-   Humidity
-   Atmospheric pressure
-   Wind conditions
-   Cloud cover
-   Visibility
-   Time-based variables
-   AQI lag features
-   AQI rolling statistics

Lag and rolling features provide the model with recent air-quality
behavior while current environmental variables provide the latest
atmospheric context.

## Machine Learning

Multiple stages of the project support model training, evaluation,
registration, inference, and monitoring.

The current production model is:

  Item               Value
  ------------------ ----------
  Model              LightGBM
  Registry version   v1
  RMSE               6.7691
  MAE                4.1939
  R-squared          0.9664

The model artifact is stored through the project's model-registry
structure and is loaded dynamically by the forecasting and monitoring
pipelines.

## 72-Hour Forecasting

The production forecasting pipeline generates recursive AQI predictions
for the next three days.

For every supported city, it:

1.  Loads the registered production model.
2.  Checks whether sufficient real-time feature history is available.
3.  Uses real-time data when the live feature history satisfies the
    forecasting requirements.
4.  Falls back to validated historical feature data when required.
5.  Generates predictions at three-hour intervals.
6.  Updates AQI lag and rolling features recursively as forecasting
    progresses.
7.  Assigns AQI categories, alert levels, and health guidance.
8.  Produces 24 predictions covering the next 72 hours.

The forecast pipeline explicitly reports whether a prediction was
generated using `REALTIME` data or `HISTORICAL_FALLBACK`.

## Real-Time Data Pipeline

The real-time pipeline collects current environmental conditions and
converts them into model-compatible features.

Important modules include:

``` text
src/data_collection/collect_data.py
src/realtime/realtime_features.py
src/realtime/feature_group.py
src/realtime/realtime_monitor.py
src/realtime/aqi_calculation.py
```

The real-time feature history is accumulated over time so the
forecasting system can transition from historical fallback to live
forecasting.

## Hopsworks Feature Store

Hopsworks is used as part of the project's feature-management workflow.

The project includes a historical AQI feature group and supporting
real-time feature-group integration. This provides a structured
feature-store layer between data engineering and machine-learning
components.

Sensitive Hopsworks credentials are supplied through environment
variables or deployment secrets rather than being stored directly in the
source code.

## Model Registry

The production model is maintained through a versioned registry
structure.

``` text
models/
├── registry.json
└── registry/
    └── LightGBM/
        └── v1/
            └── model.pkl
```

The forecasting and monitoring code resolves the production model path
from registry metadata, allowing the application to use the registered
production version instead of hard-coding a model artifact.

## Model Monitoring

The project includes monitoring for both feature-distribution changes
and predictive performance.

Observed monitoring results include:

  Monitoring Metric                    Result
  -------------------------------- ----------
  Numeric features showing drift       35.71%
  Mean PSI                           0.809088
  RMSE                                6.76914
  MAE                                 4.19394
  R-squared                           0.96636

The retraining decision does not rely on drift alone. The pipeline
evaluates drift together with model-performance degradation before
deciding whether retraining is required.

In the validated monitoring run, drift was detected but the model's RMSE
had not degraded meaningfully, so automatic retraining was not required.

## FastAPI Service

A FastAPI backend was developed to expose production information,
current conditions, and forecasts.

Available routes include:

``` text
GET  /
GET  /health
GET  /model
GET  /cities
GET  /current/{city}
POST /forecast
GET  /forecast/{city}
POST /refresh
```

The API test suite contains 12 tests covering the current API contract,
and the validated test run completed successfully.

``` bash
python -m pytest tests/test_api.py -v
```

Validated result:

``` text
12 passed
```

## Streamlit Dashboard

The Streamlit application provides a user-facing interface for the
production system.

Main capabilities include:

-   City selection
-   Current AQI monitoring
-   PM2.5 and PM10 monitoring
-   Additional pollutant information
-   Current weather conditions
-   AQI health classification
-   Health guidance
-   72-hour AQI forecast
-   Dynamic forecast visualization
-   Daily forecast summary
-   Forecast statistics
-   AQI risk assessment
-   Complete forecast table
-   CSV forecast export
-   Production model and forecast-source information

The dashboard directly uses the project's production forecasting logic
and is deployed through Streamlit Community Cloud.

## Automation and CI/CD

GitHub Actions is used to automate recurring MLOps tasks.

The repository includes:

``` text
.github/workflows/hourly_pipeline.yml
.github/workflows/daily_monitoring.yml
.github/workflows/retraining_pipeline.yml
```

These workflows support:

### Hourly Pipeline

Runs the recurring data pipeline required to maintain up-to-date
real-time feature history.

### Daily Monitoring

Executes model and data monitoring to identify drift and performance
changes.

### Retraining Pipeline

Supports the model retraining workflow when retraining criteria are
satisfied.

This automation allows the project to operate as a pipeline rather than
as a manually executed notebook-only solution.

## Project Structure

A simplified repository structure is shown below.

``` text
aqi-prediction-pipeline/
│
├── .github/
│   └── workflows/
│       ├── hourly_pipeline.yml
│       ├── daily_monitoring.yml
│       └── retraining_pipeline.yml
│
├── data/
│   ├── processed/
│   └── realtime/
│
├── models/
│   ├── registry.json
│   └── registry/
│
├── results/
│   └── predictions/
│
├── src/
│   ├── api/
│   ├── dashboard/
│   ├── data_collection/
│   ├── model_training/
│   ├── monitoring/
│   └── realtime/
│
├── tests/
│
├── requirements.txt
├── requirements-automation.txt
└── README.md
```

## Local Installation

### 1. Clone the repository

``` bash
git clone <repository-url>
cd aqi-prediction-pipeline
```

### 2. Create a Python environment

Python 3.11 is recommended because it matches the environment used to
develop and validate the project.

Using Conda:

``` bash
conda create -n aqi-hopsworks python=3.11
conda activate aqi-hopsworks
```

### 3. Install dependencies

``` bash
python -m pip install -r requirements.txt
```

### 4. Configure environment variables

The project requires API credentials for external services.

Set the following variables in the local environment or an appropriate
`.env` configuration:

``` text
OPENWEATHER_API_KEY
HOPSWORKS_API_KEY
```

Do not commit API keys to the repository.

## Running the Dashboard Locally

From the project root:

``` bash
python -m streamlit run src/dashboard/app.py
```

Streamlit will provide a local URL in the terminal.

## Running the FastAPI Backend

From the project root:

``` bash
uvicorn src.api.main:app --reload
```

The API can then be accessed locally through the address displayed by
Uvicorn.

## Running the Forecast Pipeline

``` bash
python src/model_training/predict.py
```

The generated forecast is saved under:

``` text
results/predictions/aqi_3day_forecast.csv
```

## Running Tests

``` bash
python -m pytest tests/test_api.py -v
```

## Deployment

The production dashboard is deployed on Streamlit Community Cloud from
the repository's `main` branch.

Deployment configuration:

``` text
Main application: src/dashboard/app.py
Python: 3.11
```

Production credentials are configured using Streamlit secrets rather
than committed source-code values.

**Live application:**
https://aqi-prediction-pipeline-b8va3kbak5yemfvgmstjgs.streamlit.app/

## Technology Stack

  Area                   Technology
  ---------------------- ---------------------------
  Programming language   Python
  Data processing        Pandas, NumPy
  Machine learning       LightGBM, scikit-learn
  Feature store          Hopsworks
  API                    FastAPI
  Dashboard              Streamlit
  Visualization          Plotly
  Model serialization    Joblib
  Testing                Pytest
  Automation             GitHub Actions
  Deployment             Streamlit Community Cloud
  Version control        Git and GitHub

## Key Achievements

The completed project demonstrates:

-   End-to-end AQI machine-learning pipeline development
-   Forecasting for 10 Pakistani cities
-   72-hour recursive forecasts at three-hour intervals
-   Production LightGBM model with RMSE of approximately 6.77
-   Real-time and historical fallback forecasting
-   Feature-store integration
-   Versioned production model management
-   Model-performance monitoring
-   PSI-based data-drift detection
-   Conditional retraining logic
-   Automated recurring workflows with GitHub Actions
-   Tested FastAPI endpoints
-   Production Streamlit dashboard
-   Public cloud deployment

## Limitations

The current forecasting approach has several practical limitations:

-   Future pollutant and weather variables are not independently
    forecast for every future step.
-   Recursive AQI prediction can accumulate uncertainty over longer
    horizons.
-   Real-time data availability depends on external data services.
-   Forecast quality can change as environmental patterns differ from
    the historical training distribution.
-   Monitoring identifies distribution changes, but continued collection
    of newer labelled observations is required for robust future
    retraining.

## Future Improvements

Potential extensions include:

-   Integrating future weather forecasts instead of persistence
    assumptions
-   Forecasting pollutant concentrations alongside AQI
-   Expanding coverage to additional cities
-   Increasing the historical training period
-   Adding formal model explainability
-   Improving automated model promotion and rollback
-   Adding forecast uncertainty intervals
-   Expanding integration and end-to-end testing
-   Introducing additional observability and alerting
-   Comparing the production model with time-series and deep-learning
    approaches

## Conclusion

This project implements a complete AQI forecasting workflow from data
collection through production deployment. It goes beyond standalone
model development by integrating real-time data processing, feature
management, model registration, recursive forecasting, API services,
monitoring, automated workflows, testing, and a publicly accessible
dashboard.

The result is a practical end-to-end MLOps system for monitoring and
forecasting air quality across major cities in Pakistan.
