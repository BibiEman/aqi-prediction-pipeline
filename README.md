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

The project demonstrates a complete machine-learning lifecycle rather
than only model training. It includes:

- Historical air-quality data preparation
- Real-time environmental data collection
- AQI calculation and feature engineering
- Feature storage and management with Hopsworks
- Exploratory data analysis and temporal trend analysis
- Machine-learning model training and evaluation
- Statistical forecasting with ARIMA
- Deep-learning forecasting with LSTM
- SHAP global and local model explainability
- Production model registration
- 72-hour recursive AQI forecasting
- Real-time forecasting with historical fallback
- AQI categories, health guidance, and hazardous-condition alerts
- FastAPI prediction services
- Streamlit visualization and monitoring dashboard
- Data-drift and model-performance monitoring
- Automated retraining decision logic
- GitHub Actions workflows
- Cloud deployment

## Forecasting Scope

The production system supports 10 Pakistani cities: Faisalabad,
Hyderabad, Islamabad, Karachi, Lahore, Multan, Peshawar, Quetta,
Rawalpindi, and Sialkot.

For each city, the forecasting pipeline generates a 72-hour forecast at
3-hour intervals, producing 24 predictions per city. A complete
forecasting run therefore produces 240 predictions across all supported
cities.

## System Architecture

```text
Historical Data                         Real-Time Data
      |                                      |
      v                                      v
Data Cleaning                       OpenWeather Collection
      |                                      |
      v                                      v
Feature Engineering <-------- Real-Time Feature Engineering
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
       Forecast Pipeline      Model Monitoring
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

- 43,170 records
- 33 columns
- 10 cities
- Historical period from January 2025 to June 2025
- AQI forecasting target: `target_aqi`

The project also maintains a continuously accumulated real-time feature
dataset used by the production forecasting pipeline when sufficient live
history is available.

## Feature Engineering

The forecasting dataset combines air-pollution, weather, temporal, lag,
and rolling-window information. The production feature pipeline includes
AQI, PM2.5, PM10, carbon monoxide, nitrogen dioxide, ozone, sulphur
dioxide, temperature, humidity, atmospheric pressure, wind conditions,
cloud cover, precipitation, time-based variables, AQI lag features, and
AQI rolling statistics.

Lag and rolling features provide the model with recent air-quality
behavior, while current environmental variables provide the latest
atmospheric context.

## Exploratory Data Analysis

A dedicated EDA pipeline is implemented in:

```text
src/eda/exploratory_data_analysis.py
```

The analysis validates data quality and investigates AQI behavior across
cities, time periods, hours, AQI categories, pollutants, weather
variables, and engineered lag/rolling features.

Key findings include:

- The processed dataset contains 43,170 observations with no duplicate
  rows and no missing values in the validated training data.
- Mean target AQI is approximately 119.07 and the median is 110, with
  a right-skewed distribution and a maximum AQI of 538.
- Faisalabad, Lahore, and Multan have the highest average historical
  AQI among the supported cities.
- January shows particularly elevated AQI in several cities, followed
  by a decline during February and March and increases again in some
  cities during May and June.
- Hourly AQI tends to rise during the late afternoon and evening, with
  the highest average values around 18:00-19:00.
- 55.82% of historical observations are above the Moderate AQI
  category.
- The historical dataset contains 291 Hazardous observations with AQI
  above 300.
- Recent AQI history has the strongest linear association with the
  prediction target; PM2.5 is the strongest pollutant correlation.
- Islamabad and Rawalpindi have identical historical AQI series in the
  validated dataset. This is documented as a data-quality limitation
  rather than silently modifying the training data.

EDA artifacts are generated under `results/eda/`, including AQI
distributions, city comparisons, temporal trends, hourly patterns,
category distributions, correlation tables, and a correlation heatmap.

## Machine Learning

Seven conventional machine-learning models were evaluated using MAE,
RMSE, and R-squared.

    Rank Model                   MAE      RMSE   R-squared

---

       1 LightGBM             4.1939    6.7691      0.9664
       2 ExtraTrees           4.0128    6.8341      0.9657
       3 RandomForest         4.0226    6.9461      0.9646
       4 GradientBoosting     4.4233    7.0672      0.9633
       5 CatBoost             5.1222    7.7025      0.9564
       6 XGBoost              4.3077    7.7887      0.9555
       7 Ridge                7.9320   11.5007      0.9029

LightGBM was selected as the production model based primarily on the
lowest RMSE and is integrated with the production registry, recursive
forecasting, monitoring, API, and dashboard.

Production Item Value

---

Model LightGBM
Registry version v1
RMSE 6.7691
MAE 4.1939
R-squared 0.9664

## Statistical and Deep-Learning Forecasting Experiments

### ARIMA Statistical Baseline

An ARIMA(1,1,1) model is fitted independently for each city using a
chronological 80/20 train/test split.

Metric ARIMA

---

MAE 30.6199
RMSE 39.8168
R-squared -0.1641

The simple univariate ARIMA baseline performs poorly over its long
holdout period, illustrating the difficulty of representing AQI dynamics
with a basic autoregressive formulation alone.

### LSTM Deep-Learning Baseline

A univariate LSTM uses 24 historical AQI observations as sequence
context. The scaler is fitted only on training data, the split is
chronological, training uses early stopping, and each city is modeled
independently.

Metric LSTM

---

MAE 3.5063
RMSE 5.5521
R-squared 0.9774

The LSTM results demonstrate strong one-step-ahead sequence prediction.
However, these metrics should not be treated as a direct production
comparison with LightGBM because the evaluation protocols differ. The
experimental LSTM uses observed AQI context during test evaluation,
whereas the deployed LightGBM pipeline recursively forecasts the next 72
hours using richer engineered and environmental features. LightGBM
therefore remains the production model.

## 72-Hour Forecasting

The production forecasting pipeline generates recursive AQI predictions
for the next three days. For every supported city, it:

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

```text
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
real-time feature-group integration, providing a structured
feature-store layer between data engineering and machine-learning
components.

Sensitive Hopsworks credentials are supplied through environment
variables or deployment secrets rather than being stored directly in
source code.

## Model Registry

The production model is maintained through a versioned registry
structure:

```text
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

## Model Explainability with SHAP

The project includes a formal SHAP explainability pipeline in:

```text
src/explainability/shap_explainer.py
```

The pipeline loads the trained LightGBM model and chronological
evaluation data, applies the fitted preprocessing pipeline, computes
TreeSHAP values, and produces both global and local explanations.

The validated SHAP run used 2,000 test observations and 47 transformed
features. The leading global features by mean absolute SHAP value were:

    Rank Feature         Mean Absolute SHAP

---

       1 `us_aqi`                   27.0146
       2 `ozone`                     5.5398
       3 `hour`                      2.9071
       4 `pm2_5`                     2.4659
       5 `aqi_lag_1`                 1.1253

Additional influential variables include PM10, rolling AQI statistics,
AQI lag features, nitrogen dioxide, temperature, pressure, humidity, and
seasonal/time features. SHAP values are interpreted as model
contributions rather than causal effects.

Generated explainability artifacts include a global feature-importance
CSV, SHAP beeswarm summary plot, SHAP bar plot, local waterfall
explanation, and local contribution CSV under `results/explainability/`.

## Model Monitoring

The project includes monitoring for both feature-distribution changes
and predictive performance.

Monitoring Metric Result

---

Numeric features showing drift 35.71%
Mean PSI 0.809088
RMSE 6.76914
MAE 4.19394
R-squared 0.96636

The retraining decision does not rely on drift alone. The pipeline
evaluates drift together with model-performance degradation before
deciding whether retraining is required.

In the validated monitoring run, drift was detected but the model's RMSE
had not degraded meaningfully, so automatic retraining was not required.

## AQI Categories, Health Guidance, and Hazardous Alerts

Every production forecast includes an AQI category, operational alert
level, and health guidance. Predictions above AQI 300 are classified as
`Hazardous` and assigned an `Emergency` alert level.

The Streamlit dashboard exposes these fields in the forecast table and
performs a 72-hour risk assessment. If any forecast point exceeds AQI
300, the dashboard displays an explicit warning that Hazardous AQI
conditions are forecast during the next 72 hours.

## FastAPI Service

A FastAPI backend was developed to expose production information,
current conditions, and forecasts.

```text
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
and the validated test run completed successfully:

```bash
python -m pytest tests/test_api.py -v
```

Validated result:

```text
12 passed
```

## Streamlit Dashboard

The Streamlit application provides a user-facing interface for the
production system. Main capabilities include:

- City selection
- Current AQI monitoring
- PM2.5 and PM10 monitoring
- Additional pollutant information
- Current weather conditions
- AQI health classification
- Health guidance
- 72-hour AQI forecast
- Dynamic forecast visualization
- Daily forecast summary
- Forecast statistics
- AQI risk assessment
- Complete forecast table
- CSV forecast export
- Production model and forecast-source information

The dashboard directly uses the project's production forecasting logic
and is deployed through Streamlit Community Cloud.

## Automation and CI/CD

GitHub Actions is used to automate recurring MLOps tasks.

```text
.github/workflows/hourly_pipeline.yml
.github/workflows/daily_monitoring.yml
.github/workflows/retraining_pipeline.yml
```

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

```text
aqi-prediction-pipeline/
├── .github/
│   └── workflows/
│       ├── hourly_pipeline.yml
│       ├── daily_monitoring.yml
│       └── retraining_pipeline.yml
├── data/
│   ├── processed/
│   └── realtime/
├── models/
│   ├── registry.json
│   └── registry/
├── results/
│   ├── eda/
│   ├── explainability/
│   ├── model_comparison/
│   └── predictions/
├── src/
│   ├── api/
│   ├── dashboard/
│   ├── data_collection/
│   ├── eda/
│   ├── explainability/
│   ├── forecasting_models/
│   ├── model_training/
│   ├── monitoring/
│   └── realtime/
├── tests/
├── requirements.txt
├── requirements-automation.txt
└── README.md
```

## Local Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd aqi-prediction-pipeline
```

### 2. Create a Python Environment

Python 3.11 is recommended because it matches the environment used to
develop, validate, and deploy the project.

```bash
conda create -n aqi-hopsworks python=3.11
conda activate aqi-hopsworks
```

### 3. Install Production Dependencies

```bash
python -m pip install -r requirements.txt
```

ARIMA, LSTM, and SHAP are experimental/analysis components. Their
additional packages such as Statsmodels, TensorFlow, and SHAP should be
installed only when those experiments are executed; they are not
required by the deployed Streamlit application.

### 4. Configure Environment Variables

Configure the following credentials locally or through deployment
secrets:

```text
OPENWEATHER_API_KEY
HOPSWORKS_API_KEY
```

Do not commit API keys to the repository.

## Running the Dashboard Locally

```bash
python -m streamlit run src/dashboard/app.py
```

## Running the FastAPI Backend

```bash
uvicorn src.api.main:app --reload
```

## Running the Forecast Pipeline

```bash
python src/model_training/predict.py
```

Forecast output:

```text
results/predictions/aqi_3day_forecast.csv
```

## Running EDA

```bash
python -m src.eda.exploratory_data_analysis
```

## Running SHAP Explainability

```bash
python -m src.explainability.shap_explainer
```

## Running Experimental Forecasting Baselines

ARIMA:

```bash
python -m src.forecasting_models.statistical_forecast
```

LSTM:

```bash
python -m src.forecasting_models.lstm_forecast
```

## Running Tests

```bash
python -m pytest tests/test_api.py -v
```

## Deployment

The production dashboard is deployed on Streamlit Community Cloud from
the repository's `main` branch.

```text
Main application: src/dashboard/app.py
Python: 3.11
```

Production credentials are configured using Streamlit secrets rather
than committed source-code values.

**Live Application:**
https://aqi-prediction-pipeline-b8va3kbak5yemfvgmstjgs.streamlit.app/

## Technology Stack

---

Area Technology

---

Programming language Python 3.11

Data processing Pandas, NumPy

Production machine learning LightGBM, scikit-learn

Additional ML experiments Random Forest, ExtraTrees, Gradient
Boosting, XGBoost, CatBoost, Ridge

Statistical forecasting Statsmodels / ARIMA

Deep learning TensorFlow / Keras LSTM

Explainability SHAP

Feature store Hopsworks

API FastAPI

Dashboard Streamlit

Visualization Plotly, Matplotlib

Model serialization Joblib

Testing Pytest

Automation GitHub Actions

Deployment Streamlit Community Cloud

Version control Git and GitHub

---

## Key Achievements

- End-to-end AQI machine-learning and MLOps pipeline
- Forecasting for 10 Pakistani cities
- 72-hour recursive forecasts at three-hour intervals
- Production LightGBM model with RMSE of approximately 6.77
- Seven conventional machine-learning models evaluated
- ARIMA statistical forecasting baseline
- LSTM deep-learning forecasting experiment
- Comprehensive EDA and trend analysis
- SHAP global and local model explainability
- Real-time and historical fallback forecasting
- Feature-store integration with Hopsworks
- Versioned production model management
- Model-performance monitoring
- PSI-based data-drift detection
- Conditional retraining logic
- Hazardous AQI alerts and health guidance
- Automated recurring workflows with GitHub Actions
- Tested FastAPI endpoints
- Production Streamlit dashboard
- Public cloud deployment

## Limitations

The current system has several practical limitations:

- Future pollutant and weather variables are not independently
  forecast for every future step.
- Recursive AQI prediction can accumulate uncertainty over longer
  horizons.
- Real-time data availability depends on external data services.
- Forecast quality can change as environmental patterns differ from
  the historical training distribution.
- Islamabad and Rawalpindi have identical historical AQI series in the
  validated historical dataset, so their historical behavior should
  not be interpreted as fully independent.
- The ARIMA, LSTM, and production LightGBM headline metrics should not
  be interpreted as perfectly apples-to-apples because their
  evaluation protocols differ.
- Continued collection of newer labelled observations is required for
  robust future retraining.

## Future Improvements

Potential extensions include:

- Integrating future weather forecasts instead of persistence
  assumptions
- Forecasting pollutant concentrations alongside AQI
- Expanding coverage to additional cities
- Increasing the historical training period
- Improving automated model promotion and rollback
- Adding forecast uncertainty intervals
- Expanding integration and end-to-end testing
- Introducing additional production observability and alerting
- Developing a standardized evaluation protocol for direct comparison
  of statistical, deep-learning, and production forecasting approaches

## Conclusion

This project implements a complete AQI forecasting workflow from data
collection through production deployment. It goes beyond standalone
model development by integrating real-time data processing, feature
management, EDA, multiple machine-learning approaches, statistical and
deep-learning forecasting experiments, SHAP explainability, model
registration, recursive forecasting, AQI health alerts, API services,
monitoring, automated workflows, testing, and a publicly accessible
dashboard.

The result is a practical end-to-end MLOps system for monitoring and
forecasting air quality across major cities in Pakistan.
