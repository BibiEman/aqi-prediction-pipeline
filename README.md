# **AQI Prediction Pipeline**

An end-to-end MLOps project for real-time air-quality monitoring and
72-hour Air Quality Index (AQI) forecasting across major cities in
Pakistan.

The system combines historical and real-time environmental data, feature
engineering, machine learning, model monitoring, automated workflows,
explainability, and a deployed Streamlit dashboard.

## **Live Application**

**Live Dashboard:**\
https://aqi-prediction-pipeline-b8va3kbak5yemfvgmstjgs.streamlit.app/

The dashboard provides current AQI and pollutant conditions, weather
information, health guidance, AQI forecasts, local model explanations,
activity recommendations, and multi-city comparison.

## **Project Overview**

The project implements a complete machine-learning lifecycle, including:

-   Historical and real-time air-quality data processing
-   Feature engineering and AQI calculation
-   Hopsworks feature-store integration
-   Exploratory data analysis
-   Machine-learning model training and evaluation
-   ARIMA and LSTM forecasting experiments
-   SHAP model explainability
-   Production model registration
-   72-hour recursive AQI forecasting
-   Real-time forecasting with historical fallback
-   Data-drift and model-performance monitoring
-   Automated retraining decision logic
-   FastAPI prediction services
-   GitHub Actions automation
-   Streamlit dashboard
-   Cloud deployment

## **Forecasting Scope**

The system supports **10 Pakistani cities**:

Faisalabad, Hyderabad, Islamabad, Karachi, Lahore, Multan, Peshawar,
Quetta, Rawalpindi, and Sialkot.

For each city, the production pipeline generates a **72-hour AQI
forecast at 3-hour intervals**, producing 24 predictions per city and
240 predictions in a complete forecasting run.

## **System Architecture**

``` text
Historical Data                 Real-Time Data
      |                               |
      v                               v
Data Preparation              OpenWeather Collection
      |                               |
      v                               v
Feature Engineering <---- Real-Time Feature Engineering
      |                               |
      +---------------+---------------+
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
             Production Registry
                  LightGBM
                      |
             +--------+--------+
             |                 |
             v                 v
      Forecast Pipeline    Model Monitoring
             |                 |
             v                 v
        FastAPI Service    Drift Detection
             |                 |
             +--------+--------+
                      |
                      v
             Streamlit Dashboard
                      |
                      v
                Cloud Deployment
```

## **Dataset**

The processed training dataset contains:

-   **43,170 records**
-   **33 columns**
-   **10 cities**
-   Historical period: **January 2025 to June 2025**
-   Prediction target: `target_aqi`

The system also maintains an accumulated real-time feature dataset for
production forecasting.

## **Feature Engineering**

The production feature pipeline includes AQI and pollutant measurements,
PM2.5, PM10, carbon monoxide, nitrogen dioxide, ozone, sulphur dioxide,
weather variables, temporal features, AQI lag features, and AQI rolling
statistics.

These features provide both recent AQI behavior and current
environmental context to the forecasting model.

## **Machine Learning**

Seven conventional machine-learning models were evaluated using MAE,
RMSE, and R-squared.

  Model                       MAE         RMSE           R²
  ------------------ ------------ ------------ ------------
  **LightGBM**         **4.1939**   **6.7691**   **0.9664**
  ExtraTrees               4.0128       6.8341       0.9657
  RandomForest             4.0226       6.9461       0.9646
  GradientBoosting         4.4233       7.0672       0.9633
  CatBoost                 5.1222       7.7025       0.9564
  XGBoost                  4.3077       7.7887       0.9555
  Ridge                    7.9320      11.5007       0.9029

**LightGBM v1** was selected as the production model based primarily on
RMSE and is integrated with the model registry, forecasting pipeline,
monitoring system, API, and dashboard.

## **Additional Forecasting Experiments**

Two additional forecasting approaches were implemented:

-   **ARIMA(1,1,1)** statistical forecasting baseline
-   **LSTM** deep-learning sequence forecasting

These experiments use different evaluation protocols from the production
LightGBM model, so their headline metrics should not be interpreted as
direct production comparisons.

## **72-Hour Forecasting**

For each supported city, the production pipeline:

1.  Loads the registered LightGBM model.
2.  Checks the available real-time feature history.
3.  Uses real-time data when sufficient history is available.
4.  Falls back to validated historical data when necessary.
5.  Generates AQI predictions every three hours.
6.  Recursively updates AQI lag and rolling features.
7.  Assigns AQI categories, alerts, and health guidance.
8.  Produces 24 predictions covering 72 hours.

Forecasts identify their source as either `REALTIME` or
`HISTORICAL_FALLBACK`.

## **Model Explainability**

SHAP is used to explain the LightGBM model.

The project provides:

-   Global feature importance
-   SHAP summary visualizations
-   Local prediction explanations
-   Feature contribution analysis
-   Dashboard-integrated explanation of the next 3-hour AQI prediction

SHAP contributions indicate how individual features move a prediction
relative to the model baseline and should not be interpreted as causal
effects.

## **Model Monitoring**

The monitoring pipeline evaluates feature-distribution drift, RMSE, MAE,
R-squared, and retraining conditions.

The retraining decision considers both data drift and model-performance
degradation rather than retraining solely because drift is detected.

## **FastAPI Service**

The project includes a FastAPI backend with the following endpoints:

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

API validation:

``` bash
python -m pytest tests/test_api.py -v
```

Validated result: **12 passed**

## **Streamlit Dashboard**

The deployed dashboard provides:

-   City selection and dashboard generation
-   Current AQI and AQI category
-   Current pollutant measurements
-   Current weather conditions
-   Health guidance
-   Next 24-hour AQI forecast
-   3-day / 72-hour AQI forecasting
-   Hazardous AQI alerts
-   Local SHAP model explainability
-   Smart Activity Planner
-   Comparison across all 10 cities
-   Forecast statistics and insights
-   Detailed forecast table
-   CSV forecast export
-   Production model and forecast-source information

The application includes dedicated pages for **Dashboard, Forecast, City
Comparison, Smart Planner, Model Explainability, Data & Insights, and
About**.

## **Automation**

GitHub Actions automates the main MLOps workflows:

``` text
.github/workflows/
├── hourly_pipeline.yml
├── daily_monitoring.yml
└── retraining_pipeline.yml
```

The workflows support recurring real-time data processing, daily
monitoring, and conditional model retraining.

## **Project Structure**

``` text
aqi-prediction-pipeline/
├── .github/
│   └── workflows/
├── data/
│   ├── processed/
│   └── realtime/
├── models/
│   ├── registry.json
│   └── registry/
├── results/
│   ├── eda/
│   ├── explainability/
│   ├── monitoring/
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

## **Local Installation**

### **1. Clone the Repository**

``` bash
git clone https://github.com/BibiEman/aqi-prediction-pipeline.git
cd aqi-prediction-pipeline
```

### **2. Create the Environment**

Python 3.11 is recommended.

``` bash
conda create -n aqi-hopsworks python=3.11
conda activate aqi-hopsworks
```

### **3. Install Dependencies**

``` bash
python -m pip install -r requirements.txt
```

SHAP is included in the production dependencies because it is used by
the deployed dashboard. ARIMA and LSTM experiments may require
additional dependencies such as Statsmodels and TensorFlow.

### **4. Configure Environment Variables**

``` text
OPENWEATHER_API_KEY
HOPSWORKS_API_KEY
```

API keys should be stored locally or through deployment secrets and
should never be committed to the repository.

## **Running the Project**

### **Dashboard**

``` bash
python -m streamlit run src/dashboard/app.py
```

### **FastAPI**

``` bash
uvicorn src.api.main:app --reload
```

### **72-Hour Forecast**

``` bash
python src/model_training/predict.py
```

### **SHAP Explainability**

``` bash
python -m src.explainability.shap_explainer
```

### **API Tests**

``` bash
python -m pytest tests/test_api.py -v
```

## **Deployment**

The production dashboard is deployed on **Streamlit Community Cloud**
from the repository's `main` branch.

``` text
Main application: src/dashboard/app.py
Python: 3.11
```

Production credentials are configured using Streamlit secrets.

**Live Dashboard:**\
https://aqi-prediction-pipeline-b8va3kbak5yemfvgmstjgs.streamlit.app/

## **Technology Stack**

  Area                      Technology
  ------------------------- ---------------------------
  Programming               Python 3.11
  Data Processing           Pandas, NumPy
  Machine Learning          LightGBM, scikit-learn
  Deep Learning             TensorFlow / Keras
  Statistical Forecasting   Statsmodels / ARIMA
  Explainability            SHAP
  Feature Store             Hopsworks
  API                       FastAPI
  Dashboard                 Streamlit
  Visualization             Plotly, Matplotlib
  Testing                   Pytest
  Automation                GitHub Actions
  Deployment                Streamlit Community Cloud
  Version Control           Git, GitHub

## **Key Achievements**

-   End-to-end AQI forecasting and MLOps pipeline
-   Forecasting across 10 Pakistani cities
-   72-hour recursive AQI predictions
-   Production LightGBM model with RMSE ≈ 6.77
-   Real-time forecasting with historical fallback
-   Hopsworks feature-store integration
-   Model registry and monitoring
-   PSI-based drift detection
-   Conditional retraining logic
-   SHAP global and local explainability
-   Smart Activity Planner
-   Multi-city forecast comparison
-   Automated GitHub Actions workflows
-   Tested FastAPI service
-   Deployed Streamlit dashboard

## **Limitations**

-   Future pollutant and weather variables are not independently
    forecast at every future step.
-   Recursive forecasting may accumulate uncertainty over longer
    horizons.
-   Real-time data availability depends on external services.
-   Forecast quality may change as environmental conditions drift from
    the historical training distribution.
-   Islamabad and Rawalpindi contain identical historical AQI series in
    the validated historical dataset.
-   Smart Activity Planner recommendations are general AQI-based
    guidance rather than personalized medical advice.
-   Continued collection of new labelled observations is required for
    robust future retraining.

## **Future Improvements**

-   Integrate actual future weather forecasts
-   Forecast pollutant concentrations alongside AQI
-   Add forecast uncertainty intervals
-   Expand coverage to additional cities
-   Increase the historical training period
-   Improve automated model promotion and rollback
-   Expand integration and end-to-end testing

## **Conclusion**

This project demonstrates a complete AQI forecasting and MLOps workflow,
from data collection and feature engineering through model training,
monitoring, explainability, automation, and cloud deployment.

The final system provides real-time AQI monitoring, 72-hour forecasting,
health guidance, SHAP explanations, activity recommendations, and
multi-city comparison through a publicly accessible Streamlit
application.
