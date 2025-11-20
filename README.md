![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)

# Weather-advisor
This is a side project for weather forecasting with different tools.

## Roadmap

1. Data exploration
   - [x] identify features: temperature, humidity, pressure, wind, wind direction 
   - [x] get historical data and store locally for training
   - [x] data visualizations
   - [x] clean and format data
   - [ ] feature engineering: find/calculate other features 
     - [x] values at t-1,...
     - [x] rolling mean and std values
     - [ ] seasons
2. Create dataset
    - [x] Length of historic data / length of forecasting data?
    - [x] Training/validation/test splits
3. Create forecasting models
   - [ ] Prophet
   - [ ] ARIMA
   - [ ] RandomForest
   - [x] simple LSTM
   - [x] simple Transformer
   - [ ] Transformer architectures (FEDformer,...)
4. Model training and evaluation
   - [x] training pipeline
   - [ ] evaluation pipeline
     - [ ] include metrics (MAE/MSE) 
     - [ ] measure inference times
   - [ ] include model versioning
5. Deploy the forecasting system for real-time predictions
   - [ ] serve model via API
   - [ ] real-time data pipeline
   - [ ] dashboard
   - [ ] containerization
   - [ ] monitoring and logging
6. Translate model outputs into human-readable advice using LLM and create interface for user interaction
   - [ ] mapping from model output to summary description
   - [ ] local LLM integration for user advice
   - [ ] API endpoint
   - [ ] UI
7. Model lifecycle
   - [ ] retraining schedule
   - [ ] model performance monitoring
   - [ ] model registry and promotion



## Setup

Setup the virtual environment 

```shell
python -m venv venv
pip install -r requirements.txt
```


