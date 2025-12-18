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
   - [x] Prophet
   - [ ] ARIMA
   - [ ] RandomForest
   - [x] simple LSTM
   - [ ] Transformer architectures (FEDformer,...)
     - [x] simple Transformer
     - [x] Deep Transformer
     - [x] Autoformer
     - [ ] FEDformer
4. Model training and evaluation
   - [x] training pipeline
   - [x] evaluation pipeline
     - [x] include metrics (MAE/MSE)
     - [ ] time measurements
   - [x] inference pipeline
   - [ ] include model versioning
5. Deploy the forecasting system for real-time predictions
   - [x] serve model via API
   - [x] real-time data pipeline
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
## Start

Run training and evaluation. Best trained models will be saved as .pt-files

```shell
source venv/bin activate
python main.py --mode train --model deetransformer
python main.py --mode eval --model deetransformer
```

Start App
```shell
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

