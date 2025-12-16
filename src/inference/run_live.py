from src.data_factory.live_weather import fetch_live_weather
from src.data_factory.feature_engineering import FeatureGenerator
#from src.inference.window_builder import build_last_window
from src.inference.model_loader import ForecastModelWrapper
from src.models.deep_transformer_model import DeepTransformerForecastModel

# 1. Fetch live data
df_live = fetch_live_weather(
    latitude=48.1351,
    longitude=11.5820,
    hourly_vars=[
        "temperature_2m","relative_humidity_2m","dew_point_2m",
        "wind_speed_10m","wind_direction_10m",
        "surface_pressure","rain","precipitation",
        "cloudcover_low","cloudcover_mid","cloudcover_high"
    ],
    lookback_hours=200
)

# 2. Load FeatureGenerator
fg = FeatureGenerator.load("artifacts/feature_generator.pkl")

# 3. Feature engineering
fg.dataframe = df_live
fg.create_features()
df_feat = fg.transform(fg.dataframe)

# 4. Window
x = build_last_window(df_feat, input_length=168)

# 5. Load model
model = ForecastModelWrapper(
    model_class=DeepTransformerForecastModel,
    checkpoint_path="artifacts/deeptransformer.pt",
    model_kwargs={...}
)

# 6. Predict
forecast = model.predict(x)
print("Forecast:", forecast)