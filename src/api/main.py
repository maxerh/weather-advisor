from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from copy import deepcopy
from contextlib import asynccontextmanager

from src.data_factory.live_weather import fetch_live_weather
from src.data_factory.feature_engineering import FeatureGenerator
from src.inference.window_builder import build_last_window
from src.inference.model_loader import ForecastModelWrapper
from src.models.deep_transformer_model import DeepTransformerForecastModel

LATITUDE = 48.1351
LONGITUDE = 11.5820

HOURLY_VARS = [
    "temperature_2m","relative_humidity_2m","dew_point_2m",
    "wind_speed_10m","wind_direction_10m",
    "surface_pressure","rain","precipitation",
    "cloudcover_low","cloudcover_mid","cloudcover_high"
]

INPUT_LENGTH = 168
PREDICTION_LENGTH = 24

FG_PATH = "artifacts/feature_generator.pkl"
MODEL_PATH = "trained_models/deeptransformer.pt"

class ForecastResponse(BaseModel):
    forecast: list


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Load FeatureGenerator
    fg_base = FeatureGenerator.load(FG_PATH)

    # 2. Fetch minimal live data to infer feature dimensionality
    df_bootstrap = fetch_live_weather(
        latitude=LATITUDE,
        longitude=LONGITUDE,
        hourly_vars=HOURLY_VARS,
        lookback_hours=INPUT_LENGTH + 1
    )

    fg_tmp = deepcopy(fg_base)
    fg_tmp.dataframe = df_bootstrap
    fg_tmp.create_features()

    time = fg_tmp.dataframe['time']
    fg_tmp.dataframe = fg_tmp.dataframe.drop(columns=['time'])
    df_feat = fg_tmp.transform(fg_tmp.dataframe)
    input_dim = df_feat.shape[1]

    # 3. Load model
    model = ForecastModelWrapper(
        model_class=DeepTransformerForecastModel,
        checkpoint_path=MODEL_PATH,
        model_kwargs=dict(
            input_dim=input_dim,
            output_dim=1,
            input_length=INPUT_LENGTH,
            pred_len=PREDICTION_LENGTH,
            embed_dim=128,
            num_heads=8,
            ff_dim=256,
            num_layers=4,
            dropout=0.1,
        )
    )

    app.state.fg = fg_base
    app.state.model = model
    app.state.input_dim = input_dim

    yield

app = FastAPI(
    title="Weather Forecast API",
    lifespan=lifespan
)

@app.get("/predict", response_model=ForecastResponse)
def predict_now():
    try:
        fg = app.state.fg
        model = app.state.model

        # 1. Fetch live data
        print(1)
        df_live = fetch_live_weather(
            latitude=LATITUDE,
            longitude=LONGITUDE,
            hourly_vars=HOURLY_VARS,
            lookback_hours=INPUT_LENGTH + 24
        )

        # 2. Feature engineering
        print(2)
        fg.dataframe = df_live
        fg.create_features()
        time = fg.dataframe['time']
        fg.dataframe = fg.dataframe.drop(columns=['time'])
        df_feat = fg.transform(fg.dataframe)

        # 3. Window
        print(3)
        x = build_last_window(df_feat, INPUT_LENGTH)
        print(x.shape)
        print(x[0])
        # count nan values in x
        nan_count = (x != x).sum()
        print(f"Number of NaN values in input window: {nan_count}")
        # TODO: handle nan values properly before prediction

        # 4. Predict
        print(4)
        y = model.predict(x)
        print(y)

        return {"forecast": y.squeeze().tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": app.state.model is not None,
        "num_features": len(app.state.fg.dataframe.columns) if app.state.fg else 0,
    }