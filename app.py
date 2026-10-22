import mlflow
import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import Request

# Connect to MLflow
mlflow.set_tracking_uri("http://host.docker.internal:5000")

# Load registered model
def load_model():
    return mlflow.sklearn.load_model(
        "models:/House_price_prediction/1"
    )

app = FastAPI(title='House price prediction API')

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()
    yield

app = FastAPI(
    title="House Price Prediction API",
    lifespan=lifespan
)

class HouseFeatures(BaseModel):
    bedrooms: float
    bathrooms: float
    sqft_living: float
    sqft_lot: float
    floors: float
    waterfront: float
    view: float
    condition: float
    sqft_above: float
    sqft_basement: float
    yr_built: float
    yr_renovated: float
    street: str
    city: str
    statezip: str
    country: str


@app.get('/')
def home():
    return {'message':"house price predictin api is running.!!"}

@app.post('/predict')
def predict(house:HouseFeatures, request:Request):

    input_data = pd.DataFrame([house.model_dump()])

    model = request.app.state.model

    prediction = model.predict(input_data)

    return {
        "predicted_price": round(float(prediction[0]), 2)
    }

