import mlflow
import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

#connect to mlflow
# mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_tracking_uri("http://host.docker.internal:5000")

#load registered model
model = mlflow.sklearn.load_model(
    "models:/House_price_prediction/1"
)

app = FastAPI(title='House price prediction API')

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
def predict(house:HouseFeatures):

    input_data = pd.DataFrame([house.model_dump()])

    prediction = model.predict(input_data)

    return {
        "predicted_price": round(float(prediction[0]), 2)
    }

