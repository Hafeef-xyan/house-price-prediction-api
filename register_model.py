import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri('http://127.0.0.1:5000')

client = MlflowClient()

best_run_id = "9daa537fda6143ef98a2c600544f191c"

model_uri = f"runs:/{best_run_id}/house_price_model" #model name

registered_model = mlflow.register_model(
    model_uri=model_uri,
    name='House_price_prediction'
)

print(f"Registered model: {registered_model.name}")
print(f"Model version: {registered_model.version}")

