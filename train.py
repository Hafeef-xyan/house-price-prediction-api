import numpy 
import pandas as pd
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Load the dataset
df = pd.read_csv('data/house_price_data.csv')

#Separate features and target
x = df.drop(columns=['price','date'])
y = df['price']

#Identify categorical and numerical columns
categorical_cols = x.select_dtypes(include=['object']).columns.tolist()
numerical_cols = x.select_dtypes(exclude=['object']).columns.tolist()

#preprocess the features
preprocessor = ColumnTransformer(
    transformers=[
        (
            'numerical',
            Pipeline([
                ('imputer',SimpleImputer(strategy='median'))
            ]),
            numerical_cols
        ),
        (
            'categorical',
            Pipeline([
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('encoder',OneHotEncoder(handle_unknown='ignore'))
            ]),
            categorical_cols
        ),
    ]
)

# Build the model pipeline
model = Pipeline([
    ('preprocessor',preprocessor),
    ('regressor',LinearRegression())
])

#split the data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

#congigure MLFLOW
mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('house_price_prediction_comparison_with_two_parameters')

#simple mlflow tracking
'''with mlflow.start_run(run_name='linear_regression_v1'):
    model.fit(x_train,y_train)

    predictions = model.predict(x_test)

    #evaluation
    mae = mean_absolute_error(y_test,predictions)
    mse = mean_squared_error(y_test,predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test,predictions)

    #Log metrics and parameters
    mlflow.log_param('model_type',"LinearRegression")
    mlflow.log_param('test_size',0.2)
    mlflow.log_param('random_state',42)
    mlflow.log_metric('mae',mae)
    mlflow.log_metric('mse',mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric('r2',r2)

    #Log the trained model
    mlflow.sklearn.log_model(model,name='house_price_model',skops_trusted_types=['numpy.dtype'])

    # Register this run's model in the Model Registry
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/house_price_model"

    mlflow.register_model(
        model_uri=model_uri,
        name="HousePricePrediction",
    )

    print(f"MAE:  {mae:.2f}")
    print(f"MSE:  {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.4f}")
'''
#comparing experiment tracking
for run_name, fit_intercept in [
    ('linear_regression_v1',True),
    ('linear_regressin_v2',False)
]:
    model = Pipeline([
        ('preprocessor',preprocessor),
        ('regressor',LinearRegression(fit_intercept=fit_intercept))
    ])

    with mlflow.start_run(run_name=run_name):
        model.fit(x_train,y_train)
    
        predictions = model.predict(x_test)
    
        #evaluation
        mae = mean_absolute_error(y_test,predictions)
        mse = mean_squared_error(y_test,predictions)
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        r2 = r2_score(y_test,predictions)
    
        #Log metrics and parameters
        mlflow.log_param('model_type',"LinearRegression")
        mlflow.log_param('test_size',0.2)
        mlflow.log_param('random_state',42)
        mlflow.log_param('fit_intercept',fit_intercept)
        mlflow.log_metric('mae',mae)
        mlflow.log_metric('mse',mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric('r2',r2)
    
        #Log the trained model
        mlflow.sklearn.log_model(model,name='house_price_model',skops_trusted_types=['numpy.dtype'])
    
        print(f"MAE:  {mae:.2f}")
        print(f"MSE:  {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R²:   {r2:.4f}")

print('Training completed and results logged to MLflow.')