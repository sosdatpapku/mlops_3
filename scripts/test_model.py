from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
import pickle
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://0.0.0.0:5000")
mlflow.set_experiment("test_model")

with mlflow.start_run():
    X_test = pd.read_csv('/home/airflow/mlops_3/datasets/X_test.csv')
    y_test = pd.read_csv('/home/airflow/mlops_3/datasets/y_test.csv')

    with open('/home/airflow/mlops_3/models/LR.pickle', 'rb') as model_file:
        model = pickle.load(model_file)
        y_predict=model.predict(X_test)
        score = mse(y_test,y_predict,squared=False)
        mlflow.log_artifact(local_path='/home/airflow/mlops_3/scripts/test_model.py',artifact_path="test_model code")
        mlflow.log_metric("score", score)
        mlflow.end_run()
