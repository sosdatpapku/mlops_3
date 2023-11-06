import requests
import json
import os
import mlflow
from mlflow.tracking import MlflowClient
 
mlflow.set_tracking_uri("http://0.0.0.0:5000")
mlflow.set_experiment("get_model")
with mlflow.start_run():
    df = requests.get("https://raw.githubusercontent.com/sosdatpapku/mlops_3/main/cars_moldova%20(2).csv")
    
    with open("/home/airflow/mlops_3/datasets/data.csv", "w") as f:
        f.write(df.text)
        mlflow.log_artifact(local_path="/home/airflow/mlops_3/scripts/get_data.py",artifact_path="get_data code")
        mlflow.end_run()
