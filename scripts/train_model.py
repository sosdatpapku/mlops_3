import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://0.0.0.0:5000")
mlflow.set_experiment("train_data")
with mlflow.start_run():

    X_train = pd.read_csv('/home/airflow/mlops_3/datasets/X_train.csv')
    y_train = pd.read_csv('/home/airflow/mlops_3/datasets/y_train.csv')
    X_test = pd.read_csv('/home/airflow/mlops_3/datasets/X_test.csv')
    y_test = pd.read_csv('/home/airflow/mlops_3/datasets/y_test.csv')
    LR = LinearRegression(fit_intercept=True)

    LR.fit(X_train, y_train)
    mlflow.log_artifact(local_path="/home/airflow/mlops_3/scripts/train_model.py",artifact_path="train_model code")
    mlflow.end_run()
    # Save the model using pickle
    with open('/home/airflow/mlops_3/models/LR.pickle', 'wb') as model_file:
        pickle.dump(LR, model_file)
