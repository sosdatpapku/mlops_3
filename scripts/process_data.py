import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
 
mlflow.set_tracking_uri("http://0.0.0.0:5000")
mlflow.set_experiment("process_data")
with mlflow.start_run():
    df = pd.read_csv('/home/airflow/mlops_3/datasets/data.csv')

    cat_columns = []
    num_columns = []

    for column_name in df.columns:
        if (df[column_name].dtypes == object):
            cat_columns +=[column_name]
        else:
            num_columns +=[column_name]
    # Удалим те объекты у которых Расстояние равно 0
    question_dist = df[df.Distance == 0]
    df = df.drop(question_dist.index)

    # здравый смысл
    question_dist = df[(df.Year <2021) & (df.Distance < 1100)]
    df = df.drop(question_dist.index)

    # анализ гистограмм
    question_dist = df[(df.Distance > 1e6)]
    df = df.drop(question_dist.index)

    # здравый смысл
    question_engine = df[df["Engine_capacity(cm3)"] < 600]
    df = df.drop(question_engine.index)

    # здравый смысл
    question_engine = df[df["Engine_capacity(cm3)"] > 5000]
    df = df.drop(question_engine.index)

    # здравый смысл
    question_price = df[(df["Price(euro)"] < 101)]
    df = df.drop(question_price.index)

    # анализ гистограмм
    question_price = df[df["Price(euro)"] > 1e5]
    df = df.drop(question_price.index)

    #анализ гистограмм
    question_year = df[df.Year < 1971]
    df = df.drop(question_year.index)

    df = df.reset_index(drop=True)
    df[num_columns].apply(pd.to_numeric, errors = 'coerce',)
    
    df["Year"] = df["Year"].astype("uint16")
    df["Distance"] = df["Distance"].astype("uint32")
    df["Engine_capacity(cm3)"] = df["Engine_capacity(cm3)"].astype("uint16")
    df["Price(euro)"] = df["Price(euro)"].astype("uint32")
    
    
    df["Age"] = 2022 - df.Year

    df["km_year"] = df.Distance/df.Age # добавляем признак из лекции
    question_km_year = df[df.km_year > 50e3]
    df = df.drop(question_km_year.index)
    question_km_year = df[df.km_year < 100]
    df = df.drop(question_km_year.index)
    df = df.reset_index(drop=True)
    
    num_columns.append("km_year") # добавляем в число числовых признаков
    df["km_year"] = df["km_year"].astype("uint16") # приводим к менее затратному по памяти типу
    
    q1, q2, q3 = np.percentile(df["km_year"], [25, 50, 75])
    
    df["mileage"] = q2
    
    df.loc[(df["km_year"] < q1), "mileage"] = "low"
    df.loc[(df["km_year"] > q3), "mileage"] = "high"
    df.loc[((df["km_year"] <= q3) & (df["km_year"] >= q1)), "mileage"] = "medium"
    
    cat_columns.append("mileage")
    
    counts = df.Make.value_counts()
    rare =  counts[(counts.values < 25)]
    df["Make"] = df["Make"].replace(rare.index.values, "Rare")
    
    price_q1, price_q3 = np.percentile(df["Price(euro)"], [25, 75])
    
    df.loc[((df["Price(euro)"] < price_q1) & (df["Make"] == "Rare")), "Make"] = "Cheap_rare"
    df.loc[((df["Price(euro)"] > price_q3) & (df["Make"] == "Rare")), "Make"] = "Expensive_rare"
    
    df["Engine_class"] = np.nan
    
    df.loc[(df["Engine_capacity(cm3)"] <= 1099), "Engine_class"] = "Microlitre"
    df.loc[((df["Engine_capacity(cm3)"] > 1099) & (df["Engine_capacity(cm3)"] <= 1799)), "Engine_class"] = "Low-capacity"
    df.loc[((df["Engine_capacity(cm3)"] > 1799) & (df["Engine_capacity(cm3)"] <= 3499)), "Engine_class"] = "Mid-capacity"
    df.loc[(df["Engine_capacity(cm3)"] > 3499), "Engine_class"] = "Large-capacity"
    
    cat_columns.append("Engine_class")
    
    df.to_csv('/home/airflow/mlops_3/datasets/data_processed.csv',index=False)

    mlflow.log_artifact(local_path="/home/airflow/mlops_3/scripts/process_data.py",artifact_path="process_data code")
    mlflow.end_run()
