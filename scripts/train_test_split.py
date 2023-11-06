import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
 
df = pd.read_csv('/home/airflow/mlops_3/datasets/data_processed.csv')

num_columns = ['Year', 'Distance', 'Engine_capacity(cm3)', 'Price(euro)', 'km_year']

 
df_num = df[num_columns].copy()

X,y = df_num.drop(columns = ['Price(euro)']).values,df_num['Price(euro)'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train = pd.DataFrame(X_train, columns=['Year', 'Distance', 'Engine_capacity(cm3)', 'km_year'])
y_test = pd.DataFrame(y_test, columns=['Price(euro)'])
X_test = pd.DataFrame(X_test, columns=['Year', 'Distance', 'Engine_capacity(cm3)', 'km_year'])
y_train = pd.DataFrame(y_train, columns=['Price(euro)'])
 
X_train.to_csv('/home/airflow/mlops_3/datasets/X_train.csv',
                        index=None)
X_test.to_csv('/home/airflow/mlops_3/datasets/X_test.csv',
                        index=None)
y_train.to_csv('/home/airflow/mlops_3/datasets/y_train.csv',
                        index=None)
y_test.to_csv('/home/airflow/mlops_3/datasets/y_test.csv',
                        index=None)
