#!/usr/bin/env python
# coding: utf-8

# In[8]:

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.express as px
import mlflow # New
import mlflow.sklearn # New

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


df = pd.read_csv("germany.csv", parse_dates=[0], index_col=0)


# In[10]:


X = df[["windspeed", "temperature", "rad_horizontal", "rad_diffuse"]]
y = df[["solar_GW", "wind_GW"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



with mlflow.start_run(): # New, run_name optional
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_predict = model.predict(X_test)
    rmse, mae, r2 = eval_metrics(y_test, y_predict)

    print(f"RMSE: {rmse:0.2f}")
    print(f"MAE: {mae:0.2f}")
    print(f"r2: {r2:0.2f}")
    
    mlflow.log_metric("rmse", rmse) # New
    mlflow.log_metric("mae", mae) # New
    mlflow.log_metric("r2", r2) # New
    mlflow.sklearn.log_model(model, "model") # New




