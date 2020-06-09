#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, metrics
import matplotlib.pyplot as plt
import plotly.express as px
from keras.callbacks import Callback
import mlflow # New
import mlflow.tensorflow # New

mlflow.set_experiment("predicting_wind_solar") # New, optional

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


# In[2]:


df = pd.read_csv("germany.csv", parse_dates=[0], index_col=0)
df.head()


# In[3]:


X = df[["windspeed", "temperature", "rad_horizontal", "rad_diffuse"]]
y = df[["solar_GW", "wind_GW"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
tf.random.set_seed(42)


# Lets not even use an sklearn model, we'll make something very simple with Keras.

# In[13]:


def get_model():
    model = keras.Sequential([
        layers.Dense(32, activation='relu', input_shape=[X_train.shape[1]]),
        layers.Dense(32, activation='relu'),
        layers.Dense(2)
      ])
    optimizer = tf.keras.optimizers.Adam(lr=0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=[
        metrics.RootMeanSquaredError(name="rmse"), # Notice I add the names here to make consistent
        metrics.MeanAbsoluteError(name="mae") # Notice I add the names here to make consistent
    ])
    return model


# In[16]:


with mlflow.start_run(run_name="keras"): # New, run_name optional
    mlflow.tensorflow.autolog(every_n_iter=1)

    model = get_model()
    model.summary()
    history = model.fit(X_train, y_train, epochs=200, batch_size=256, validation_split=0.2)
    
    y_predict = model.predict(X_test)
    rmse, mae, r2 = eval_metrics(y_test, y_predict)
    


# In[10]:


fig, axes = plt.subplots(ncols=3, figsize=(10, 4))
axes[0].plot(history.history["loss"])
axes[1].plot(history.history["root_mean_squared_error"])
axes[2].plot(history.history["mean_absolute_error"])
axes[0].set_title("loss"), axes[1].set_title("RMSE"), axes[2].set_title("MAE");


# In[12]:


fig = px.scatter(x=y_test["solar_GW"], y=y_predict[:, 0], 
                 color=X_test["temperature"], range_color=[-5, 25],
                 color_continuous_scale='Inferno')
fig.update_layout(xaxis_title="Actual Solar", 
                  yaxis_title="Predicted Solar", 
                  coloraxis_colorbar=dict(title="temperature"))
fig.show()


# In[13]:


fig = px.scatter(x=y_test["wind_GW"], y=y_predict[:, 1], 
                 color=X_test["windspeed"], range_color=[0, 8],
                 color_continuous_scale='Inferno')
fig.update_layout(xaxis_title="Actual Wind", 
                  yaxis_title="Predicted Wind",
                  coloraxis_colorbar=dict(title="windspeed"))
fig.show()


# In[ ]:




