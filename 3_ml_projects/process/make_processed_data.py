#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
import seaborn as sb


# # Power Generation Data
# 
# 
# 
# Data from [Open Power System Data](https://data.open-power-system-data.org/time_series/). Direct link to the specific file [here](https://data.open-power-system-data.org/time_series/2019-06-05/time_series_60min_singleindex.csv). https://bit.ly/2AGlXQw for anyone without the notebook.
# 
# **Open Power System Data.** 2019. Data Package Time series. *Version 2019-06-05.* https://doi.org/10.25832/time_series/2019-06-05.

# In[2]:


df_power_all = pd.read_csv("time_series_60min_singleindex.csv", parse_dates=[0], index_col=0)


# In[3]:


df_power_all.head()


# In[4]:


df_power_germany = df_power_all[["DE_solar_generation_actual", "DE_wind_generation_actual"]]
df_power_germany


# In[5]:


df_power_germany.plot();


# In[6]:


df_power_restricted = df_power_germany["2013":]
df_power_restricted.columns = ["solar_MW", "wind_MW"]
df_power_restricted.plot();


# In[7]:


df_power_restricted["2015"].rolling(24 * 7, center=True).mean().plot();


# # Weather Data
# 
# Again, data from [Open Power System Data](https://data.open-power-system-data.org/weather_data/). Direct link to the specific file [here](https://data.open-power-system-data.org/weather_data/opsd-weather_data-2019-04-09.zip). https://bit.ly/2U7TGJU for anyone without the notebook.
# 
# **Open Power System Data.** 2019. Data Package Weather Data. *Version 2019-04-09.* https://doi.org/10.25832/weather_data/2019-04-09.
# 

# In[8]:


df_all_weather = pd.read_csv("weather_data.csv", parse_dates=[0], index_col=0)
df_all_weather.head()


# In[9]:


df_weather_germany = df_all_weather.loc["2013":, ["DE_windspeed_10m", "DE_temperature", "DE_radiation_direct_horizontal", "DE_radiation_diffuse_horizontal"]]
df_weather_germany


# In[49]:


df_weather_germany["DE_temperature"].plot();


# In[10]:


pd.plotting.scatter_matrix(df_weather_germany["2015"], figsize=(9,9));


# # Combining the data
# 
# Because our data comes from the same source, we're fortunate its already on the same index and we can just join. Otherwise we'd probably make use of aggregating to the coarsest scale or using `merge_asof`.

# In[11]:


df_combined_germany = df_power_restricted.join(df_weather_germany)
df_combined_germany


# In[12]:


df_combined_germany.info()


# In[45]:


# And for simplicity, lets dropna instead of trying some fancy imputation or augmentation
df_final = df_combined_germany.dropna()
df_final.loc[:, "solar_MW"] *= 0.001
df_final.loc[:, "wind_MW"] *= 0.001
df_final.columns = ["solar_GW", "wind_GW", "windspeed", "temperature", "rad_horizontal", "rad_diffuse"]
df_final


# In[43]:


sb.pairplot(data=df_final);


# In[46]:


df_final.to_csv("germany.csv")


# In[ ]:




