#!/usr/bin/env python
# coding: utf-8

# # **Import Libraries**

# In[1]:


import pandas as pd
import numpy as np
import math
import datetime as dt
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, GRU

from itertools import cycle

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# # **Import Dataset**

# In[2]:


# Import dataset
data = pd.read_csv("GME.csv")
data.head()


# # **Data Preprocessing**

# In[3]:


# check columns of dataset
data.columns


# In[4]:


# check info about dtypes etc of dataset
data.info()


# In[5]:


data.describe()


# In[6]:


# check dimension of dataset
data.shape


# In[7]:


# convert date field from string to Date format index
data['Date'] = pd.to_datetime(data.Date)
data.head()


# In[8]:


print("Starting date: ",data.iloc[0][0])
print("Ending date: ", data.iloc[-1][0])
print("Duration: ", data.iloc[-1][0]-data.iloc[0][0])


# In[9]:


dataavg= data.groupby(data['Date'].dt.strftime('%Y'))[['Open','Close']].mean().sort_values(by='Close')
dataavg.head()


# In[10]:


fig = go.Figure()

fig.add_trace(go.Bar(
    x=dataavg.index,
    y=dataavg['Open'],
    name='Stock Open Price',
    marker_color='blue'
))
fig.add_trace(go.Bar(
    x=dataavg.index,
    y=dataavg['Close'],
    name='Stock Close Price',
    marker_color='red'
))

fig.update_layout(barmode='group', xaxis_tickangle=-45,
                  title='Annual Comparison Between Stock Actual Open And Close Price')
fig.show()


# In[11]:


data.groupby(data['Date'].dt.strftime('%Y'))['Low'].min()


# In[12]:


dataavg_high= data.groupby(data['Date'].dt.strftime('%Y'))['High'].max()
dataavg_low= data.groupby(data['Date'].dt.strftime('%Y'))['Low'].min()


# In[13]:


fig = go.Figure()
fig.add_trace(go.Bar(
    x=dataavg_high.index,
    y=dataavg_high,
    name='Stock high Price',
    marker_color='blue'
))
fig.add_trace(go.Bar(
    x=dataavg_low.index,
    y=dataavg_low,
    name='Stock low Price',
    marker_color='red'
))

fig.update_layout(barmode='group',
                  title='Annual High and Low Stock Price')
fig.show()


# In[14]:


names = cycle(['Stock Open Price','Stock Close Price','Stock High Price','Stock Low Price'])

fig = px.line(data, x=data.Date, y=[data['Open'], data['Close'],
                                          data['High'], data['Low']],
             labels={'Date': 'Date','value':'Stock value'})
fig.update_layout(title_text='Stock Analysis Chart', font_size=15, font_color='black',legend_title_text='Stock Parameters')
fig.for_each_trace(lambda t:  t.update(name = next(names)))
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)

fig.show()


# In[15]:


closedf = data[['Date','Close']]
print("Shape of close dataframe:", closedf.shape)


# In[16]:


fig = px.line(closedf, x=closedf.Date, y=closedf.Close,labels={'Date':'Date','Close':'Close Stock'})
fig.update_traces(marker_line_width=2, opacity=0.6)
fig.update_layout(title_text='Stock Close Price Chart', plot_bgcolor='white', font_size=15, font_color='black')
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()


# In[17]:


close_stock = closedf.copy()
del closedf['Date']
scaler=MinMaxScaler(feature_range=(0,1))
closedf=scaler.fit_transform(np.array(closedf).reshape(-1,1))
print(closedf.shape)


# # **Splliting into Training and testing**

# In[18]:


training_size=int(len(closedf)*0.8)
test_size=len(closedf)-training_size
train_data,test_data=closedf[0:training_size,:],closedf[training_size:len(closedf),:1]
print("train_data: ", train_data.shape)
print("test_data: ", test_data.shape)


# In[19]:


# convert an array of values into matrix
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


# In[20]:


# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 15
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

print("X_train: ", X_train.shape)
print("y_train: ", y_train.shape)
print("X_test: ", X_test.shape)
print("y_test", y_test.shape)


# # **Building the LSTM Network**

# In[21]:


from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout


# In[22]:


#initialisizng the model
regression= Sequential()

#First Input layer and LSTM layer with 0.2% dropout
regression.add(LSTM(units=50,return_sequences=True,kernel_initializer='glorot_uniform',input_shape=(X_train.shape[1],1)))
regression.add(Dropout(0.2))

# Where:
#     return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence.

# Second LSTM layer with 0.2% dropout
regression.add(LSTM(units=50,kernel_initializer='glorot_uniform',return_sequences=True))
regression.add(Dropout(0.2))

#Third LSTM layer with 0.2% dropout
regression.add(LSTM(units=50,kernel_initializer='glorot_uniform',return_sequences=True))
regression.add(Dropout(0.2))

#Fourth LSTM layer with 0.2% dropout, we wont use return sequence true in last layers as we dont want to previous output
regression.add(LSTM(units=50,kernel_initializer='glorot_uniform'))
regression.add(Dropout(0.2))
#Output layer , we wont pass any activation as its continous value model
regression.add(Dense(units=1))

#Compiling the network
regression.compile(optimizer='adam',loss='mean_squared_error')


#fitting the network
regression.fit(X_train,y_train,batch_size=100,epochs=250)


# # **Predicting for test data**

# In[23]:


# Prediction

train_predict=regression.predict(X_train)
test_predict=regression.predict(X_test)

train_predict = train_predict.reshape(-1,1)
test_predict = test_predict.reshape(-1,1)

print("Train data prediction:", train_predict.shape)
print("Test data prediction:", test_predict.shape)


# In[24]:


# Transform back to original form

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1))
original_ytest = scaler.inverse_transform(y_test.reshape(-1,1))


# # **Metrices Evaluation**

# In[25]:


#Evaluation metrices
print("Train data RMAPE: ", math.sqrt(mean_absolute_error(original_ytrain, train_predict)*100))
print("Train data MAPE: ", mean_absolute_error(original_ytrain,train_predict)*100)
print("-------------------------------------------------------------------------------------")
print("Test data RMAPE: ", math.sqrt(mean_absolute_error(original_ytest, test_predict)*100))
print("Test data MAPE: ", mean_absolute_error(original_ytest,test_predict)*100)


# # **Plot train predictions**

# In[26]:


# Plot train predictions

look_back=time_step
trainPredictPlot = np.empty_like(closedf)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
print("Train predicted data: ", trainPredictPlot.shape)

# Plot test predictions
testPredictPlot = np.empty_like(closedf)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(closedf)-1, :] = test_predict
print("Test predicted data: ", testPredictPlot.shape)

names = cycle(['Actual close price','Train predicted close price','Test predicted close price'])

plotdf = pd.DataFrame({'Date': close_stock['Date'],
                       'original_close': close_stock['Close'],
                      'train_predicted_close': trainPredictPlot.reshape(1,-1)[0].tolist(),
                      'test_predicted_close': testPredictPlot.reshape(1,-1)[0].tolist()})

fig = px.line(plotdf,x=plotdf['Date'], y=[plotdf['original_close'],plotdf['train_predicted_close'],
                                          plotdf['test_predicted_close']],
              labels={'value':'Stock price','Date': 'Date'})
fig.update_layout(title_text='Actual Close Price VS Predicted Close Price',
                  plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Close Price')
fig.for_each_trace(lambda t:  t.update(name = next(names)))

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()
