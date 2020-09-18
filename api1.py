import random
import bokeh.palettes
from bokeh.models import (HoverTool, FactorRange, Plot, LinearAxis, Grid, Range1d)
from bokeh.models.glyphs import VBar
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.models.sources import ColumnDataSource
from bokeh.io import output_file, show
# from bokeh.palettes import Spectral6
from bokeh.plotting import figure
from flask import Flask, render_template
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
import statsmodels
from statsmodels.tsa.arima_model import ARIMAResults
from pandas.api.types import is_numeric_dtype
# import matplotlib
# matplotlib.use('Agg')
# from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib 
import pandas as pd
from numpy import nan
from numpy import isnan
# import matplotlib.pyplot as plt
import itertools
import warnings
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf
from statsmodels.tsa.arima_model import ARIMA

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,r2_score

import keras
import tensorflow as tf
# import tensorflow.keras
from tensorflow_core.python.keras import Sequential
from tensorflow_core.python.keras.layers import Dense
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.optimizers import SGD 
# from tensorflow.keras.layers import LSTM
# from tensorflow.keras.layers import Dropout
# from tensorflow.keras import losses
from tensorflow_core.python.keras.optimizers import SGD 
from tensorflow_core.python.keras.layers import LSTM
from tensorflow_core.python.keras.layers import Dropout
from tensorflow_core.python.keras import losses
from datetime import datetime



import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from pandas.api.types import is_numeric_dtype
import base64
from pandas import DataFrame
from pandas import concat
# from tensorflow.keras.models import load_model
from tensorflow_core.python.keras.models import load_model

app = Flask(__name__)

# def lstmPrediction(df, offset):
#     df = df.set_index('datetime')

#     df = remove_outlier(df)

#     values = df.values
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled = scaler.fit_transform(values)
#     train_sup = series_to_supervised(scaled, 1, 1)
#     train_sup.drop(train_sup.columns[[8,9,10,11,12,13]], axis=1, inplace=True)
    
#     train_X, train_y, test_X, test_y = TrainTestSplit(df, offset)

#     model = Sequential()           
#     model.add(LSTM(80, input_shape=(train_X.shape[1], train_X.shape[2]),return_sequences=True))
#     model.add(Dropout(0.15))
#     model.add(LSTM(50))
#     model.add(Dropout(0.15))
#     model.add(Dense(1))
#     model.compile(loss=losses.mean_squared_error, optimizer='adam')

#     history = model.fit(train_X, train_y, epochs=25, batch_size=60, validation_data=(test_X, test_y),verbose=2, shuffle=False)

        
#     yhat = model.predict(test_X)
#     test_X = test_X.reshape((test_X.shape[0], 7))

#     # invert scaling for forecast
#     inv_yhat = np.concatenate((yhat, test_X[:, -6:]), axis=1)
#     inv_yhat = scaler.inverse_transform(inv_yhat)
#     inv_yhat = inv_yhat[:,0]

#     # invert scaling for actual
#     test_y = test_y.reshape((len(test_y), 1))
#     inv_y = np.concatenate((test_y, test_X[:, -6:]), axis=1)
#     inv_y = scaler.inverse_transform(inv_y)
#     inv_y = inv_y[:,0]

#     # calculate RMSE
#     rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
#     print('Test RMSE: %.3f' % rmse)


#     return rmse

# def TrainTestSplit(df,offset):
#     values = df.values
#     n_train_time = offset

#     train = values[:n_train_time, :]
#     test = values[n_train_time:, :]

#     # split into input and outputs
#     train_X, train_y = train[:, :-1], train[:, -1]
#     test_X, test_y = test[:, :-1], test[:, -1]

#     # reshape input to be 3D [samples, timesteps, features]
#     train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
#     test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
#     return train_X, train_y, test_X, test_y

# def series_to_supervised(data, lag=1, lead=1, dropnan=True):
#     '''
#         an auxillary function to prepare the dataset with given lag and lead using pandas shift function.
#     '''
#     n_vars = data.shape[1]
#     dff = pd.DataFrame(data)
#     cols, names = [],[]
    
#     for i in range(lag, 0, -1):
#         cols.append(dff.shift(i))
#         names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

#     for i in range(0, lead):
#         cols.append(dff.shift(-i))
#         if i == 0:
#             names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
#         else:
#             names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

#     total = pd.concat(cols, axis=1)
#     total.columns = names
#     if dropnan:
#         total.dropna(inplace=True)
#     return total

# def remove_outlier(df_in):
#     for col_name in df_in.columns:
#         q1 = df_in[col_name].quantile(0.25)
#         q3 = df_in[col_name].quantile(0.75)
#         iqr = q3-q1
#         fence_low  = q1-1.5*iqr
#         fence_high = q3+1.5*iqr
#         df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
#     return df_out





def get_lstm_model(train_X):
    model = Sequential()           
    model.add(LSTM(80, input_shape=(train_X.shape[1], train_X.shape[2]),return_sequences=True))
    model.add(Dropout(0.15))
    model.add(LSTM(50))
    model.add(Dropout(0.15))
    model.add(Dense(1))
    model.compile(loss=losses.mean_squared_error, optimizer='adam')
    return model

def lstmPrediction(df, look_back):


    # df = df.drop(['Rest_active_power'],axis=1)
    # df = df.set_index('datetime')
    # df.index = pd.to_datetime(df.index)
    df.info()
    data = df.TotalActivePower.values
    data = data.astype('float32')

    # reshaping
    data = np.reshape(data, (-1, 1))
    minmax = MinMaxScaler(feature_range=(0, 1))

    # need to scale the data on training set, then transform the unseen data using this
    # otherwise the model would overfit and achieve better results, which is not ideal for
    # real world use of the model
    train_size = int(len(data) * 0.80)
    test_size = len(data) - train_size
    # train_test_split
    train, test = data[0:train_size,:], data[train_size:len(data),:]
    # scaling
    train = minmax.fit_transform(train)
    test = minmax.transform(test)
    # reshaping into X=t and y=t+1
    # look_back = 100
    
    X_train, y_train = create_dataset(train, look_back)
    X_test, y_test = create_dataset(test, look_back)

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    # reshaping input into [samples, time_steps, features] format
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    model = Sequential()
    model.add(LSTM(100, return_sequences=True,
                input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    history = model.fit(X_train, y_train, epochs=20, batch_size=256, validation_data=(X_test, y_test), verbose=1, shuffle=False)
  
    # test_predict = model.predict(X_test)
    # # invert predictions
    # test_predict = minmax.inverse_transform(test_predict)
    # y_test = minmax.inverse_transform([y_test])
    
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    # invert predictions
    train_predict = minmax.inverse_transform(train_predict)
    y_train = minmax.inverse_transform([y_train])
    test_predict = minmax.inverse_transform(test_predict)
    y_test = minmax.inverse_transform([y_test])
    
    result=pd.DataFrame()
    result['datetime']= df.datetime.iloc[train_size+look_back+1:]
    # print(len(result.datetime))
    result['predicted']=test_predict
    # print(len(result.predicted))
    result['actual']=y_test[0][:]
    print(len(result.actual))
    result['datetime'] = pd.to_datetime(result['datetime'])
    source = ColumnDataSource(result)
    p = figure(x_axis_type="datetime", height=280, width=400)
    p.line(x='datetime', y='predicted', legend='Predicted',line_width=2, source=source)
    p.line(x='datetime', y='actual',color='red',legend='Actual', line_width=2, source=source)
    p.xaxis.axis_label = "Year"
    p.yaxis.axis_label = "Energy Consumption"
    script, div = components(p)
    return script,div

# need to create an array of values into a dataset matrix

def create_dataset(dataset, look_back=1):
    X, y = list(), list()
    print(len(dataset)-look_back-1)
    for i in range(len(dataset)-look_back-1):            
        a = (dataset[i:(i+look_back), 0])
        
        X.append(a)
        y.append(dataset[i+look_back, 0])
    
    return np.array(X), np.array(y)

def TrainTestSplit(df):
    # converting to float32 as it takes up less memory and operations can be faster 
    data = df.TotalActivePower.values
    data = data.astype('float32')

    # reshaping
    data = np.reshape(data, (-1, 1))
    minmax = MinMaxScaler(feature_range=(0, 1))

    # need to scale the data on training set, then transform the unseen data using this
    # otherwise the model would overfit and achieve better results, which is not ideal for
    # real world use of the model
    train_size = int(len(data) * 0.80)
    test_size = len(data) - train_size
    # train_test_split
    train, test = data[0:train_size,:], data[train_size:len(data),:]
    # scaling
    train = minmax.fit_transform(train)
    test = minmax.transform(test)
    return train,test


def get_model(df,season):
    model = sm.tsa.statespace.SARIMAX(df.TotalActivePower,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1,season),
                                enforce_stationarity=True,
                                enforce_invertibility=False)
    print("**********")
    results = model.fit()
    return results

def get_predictions_month(df, offset):
    df = df.drop(['TotalReactivePower','CurrentIntensity','Device_1','Device_2','Device_3','Voltage'],axis=1)  
    df.set_index('datetime')
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    
    results = ARIMAResults.load('ARIMAmodelMonth.pkl')
    print("-----------------------------")
    pred = results.get_prediction(start=offset, dynamic=False)
    predicted = pred.predicted_mean
    actual = df[offset:].TotalActivePower
    result = pd.DataFrame()
    steps=12
    forecast = results.forecast(steps)
    print(forecast)
    result=pd.DataFrame()
    # result['datetime']= df.datetime.iloc[offset:]
    data = pd.date_range('2019-11-27', periods = steps, freq ='M')
    print(data)
    result['predicted']=forecast
    
    print(len(result.predicted))
    # result['actual']=actual
    result['datetime'] = pd.to_datetime(data)
    source = ColumnDataSource(result)
    p = figure(x_axis_type="datetime", plot_height=250, plot_width=350,title='Estimate Usage' )
    line=p.line(x='datetime', y='predicted', legend='Predicted', line_width=2, source=source)
    # p.line(x='datetime', y='actual',color='red',  legend='Actual',line_width=2, source=source)
    p.xaxis.axis_label = "Month"
    p.yaxis.axis_label = "Energy Consumption (kWH)"
    hover_tool = HoverTool(tooltips=[
            ('Value', '@predicted'),
            ('Date', '@datetime{%F}'),
        ],formatters={'@datetime': 'datetime'}, renderers=[line])
    p.tools.append(hover_tool)
    script, div = components(p)
    # result.to_csv('SwimmingPoolMonthsPrediction.csv')
    return script, div




def get_predictions_day(df, offset):
    df = df.drop(['TotalReactivePower','CurrentIntensity','Device_1','Device_2','Device_3','Voltage'],axis=1)  
    df.set_index('datetime')
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    
    results = ARIMAResults.load('ARIMAmodelDay.pkl')
    print("-----------------------------")
    pred = results.get_prediction(start=offset, dynamic=False)
    predicted = pred.predicted_mean
    actual = df[offset:].TotalActivePower
    result = pd.DataFrame()
    steps=30
    forecast = results.forecast(steps)
    print(forecast)
    result=pd.DataFrame()
    # result['datetime']= df.datetime.iloc[offset:]
    data = pd.date_range('2019-11-27', periods = steps, freq ='D')
    # print(data)
    result['predicted']=forecast
    
    print(len(result.predicted))
    # result['actual']=actual
    result['datetime'] = pd.to_datetime(data)
    source = ColumnDataSource(result)
    p = figure(x_axis_type="datetime", plot_height=250, plot_width=350,title='Estimate Usage' )
    line=p.line(x='datetime', y='predicted', legend='Predicted', line_width=2, source=source)
    # p.line(x='datetime', y='actual',color='red',  legend='Actual',line_width=2, source=source)
    p.xaxis.axis_label = "Day"
    p.yaxis.axis_label = "Energy Consumption (kWH)"
    hover_tool = HoverTool(tooltips=[
            ('Value', '@predicted'),
            ('Date', '@datetime{%F}'),
        ],formatters={'@datetime': 'datetime'}, renderers=[line])
    p.tools.append(hover_tool)
    script, div = components(p)
    # result.to_csv('SwimmingPoolDaysPrediction.csv')
    return script, div



# @app.route("/")
# def home():

#     return render_template('chart.html')
@app.route("/")
def home():
    date=datetime.date(datetime.now())
    return render_template('chart.html',date=date)

@app.route("/appliances")
def appliances():
    date=datetime.date(datetime.now())
    return render_template('appliances.html',date=date)

@app.route("/day")
def day():
    df = pd.read_csv(r'C:\Users\Aditi\Desktop\projectFy\flaskapp\flaskapp\SwimmingPoolDays.csv', delimiter=',') 
    future= pd.DataFrame()
    future= pd.read_csv(r'C:\Users\Aditi\Desktop\projectFy\flaskapp\flaskapp\SwimmingPoolDaysPrediction.csv', delimiter=',') 
    offset = 356 * 3
    script, div=get_predictions_day(df,offset)
    script1,div1=active_graph(df,26)
    script2, div2=cost_graph(future,df,26,30)
    return render_template('day.html',script=script, div=div,script1=script1,div1=div1,script2=script2, div2=div2)

@app.route("/month")
def month():
    df = pd.read_csv(r'C:\Users\Aditi\Desktop\projectFy\flaskapp\flaskapp\SwimmingPoolMonths.csv', delimiter=',')
    future= pd.DataFrame()
    future= pd.read_csv(r'C:\Users\Aditi\Desktop\projectFy\flaskapp\flaskapp\SwimmingPoolMonthsPrediction.csv', delimiter=',') 
    offset = 12 * 3
    script, div=get_predictions_month(df,  offset)
    script1,div1=active_graph(df,11)
    script2, div2=cost_graph(future,df,11,12)
    return render_template('month.html',script=script, div=div,script1=script1,div1=div1,script2=script2, div2=div2)


@app.route("/hour",methods=['GET'])
def hour():
    df = pd.read_csv(r'C:\Users\Aditi\Desktop\projectFy\flaskapp\flaskapp\SwimmingPoolHours.csv', delimiter=',') 
    future= pd.DataFrame()
    future= pd.read_csv(r'C:\Users\Aditi\Desktop\projectFy\flaskapp\flaskapp\SwimmingPoolHourPrediction.csv', delimiter=',') 
    df.set_index('datetime')
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    script, div=hour_prediction_new()
    script1,div1=active_graph(df,22)
    script2, div2=cost_graph(future,df,22,24)
    # script2, div2 =costplot(df,future,22)
    # script, div=yearappgraphplot()
    return render_template('hour.html',script=script, div=div,script1=script1,div1=div1,script2=script2, div2=div2)   

@app.route("/dayapp")
def dayapp():
    
    return render_template('dayapp.html')

@app.route("/monthapp")
def monthapp():
   
    return render_template('monthapp1.html')


@app.route("/hourapp",methods=['GET'])
def hourapp():
    
    return render_template('hourapp.html')   


# def cost_graph(future,df,present_count,past_count):
#     #presnt_count=22, past_count = 24 for hour
#     #presnt_count=26, past_count = 30 for day
#     # presnt_count= 11, past_count = 12 for month
#     # present=[]
#     # past=[]
#     # present =  df.TotalActivePower.iloc[:-present_count].sum()
#     # past =   df.TotalActivePower.iloc[-(past_count+present_count):-present_count].sum()

#     present=pd.DataFrame()
#     past=pd.DataFrame()
#     present =  df.iloc[-present_count:]
#     past =   df.iloc[-(past_count+present_count):-present_count]
#     # future= pd.DataFrame()
#     # future= pd.read_csv('/home/dell/Documents/Btech Project/flaskapp/SwimmingPoolHoursPrediction.csv', delimiter=',') 
#     pre=present['TotalActivePower'].sum()*0.101
#     pas=past['TotalActivePower'].sum()*0.101
#     fut=future['predicted'].sum()*0.101
    
#     # print("*********",pre)
#     # print("----------",pas)
#     print("----------",fut)

#     dates =['Past','Present','Predicted']
#     # dates=[df.datetime.iloc[:-past_count+1],df.datetime.iloc[:-1]]
#     Value =[pas,pre,fut]  
#     source = ColumnDataSource(data=dict(dates=dates, Value=Value))

#     p = figure(x_range=dates, plot_height=280, title="Cost Graph")

#     # p.vbar(x='Devices', top='Value', width=0.9, legend_field="Device", source=source)
#     line=p.vbar(x=dates, top=Value, width=0.2)
#     p.xgrid.grid_line_color = None
#     p.legend.orientation = "horizontal"
#     p.legend.location = "top_center"
#     hover_tool = HoverTool(tooltips=[
#             ('Value', '@top'),
           
#         ], renderers=[line])
#     p.tools.append(hover_tool)

#     script, div = components(p)
#     return script,div





# def active_graph(df,offset):
#     #22 for hour
#     #11 for month
#     #26 for days
#     result = pd.DataFrame()

#     result['Pump'] = df.Device_1.iloc[:-offset]
#     result['Heater'] = df.Device_2.iloc[:-offset]
#     result['Lights'] = df.Device_3.iloc[:-offset]

#     Device = ['Heater','Lights','Pump']
#     Value = [result['Pump'].sum(),result['Heater'].sum(),result['Lights'].sum()]
    

#     source = ColumnDataSource(data=dict(Device=Device, Value=Value))

#     p = figure(x_range=Device, plot_height=280, title="Active Appliances")

#     # p.vbar(x='Devices', top='Value', width=0.9, legend_field="Device", source=source)
#     line=p.vbar(x=Device, top=Value, width=0.2)
#     p.xgrid.grid_line_color = None
#     p.legend.orientation = "horizontal"
#     p.legend.location = "top_center"
#     p.xaxis.axis_label = "Appliances"
#     p.yaxis.axis_label = "Energy Consumption (WH)"
#     hover_tool = HoverTool(tooltips=[
#             ('Value', '@top'),
           
#         ], renderers=[line])
#     p.tools.append(hover_tool)
#     script, div = components(p)
#     return script,div


def cost_graph(future,df,present_count,past_count):
    #presnt_count=22, past_count = 24 for hour
    #presnt_count=26, past_count = 30 for day
    # presnt_count= 11, past_count = 12 for month
    # present=[]
    # past=[]
    # present =  df.TotalActivePower.iloc[:-present_count].sum()
    # past =   df.TotalActivePower.iloc[-(past_count+present_count):-present_count].sum()

    present=pd.DataFrame()
    past=pd.DataFrame()
    present =  df.iloc[-present_count:]
    past =   df.iloc[-(past_count+present_count):-present_count]
    # future= pd.DataFrame()
    # future= pd.read_csv('/home/dell/Documents/Btech Project/flaskapp/SwimmingPoolHoursPrediction.csv', delimiter=',') 
    pre=present['TotalActivePower'].sum()*0.002683
    pas=past['TotalActivePower'].sum()*0.002683
    fut=future['predicted'].sum()*0.002683
    
    # print("***",pre)
    # print("----------",pas)
    print("----------",fut)

    dates =['Past','Present','Predicted']
    # dates=[df.datetime.iloc[:-past_count+1],df.datetime.iloc[:-1]]
    Value =[pas,pre,fut]  
    source = ColumnDataSource(data=dict(dates=dates, Value=Value))

    p = figure(x_range=dates, plot_height=250, plot_width=350, title="Cost Graph")

    # p.vbar(x='Devices', top='Value', width=0.9, legend_field="Device", source=source)
    line=p.vbar(x=dates, top=Value, width=0.2)
    p.xgrid.grid_line_color = None
    p.legend.orientation = "horizontal"
    p.legend.location = "top_center"
    p.xaxis.axis_label = "Timeline"
    p.yaxis.axis_label = "Cost ($)"
    hover_tool = HoverTool(tooltips=[
            ("Value", "@top{1.111}"),
           
        ], renderers=[line])
    p.tools.append(hover_tool)

    script, div = components(p)
    return script,div





def active_graph(df,offset):
    #22 for hour
    #11 for month
    #26 for days
    result = pd.DataFrame()

    result['Pump'] = df.Device_1.iloc[-offset:]
    result['Heater'] = df.Device_2.iloc[-offset:]
    result['Lights'] = df.Device_3.iloc[-offset:]

    Device = ['Heater','Lights','Pump']
    Value = [(result['Pump'].sum())/1000,(result['Heater'].sum())/1000,(result['Lights'].sum())/1000]
    

    source = ColumnDataSource(data=dict(Device=Device, Value=Value))

    p = figure(x_range=Device, plot_height=250, plot_width=350, title="Active Appliances")

    # p.vbar(x='Devices', top='Value', width=0.9, legend_field="Device", source=source)
    line=p.vbar(x=Device, top=Value, width=0.2)
    p.xgrid.grid_line_color = None
    p.legend.orientation = "horizontal"
    p.legend.location = "top_center"
    p.xaxis.axis_label = "Appliances"
    p.yaxis.axis_label = "Energy Consumption (kWH)"
    hover_tool = HoverTool(tooltips=[
            ('Energy ', '@top{1.111} kWH'),
           
        ], renderers=[line])
    p.tools.append(hover_tool)
    script, div = components(p)
    return script,div

# salluuuuuuuuu
def remove_outlier(df_in):
    for col_name in df_in.columns:
        q1 = df_in[col_name].quantile(0.25)
        q3 = df_in[col_name].quantile(0.75)
        iqr = q3-q1
        fence_low  = q1-1.5*iqr
        fence_high = q3+1.5*iqr
        df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out

def series_to_supervised(data, lag=1, lead=1, dropnan=True):
    '''
        an auxillary function to prepare the dataset with given lag and lead using pandas shift function.
    '''
    n_vars = data.shape[1]
    dff = pd.DataFrame(data)
    cols, names = [],[]
    
    for i in range(lag, 0, -1):
        cols.append(dff.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

    for i in range(0, lead):
        cols.append(dff.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

    total = pd.concat(cols, axis=1)
    total.columns = names
    if dropnan:
        total.dropna(inplace=True)
    return total


def hour_prediction_new():
    
    result=pd.read_csv(r'C:\Users\Aditi\Desktop\projectFy\flaskapp\flaskapp\SwimmingPoolHourPrediction.csv', delimiter=',') 

    print(result.head())
    result['datetime'] = pd.to_datetime(result['datetime'])
    source = ColumnDataSource(result)
    p = figure(x_axis_type="datetime", plot_height=250, plot_width=350,title='Estimate Usage' )
    line=p.line(x='datetime', y='predicted', legend='Predicted', line_width=2, source=source)
    # p.line(x='datetime', y='actual',color='red',  legend='Actual',line_width=2, source=source)
    p.xaxis.axis_label = "Time"
    p.yaxis.axis_label = "Energy Consumption (kWH)"
    hover_tool = HoverTool(tooltips=[
            ('Value', '$y'),
            ('Date', '@datetime{%F}'),
        ],formatters={'@datetime': 'datetime'}, renderers=[line])
    p.tools.append(hover_tool)
    script, div = components(p)
    # script, div=active_graph(df,22)
    # script, div=cost_graph(df,22,24)
    # result.to_csv('SwimmingPoolHourPrediction.csv')

    return script,div

def forecast(test_X,future,scaler,model):
    predictions = []
    # predict for #days ahead

    ## we require last 7 days of actual test data to start predicting for next 7 days ahead. 
    ## hence test_X[-7,:]

    for ix in range(future,0,-1):
        row = test_X[-ix,:]
        row = row.reshape(1,1,7)
        out = model.predict(row)
        #print(out)
        row = row.reshape((1,7))

        inv_yhat = np.concatenate((out, row[:, -6:]), axis=1)
        inv_yhat = scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:,0]

        predictions.append(inv_yhat[0])
        
    return predictions


# naaruuuuuuuuu
# def yearappgraphplot():
#     ddf= pd.read_csv('/home/dell/Documents/Btech Project/DatasetLatestDates/SwimmingPoolMonths.csv', delimiter=',')
#     df= ddf.loc[1:14, ['datetime', 'Device_1']] 
#     result=pd.DataFrame()
   
#     result['datetime']=df['datetime']
#     result['device1']=df['Device_1']
#     print(result)
#     #print(len(result.predicted))
#     # result['actual']=y_test[0][:]
#     # result['datetime'] = pd.to_datetime(result['datetime'])
#     result['datetime'] = pd.to_datetime(result['datetime'], errors='coerce')
#     # print(len(result['datetime']))
#     # datetime1=result['datetime'].dt.date
#     source = ColumnDataSource(result)
#     # source = ColumnDataSource(df)
#     p = figure(x_axis_type="datetime", height=280, width=400,title='Estimate Usage' )
#     line=p.line(x='datetime', y='device1', legend='energy for device 1', line_width=2, source=source)
#     # p.line(x='datetime', y='actual',color='red',  legend='Actual',line_width=2, source=source)
#     p.xaxis.axis_label = "Year"
#     p.yaxis.axis_label = "Energy Consumption by device 1"

#     script, div = components(p)
#     # result.to_csv('SwimmingPoolDaysPrediction.csv')
#     return script, div



def deviceplots(d,v):

    Device=d
    Value=v
    source = ColumnDataSource(data=dict(Device=Device, Value=Value))

    p = figure(x_range=Device, plot_height=350, title="Device Wise Energy Consumption")

# p.vbar(x='Devices', top='Value', width=0.9, legend_field="Device", source=source)
    line=p.vbar(x=Device, top=Value, width=0.2)
    p.xgrid.grid_line_color = None
    p.legend.orientation = "horizontal"
    p.legend.location = "top_center"
    p.xaxis.axis_label = "Date"
    p.xaxis.axis_label_text_font_size = "12pt"
    p.xaxis.major_label_orientation = 'vertical'
    p.yaxis.axis_label = "Energy Consumption (WH)"
    hover_tool = HoverTool(tooltips=[('Value', '@top'),], renderers=[line])
    p.tools.append(hover_tool)
    script, div = components(p)
    return script,div   


# def costplot(df,future,offset):
#     #presnt_count=22, past_count = 24 for hour
#     #presnt_count=26, past_count = 30 for day
#     # presnt_count= 11, past_count = 12 for month
#     result=pd.DataFrame()
#     result['datetime'] = df['datetime'].iloc[-offset:]
#     result['datetime'] = pd.to_datetime(result['datetime'], errors='coerce')
#     result['predicted']=df['TotalActivePower'].iloc[-offset:]
#     result.append(future, ignore_index = True)
#     print(result.tail())
#     source = ColumnDataSource(result)

#     p = figure(x_range="datetime", plot_height=350, title="Device wise Energy")

# # p.vbar(x='Devices', top='Value', width=0.9, legend_field="Device", source=source)
#     line=p.vbar(x='datetime', top='predicted', width=0.2,source=source)
#     p.xgrid.grid_line_color = None
#     p.legend.orientation = "horizontal"
#     p.legend.location = "top_center"
#     p.xaxis.axis_label = "Date"
#     p.xaxis.axis_label_text_font_size = "12pt"
#     p.xaxis.major_label_orientation = 'vertical'
#     p.yaxis.axis_label = "Energy Consumption (WH)"
#     hover_tool = HoverTool(tooltips=[('Value', '@Value'),], renderers=[line])
#     p.tools.append(hover_tool)
#     script, div = components(p)
#     return script,div


def deviceoneplot(result):
    
    source = ColumnDataSource(result)
    
    p = figure(x_axis_type="datetime", height=350, width=400,title='Estimate Usage' )
    line=p.line(x='datetime', y='device1', legend='energy for device 1', line_width=2, source=source)
    # p.line(x='datetime', y='actual',color='red',  legend='Actual',line_width=2, source=source)
    p.xaxis.axis_label = "Year"
    p.yaxis.axis_label = "Energy Consumption by device 1"

    script, div = components(p)
    # result.to_csv('SwimmingPoolDaysPrediction.csv')
    return script, div


@app.route('/yeardeviceplot/<int:post_id>')
def yeardeviceplot(post_id):
    
    b=post_id
    print(b)
    if(b==1):
        ddf= pd.read_csv(r'C:\Users\Aditi\Desktop\projectFy\flaskapp\flaskapp\SwimmingPoolMonths.csv', delimiter=',')
        df= ddf.loc[39:, ['datetime', 'Device_1']] 
        result=pd.DataFrame()
        result = pd.DataFrame()
        result['Pump'] = df['Device_1'] 
        Device =df['datetime']
        Value = result['Pump']
        script, div=deviceplots(Device,Value)
        return render_template('yeardeviceplot.html' ,script=script, div=div)
    if(b==2):
        ddf= pd.read_csv(r'C:\Users\Aditi\Desktop\projectFy\flaskapp\flaskapp\SwimmingPoolMonths.csv', delimiter=',')
        df= ddf.loc[39:, ['datetime', 'Device_2']] 
        result=pd.DataFrame()
        result = pd.DataFrame()
        result['Lighting'] = df['Device_2'] 
        Device =df['datetime']
        Value = result['Lighting']
        script, div=deviceplots(Device,Value)
        return render_template('yeardeviceplot.html' ,script=script, div=div)
    if(b==3):
        ddf= pd.read_csv(r'C:\Users\Aditi\Desktop\projectFy\flaskapp\flaskapp\SwimmingPoolMonths.csv', delimiter=',')
        df= ddf.loc[39:, ['datetime', 'Device_3']] 
        result=pd.DataFrame()
        result = pd.DataFrame()
        result['Pump'] = df['Device_3'] 
        Device =df['datetime']
        Value = result['Pump']
        script, div=deviceplots(Device,Value)
        return render_template('yeardeviceplot.html' ,script=script, div=div)



@app.route('/monthdeviceplot/<int:post_id>')
def monthdeviceplot(post_id):
    
    b=post_id
    print(b)
    if(b==1):
        ddf= pd.read_csv(r'C:\Users\Aditi\Desktop\projectFy\flaskapp\flaskapp\SwimmingPoolDays.csv', delimiter=',')
        df= ddf.loc[1418:, ['datetime', 'Device_1']] 
        result=pd.DataFrame()
        result = pd.DataFrame()
        result['Pump'] = df['Device_1'] 
        Device =df['datetime']
        Value = result['Pump']
        script, div=deviceplots(Device,Value)
        return render_template('monthdeviceplot.html' ,script=script, div=div)
    if(b==2):
        ddf= pd.read_csv(r'C:\Users\Aditi\Desktop\projectFy\flaskapp\flaskapp\SwimmingPoolDays.csv', delimiter=',')
        df= ddf.loc[1418:, ['datetime', 'Device_2']] 
        result=pd.DataFrame()
        result = pd.DataFrame()
        result['Lighting'] = df['Device_2'] 
        Device =df['datetime']
        Value = result['Lighting']
        script, div=deviceplots(Device,Value)
        return render_template('monthdeviceplot.html' ,script=script, div=div)
    if(b==3):
        ddf= pd.read_csv(r'C:\Users\Aditi\Desktop\projectFy\flaskapp\flaskapp\SwimmingPoolDays.csv', delimiter=',')
        df= ddf.loc[1418:, ['datetime', 'Device_3']] 
        result=pd.DataFrame()
        result = pd.DataFrame()
        result['Pump'] = df['Device_3'] 
        Device =df['datetime']
        Value = result['Pump']
        script, div=deviceplots(Device,Value)
        return render_template('monthdeviceplot.html' ,script=script, div=div)


@app.route('/todaydeviceplot/<int:post_id>')
def todaydeviceplot(post_id):
    
    b=post_id
    print(b)
    if(b==1):
        ddf= pd.read_csv(r'C:\Users\Aditi\Desktop\projectFy\flaskapp\flaskapp\SwimmingPoolHours.csv', delimiter=',')
        df= ddf.loc[34569:, ['datetime', 'Device_1']] 
        result=pd.DataFrame()
        result = pd.DataFrame()
        result['Pump'] = df['Device_1'] 
        Device =df['datetime']
        Value = result['Pump']
        script, div=deviceplots(Device,Value)
        return render_template('todaydeviceplot.html' ,script=script, div=div)
    if(b==2):
        ddf= pd.read_csv(r'C:\Users\Aditi\Desktop\projectFy\flaskapp\flaskapp\SwimmingPoolHours.csv', delimiter=',')
        df= ddf.loc[34569:, ['datetime', 'Device_2']] 
        result=pd.DataFrame()
        result = pd.DataFrame()
        result['Lighting'] = df['Device_2'] 
        Device =df['datetime']
        Value = result['Lighting']
        script, div=deviceplots(Device,Value)
        return render_template('todaydeviceplot.html' ,script=script, div=div)
    if(b==3):
        ddf= pd.read_csv(r'C:\Users\Aditi\Desktop\projectFy\flaskapp\flaskapp\SwimmingPoolHours.csv', delimiter=',')
        df= ddf.loc[34569:, ['datetime', 'Device_3']] 
        result=pd.DataFrame()
        result = pd.DataFrame()
        result['Pump'] = df['Device_3'] 
        Device =df['datetime']
        Value = result['Pump']
        script, div=deviceplots(Device,Value)
        return render_template('todaydeviceplot.html' ,script=script, div=div)




if __name__ == "__main__":
    app.run(debug=True,port="5002",threaded=False)
