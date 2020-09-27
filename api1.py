from bokeh.models import (HoverTool, FactorRange, Plot, LinearAxis, Grid, Range1d)
from bokeh.models.glyphs import VBar
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.models.sources import ColumnDataSource
from bokeh.plotting import figure

from flask import Flask, render_template

from statsmodels.tsa.arima_model import ARIMAResults


import pandas as pd
from pandas import read_csv
import numpy as np
from pandas import DataFrame

from datetime import datetime


app = Flask(__name__)

filepath = "C:\\Users\\Aditi\\Desktop\\projectFy\\flaskapp\\flaskapp\\"
SwimmingPoolDays = "SwimmingPoolDays.csv"
SwimmingPoolMonths = "SwimmingPoolMonths.csv"
SwimmingPoolHours = "SwimmingPoolHours.csv"
SwimmingPoolDaysPrediction = "SwimmingPoolDaysPrediction.csv"
SwimmingPoolMonthsPrediction = "SwimmingPoolMonthsPrediction.csv"
SwimmingPoolHourPrediction = "SwimmingPoolHourPrediction.csv"


def get_predictions_hour():
    
    result=pd.read_csv(filepath + SwimmingPoolHourPrediction , delimiter=',') 

    print(result.head())
    result['datetime'] = pd.to_datetime(result['datetime'])
    source = ColumnDataSource(result)
    p = figure(x_axis_type="datetime", plot_height=250, plot_width=350,title='Estimate Usage' )
    line=p.line(x='datetime', y='predicted', legend='Predicted', line_width=2, source=source)
    p.xaxis.axis_label = "Time"
    p.yaxis.axis_label = "Energy Consumption (kWH)"
    hover_tool = HoverTool(tooltips=[
            ('Value', '$y{1.11}'),
            ('Date', '@datetime{%F}'),
        ],formatters={'@datetime': 'datetime'}, renderers=[line])
    p.tools.append(hover_tool)
    script, div = components(p)

    return script,div


def get_predictions_month(df, offset):
    df = df.drop(['TotalReactivePower','CurrentIntensity','Device_1','Device_2','Device_3','Voltage'],axis=1)  
    df.set_index('datetime')
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    
    results = ARIMAResults.load('ARIMAmodelMonth.pkl')

    pred = results.get_prediction(start=offset, dynamic=False)
    predicted = pred.predicted_mean
    actual = df[offset:].TotalActivePower
    result = pd.DataFrame()
    steps=12
    forecast = results.forecast(steps)
    print(forecast)
    result=pd.DataFrame()
    data = pd.date_range('2019-11-27', periods = steps, freq ='M')
    print(data)
    result['predicted']=forecast
    
    print(len(result.predicted))
    result['datetime'] = pd.to_datetime(data)
    source = ColumnDataSource(result)
    p = figure(x_axis_type="datetime", plot_height=250, plot_width=350,title='Estimate Usage' )
    line=p.line(x='datetime', y='predicted', legend='Predicted', line_width=2, source=source)
    p.xaxis.axis_label = "Month"
    p.yaxis.axis_label = "Energy Consumption (kWH)"
    hover_tool = HoverTool(tooltips=[
            ('Value', '@predicted{1.11}'),
            ('Date', '@datetime{%F}'),
        ],formatters={'@datetime': 'datetime'}, renderers=[line])
    p.tools.append(hover_tool)
    script, div = components(p)
    #result.to_csv('SwimmingPoolMonthsPrediction.csv')
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
    data = pd.date_range('2019-11-27', periods = steps, freq ='D')
    result['predicted']=forecast
    
    print(len(result.predicted))
    result['datetime'] = pd.to_datetime(data)
    source = ColumnDataSource(result)
    p = figure(x_axis_type="datetime", plot_height=250, plot_width=350,title='Estimate Usage' )
    line=p.line(x='datetime', y='predicted', legend='Predicted', line_width=2, source=source)
    p.xaxis.axis_label = "Day"
    p.yaxis.axis_label = "Energy Consumption (kWH)"
    hover_tool = HoverTool(tooltips=[
            ('Value', '@predicted{1.11}'),
            ('Date', '@datetime{%F}'),
        ],formatters={'@datetime': 'datetime'}, renderers=[line])
    p.tools.append(hover_tool)
    script, div = components(p)
    # result.to_csv('SwimmingPoolDaysPrediction.csv')
    return script, div



def cost_graph(future,df,present_count,past_count):

    present=pd.DataFrame()
    past=pd.DataFrame()
    present =  df.iloc[-present_count:]
    past =   df.iloc[-(past_count+present_count):-present_count]
    pre=present['TotalActivePower'].sum()*0.002683
    pas=past['TotalActivePower'].sum()*0.002683
    fut=future['predicted'].sum()*0.002683
    
    print("----------",fut)

    dates =['Past','Present','Predicted']
    Value =[pas,pre,fut]  
    source = ColumnDataSource(data=dict(dates=dates, Value=Value))

    p = figure(x_range=dates, plot_height=250, plot_width=350, title="Cost Graph")

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
   
    result = pd.DataFrame()
    result['Pump'] = df.Device_1.iloc[-offset:]
    result['Heater'] = df.Device_2.iloc[-offset:]
    result['Lights'] = df.Device_3.iloc[-offset:]
    Device = ['Heater','Lights','Pump']
    Value = [(result['Pump'].sum())/1000,(result['Heater'].sum())/1000,(result['Lights'].sum())/1000]
    source = ColumnDataSource(data=dict(Device=Device, Value=Value))
    p = figure(x_range=Device, plot_height=250, plot_width=350, title="Active Appliances")
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




def forecast(test_X,future,scaler,model):
    predictions = []

    for ix in range(future,0,-1):
        row = test_X[-ix,:]
        row = row.reshape(1,1,7)
        out = model.predict(row)
        row = row.reshape((1,7))
        inv_yhat = np.concatenate((out, row[:, -6:]), axis=1)
        inv_yhat = scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:,0]

        predictions.append(inv_yhat[0])
        
    return predictions




def deviceplots(d,v):

    Device=d
    Value=v
    source = ColumnDataSource(data=dict(Device=Device, Value=Value))

    p = figure(x_range=Device, plot_height=350, title="Device Wise Energy Consumption")

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
    df = pd.read_csv(filepath + SwimmingPoolDays, delimiter=',') 
    future= pd.DataFrame()
    future= pd.read_csv(filepath + SwimmingPoolDaysPrediction, delimiter=',') 
    offset = 356 * 3
    script, div=get_predictions_day(df,offset)
    script1,div1=active_graph(df,26)
    script2, div2=cost_graph(future,df,26,30)
    return render_template('day.html',script=script, div=div,script1=script1,div1=div1,script2=script2, div2=div2)

@app.route("/month")
def month():
    df = pd.read_csv(filepath + SwimmingPoolMonths, delimiter=',')
    future= pd.DataFrame()
    future= pd.read_csv(filepath + SwimmingPoolMonthsPrediction , delimiter=',') 
    offset = 12 * 3
    script, div=get_predictions_month(df,  offset)
    script1,div1=active_graph(df,11)
    script2, div2=cost_graph(future,df,11,12)
    return render_template('month.html',script=script, div=div,script1=script1,div1=div1,script2=script2, div2=div2)


@app.route("/hour",methods=['GET'])
def hour():
    df = pd.read_csv(filepath + SwimmingPoolHours , delimiter=',') 
    future= pd.DataFrame()
    future= pd.read_csv(filepath + SwimmingPoolHourPrediction, delimiter=',') 
    df.set_index('datetime')
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    script, div=get_predictions_hour()
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




@app.route('/yeardeviceplot/<int:post_id>')
def yeardeviceplot(post_id):
    
    b=post_id
    print(b)
    if(b==1):
        ddf= pd.read_csv(filepath + SwimmingPoolMonths, delimiter=',')
        df= ddf.loc[39:, ['datetime', 'Device_1']] 
        result=pd.DataFrame()
        result = pd.DataFrame()
        result['Pump'] = df['Device_1'] 
        Device =df['datetime']
        Value = result['Pump']
        script, div=deviceplots(Device,Value)
        return render_template('yeardeviceplot.html' ,script=script, div=div)
    if(b==2):
        ddf= pd.read_csv(filepath + SwimmingPoolMonths, delimiter=',')
        df= ddf.loc[39:, ['datetime', 'Device_2']] 
        result=pd.DataFrame()
        result = pd.DataFrame()
        result['Lighting'] = df['Device_2'] 
        Device =df['datetime']
        Value = result['Lighting']
        script, div=deviceplots(Device,Value)
        return render_template('yeardeviceplot.html' ,script=script, div=div)
    if(b==3):
        ddf= pd.read_csv(filepath + SwimmingPoolMonths, delimiter=',')
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
        ddf= pd.read_csv(filepath + SwimmingPoolDays, delimiter=',')
        df= ddf.loc[1418:, ['datetime', 'Device_1']] 
        result=pd.DataFrame()
        result = pd.DataFrame()
        result['Pump'] = df['Device_1'] 
        Device =df['datetime']
        Value = result['Pump']
        script, div=deviceplots(Device,Value)
        return render_template('monthdeviceplot.html' ,script=script, div=div)
    if(b==2):
        ddf= pd.read_csv(filepath + SwimmingPoolDays, delimiter=',')
        df= ddf.loc[1418:, ['datetime', 'Device_2']] 
        result=pd.DataFrame()
        result = pd.DataFrame()
        result['Lighting'] = df['Device_2'] 
        Device =df['datetime']
        Value = result['Lighting']
        script, div=deviceplots(Device,Value)
        return render_template('monthdeviceplot.html' ,script=script, div=div)
    if(b==3):
        ddf= pd.read_csv(filepath + SwimmingPoolDays, delimiter=',')
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
        ddf= pd.read_csv(filepath + SwimmingPoolHours, delimiter=',')
        df= ddf.loc[34569:, ['datetime', 'Device_1']] 
        result=pd.DataFrame()
        result = pd.DataFrame()
        result['Pump'] = df['Device_1'] 
        Device =df['datetime']
        Value = result['Pump']
        script, div=deviceplots(Device,Value)
        return render_template('todaydeviceplot.html' ,script=script, div=div)
    if(b==2):
        ddf= pd.read_csv(filepath + SwimmingPoolHours, delimiter=',')
        df= ddf.loc[34569:, ['datetime', 'Device_2']] 
        result=pd.DataFrame()
        result = pd.DataFrame()
        result['Lighting'] = df['Device_2'] 
        Device =df['datetime']
        Value = result['Lighting']
        script, div=deviceplots(Device,Value)
        return render_template('todaydeviceplot.html' ,script=script, div=div)
    if(b==3):
        ddf= pd.read_csv(filepath + SwimmingPoolHours, delimiter=',')
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
