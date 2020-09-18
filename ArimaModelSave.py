import random
from bokeh.models import (HoverTool, FactorRange, Plot, LinearAxis, Grid, Range1d)
from bokeh.models.glyphs import VBar
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.models.sources import ColumnDataSource
from flask import Flask, render_template
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
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
from sklearn.externals import joblib

from statsmodels.tsa.arima_model import ARIMAResults
 
# monkey patch around bug in ARIMA class
def _getnewargs_(self):
	return ((self.endog),(self.k_lags, self.k_diff, self.k_ma))
ARIMA._getnewargs_ = _getnewargs_


def get_model(df,season):

    model = sm.tsa.statespace.SARIMAX(df.TotalActivePower,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1,season),
                                enforce_stationarity=True,
                                enforce_invertibility=False)
    # print("**********")
    results = model.fit()
    return results


df_days = pd.read_csv(r'C:\Users\Aditi\Desktop\projectFy\flaskapp\flaskapp\SwimmingPoolDays.csv', delimiter=',') 
# Save the model as a pickle in a file 
model_day =get_model(df_days,30)
# save model
model_day.save('ARIMAmodelDay.pkl')

df_months = pd.read_csv(r'C:\Users\Aditi\Desktop\projectFy\flaskapp\flaskapp\SwimmingPoolMonths.csv', delimiter=',') 
# Save the model as a pickle in a file 
model_day =get_model(df_months,4)
# save model
model_day.save('ARIMAmodelMonth.pkl')

