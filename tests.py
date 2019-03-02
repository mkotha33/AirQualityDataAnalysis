# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 10:30:19 2018

@author: Sowmya Vasuki
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
# Import statsmodel
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import adfuller
import math

AIRQUALITY_FILE = os.path.join("airquality.csv")
df = pd.read_csv(AIRQUALITY_FILE,header=0,parse_dates={'Datetime':['Date','Time']})

df = df.fillna(df.mean())
df = df.where(df!=-200)
df = df.fillna(df.mean())


datetime = pd.Series(df.Datetime)
df.index = datetime

df.columns = ["V"+str(i) for i in range(1, len(df.columns)+1)]

data = df.V13
data1=df.V14

#normal plotting
fig, axes = plt.subplots(2,1, figsize=(10,15))

axes[0].plot(df.V13)
axes[0].set_title ('Relative Humidity')
axes[1].plot(df.V14)
axes[1].set_title ('Absolute Humidity')


def adfu(ts):
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(ts)
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

adfu(data)
adfu(data1)


from pandas.plotting import lag_plot
lag_plot(df.V13)
lag_plot(df.V14)

from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df.V13)
autocorrelation_plot(df.V14)

from statsmodels.graphics.tsaplots import plot_acf
plot_acf(df.V13)

df['V14log'] = np.log(df.V14)
df["V14LogShi1"] = df.V14log.shift()
df = df.fillna(0)
df["Diff"] = df.V14log - df.V14LogShi1
ts_diff = df.Diff
ts_diff.dropna(inplace = True)




from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(ts_diff, nlags=20)
lag_pacf = pacf(ts_diff, nlags=20, method='ols')
#Plot ACF: 
#plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')
plt.show()
#Plot PACF:
#plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.show()
#p=2 q=1

#ARIMA
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(ts_diff, order=(4,0,3))  
results_ARIMA = model.fit(disp=-1)  
plt.plot(ts_diff)
plt.plot(results_ARIMA.fittedvalues, color='blue')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_diff)**2))