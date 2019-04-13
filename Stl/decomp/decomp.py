#-*- coding:utf-8 -*-
'''
周期性时间序列预测
'''
import os
import numpy as np
from test_stationarity import *
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from datetime import timedelta


from statsmodels.tsa.stattools import adfuller


def test_stationarity(ts):
    # Determing rolling statistics
    rolmean = ts.rolling(window=7).mean()
    rolstd = ts.rolling(window=7).std()

    # Plot rolling statistics:
    orig = plt.plot(ts, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(ts, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

ts = pd.read_csv('../data/average/ra.csv', sep=',', engine='python',
                   encoding='utf-8',
                   parse_dates=['report_date'], index_col=['report_date'])
ts_pred = ts['2014-08':]
ts = ts['2014-04':'2014-08']
ts.plot()

decomposition = seasonal_decompose(ts, freq=7)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
decomposition.plot()
plt.savefig('decomposed.png')

plt.show()

trend.to_csv("./trend/trend.csv")
residual.to_csv("./residual/residual.csv")
seasonal.to_csv("./seasonal/seasonal.csv")



seasonal_part = []
for i in ts_pred.index:
    seasonal_part.append(((seasonal[seasonal.index == ((i - timedelta(days=35)).strftime("%Y-%m-%d"))]).values)[0])

print(seasonal_part)
seasonal_part = pd.DataFrame(seasonal_part)

seasonal_part.to_csv("./seasonal/seasonal_pred.csv")

'''
ts_log = np.log(ts)
ts_log.plot()
decomposition = seasonal_decompose(ts_log, freq=7)
trend_log = decomposition.trend
seasonal_log = decomposition.seasonal
residual_log = decomposition.resid
decomposition.plot()
plt.savefig('decomposed_log.png')
plt.show()

trend_log.to_csv("./trend_log/trend_log.csv")
residual_log.to_csv("./residual_log/residual_log.csv")
seasonal_log.to_csv("./seasonal_log/seasonal_log.csv")



seasonal_part = []
for i in ts_pred.index:
    seasonal_part.append(((seasonal_log[seasonal_log.index == ((i - timedelta(days=35)).strftime("%Y-%m-%d"))]).values)[0])

print(seasonal_part)
seasonal_part = pd.DataFrame(seasonal_part)

seasonal_part.to_csv("./seasonal_log/seasonal_log_pred.csv")

'''
