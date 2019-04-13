import os
import time
import pandas as pd
import numpy  as np
import math
# from scipy import signal
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
import statsmodels.api as sm
import warnings

# 定义变量
pd.options.mode.chained_assignment = None
train_type = "1403to1408"


# ------------------------------------------
# 读取数据文件
# 用户申购赎回数数据
purchase_data1403to1408 = pd.read_csv(r"../../data/processed/purchase_data1403to1408.csv", sep=',', engine='python',
                                      encoding='utf-8',
                                      parse_dates=['report_date'])
redeem_data1403to1408 = pd.read_csv(r"../../data/processed/redeem_data1403to1408.csv", sep=',', engine='python',
                                    encoding='utf-8',
                                    parse_dates=['report_date'])
purchase_data1403to1408.set_index(['report_date'],inplace=True)

redeem_data1403to1408.set_index(['report_date'],inplace=True)

purchase_data1403to1408_diff = purchase_data1403to1408.diff(1)
redeem_data1403to1408_diff = redeem_data1403to1408.diff(1)
# ------------------------------------------

redeem_data1403to1408 = redeem_data1403to1408.astype('float64')
purchase_data1403to1408 = purchase_data1403to1408.astype('float64')
# ------------------------------------------


print('begin arma')
text_file = open("./arma.txt", "w")
warnings.filterwarnings("ignore")
for p in range(0,10):
    for q in range(0,10):
        if(p!=q):
            try:
                arma_model = sm.tsa.ARMA(redeem_data1403to1408,(p,q)).fit()
                text_file.write('redeem ARMA(%d,%d)AIC:%f BIC:%f HQOC:%f'%(p,q,arma_model.aic,arma_model.bic,arma_model.hqic))
                text_file.write('\n')
            except:
                continue

for p in range(0,10):
    for q in range(0,10):
        if(p!=q):
            try:
                arma_model = sm.tsa.ARMA(purchase_data1403to1408,(p,q)).fit()
                text_file.write('purchase ARMA(%d,%d)AIC:%f BIC:%f HQOC:%f'%(p,q,arma_model.aic,arma_model.bic,arma_model.hqic))
                text_file.write('\n')
            except:
                continue
text_file.close()
# ------------------------------------------

print('begin arima')
text_file = open("./arima.txt", "w")
warnings.filterwarnings("ignore")
for p in range(0, 10):
    for q in range(0, 10):
        if (p != q):
            try:
                arima_model = sm.tsa.ARIMA(redeem_data1403to1408, [p, 1, q]).fit()
                text_file.write(
                    'redeem ARIMA(%d,1,%d)AIC:%f BIC:%f HQIC:%f' % (
                    p, q, arima_model.aic, arima_model.bic, arima_model.hqic))
                text_file.write('\n')
            except:
                continue

for p in range(0, 10):
    for q in range(0, 10):
        if (p != q):
            try:
                arima_model = sm.tsa.ARIMA(purchase_data1403to1408, [p, 1, q]).fit()
                text_file.write(
                    'purchase ARIMA(%d,1,%d)AIC:%f BIC:%f HQIC:%f' % (
                        p, q, arima_model.aic, arima_model.bic, arima_model.hqic))
                text_file.write('\n')
            except:
                continue

text_file.close()
