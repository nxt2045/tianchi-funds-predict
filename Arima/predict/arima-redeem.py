import os
import time
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

pd.options.mode.chained_assignment = None

# 定义变量
redeem_p = 4
redeem_d = 1
redeem_q = 5

train_type = "1403to1408"

# 开始计时
start = time.clock()
test_date = time.time()
redeem_test_path = "./output/redeem(" + str(redeem_p) + "," + str(redeem_d) + "," + str(redeem_q) + ")"
if not os.path.exists(redeem_test_path):
    os.makedirs(redeem_test_path)


# ------------------------------------------
# 读取数据文件
# 用户申购赎回数数据

redeem_data1403to1408 = pd.read_csv(r"../data/processed/redeem_data1403to1408.csv", sep=',', engine='python',
                                    encoding='utf-8',
                                    parse_dates=['report_date'])

redeem_data1403to1408.set_index(['report_date'],inplace=True)
redeem_data1403to1408_diff = redeem_data1403to1408  # redeem_data1403to1408.diff(1)
# ------------------------------------------

# 计算ACF和PACF
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(redeem_data1403to1408_diff, lags=40, title='total_redeem_amt acf', ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(redeem_data1403to1408_diff, lags=40, title='total_redeem_amt pacf', ax=ax2)
plt.savefig(redeem_test_path + "/total_redeem_amt_acf+pacf" + ".png")

redeem_data1403to1408 = redeem_data1403to1408.astype('float64')

redeem_data1403to1408_diff = redeem_data1403to1408_diff.astype('float64')
redeem_data1403to1408_diff.dropna(axis=0, how='any', inplace=True)

# ------------------------------------------
# 建立模型

warnings.filterwarnings("ignore")
redeem_arima_model = sm.tsa.ARIMA(redeem_data1403to1408.astype('float64'), [redeem_p, redeem_d, redeem_q]).fit()

# ------------------------------------------

# 计数残差
redeem_resid = redeem_arima_model.resid

# 残差检验
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(redeem_resid.values.squeeze(), lags=40, title='redeem_resid acf', ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(redeem_resid, lags=40, title='redeem_resid pacf', ax=ax2)
plt.savefig(
    redeem_test_path + "/redeem_resid pacf" + ".png")


# D-W检验
print(sm.stats.durbin_watson(redeem_resid.values))

# 观察是否符合正态分布
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
fig = sm.qqplot(redeem_resid, line='q', ax=ax, fit=True)
plt.title('redeem_resid Q-Q')
plt.savefig(
    redeem_test_path + "/redeem_resid Q-Q" + ".png")


# ------------------------------------------

# 预测9月
redeem_predict_diff = redeem_arima_model.predict('2014-09-01', '2014-09-30', dynamic=False)
redeem_predict = redeem_predict_diff.cumsum().add(redeem_data1403to1408.ix['2014-08-31', 'total_redeem_amt'])

# 结束计时
end = time.clock()

# 输出预测值to csv
redeem_predict.to_csv(redeem_test_path + "/predict.csv")

# 画预测图
fig, ax = plt.subplots(figsize=(12, 8))
redeem_predict.plot(ax=ax, label='predict')
plt.legend()
plt.title('Sep\'s total_redeem_amt predict')
plt.draw()
plt.savefig(
    redeem_test_path + "/redeem-predict.png")


# ------------------------------------------
# 显示图像
# plt.show()
