
import os
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

pd.options.mode.chained_assignment = None

# 定义变量

redeem_p = 4
redeem_d = 1
redeem_q = 5
train_type = "1403to1407"

# 开始计时
start = time.clock()
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
redeem_data1403to1407 = redeem_data1403to1408['2014-03':'2014-07']
prtwd1408 = redeem_data1403to1408['2014-08']
redeem_data1403to1407_diff = redeem_data1403to1407.diff(1)

# ------------------------------------------

# 计算ACF和PACF
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(redeem_data1403to1407_diff, title='total_redeem_amt acf', lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(redeem_data1403to1407_diff, title='total_redeem_amt pacf', lags=40, ax=ax2)
plt.savefig(redeem_test_path + "/total_redeem_amt_acf+pacf" + ".png")


redeem_data1403to1407 = redeem_data1403to1407.astype('float64')

redeem_data1403to1407_diff = redeem_data1403to1407_diff.astype('float64')
redeem_data1403to1407_diff.dropna(axis=0, how='any', inplace=True)

# ------------------------------------------

# 建立模型

warnings.filterwarnings("ignore")
redeem_arima_model = sm.tsa.ARIMA(redeem_data1403to1407.astype('float64'), [redeem_p, redeem_d, redeem_q]).fit()

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

# 预测8月
redeem_predict_diff = redeem_arima_model.predict('2014-08-01', '2014-08-31', dynamic=False)
redeem_predict = redeem_predict_diff.cumsum().add(redeem_data1403to1407.ix['2014-07-30', 'total_redeem_amt'])

# 结束计时
end = time.clock()

# 输出预测值to csv
redeem_predict.to_csv(redeem_test_path + "/predict.csv")
# 画预测图
fig, ax = plt.subplots(figsize=(12, 8))
redeem_predict.plot(ax=ax, label='predict')
prtwd1408['total_redeem_amt'].plot(ax=ax, label='test')
plt.legend()
plt.title('8\'s redeem predict by arima')
plt.draw()
plt.savefig(
    redeem_test_path + "/redeem-predict.png")

# ------------------------------------------

# 计算误差
redeem_error = np.divide(np.abs(np.array(prtwd1408['total_redeem_amt']) - np.array(redeem_predict).T),
                           np.array(prtwd1408['total_redeem_amt']))
print('redeem_error')
print(redeem_error)
redeem_score = 0
for item in redeem_error:
    print(item)
    if item <= 0.3:
        redeem_score += 10 - (100.0 / 3) * item
print('redeem_score:\n', redeem_score)
text_file = open("./output/redeem_score.txt", "a+")
text_file.write('redeem(arima) p:%d d:%d q:%d time:%f score:%f' % (
    redeem_p, redeem_d, redeem_q, end - start, redeem_score))
text_file.write('\n')
text_file.close()

# 显示图像
# plt.show()
