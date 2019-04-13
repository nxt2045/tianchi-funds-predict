import os
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

pd.options.mode.chained_assignment = None

# 定义变量


purchase_p = 6
purchase_q = 4
train_type = "1403to1407"

# 开始计时
start = time.clock()
purchase_test_path = "./output/purchase(" + str(purchase_p) + "," + str(purchase_q) + ")"

if not os.path.exists(purchase_test_path):
    os.makedirs(purchase_test_path)

# ------------------------------------------
# 读取数据文件
# 用户申购赎回数数据
purchase_data1403to1408 = pd.read_csv(r"../data/processed/purchase_data1403to1408.csv", sep=',', engine='python',
                                      encoding='utf-8',
                                      parse_dates=['report_date'])

purchase_data1403to1408.set_index(['report_date'], inplace=True)
purchase_data1403to1407 = purchase_data1403to1408['2014-03':'2014-07']
prtwd1408 = purchase_data1403to1408['2014-08']

purchase_data1403to1407_diff = purchase_data1403to1407  # purchase_data1403to1407.diff(1)
# ------------------------------------------
# 计算ACF和PACF
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(purchase_data1403to1407_diff, title='total_purchase_amt acf', lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(purchase_data1403to1407_diff, title='total_purchase_amt pacf', lags=40, ax=ax2)
plt.savefig(purchase_test_path + "/total_purchase_amt_acf+pacf" + ".png")

purchase_data1403to1407 = purchase_data1403to1407.astype('float64')

# ------------------------------------------

# 建立模型
warnings.filterwarnings("ignore")
purchase_arma_model = sm.tsa.ARMA(purchase_data1403to1407, (purchase_p, purchase_q)).fit()

# ------------------------------------------

# 计数残差
purchase_resid = purchase_arma_model.resid

# 残差检验

fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(purchase_resid.values.squeeze(), lags=40, title='purchase_resid acf', ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(purchase_resid, lags=40, title='purchase_resid pacf', ax=ax2)
plt.savefig(
    purchase_test_path + "/purchase_resid pacf" + ".png")

# D-W检验
print(sm.stats.durbin_watson(purchase_resid.values))

# 观察是否符合正态分布

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
fig = sm.qqplot(purchase_resid, line='q', ax=ax, fit=True)
plt.title('purchase_resid Q-Q')
plt.savefig(
    purchase_test_path + "/purchase_resid Q-Q" + ".png")

# ------------------------------------------

# 预测8月
purchase_predict = purchase_arma_model.predict('2014-08-01', '2014-08-31', dynamic=False)

# 结束计时
end = time.clock()

# 输出预测值to csv
purchase_predict.to_csv(purchase_test_path + "/predict.csv")

# 画预测图
fig, ax = plt.subplots(figsize=(12, 8))
purchase_predict.plot(ax=ax, label='predict')
prtwd1408['total_purchase_amt'].plot(ax=ax, label='test')
plt.legend()
plt.title('8\'s purchase predict by arma')
plt.draw()
plt.savefig(
    purchase_test_path + "/purchase-predict.png")

# ------------------------------------------

# 计算误差

purchase_error = np.divide(np.abs(np.array(prtwd1408['total_purchase_amt']) - np.array(purchase_predict).T),
                           np.array(prtwd1408['total_purchase_amt']))
print('purchase_error\n')
print(purchase_error)
purchase_score = 0
for item in purchase_error:
    print(item)
    if item <= 0.3:
        purchase_score += 10 - (100.0/3)*item
print('purchase_score:\n', purchase_score)
text_file = open("./output/purchase_score.txt", "a+")
text_file.write('purchase(arma) p:%d q:%d time:%f score:%f' % (
    purchase_p, purchase_q, end - start, purchase_score))
text_file.write('\n')
text_file.close()
# 显示图像
plt.show()
