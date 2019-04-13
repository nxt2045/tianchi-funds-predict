import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



def get_score(test_y, test_predict):
    test_error = np.divide(np.abs(test_y - test_predict), np.abs(test_y))
    test_score = 0
    for item in test_error:
        if item <= 0.3:
            test_score += 10 - (100.0 / 3) * item
    return test_score


# --------------------------------------------------------------------

save_path = "./"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# --------------------------------------------------------------------

df = pd.read_csv(r"./trend_log_redeem.csv", sep=',', engine='python',
                 encoding='utf-8',
                 parse_dates=['report_date'])
df.set_index(['report_date'], inplace=True)
print('data.head()', df.head())
print('data.tail()', df.tail())
df = df.fillna(0)
data = df['y']


# --------------------------------------------------------------------
df = pd.read_csv(r"./trend_log_pred_redeem.csv", sep=',', engine='python',
                 encoding='utf-8',
                 parse_dates=['report_date'])
df.set_index(['report_date'], inplace=True)
print('data.head()', df.head())
print('data.tail()', df.tail())
df = df.fillna(0)
data_pred = df['y']

# --------------------------------------------------------------------

# 画预测图


fig, ax = plt.subplots(figsize=(24, 4))
data_pred.plot(ax=ax, label='predict')
data.plot(ax=ax, label='real')
plt.legend()
plt.title('trend_log predict by lstm')
plt.draw()
plt.savefig('./trend_log_pred_redeem.png')

plt.show()
