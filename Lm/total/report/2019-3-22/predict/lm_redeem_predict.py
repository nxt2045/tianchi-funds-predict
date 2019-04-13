import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



def get_score(test_y, test_predict):
    test_error = np.divide(np.abs(test_y - test_predict), test_y)
    test_score = 0
    for item in test_error:
        print(item)
        if item <= 0.3:
            test_score += 10 - (100.0 / 3) * item
    return test_score


# --------------------------------------------------------------------

save_path = "./redeem/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# --------------------------------------------------------------------

df = pd.read_csv(r"../../data/processed/lm_1403to1409.csv", sep=',', engine='python',
                 encoding='utf-8',
                 parse_dates=['report_date'])
print()
df.set_index(['report_date'], inplace=True)
print('data.head()', df.head())
print('data.tail()', df.tail())
df = df.fillna(0)
data = df[:'2014-08']
data_test = df['2014-09']

# --------------------------------------------------------------------


# --------------------------------------------------------------------
x_cols = ['workday','weekend','1st','early','middle','late','mon_or_fri_work','day_after_holiday','holiday','holiday_2','day_after_holiday_2','day_before_holiday_3','holiday_3','day_after_holiday_3','holiday_1st_or_3rd','day_before_or_after_holiday_normal_work','day_before_holiday_2','day_before_holiday_7','sunday_work']
# x_cols = ['workday','weekend','1st','early','middle','late','mon_or_fri_work','day_after_holiday','holiday','holiday_2','day_after_holiday_2','day_before_holiday_3','holiday_3','day_after_holiday_3','holiday_1st_or_3rd','day_before_or_after_holiday_normal_work','day_before_holiday_2','day_before_holiday_7','sunday_work','workday2','workday3']
y_cols = ['total_redeem_amt']
# --------------------------------------------------------------------
X_train = data[x_cols].values
y_train = data[y_cols].values
X_test = data_test[x_cols].values
# --------------------------------------------------------------------
print('data_train[x_cols].head()', X_train)
model = LinearRegression(fit_intercept=True, normalize=False)
# model = LinearRegression(fit_intercept=False)
model.fit(X_train, y_train)
train_score = model.score(X_train, y_train)  # 1 最好；-1 最差；0 无关
print('train_score', train_score)
coef = pd.DataFrame((model.coef_.flatten()).tolist(), x_cols)
coef = pd.DataFrame(coef)
coef.to_csv(save_path + 'coef.csv')
print('model.coef', coef)
# --------------------------------------------------------------------
y_pred = model.predict(X_test)
print(y_pred)
result = pd.DataFrame(y_pred, columns=['redeem'], index=data_test.index)
result.to_csv(save_path + 'redeem.csv')

y_pred_all = model.predict(df[x_cols].values)

# 画预测图
y_pred_all = pd.DataFrame(y_pred_all, columns=['redeem'], index=df.index)
fig, ax = plt.subplots(figsize=(24, 4))
y_pred_all.plot(ax=ax, label='predict')
data[y_cols].plot(ax=ax, label='real')
plt.legend()
plt.title('redeem predict by lm')
plt.draw()
plt.savefig(save_path + 'redeem.png')

plt.show()

