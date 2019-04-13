import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def get_score(test_y, test_predict):
    test_error = np.divide(np.abs(test_y - test_predict), test_y)
    print('test_error',test_error[:,0])
    test_score = 0
    for item in test_error:
        if item <= 0.3:
            test_score += 10 - (100.0 / 3) * item
    return test_score*30/test_y.size




ts = pd.read_csv('../data/average/user_amt.csv', sep=',', engine='python',
                   encoding='utf-8',
                   parse_dates=['report_date'], index_col=['report_date'])

y_train = ts['2014-06':'2014-08'].values


X_train = np.linspace(1,y_train.shape[0],y_train.shape[0]).reshape(-1, 1)
print(X_train)
X_test = np.linspace(y_train.shape[0]+1,30+y_train.shape[0],30).reshape(-1, 1)
print(X_test)

model = LinearRegression(fit_intercept=True, normalize=False)
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)  # 1 最好；-1 最差；0 无关
print('train_score', train_score)
coef = pd.DataFrame((model.coef_.flatten()).tolist())
print('model.coef', coef)
# --------------------------------------------------------------------
y_pred = model.predict(X_train)
print('train_error', y_train - y_pred)
y_pred = model.predict(X_test)
print(y_pred)

y_pred = pd.DataFrame(y_pred)
y_pred.to_csv("user_amt_pred.csv")


