# Arima
Arima model for yu-e-bao prediction
## Notes
- 加法模型：时间序列=季节性+趋势+随机 
译注：如果时间序列的波峰波谷的差距一直差不多，就用加法模型。
- 乘法模型：时间序列=趋势*季节性*随机 
译注：如果时间序列的波峰波谷的差距随着时间推移而一直加大，就用乘法模型。
- 在处理季节性影响时，我们利用季节性 ARIMA，表示为ARIMA(p,d,q)(P,D,Q)s 。 这里， (p, d, q)是上述非季节性参数，而(P, D, Q)遵循相同的定义，但适用于时间序列的季节分量。 术语s是时间序列的周期（季度为4 ，年度为12 ，等等）。

## Directory
- /test/redeem/purchase_score.txt 放的是分数    
每个分数 对应一个test_date(运行代码的时间) import os os.time.time()的返回值
- /test/best 放的是最佳的结果