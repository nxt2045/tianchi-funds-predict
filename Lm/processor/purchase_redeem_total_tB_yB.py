import pandas as pd

# 定义变量
pd.options.mode.chained_assignment = None

# ------------------------------------------

# 读取数据文件

# 用户申购赎回数数据
user_balance_table = pd.read_csv(r"../data/user_balance_table.csv", sep=',', engine='python', encoding='utf-8',
                                 parse_dates=['report_date'])


# ------------------------------------------

# user_balance_table 数据缺失值处理 填充0(众数)
user_balance_table = user_balance_table.fillna(0)

# 购买赎回总量计算
user_balance = user_balance_table.groupby(['report_date'])
purchase_redeem_total = user_balance['total_purchase_amt', 'total_redeem_amt', 'tBalance', 'yBalance'].sum()

purchase_redeem_total = purchase_redeem_total.fillna(method='pad')

purchase_redeem_total.to_csv("../data/processed/prt_tB_yB.csv")