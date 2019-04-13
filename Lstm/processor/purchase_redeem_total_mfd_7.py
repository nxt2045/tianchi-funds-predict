import pandas as pd

# 定义变量
pd.options.mode.chained_assignment = None

# ------------------------------------------
# 读取数据文件

# 用户申购赎回数数据
purchase_redeem_total = pd.read_csv(r"../data/processed/prt.csv", sep=',', engine='python',
                                    encoding='utf-8',
                                    parse_dates=['report_date'])

purchase_redeem_total.set_index(['report_date'], inplace=True)
# 收益率：日期 万分收益 七日年化收益
mfd_day_share_interest = pd.read_csv(r"../data/mfd_day_share_interest.csv", sep=',', engine='python', encoding='utf-8',
                                     parse_dates=['mfd_date'])
# ------------------------------------------


# 收益
mfd_7daily_yield = mfd_day_share_interest.get(['mfd_date', 'mfd_7daily_yield'])
mfd_7daily_yield.columns = ['report_date', 'mfd_7daily_yield']
mfd_7daily_yield.set_index(['report_date'], inplace=True)

purchase_redeem_total_mfd_7 = pd.merge(purchase_redeem_total, mfd_7daily_yield, on='report_date')

purchase_redeem_total_mfd_7 = purchase_redeem_total_mfd_7.fillna(method='pad')

purchase_redeem_total_mfd_7.to_csv("../data/processed/prt_mfd_7.csv", float_format='%.3f')
