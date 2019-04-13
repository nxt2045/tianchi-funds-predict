import pandas as pd

# from scipy import signal

# 定义变量
pd.options.mode.chained_assignment = None


# ------------------------------------------

# 读取数据文件
# 用户数据:id 性别 城市 星座
user_profile_table = pd.read_csv(r"./user_profile_table.csv", sep=',', engine='python', encoding='utf-8')
# 用户申购赎回数数据
user_balance_table = pd.read_csv(r"./user_balance_table.csv", sep=',', engine='python', encoding='utf-8',
                                 parse_dates=['report_date'])
# 收益率：日期 万分收益 七日年化收益
mfd_day_share_interest = pd.read_csv(r"./mfd_day_share_interest.csv", sep=',', engine='python', encoding='utf-8',
                                     parse_dates=['mfd_date'])
# 银行拆放利率:日期 隔夜利率（%）1周利率（%）2周利率（%）1个月利率（%）3个月利率（%）6个月利率（%）9个月利率（%）
mfd_bank_shibor = pd.read_csv(r"./mfd_bank_shibor.csv", sep=',', engine='python', encoding='utf-8',
                              parse_dates=['mfd_date'])

# ------------------------------------------

# user_balance_table 数据缺失值处理 填充0(众数)
user_balance_table = user_balance_table.fillna(0)

# 购买赎回总量计算
user_balance = user_balance_table.groupby(['report_date'])
purchase_redeem_total = user_balance['total_purchase_amt', 'total_redeem_amt'].sum()

# 计算星期列
date = pd.DataFrame(purchase_redeem_total.index)
# date['day_of_week']=date['report_date'].dt.dayofweek
date['day_of_week'] = date['report_date'].dt.weekday_name
tt_date = date.groupby(['report_date'])
tt_date = tt_date['day_of_week'].sum()
# 转化为0 1 哑变量 并进行DF拼接
purchase_redeem_total_with_week_day = pd.concat([pd.get_dummies(tt_date, columns='day_of_week'), purchase_redeem_total],
                                                axis=1)

# 收益
time_mfd_day_share_interest = mfd_day_share_interest.groupby(['mfd_date'])
share_interest = time_mfd_day_share_interest['mfd_daily_yield', 'mfd_7daily_yield'].sum()
# 拆放利率
t_mfd_bank_shibor = (mfd_bank_shibor.groupby(['mfd_date']))
time_mfd_bank_shibor = t_mfd_bank_shibor['Interest_O_N'].sum()
time_mfd_bank_shibor = time_mfd_bank_shibor.fillna(method='pad')

prtwd = pd.concat([purchase_redeem_total_with_week_day, share_interest, time_mfd_bank_shibor, tt_date], axis=1)
prtwd = prtwd.fillna(method='pad')

prtwd = purchase_redeem_total
prtwd = prtwd.fillna(method='pad')

# 选出2014年 3月份到7月的数据 8月的数据
prtwd1403to1408 = prtwd['2014-03':'2014-08']

# 添加节假日
prtwd1403to1408['holiday_festival'] = 0
prtwd1403to1408.ix['2014-04-05':'2014-04-07', 'holiday_festival'] = 1
prtwd1403to1408.ix['2014-05-01':'2014-05-03', 'holiday_festival'] = 1
prtwd1403to1408.ix['2014-05-31':'2014-06-02', 'holiday_festival'] = 1
prtwd1403to1408.ix['2014-08-31', 'holiday_festival'] = 1

# 添加月初1周 和月末1周
prtwd1403to1408['early_mouth'] = 0
prtwd1403to1408.ix['2014-03-01':'2014-03-07', 'early_mouth'] = 1
prtwd1403to1408.ix['2014-04-01':'2014-04-07', 'early_mouth'] = 1
prtwd1403to1408.ix['2014-05-01':'2014-05-07', 'early_mouth'] = 1
prtwd1403to1408.ix['2014-06-01':'2014-06-07', 'early_mouth'] = 1
prtwd1403to1408.ix['2014-07-01':'2014-07-07', 'early_mouth'] = 1
prtwd1403to1408.ix['2014-08-01':'2014-08-07', 'early_mouth'] = 1

prtwd1403to1408['late_mouth'] = 0
prtwd1403to1408.ix['2014-03-25':'2014-03-31', 'late_mouth'] = 1
prtwd1403to1408.ix['2014-04-24':'2014-04-30', 'late_mouth'] = 1
prtwd1403to1408.ix['2014-05-25':'2014-05-31', 'late_mouth'] = 1
prtwd1403to1408.ix['2014-06-24':'2014-06-30', 'late_mouth'] = 1
prtwd1403to1408.ix['2014-07-25':'2014-07-31', 'late_mouth'] = 1
prtwd1403to1408.ix['2014-08-25':'2014-08-31', 'late_mouth'] = 1


purchase_data1403to1408 = pd.DataFrame(prtwd1403to1408['total_purchase_amt'])
redeem_data1403to1408 = pd.DataFrame(prtwd1403to1408['total_redeem_amt'])


redeem_data1403to1408.to_csv("./processed/redeem_data1403to1408.csv")
purchase_data1403to1408.to_csv("./processed/purchase_data1403to1408.csv")