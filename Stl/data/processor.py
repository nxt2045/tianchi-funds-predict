import pandas as pd
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None
# 读取数据文件------------------------------------------------
# 用户数据:id 性别 城市 星座
user_profile_table = pd.read_csv(r"./provided/user_profile_table.csv", sep=',', engine='python', encoding='utf-8')
# 用户申购赎回数数据
user_balance_table = pd.read_csv(r"./provided/user_balance_table.csv", sep=',', engine='python', encoding='utf-8',
                                 parse_dates=['report_date'])
# 收益率：日期 万分收益 七日年化收益
mfd_day_share_interest = pd.read_csv(r"./provided/mfd_day_share_interest.csv", sep=',', engine='python',
                                     encoding='utf-8', parse_dates=['mfd_date'])
# 银行拆放利率:日期 隔夜利率（%）1周利率（%）2周利率（%）1个月利率（%）3个月利率（%）6个月利率（%）9个月利率（%）
mfd_bank_shibor = pd.read_csv(r"./provided/mfd_bank_shibor.csv", sep=',', engine='python', encoding='utf-8',
                              parse_dates=['mfd_date'])
# user_balance_table 数据缺失值处理 填充0(众数)
user_balance_table = user_balance_table.fillna(0)
'''
# 用户申购赎回数数据
pt = pd.read_csv(r"./total/pt.csv", sep=',', engine='python', encoding='utf-8',
                 parse_dates=['report_date'], index_col=['report_date'])
rt = pd.read_csv(r"./total/rt.csv", sep=',', engine='python', encoding='utf-8',
                 parse_dates=['report_date'], index_col=['report_date'])
prt_tB_yB = pd.read_csv(r"./total/prt_tB_yB.csv", sep=',', engine='python', encoding='utf-8',
                        parse_dates=['report_date'], index_col=['report_date'])
user_amount = pd.read_csv(r"./average/user_amt.csv", sep=',', engine='python', encoding='utf-8',
                          parse_dates=['report_date'], index_col=['report_date'])
pt_1404to1408 = pt['2014-04':'2014-08']
rt_1404to1408 = rt['2014-04':'2014-08']
user_amount_1404to1408 = user_amount['2014-04':'2014-08']
# ----------------------------------------------------------------
# 用户总和计算
user_amount.plot(label='user_amount', figsize=(20, 4))
plt.legend()
plt.title('user_amount')
plt.draw()
plt.savefig('./user_amount.png')
# ----------------------------------------------------------------
# 购买总和计算
pt.plot(label='pt', figsize=(20, 4))
plt.legend()
plt.title('purchase total')
plt.draw()
plt.savefig('./pt.png')
# ----------------------------------------------------------------
# 赎回总和计算
rt.plot(label='rt', figsize=(20, 4))
plt.legend()
plt.title('redeem total')
plt.draw()
plt.savefig('./rt.png')
# ----------------------------------------------------------------
# 人均计算
# ----------------------------------------------------------------
# 购买人均计算
pa = pd.DataFrame(pt['total_purchase_amt'].values / user_amount['user_amount'].values, index=user_amount.index)
pa_1404to1408 = pd.DataFrame(pt_1404to1408['total_purchase_amt'].values / user_amount_1404to1408['user_amount'].values,
                             index=user_amount_1404to1408.index)
pa.to_csv('./average/pa.csv', index=True)
pa_1404to1408.to_csv('./average/pa_1404to1408.csv', index=True)
pa.plot(label='ra', figsize=(20, 4))
plt.legend()
plt.title('purchase average')
plt.draw()
plt.savefig('../anaylse/pa.png')
# ----------------------------------------------------------------
# 赎回人均计算
ra = pd.DataFrame(rt['total_redeem_amt'].values / user_amount['user_amount'].values, index=user_amount.index)
ra_1404to1408 = pd.DataFrame(rt_1404to1408['total_redeem_amt'].values / user_amount_1404to1408['user_amount'].values,
                             index=user_amount_1404to1408.index)
ra.to_csv('./average/ra.csv', index=True)
ra_1404to1408.to_csv('./average/ra_1404to1408.csv', index=True)
ra.plot(label='ra', figsize=(20, 4))
plt.legend()
plt.title('redeem average')
plt.draw()
plt.savefig('../anaylse/ra.png')
# ----------------------------------------------------------------
# today余额总和计算
tBt = pd.DataFrame(prt_tB_yB['tBalance'].values, index=user_amount.index)
tBt.to_csv('./average/tBt.csv', index=True)
tBt.plot(label='tBt', figsize=(20, 4))
plt.legend()
plt.title('total today balance')
plt.draw()
plt.savefig('../anaylse/tBt.png')
# ----------------------------------------------------------------
# today余额人均计算
tBa = pd.DataFrame(prt_tB_yB['tBalance'].values / user_amount['user_amount'].values, index=user_amount.index)
tBa.to_csv('./average/tBa.csv', index=True)
tBa.plot(label='tBa', figsize=(20, 4))
plt.legend()
plt.title('average today balance')
plt.draw()
plt.savefig('../anaylse/tBa.png')

'''

# ----------------------------------------------------------------
# 穷富人群划分
# ----------------------------------------------------------------
bound = 5000000
# 单笔大于5W的赎回分析
more_than_bound = user_balance_table[(user_balance_table['total_redeem_amt'] >= bound)]
rich_balance_table = user_balance_table.loc[user_balance_table['user_id'].isin(more_than_bound['user_id'])]
# ----------------------------------------------------------------
user_rich_time_group = rich_balance_table.groupby(['report_date'])
prt_rich = user_rich_time_group['total_purchase_amt', 'total_redeem_amt', 'consume_amt', 'transfer_amt'].sum()
user_rich_amount = pd.DataFrame(user_rich_time_group.size())
user_rich_amount.columns = ['user_amount']
user_rich_amount.to_csv('./divided/rich_user_amt.csv', index=True)
user_rich_amount.plot(label='user_rich_amount', figsize=(20, 4))
plt.legend()
plt.title('user_rich_amount')
plt.draw()
plt.savefig('../anaylse/user_rich_amount.png')
# ----------------------------------------------------------------
pt_rich = pd.DataFrame(prt_rich['total_purchase_amt'].values, index=user_rich_amount.index)
pt_rich.to_csv('./divided/pt_rich.csv', index=True)
pt_rich.plot(label='pt', figsize=(20, 4))
plt.legend()
plt.title('purchase total rich')
plt.draw()
plt.savefig('../anaylse/pt_rich.png')
# ----------------------------------------------------------------
pa_rich = pd.DataFrame(prt_rich['total_purchase_amt'].values / user_rich_amount['user_amount'].values, index=user_rich_amount.index)
pa_rich.to_csv('./divided/pa_rich.csv', index=True)
pa_rich.plot(label='pa', figsize=(20, 4))
plt.legend()
plt.title('purchase average rich')
plt.draw()
plt.savefig('../anaylse/pa_rich.png')
# ----------------------------------------------------------------
rt_rich = pd.DataFrame(prt_rich['total_redeem_amt'].values, index=user_rich_amount.index)
rt_rich.to_csv('./divided/rt_rich.csv', index=True)
rt_rich.plot(label='rt', figsize=(20, 4))
plt.legend()
plt.title('redeem total rich')
plt.draw()
plt.savefig('../anaylse/rt_rich.png')
# ----------------------------------------------------------------
ra_rich = pd.DataFrame(prt_rich['total_redeem_amt'].values / user_rich_amount['user_amount'].values, index=user_rich_amount.index)
ra_rich.to_csv('./divided/ra_rich.csv', index=True)
ra_rich.plot(label='ra', figsize=(20, 4))
plt.legend()
plt.title('redeem average rich')
plt.draw()
plt.savefig('../anaylse/ra_rich.png')
# ----------------------------------------------------------------
ct_rich = pd.DataFrame(prt_rich['consume_amt'].values, index=user_rich_amount.index)
ct_rich.to_csv('./divided/ct_rich.csv', index=True)
ct_rich.plot(label='ct', figsize=(20, 4))
plt.legend()
plt.title('consume total rich')
plt.draw()
plt.savefig('../anaylse/ct_rich.png')
# ----------------------------------------------------------------
ca_rich = pd.DataFrame(prt_rich['consume_amt'].values / user_rich_amount['user_amount'].values, index=user_rich_amount.index)
ca_rich.to_csv('./divided/ca_rich.csv', index=True)
ca_rich.plot(label='ca', figsize=(20, 4))
plt.legend()
plt.title('consume average rich')
plt.draw()
plt.savefig('../anaylse/ca_rich.png')
# ----------------------------------------------------------------
tt_rich = pd.DataFrame(prt_rich['transfer_amt'].values / user_rich_amount['user_amount'].values, index=user_rich_amount.index)
tt_rich.to_csv('./divided/tt_rich.csv', index=True)
tt_rich.plot(label='tt', figsize=(20, 4))
plt.legend()
plt.title('transfer total rich')
plt.draw()
plt.savefig('../anaylse/tt_rich.png')
# ----------------------------------------------------------------
ta_rich = pd.DataFrame(prt_rich['transfer_amt'].values / user_rich_amount['user_amount'].values, index=user_rich_amount.index)
ta_rich.to_csv('./divided/ta_rich.csv', index=True)
ta_rich.plot(label='ta', figsize=(20, 4))
plt.legend()
plt.title('transfer average rich')
plt.draw()
plt.savefig('../anaylse/ta_rich.png')
# ----------------------------------------------------------------
# 单笔小于5W的赎回分析
poor_balance_table = user_balance_table.loc[~user_balance_table['user_id'].isin(more_than_bound['user_id'])]
# ----------------------------------------------------------------
user_poor_time_group = poor_balance_table.groupby(['report_date'])
prt_poor = user_poor_time_group['total_purchase_amt', 'total_redeem_amt', 'consume_amt', 'transfer_amt'].sum()
user_poor_amount = pd.DataFrame(user_poor_time_group.size())
user_poor_amount.columns = ['user_amount']
user_poor_amount.to_csv('./divided/poor_user_amt.csv', index=True)
user_poor_amount.plot(label='user_poor_amount', figsize=(20, 4))
plt.legend()
plt.title('user_poor_amount')
plt.draw()
plt.savefig('../anaylse/user_poor_amount.png')
# ----------------------------------------------------------------
pt_poor = pd.DataFrame(prt_poor['total_purchase_amt'].values, index=user_poor_amount.index)
pt_poor.to_csv('./divided/pt_poor.csv', index=True)
pt_poor.plot(label='pt', figsize=(20, 4))
plt.legend()
plt.title('purchase total poor')
plt.draw()
plt.savefig('../anaylse/pt_poor.png')
# ----------------------------------------------------------------
pa_poor = pd.DataFrame(prt_poor['total_purchase_amt'].values / user_poor_amount['user_amount'].values, index=user_poor_amount.index)
pa_poor.to_csv('./divided/pa_poor.csv', index=True)
pa_poor.plot(label='pa', figsize=(20, 4))
plt.legend()
plt.title('purchase average poor')
plt.draw()
plt.savefig('../anaylse/pa_poor.png')
# ----------------------------------------------------------------
rt_poor = pd.DataFrame(prt_poor['total_redeem_amt'].values, index=user_poor_amount.index)
rt_poor.to_csv('./divided/rt_poor.csv', index=True)
rt_poor.plot(label='rt', figsize=(20, 4))
plt.legend()
plt.title('redeem total poor')
plt.draw()
plt.savefig('../anaylse/rt_poor.png')
# ----------------------------------------------------------------
ra_poor = pd.DataFrame(prt_poor['total_redeem_amt'].values / user_poor_amount['user_amount'].values, index=user_poor_amount.index)
ra_poor.to_csv('./divided/ra_poor.csv', index=True)
ra_poor.plot(label='ra', figsize=(20, 4))
plt.legend()
plt.title('redeem average poor')
plt.draw()
plt.savefig('../anaylse/ra_poor.png')
# ----------------------------------------------------------------
ct_poor = pd.DataFrame(prt_poor['consume_amt'].values, index=user_poor_amount.index)
ct_poor.to_csv('./divided/ct_poor.csv', index=True)
ct_poor.plot(label='ct', figsize=(20, 4))
plt.legend()
plt.title('consume total poor')
plt.draw()
plt.savefig('../anaylse/ct_poor.png')
# ----------------------------------------------------------------
ca_poor = pd.DataFrame(prt_poor['consume_amt'].values / user_poor_amount['user_amount'].values, index=user_poor_amount.index)
ca_poor.to_csv('./divided/ca_poor.csv', index=True)
ca_poor.plot(label='ca', figsize=(20, 4))
plt.legend()
plt.title('consume average poor')
plt.draw()
plt.savefig('../anaylse/ca_poor.png')
# ----------------------------------------------------------------
tt_poor = pd.DataFrame(prt_poor['transfer_amt'].values / user_poor_amount['user_amount'].values, index=user_poor_amount.index)
tt_poor.to_csv('./divided/tt_poor.csv', index=True)
tt_poor.plot(label='tt', figsize=(20, 4))
plt.legend()
plt.title('transfer total poor')
plt.draw()
plt.savefig('../anaylse/tt_poor.png')
# ----------------------------------------------------------------
ta_poor = pd.DataFrame(prt_poor['transfer_amt'].values / user_poor_amount['user_amount'].values, index=user_poor_amount.index)
ta_poor.to_csv('./divided/ta_poor.csv', index=True)
ta_poor.plot(label='ta', figsize=(20, 4))
plt.legend()
plt.title('transfer average poor')
plt.draw()
plt.savefig('../anaylse/ta_poor.png')
# ----------------------------------------------------------------
