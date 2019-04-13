import pandas as pd
import numpy as np


df_purchase = pd.read_csv('./output/purchase(2,1,3)/predict.csv', sep=',', engine='python', encoding='utf-8',header=None, names = ['report_date','purchase'])
print(df_purchase)
df_purchase['report_date'] = pd.to_datetime(df_purchase.report_date)
df_purchase['report_date'] = df_purchase['report_date'].dt.strftime('%Y%m%d')
df_purchase['purchase'] = round(df_purchase['purchase'])
df_purchase['purchase'] = df_purchase['purchase'].astype('int')


df_redeem = pd.read_csv('./output/redeem(4,1,5)/predict.csv', sep=',', engine='python', encoding='utf-8',header=None, names = ['report_date','redeem'])
df_redeem.columns = ['report_date','redeem']

df_redeem['report_date'] = pd.to_datetime(df_redeem.report_date)
df_redeem['report_date'] = df_redeem['report_date'].dt.strftime('%Y%m%d')
df_redeem['redeem'] = round(df_redeem['redeem'])
df_redeem['redeem'] = df_redeem['redeem'].astype('int')


df = pd.merge(df_purchase, df_redeem, on='report_date')

df.to_csv("./output/purchase(2,1,3)+redeem(4,1,5).csv", index=False)
