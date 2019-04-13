import pandas as pd
import numpy as np

df_purchase = pd.read_csv('./predict/purchase/purchase.csv', sep=',', engine='python', encoding='utf-8',parse_dates=['report_date'])
df_purchase['report_date'] = df_purchase['report_date'].dt.strftime('%Y%m%d')
df_purchase['purchase'] = round(df_purchase['purchase'])
df_purchase['purchase'] = df_purchase['purchase'].astype('int')

df_redeem = pd.read_csv('./predict/redeem/redeem.csv', sep=',', engine='python', encoding='utf-8',parse_dates=['report_date'])
df_redeem['report_date'] = df_redeem['report_date'].dt.strftime('%Y%m%d')
df_redeem['redeem'] = round(df_redeem['redeem'])
df_redeem['redeem'] = df_redeem['redeem'].astype('int')


df = pd.merge(df_purchase, df_redeem, on='report_date')
print(df)

df.to_csv("./report/purchase+redeem.csv", index=False)
