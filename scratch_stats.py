import pandas as pd
import numpy as np
import sqlite3

# 統計テーブルを詳しく読む
g = pd.read_csv('outputs/yield_group_analysis/04_stats_table_global.csv')
i = pd.read_csv('outputs/yield_group_analysis/08_stats_table_intra.csv')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print('=== GLOBAL (全体比較) ===')
print(g.to_string(index=False))
print()
print('=== INTRA (同市町村内偏差比較) ===')
print(i.to_string(index=False))

# 収量の基礎統計
conn = sqlite3.connect('data/processed/FieldData_fieldid.db')
q = pd.read_sql(
    "SELECT year, yield, breed FROM Questionaire WHERE yield IS NOT NULL AND year BETWEEN 2015 AND 2018",
    conn)
conn.close()

print('\n=== 収量基礎統計（2015-2018）===')
print(q['yield'].describe().round(1))
print('\n年別統計:')
print(q.groupby('year')['yield'].agg(['mean','std','min','max','count']).round(1))

lo33 = q['yield'].quantile(0.33)
hi33 = q['yield'].quantile(0.67)
high_mean = q[q['yield'] >= hi33]['yield'].mean()
low_mean  = q[q['yield'] <= lo33]['yield'].mean()
print(f'\nLow閾値（33%ile）: {lo33:.1f} kg/10a')
print(f'High閾値（67%ile）: {hi33:.1f} kg/10a')
print(f'High群平均: {high_mean:.1f} kg/10a')
print(f'Low群平均:  {low_mean:.1f} kg/10a')
print(f'群間差:     {high_mean - low_mean:.1f} kg/10a')

# 品種ランキング（上位10）
print('\n品種出現数 上位10:')
print(q['breed'].value_counts().head(10))
