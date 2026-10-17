import sqlite3, pandas as pd, numpy as np
import matplotlib, matplotlib.pyplot as plt, matplotlib.font_manager as fm

matplotlib.use('Agg')
for _fn in ['IPAexGothic','MS Gothic','Yu Gothic','Meiryo']:
    if any(_fn.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        plt.rcParams['font.family'] = _fn; break
plt.rcParams['axes.unicode_minus'] = False

conn = sqlite3.connect('data/processed/FieldData_fieldid.db')
sm = pd.read_sql('SELECT field_id, year, date, VWC FROM SolidMoisture WHERE field_id IS NOT NULL', conn)
quest = pd.read_sql(
    'SELECT field_id, year, seed_date, harvest_date FROM Questionaire WHERE seed_date IS NOT NULL AND harvest_date IS NOT NULL',
    conn)
conn.close()

sm['field_id'] = pd.to_numeric(sm['field_id'], errors='coerce').dropna().astype(int)
sm['year'] = sm['year'].astype(int)
sm['date'] = pd.to_datetime(sm['date'], errors='coerce')

quest['field_id'] = pd.to_numeric(quest['field_id'], errors='coerce').dropna().astype(int)
quest['year'] = quest['year'].astype(int)
quest['seed_date'] = pd.to_datetime(quest['seed_date'])
quest['harvest_date'] = pd.to_datetime(quest['harvest_date'])
quest['growing_days'] = (quest['harvest_date'] - quest['seed_date']).dt.days

# 結合
df = sm.merge(quest, on=['field_id','year'], how='inner')
df['DAP'] = (df['date'] - df['seed_date']).dt.days
df['DAP_pct'] = df['DAP'] / df['growing_days'] * 100   # 0〜100%正規化
df['DBH'] = (df['harvest_date'] - df['date']).dt.days  # 収穫までの残日数

print('=== DAP の統計 ===')
print(df['DAP'].describe().round(1))
print()
print('=== DAP% の統計 ===')
print(df['DAP_pct'].describe().round(1))
print()
print('=== DBH（収穫前日数）の統計 ===')
print(df['DBH'].describe().round(1))

# 各DAP%地点での有効圃場数を集計
bins = np.arange(-10, 130, 5)  # -10%〜130%を5%刻み
df['DAP_bin'] = pd.cut(df['DAP_pct'], bins=bins, labels=False)
coverage_dap = df.groupby('DAP_bin')['field_id'].nunique()
bin_centers = bins[:-1] + 2.5

# 各DBH地点での有効圃場数
bins_dbh = np.arange(-30, 130, 5)
df['DBH_bin'] = pd.cut(df['DBH'], bins=bins_dbh, labels=False)
coverage_dbh = df.groupby('DBH_bin')['field_id'].nunique()
bin_centers_dbh = bins_dbh[:-1] + 2.5

# 可視化
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

# DAP%カバレッジ
ax1.bar(bin_centers, [coverage_dap.get(i, 0) for i in range(len(bins)-1)],
        width=4.5, color='#3498DB', alpha=0.7, edgecolor='white')
ax1.axvline(0, color='green', lw=1.5, ls='--', label='播種日 (DAP%=0)')
ax1.axvline(100, color='red', lw=1.5, ls='--', label='収穫日 (DAP%=100)')
ax1.axhline(50, color='gray', lw=1, ls=':', label='n=50 ライン')
ax1.set_xlabel('生育ステージ（DAP %）', fontsize=10)
ax1.set_ylabel('有効圃場数', fontsize=10)
ax1.set_title('DAP% 軸での圃場カバレッジ（測定データがある圃場数）', fontsize=11)
ax1.legend(fontsize=9)
ax1.grid(axis='y', ls='--', alpha=0.4)

# DBHカバレッジ
ax2.bar(bin_centers_dbh, [coverage_dbh.get(i, 0) for i in range(len(bins_dbh)-1)],
        width=4.5, color='#E74C3C', alpha=0.7, edgecolor='white')
ax2.axvline(0, color='red', lw=1.5, ls='--', label='収穫日 (DBH=0)')
ax2.axhline(50, color='gray', lw=1, ls=':', label='n=50 ライン')
ax2.set_xlabel('収穫前日数（DBH）← 大きいほど収穫から遠い', fontsize=10)
ax2.set_ylabel('有効圃場数', fontsize=10)
ax2.set_title('DBH 軸での圃場カバレッジ（収穫日を基準に整列）', fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(axis='y', ls='--', alpha=0.4)
ax2.invert_xaxis()

plt.tight_layout()
plt.savefig('outputs/soil_analysis/coverage_check.png', dpi=180, bbox_inches='tight')
plt.close()

print('\n=== DAP% Coverage (5% bins) ===')
for i, center in enumerate(bin_centers):
    n = coverage_dap.get(i, 0)
    bar = '|' * (n // 5)
    print(f'  DAP {center:6.1f}%: {n:3d} fields {bar}')
