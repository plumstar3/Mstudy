import sqlite3, pandas as pd, numpy as np

conn = sqlite3.connect('data/processed/FieldData_fieldid.db')

# テーブル構造確認
cols = pd.read_sql('PRAGMA table_info(SolidMoisture)', conn)
print('=== SolidMoisture テーブル列情報 ===')
print(cols[['cid','name','type']].to_string(index=False))

# 総レコード数と先頭行
total = pd.read_sql('SELECT COUNT(*) as n FROM SolidMoisture', conn).iloc[0,0]
print(f'\n総レコード数: {total:,} 件')

df = pd.read_sql('SELECT * FROM SolidMoisture LIMIT 10', conn)
print('\n先頭10行:')
print(df.to_string(index=False))

# 圃場数・期間
df_all = pd.read_sql('SELECT field_id, place, year, date, VWC FROM SolidMoisture', conn)
df_all['date']     = pd.to_datetime(df_all['date'], errors='coerce')
df_all['field_id'] = pd.to_numeric(df_all['field_id'], errors='coerce')
df_all = df_all.dropna(subset=['date','field_id'])

n_fields = df_all['field_id'].nunique()
n_places = df_all['place'].nunique()
print(f'\n測定圃場数（field_id）: {n_fields} 圃場')
print(f'測定地点数（place）  : {n_places} 地点')
print(f'測定年: {sorted(df_all["year"].dropna().unique().tolist())}')
print(f'測定期間: {df_all["date"].min().date()} 〜 {df_all["date"].max().date()}')
print(f'VWC の範囲: {df_all["VWC"].min():.4f} 〜 {df_all["VWC"].max():.4f}')
print(f'VWC 欠損数: {df_all["VWC"].isna().sum()} 件')

# 圃場ごとのレコード数・期間・測定間隔
print('\n=== 圃場別サマリ（field_id） ===')
rows = []
for fid, g in df_all.groupby('field_id'):
    g = g.sort_values('date')
    intervals = g['date'].diff().dt.total_seconds() / 3600
    rows.append({
        'field_id':          int(fid),
        'n_records':         len(g),
        'start':             g['date'].min().date(),
        'end':               g['date'].max().date(),
        'days_span':         (g['date'].max() - g['date'].min()).days,
        'median_interval_h': round(float(intervals.median()), 1),
        'vwc_mean':          round(float(g['VWC'].mean()), 3),
        'vwc_std':           round(float(g['VWC'].std()), 3),
    })
summary = pd.DataFrame(rows).sort_values('n_records', ascending=False)
print(summary.to_string(index=False))

# 測定間隔の全体分布
all_intervals = df_all.sort_values(['field_id','date']).groupby('field_id')['date'].diff().dt.total_seconds() / 3600
print('\n=== 測定間隔（時間）の全体統計 ===')
print(all_intervals.dropna().describe().round(2))
top_intervals = all_intervals.dropna().round(1).value_counts().head(10)
print('\n最多の測定間隔（上位10）:')
print(top_intervals.to_string())

conn.close()
