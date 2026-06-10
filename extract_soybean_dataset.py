"""
大豆データセット抽出スクリプト
  - FieldData_fieldid.db の Questionaire テーブルから field_id がある収量データを取得
  - weather_database_fieldid.db から対応する気象データ（TMP_mea, TMP_max, TMP_min,
    APCP, SSD, GSR, SD, SWE, SFW）を取得
  - 中間保存: soybean_dataset.parquet
  - ts2vec用: X.npy (N, T, 9), y.npy (N,), meta.csv
"""

import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path

# ─── パス設定 ────────────────────────────────────────────────
FIELD_DB   = r'c:\Users\amilu\Projects\vsCodeFile\Mstudy\data\processed\FieldData_fieldid.db'
WEATHER_DB = r'c:\Users\amilu\Projects\vsCodeFile\Mstudy\data\processed\weather_database_fieldid.db'
OUT_DIR    = Path(r'c:\Users\amilu\Projects\vsCodeFile\Mstudy\data\processed\soybean_ts2vec')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 気象特徴量カラム（1981年から存在するもの）
WEATHER_COLS = ['TMP_mea', 'TMP_max', 'TMP_min', 'APCP', 'SSD', 'GSR', 'SD', 'SWE', 'SFW']

# ─── Step 1: 収量データ取得（field_id・yield ともに非NULL） ──
print("Step 1: Questionaire から有効な収量データを取得...")
conn_f = sqlite3.connect(FIELD_DB)
df_yield = pd.read_sql(
    """
    SELECT field_id, year, yield
    FROM Questionaire
    WHERE field_id IS NOT NULL
      AND yield IS NOT NULL
    ORDER BY field_id, year
    """,
    conn_f
)
conn_f.close()

df_yield['field_id'] = df_yield['field_id'].astype(int)
df_yield['year']     = df_yield['year'].astype(int)
print(f"  → {len(df_yield)} 件 (field_id: {df_yield['field_id'].nunique()} 個, "
      f"year: {df_yield['year'].min()}〜{df_yield['year'].max()})")

# ─── Step 2: 気象データ取得 ────────────────────────────────────
print("\nStep 2: weather_database から対応する気象データを取得...")
fid_list   = df_yield['field_id'].unique().tolist()
year_list  = df_yield['year'].unique().tolist()

# 対象 field_id と year に絞って取得（1回のクエリで効率化）
fid_ph  = ','.join(['?' for _ in fid_list])
year_ph = ','.join(['?' for _ in year_list])

weather_cols_str = ', '.join(WEATHER_COLS)
query = f"""
    SELECT field_id,
           date,
           {weather_cols_str}
    FROM weather_data
    WHERE field_id IN ({fid_ph})
      AND CAST(SUBSTR(date, 1, 4) AS INTEGER) IN ({year_ph})
    ORDER BY field_id, date
"""

conn_w = sqlite3.connect(WEATHER_DB)
print("  SQLクエリ実行中（数分かかる場合があります）...")
df_weather = pd.read_sql(query, conn_w, params=fid_list + year_list)
conn_w.close()

df_weather['field_id'] = df_weather['field_id'].astype(int)
df_weather['date']     = pd.to_datetime(df_weather['date'])
df_weather['year']     = df_weather['date'].dt.year
print(f"  → {len(df_weather):,} 行取得")

# ─── Step 3: 中間保存（Parquet） ──────────────────────────────
print("\nStep 3: 中間データを Parquet で保存...")
# yield 情報をマージ
df_full = df_weather.merge(df_yield, on=['field_id', 'year'], how='inner')
parquet_path = OUT_DIR / 'soybean_dataset.parquet'
df_full.to_parquet(parquet_path, index=False, compression='snappy')
print(f"  → 保存: {parquet_path}")
print(f"     行数: {len(df_full):,}, カラム: {list(df_full.columns)}")

# ─── Step 4: ts2vec用 NPY 配列生成 ───────────────────────────
print("\nStep 4: ts2vec 用配列を生成...")

# 各サンプルの時系列長を揃えるため T=366（うるう年対応）、不足はNaNでパディング
T = 366

samples_X    = []
samples_y    = []
samples_meta = []

for _, row in df_yield.iterrows():
    fid  = row['field_id']
    year = row['year']
    yld  = row['yield']

    # このサンプルの気象時系列
    mask = (df_weather['field_id'] == fid) & (df_weather['year'] == year)
    ts   = df_weather.loc[mask].sort_values('date')

    # 9特徴量を抽出（欠損はNaN）
    arr = ts[WEATHER_COLS].to_numpy(dtype=np.float32)  # shape: (n_days, 9)

    # T=366にパディング（不足分をNaNで埋める）
    if len(arr) < T:
        pad = np.full((T - len(arr), len(WEATHER_COLS)), np.nan, dtype=np.float32)
        arr = np.vstack([arr, pad])
    else:
        arr = arr[:T]  # 366日超（基本なし）は切り捨て

    samples_X.append(arr)
    samples_y.append(yld)
    samples_meta.append({'field_id': fid, 'year': year, 'yield': yld})

X = np.stack(samples_X, axis=0)   # (N, T, 9)
y = np.array(samples_y, dtype=np.float32)  # (N,)

print(f"  X.shape: {X.shape}  (サンプル数, タイムステップ, 気象変数数)")
print(f"  y.shape: {y.shape}")
print(f"  y 統計: mean={y.mean():.2f}, std={y.std():.2f}, "
      f"min={y.min():.2f}, max={y.max():.2f}")
print(f"  NaN 含有率: {np.isnan(X).mean()*100:.2f}%")

# 保存
np.save(OUT_DIR / 'X.npy', X)
np.save(OUT_DIR / 'y.npy', y)
print(f"  → 保存: {OUT_DIR / 'X.npy'}")
print(f"  → 保存: {OUT_DIR / 'y.npy'}")

# メタデータ CSV
df_meta = pd.DataFrame(samples_meta)
df_meta['feature_names'] = str(WEATHER_COLS)  # ヘッダ情報として参考保存
meta_path = OUT_DIR / 'meta.csv'
df_meta[['field_id', 'year', 'yield']].to_csv(meta_path, index=False)
print(f"  → 保存: {meta_path}")

# ─── Step 5: サマリー ─────────────────────────────────────────
print("\n" + "="*50)
print("抽出完了サマリー")
print("="*50)
print(f"  サンプル数  : {len(X)}")
print(f"  タイムステップ: {T} 日（うるう年対応パディング）")
print(f"  気象変数    : {WEATHER_COLS}")
print(f"  出力ディレクトリ: {OUT_DIR}")
print(f"  ファイル一覧:")
for f in sorted(OUT_DIR.iterdir()):
    size_mb = f.stat().st_size / 1024 / 1024
    print(f"    {f.name:30s}  {size_mb:7.2f} MB")
