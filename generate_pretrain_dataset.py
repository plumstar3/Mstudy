"""
事前学習用データセット生成スクリプト
  - weather_database_fieldid.db から 1981〜2018年の全圃場×全年の気象データを抽出
  - ラベル（収量）不要
  - 出力: pretrain_X.npy  shape: (N, 366, 9)
  - 出力: pretrain_meta.csv  各サンプルの field_id, year 対応表
"""

import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
import time

WEATHER_DB   = r'c:\Users\amilu\Projects\vsCodeFile\Mstudy\data\processed\weather_database_fieldid.db'
OUT_DIR      = Path(r'c:\Users\amilu\Projects\vsCodeFile\Mstudy\data\processed\soybean_ts2vec')
WEATHER_COLS = ['TMP_mea', 'TMP_max', 'TMP_min', 'APCP', 'SSD', 'GSR', 'SD', 'SWE', 'SFW']
YEAR_START   = 1981
YEAR_END     = 2018
T            = 366  # うるう年対応（不足分NaNパディング）

t0 = time.time()

conn = sqlite3.connect(WEATHER_DB)

# ─── Step 1: 対象の (field_id, year) ペアを取得 ─────────────
print("Step 1: 対象 (field_id, year) ペアを確認...")
pairs_df = pd.read_sql(
    f"""
    SELECT DISTINCT field_id,
           CAST(SUBSTR(date, 1, 4) AS INTEGER) AS year
    FROM weather_data
    WHERE CAST(SUBSTR(date, 1, 4) AS INTEGER) BETWEEN {YEAR_START} AND {YEAR_END}
    ORDER BY field_id, year
    """,
    conn
)
N = len(pairs_df)
print(f"  → 対象サンプル数 : {N:,} ({pairs_df['field_id'].nunique()} 圃場 × {pairs_df['year'].nunique()} 年)")
print(f"  → year 範囲     : {pairs_df['year'].min()} 〜 {pairs_df['year'].max()}")
print(f"  → 推定 NPY サイズ: {N * T * 9 * 4 / 1024**2:.1f} MB")

# ─── Step 2: 年ごとにチャンク読み込み → 3D 配列に格納 ───────
print(f"\nStep 2: 気象データを年ごとに読み込み（{YEAR_START}〜{YEAR_END}年）...")

# 事前に全サンプルのインデックスを確定
pairs_df = pairs_df.reset_index(drop=True)
pair_to_idx = {(int(r.field_id), int(r.year)): i for i, r in pairs_df.iterrows()}

# 配列を事前確保（NaNで初期化）
X_pretrain = np.full((N, T, len(WEATHER_COLS)), np.nan, dtype=np.float32)

years = sorted(pairs_df['year'].unique())
for year in years:
    year_fids = pairs_df[pairs_df['year'] == year]['field_id'].tolist()
    fid_ph = ','.join(['?' for _ in year_fids])

    df_year = pd.read_sql(
        f"""
        SELECT field_id, date, {', '.join(WEATHER_COLS)}
        FROM weather_data
        WHERE CAST(SUBSTR(date, 1, 4) AS INTEGER) = {year}
          AND field_id IN ({fid_ph})
        ORDER BY field_id, date
        """,
        conn,
        params=year_fids
    )

    # 圃場ごとに配列に格納
    for fid, grp in df_year.groupby('field_id'):
        idx = pair_to_idx.get((int(fid), year))
        if idx is None:
            continue
        arr = grp[WEATHER_COLS].to_numpy(dtype=np.float32)
        n_days = min(len(arr), T)
        X_pretrain[idx, :n_days, :] = arr[:n_days]

    elapsed = time.time() - t0
    print(f"  {year} 完了 ({len(year_fids):3d} 圃場)  経過: {elapsed:.1f}s")

conn.close()

# ─── Step 3: 保存 ────────────────────────────────────────────
print(f"\nStep 3: 保存中...")

out_x    = OUT_DIR / 'pretrain_X.npy'
out_meta = OUT_DIR / 'pretrain_meta.csv'

np.save(out_x, X_pretrain)
pairs_df.to_csv(out_meta, index=False)

size_mb = out_x.stat().st_size / 1024**2

# ─── Step 4: サマリー ─────────────────────────────────────────
print(f"\n{'='*55}")
print("事前学習用データセット 生成完了")
print(f"{'='*55}")
print(f"  pretrain_X.npy")
print(f"    shape  : {X_pretrain.shape}  (N, T, features)")
print(f"    dtype  : {X_pretrain.dtype}")
print(f"    サイズ : {size_mb:.1f} MB")
print(f"    NaN率  : {np.isnan(X_pretrain).mean()*100:.2f}%")
print(f"  pretrain_meta.csv")
print(f"    行数   : {len(pairs_df):,}")
print(f"    列     : {list(pairs_df.columns)}")
print(f"  気象変数 : {WEATHER_COLS}")
print(f"  期間     : {YEAR_START}〜{YEAR_END}年")
print(f"  総処理時間: {time.time()-t0:.1f}秒")

print(f"\n【年別サンプル数（上位/下位5年）】")
year_cnt = pairs_df.groupby('year').size()
print("  先頭5年:")
print(year_cnt.head().to_string())
print("  末尾5年:")
print(year_cnt.tail().to_string())

print(f"\n【各気象変数の統計（NaN除外）】")
print(f"  {'変数':<12} {'mean':>9} {'std':>9} {'min':>9} {'max':>9} {'NaN%':>7}")
print(f"  {'-'*57}")
for i, col in enumerate(WEATHER_COLS):
    vals = X_pretrain[:, :, i].flatten()
    v = vals[~np.isnan(vals)]
    nan_pct = np.isnan(vals).mean() * 100
    print(f"  {col:<12} {v.mean():>9.3f} {v.std():>9.3f} {v.min():>9.3f} {v.max():>9.3f} {nan_pct:>6.2f}%")

print(f"\n【ファイルまとめ】")
for f in sorted(OUT_DIR.iterdir()):
    if f.suffix in ('.npy', '.csv', '.parquet'):
        mb = f.stat().st_size / 1024**2
        print(f"  {f.name:<35} {mb:7.1f} MB")
