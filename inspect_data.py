"""
データ内容の確認スクリプト
"""
import numpy as np
import pandas as pd
import os
import datetime

DATASET_DIR = os.path.join('data', 'processed', 'soybean_ts2vec')
WEATHER_COLS = ['TMP_mea', 'TMP_max', 'TMP_min', 'APCP', 'SSD', 'GSR', 'SD', 'SWE', 'SFW']

# ── ファイル一覧 ────────────────────────────────────────────────
print("=" * 65)
print("  データファイル一覧")
print("=" * 65)
for f in sorted(os.listdir(DATASET_DIR)):
    fpath = os.path.join(DATASET_DIR, f)
    size  = os.path.getsize(fpath) / 1024 / 1024
    print(f"  {f:<35} {size:8.2f} MB")

# ── X.npy ──────────────────────────────────────────────────────
print()
print("=" * 65)
print("  X.npy  （生データ・正規化前）")
print("=" * 65)
X = np.load(os.path.join(DATASET_DIR, 'X.npy'))
print(f"  shape      : {X.shape}  → (サンプル数, タイムステップ, 気象変数数)")
print(f"  dtype      : {X.dtype}")
print(f"  NaN 含有率 : {np.isnan(X).mean()*100:.4f}%")
print(f"  非NaN 統計 : min={np.nanmin(X):.3f}  max={np.nanmax(X):.3f}  mean={np.nanmean(X):.3f}")
print()
print("  --- 各気象変数の統計（全サンプル・全時間ステップ） ---")
print(f"  {'変数':<10} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}  説明")
desc = {
    'TMP_mea': '日平均気温 (°C)',
    'TMP_max': '日最高気温 (°C)',
    'TMP_min': '日最低気温 (°C)',
    'APCP'   : '降水量 (mm)',
    'SSD'    : '日照時間 (h)',
    'GSR'    : '全天日射量 (MJ/m2)',
    'SD'     : '積雪深 (cm)',
    'SWE'    : '積雪水量 (mm)',
    'SFW'    : '降雪量 (mm)',
}
for i, col in enumerate(WEATHER_COLS):
    vals = X[:, :, i].flatten()
    vals = vals[~np.isnan(vals)]
    print(f"  {col:<10} {vals.mean():>8.3f} {vals.std():>8.3f} {vals.min():>8.3f} {vals.max():>8.3f}  {desc[col]}")

# ── サンプル0の実際の値 ────────────────────────────────────────
print()
print("=" * 65)
print("  X[0] の実際の値（サンプル0 = 最初のサンプル）")
print("=" * 65)
meta = pd.read_csv(os.path.join(DATASET_DIR, 'meta.csv'))
row0 = meta.iloc[0]
print(f"  field_id={int(row0['field_id'])}  year={int(row0['year'])}  yield={row0['yield']:.3f}")
print()
sample = X[0]   # (366, 9)
year   = int(row0['year'])
jan1   = datetime.date(year, 1, 1)

# 5/1〜12/27 の期間を切り出して表示
may1_idx  = (datetime.date(year, 5, 1)  - jan1).days
dec27_idx = (datetime.date(year, 12, 27) - jan1).days
window = sample[may1_idx:dec27_idx + 1]  # (241, 9)

print(f"  ※ 5/1〜12/27 の期間（index {may1_idx}〜{dec27_idx}、計 {len(window)} 日）を表示")
print()
print(f"  {'日付':<12} {'idx':>4} | " + " ".join(f"{c:>8}" for c in WEATHER_COLS))
print("  " + "-" * (16 + 9 * 9))
show_days = list(range(0, 10)) + ['...'] + list(range(len(window)-5, len(window)))
for d in show_days:
    if d == '...':
        print("  " + " " * 17 + "...")
        continue
    date   = jan1 + datetime.timedelta(days=may1_idx + d)
    idx_in = may1_idx + d
    vals   = sample[idx_in]
    vals_str = " ".join(
        f"{'NaN':>8}" if np.isnan(v) else f"{v:>8.3f}"
        for v in vals
    )
    print(f"  {str(date):<12}  {idx_in:>4} | {vals_str}")

# ── y.npy ──────────────────────────────────────────────────────
print()
print("=" * 65)
print("  y.npy  （収量ラベル）")
print("=" * 65)
y = np.load(os.path.join(DATASET_DIR, 'y.npy'))
print(f"  shape  : {y.shape}")
print(f"  dtype  : {y.dtype}")
print(f"  min    : {y.min():.3f}")
print(f"  max    : {y.max():.3f}")
print(f"  mean   : {y.mean():.3f}")
print(f"  std    : {y.std():.3f}")
print()
print("  --- 年度別統計 ---")
for yr in sorted(meta['year'].unique()):
    mask = meta['year'] == yr
    yvals = y[mask]
    print(f"  {int(yr)}: n={len(yvals):3d}  "
          f"mean={yvals.mean():.1f}  std={yvals.std():.1f}  "
          f"min={yvals.min():.1f}  max={yvals.max():.1f}")

# ── meta.csv ──────────────────────────────────────────────────
print()
print("=" * 65)
print("  meta.csv")
print("=" * 65)
print(f"  shape   : {meta.shape}")
print(f"  columns : {list(meta.columns)}")
print()
print("  --- 先頭10行 ---")
print(meta.head(10).to_string(index=False))

# ── parquet ────────────────────────────────────────────────────
print()
print("=" * 65)
print("  soybean_dataset.parquet  （元の中間データ）")
print("=" * 65)
df = pd.read_parquet(os.path.join(DATASET_DIR, 'soybean_dataset.parquet'))
df['date'] = pd.to_datetime(df['date'])
print(f"  shape      : {df.shape}")
print(f"  columns    : {list(df.columns)}")
print(f"  日付範囲   : {df['date'].min().date()} 〜 {df['date'].max().date()}")
print(f"  field_id   : {df['field_id'].nunique()} ユニーク")
print()
print("  --- 先頭5行 ---")
print(df.head(5).to_string(index=False))
print()
print("  --- 気象変数の記述統計 ---")
print(df[WEATHER_COLS].describe().round(3).to_string())
