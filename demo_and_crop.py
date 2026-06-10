"""
Q1: field_id を指定して収量と気象時系列を取得する例
Q2: 4月1日〜12月31日に限定した X_apr_dec.npy の生成
"""

import numpy as np
import pandas as pd
from pathlib import Path

OUT_DIR    = Path(r'c:\Users\amilu\Projects\vsCodeFile\Mstudy\data\processed\soybean_ts2vec')
WEATHER_COLS = ['TMP_mea', 'TMP_max', 'TMP_min', 'APCP', 'SSD', 'GSR', 'SD', 'SWE', 'SFW']

# ─── データ読み込み ─────────────────────────────────────────
X    = np.load(OUT_DIR / 'X.npy')           # (603, 366, 9)
y    = np.load(OUT_DIR / 'y.npy')           # (603,)
meta = pd.read_csv(OUT_DIR / 'meta.csv')    # field_id, year, yield
df   = pd.read_parquet(OUT_DIR / 'soybean_dataset.parquet')  # 日付付き生データ

# ═══════════════════════════════════════════════════════════
# Q1: field_id を指定して収量と気象時系列を取得
# ═══════════════════════════════════════════════════════════

def get_sample_by_field(field_id: int, year: int = None):
    """
    field_id（と year）を指定してサンプルを返す。

    Returns:
        idx    : サンプルインデックス
        yield_ : 収量 (kg/10a)
        ts_df  : 気象時系列 DataFrame (366日 × 9変数)
    """
    cond = meta['field_id'] == field_id
    if year is not None:
        cond &= meta['year'] == year

    rows = meta[cond]
    if len(rows) == 0:
        raise ValueError(f"field_id={field_id}, year={year} のデータが見つかりません")
    if len(rows) > 1:
        print(f"  ※ 複数の年が存在します: {rows['year'].tolist()}  → 最初の1件を返します")

    idx    = rows.index[0]
    year_  = int(rows.iloc[0]['year'])
    yield_ = float(rows.iloc[0]['yield'])

    # 日付列を付与するため Parquet から取得
    ts_df = df[(df['field_id'] == field_id) & (df['year'] == year_)]\
              .sort_values('date')\
              [['date'] + WEATHER_COLS]\
              .reset_index(drop=True)

    return idx, yield_, ts_df


# ── 実際に試す ──────────────────────────────────────────────
print("=" * 60)
print("Q1: field_id=1 の情報を取得")
print("=" * 60)

target_fid = 1
idx, yield_, ts_df = get_sample_by_field(target_fid)

print(f"  field_id  : {target_fid}")
print(f"  サンプルindex: {idx}  (X[{idx}], y[{idx}])")
print(f"  栽培年度  : {int(meta.loc[idx, 'year'])}")
print(f"  収量      : {yield_:.2f} kg/10a")
print()
print("  【年間気象時系列 最初の7日間】")
print(ts_df.head(7).to_string(index=True))
print()
print("  【年間気象時系列 最後の3日間】")
print(ts_df.tail(3).to_string(index=True))
print()
print(f"  時系列の長さ: {len(ts_df)} 日")

# X.npy から直接参照する場合
print()
print("  【X.npy から直接参照（day[0〜4]）】")
for d in range(5):
    vals = X[idx, d, :]
    print(f"    X[{idx}, {d}] = " + str(np.round(vals, 3)))

# ═══════════════════════════════════════════════════════════
# Q2: 4月1日〜12月31日に限定した配列を生成・保存
# ═══════════════════════════════════════════════════════════
print()
print("=" * 60)
print("Q2: 4月1日〜12月31日のデータを抽出")
print("=" * 60)

START_MMDD = (4, 1)    # 開始月日
END_MMDD   = (12, 31)  # 終了月日

samples_X_crop = []
samples_meta_crop = []

for i, row in meta.iterrows():
    fid  = int(row['field_id'])
    year = int(row['year'])

    # Parquet から対象圃場×年×日付範囲を抽出
    mask = (
        (df['field_id'] == fid) &
        (df['year'] == year) &
        (df['date'].dt.month > START_MMDD[0] |
         ((df['date'].dt.month == START_MMDD[0]) & (df['date'].dt.day >= START_MMDD[1]))) &
        (df['date'].dt.month <= END_MMDD[0])
    )
    # シンプルな日付フィルタ
    ts = df[
        (df['field_id'] == fid) &
        (df['year'] == year) &
        (df['date'] >= f"{year}-{START_MMDD[0]:02d}-{START_MMDD[1]:02d}") &
        (df['date'] <= f"{year}-{END_MMDD[0]:02d}-{END_MMDD[1]:02d}")
    ].sort_values('date')

    arr = ts[WEATHER_COLS].to_numpy(dtype=np.float32)  # (n_days, 9)
    samples_X_crop.append(arr)
    samples_meta_crop.append({'field_id': fid, 'year': year, 'n_days': len(arr)})

# タイムステップ数の確認
n_days_list = [s['n_days'] for s in samples_meta_crop]
print(f"  各サンプルの日数: {set(n_days_list)}")

# T を最大値で統一（うるう年=275日, 通常=275日 になるはず）
T_crop = max(n_days_list)
print(f"  固定タイムステップ T = {T_crop} 日")

# NaN パディングして揃える
X_crop = np.full((len(samples_X_crop), T_crop, len(WEATHER_COLS)), np.nan, dtype=np.float32)
for i, arr in enumerate(samples_X_crop):
    X_crop[i, :len(arr), :] = arr

print(f"  X_apr_dec.shape: {X_crop.shape}  (サンプル数, 日数, 気象変数数)")
print(f"  NaN 含有率: {np.isnan(X_crop).mean()*100:.2f}%")

# 保存
np.save(OUT_DIR / 'X_apr_dec.npy', X_crop)
print(f"  → 保存: {OUT_DIR / 'X_apr_dec.npy'}")
print()
print("  【サンプル検証: field_id=1 の4月1日〜5日】")
idx_check = meta[meta['field_id'] == 1].index[0]
for d in range(5):
    vals = X_crop[idx_check, d, :]
    print(f"    X_apr_dec[{idx_check}, {d}] (4/{d+1}) = " + str(np.round(vals, 3)))

print()
print("  ※ y.npy と meta.csv はそのまま共用できます（順番は同じ）")

# ─── 最終サマリー ────────────────────────────────────────────
print()
print("=" * 60)
print("生成ファイルの対比")
print("=" * 60)
print(f"  X.npy         : {X.shape}        1/1〜12/31（366日）")
print(f"  X_apr_dec.npy : {X_crop.shape}   4/1〜12/31（{T_crop}日）")
print(f"  y.npy         : {y.shape}        収量（共通）")
print(f"  meta.csv      : {meta.shape}        メタ情報（共通）")
