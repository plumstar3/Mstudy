"""
04_ndvi_smoothing_test.py
=========================
1圃場のNDVI時系列に対して、スプライン補間 + Savitzky-Golay フィルタを
適用して生育曲線を滑らかにするテストスクリプト。

処理の流れ:
  1. CSVと圃場DBを読み込む
  2. 対象圃場のNDVIデータを抽出
  3. 播種日（seed_date）を起点に「播種後日数（DAP）」を計算
  4. スプライン補間で毎日のNDVI値を作成
  5. SGフィルタでギザギザを平滑化（ノイズ除去）
  6. 処理前後を比較グラフで可視化

Jupyter Notebook で使う場合:
  各 [Cell N] のブロックをそれぞれのセルに貼り付けて実行。
"""

# =============================================================================
# [Cell 1] ライブラリのインポートとパラメータ設定
# =============================================================================
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

# ── Windows 日本語フォント設定 ──────────────────────────────────
_jp_fonts  = ['Yu Gothic', 'MS Gothic', 'Meiryo', 'IPAexGothic', 'Noto Sans CJK JP']
_available = {f.name for f in fm.fontManager.ttflist}
_font      = next((f for f in _jp_fonts if f in _available), None)
if _font:
    plt.rcParams['font.family'] = _font
plt.rcParams['axes.unicode_minus'] = False
# ────────────────────────────────────────────────────────────────

# ── テスト対象の設定 ─────────────────────────────────────────────
TARGET_FIELD_ID = 141  # ← 試したい field_id に変更
TARGET_YEAR     = 2015  # ← 試したい year に変更

# ── ファイルパス ─────────────────────────────────────────────────
NDVI_CSV  = '../../data/raw/hls_50m_ndvi_timeseries.csv'
FIELD_DB  = '../../data/processed/FieldData_fieldid.db'

# ── SGフィルタのパラメータ ───────────────────────────────────────
# window_length : 平滑化する窓の幅（奇数、大きいほど滑らか）
# polyorder     : 当てはめる多項式の次数（小さいほど滑らか）
# ※ window_length > polyorder の関係が必須
SG_WINDOW   = 15   # 15日幅の窓（7〜21の奇数を試してみてください）
SG_POLYORDER = 3   # 3次多項式（2〜4が実用的）

print('設定完了！')
print(f'  対象圃場: field_id={TARGET_FIELD_ID}, year={TARGET_YEAR}')
print(f'  SGフィルタ: 窓幅={SG_WINDOW}日, 多項式次数={SG_POLYORDER}次')


# =============================================================================
# [Cell 2] データの読み込みと前処理
# =============================================================================

# ── NDVIデータの読み込み（対象圃場のみ抽出）──────────────────────
ndvi_all = pd.read_csv(NDVI_CSV, parse_dates=['date'])
raw_df = ndvi_all[
    (ndvi_all['field_id'] == TARGET_FIELD_ID) &
    (ndvi_all['year']     == TARGET_YEAR)
].copy().sort_values('date').reset_index(drop=True)

if raw_df.empty:
    raise ValueError(
        f'field_id={TARGET_FIELD_ID}, year={TARGET_YEAR} のデータが '
        f'{NDVI_CSV} に見つかりません。'
    )

# ── 圃場DBから播種日（seed_date）と収穫日（harvest_date）を取得 ──
conn = sqlite3.connect(FIELD_DB)
field_info = pd.read_sql(f"""
    SELECT field_id, year, lat, lon, seed_date, harvest_date, yield
    FROM Questionaire
    WHERE field_id = {TARGET_FIELD_ID} AND year = {TARGET_YEAR}
    LIMIT 1
""", conn)
conn.close()

if field_info.empty:
    raise ValueError(f'field_id={TARGET_FIELD_ID}, year={TARGET_YEAR} がDBに見つかりません。')

seed_date    = pd.to_datetime(field_info.iloc[0]['seed_date'])
harvest_date = pd.to_datetime(field_info.iloc[0]['harvest_date'])
actual_yield = float(field_info.iloc[0]['yield'])

# ── 播種後日数（DAP: Days After Planting）を計算 ─────────────────
raw_df['DAP'] = (raw_df['date'] - seed_date).dt.days

print('=== 圃場情報 ===')
print(f'  播種日    : {seed_date.date()}')
print(f'  収穫日    : {harvest_date.date()}')
print(f'  実測収量  : {actual_yield:.1f} kg/10a')
print(f'  生育日数  : {(harvest_date - seed_date).days} 日')
print()
print(f'=== 取得済みNDVIデータ（{len(raw_df)} 件）===')
print(raw_df[['date', 'DAP', 'ndvi', 'source']].to_string())


# =============================================================================
# [Cell 3] スプライン補間 → SGフィルタの適用
# =============================================================================

# ── ステップ1：DAPの有効範囲の決定 ───────────────────────────────
# 播種日（DAP=0）から収穫日（DAP=harvest_dap）までを対象とする
harvest_dap = (harvest_date - seed_date).days
dap_min     = max(0, raw_df['DAP'].min())   # 最初の観測（DAP≥0のもの）
dap_max     = harvest_dap

print(f'補間範囲: DAP {dap_min} 〜 {dap_max} 日（1日刻み）')

# 播種前と収穫後のデータは除外（ノイズになりやすいため）
interp_df = raw_df[
    (raw_df['DAP'] >= dap_min) &
    (raw_df['DAP'] <= dap_max)
].copy()

# ── ステップ2：両端アンカー点の追加 ─────────────────────────────
# 同じDAPが複数ある場合（同じ日に複数シーン）は平均値を使用
interp_src = interp_df.groupby('DAP')['ndvi'].mean().reset_index()

# 農学的根拠に基づく「仮想裸地NDVI」
# 播種日（DAP=0）: 大豆はまだ地中 → 裸地
# 収穫日（DAP=harvest_dap）: 完全落葉・刈取後 → 裸地
# 日本農地の典型的な裸地NDVI = 0.10〜0.20（ここでは中央値0.15を使用）
BARE_SOIL_NDVI = 0.15

anchor_start = pd.DataFrame({'DAP': [0],           'ndvi': [BARE_SOIL_NDVI]})
anchor_end   = pd.DataFrame({'DAP': [harvest_dap], 'ndvi': [BARE_SOIL_NDVI]})

# 既存データにアンカーを追加（既に同じDAPの観測がある場合はアンカーを優先しない）
if 0 not in interp_src['DAP'].values:
    interp_src = pd.concat([anchor_start, interp_src], ignore_index=True)
if harvest_dap not in interp_src['DAP'].values:
    interp_src = pd.concat([interp_src, anchor_end], ignore_index=True)

interp_src = interp_src.sort_values('DAP').reset_index(drop=True)
n_points = len(interp_src)

print(f'アンカー追加後: {n_points} 点（播種日={0 in interp_src["DAP"].values}, '
      f'収穫日={harvest_dap in interp_src["DAP"].values}）')

if n_points < 2:
    raise ValueError(
        f'補間に必要な最低2点のデータが得られません。'
        f'（現在 {n_points} 点） -> 学習データからの除外を推奨'
    )

# ── ステップ3：補間処理（日次データの生成）─────────────────────
# 毎日（1日刻み）のDAP（播種日=0 〜 収穫日）
dap_daily = np.arange(0, harvest_dap + 1, 1)

if n_points >= 4:
    # 4点以上なら3次スプライン補間
    cs = CubicSpline(
        interp_src['DAP'].values,
        interp_src['ndvi'].values,
        extrapolate=False
    )
    ndvi_interp = cs(dap_daily)
    interp_method = '3次スプライン補間'
else:
    # 2〜3点なら線形補間
    ndvi_interp = np.interp(
        dap_daily,
        interp_src['DAP'].values,
        interp_src['ndvi'].values
    )
    interp_method = '線形補間'

# 物理的にあり得ない値をクリップ（NDVI は -1 〜 1 の範囲）
ndvi_interp = np.clip(ndvi_interp, -1.0, 1.0)

print(f'{interp_method}完了: {len(dap_daily)} 日分のデータを生成 (実観測: {len(interp_df)} 点)')

# ── ステップ3：SGフィルタで平滑化 ───────────────────────────────
# window_length がデータ数より大きい場合は自動調整
win = min(SG_WINDOW, len(ndvi_interp))
if win % 2 == 0:
    win -= 1   # 奇数に調整

ndvi_sg = savgol_filter(
    ndvi_interp,
    window_length=win,
    polyorder=SG_POLYORDER,
    mode='nearest'   # 端点の処理方法（mirror / nearest / wrap）
)
ndvi_sg = np.clip(ndvi_sg, -1.0, 1.0)

print(f'SGフィルタ完了: 窓幅={win}日（指定: {SG_WINDOW}日）')

# ── 結果をDataFrameにまとめる ────────────────────────────────────
smooth_df = pd.DataFrame({
    'DAP'          : dap_daily,
    'date'         : [seed_date + pd.Timedelta(days=int(d)) for d in dap_daily],
    'ndvi_interp'  : ndvi_interp,
    'ndvi_sg'      : ndvi_sg,
})

print()
print('=== 平滑化後データのサマリー ===')
print(f'  最大NDVI (Peak)  : {ndvi_sg.max():.3f}  (DAP={dap_daily[ndvi_sg.argmax()]}日)')
print(f'  平均NDVI         : {ndvi_sg.mean():.3f}')
print(f'  AUC（曲線の積分）: {np.trapz(ndvi_sg, dap_daily):.1f}')


# =============================================================================
# [Cell 4] 比較グラフの描画
# =============================================================================
import os
os.makedirs('../../outputs', exist_ok=True)

fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=False)

# ── 上段：カレンダー日付軸（生データ vs 補間 vs SG）──────────────
ax1 = axes[0]

# ① 生の観測点（Landsat / Sentinel-2 で色分け）
COLOR_MAP = {'L30': '#e67e22', 'S30': '#2ecc71'}
for src, grp in raw_df.groupby('source'):
    # 播種〜収穫期間外は薄く表示
    in_range = grp[(grp['DAP'] >= dap_min) & (grp['DAP'] <= dap_max)]
    out_range = grp[~grp.index.isin(in_range.index)]
    ax1.scatter(in_range['date'],  in_range['ndvi'],
                color=COLOR_MAP.get(src, 'gray'), s=60, zorder=4,
                edgecolors='black', linewidths=0.5, label=f'{src}（使用）')
    if not out_range.empty:
        ax1.scatter(out_range['date'], out_range['ndvi'],
                    color=COLOR_MAP.get(src, 'gray'), s=40, zorder=3,
                    edgecolors='black', linewidths=0.5, alpha=0.3,
                    label=f'{src}（範囲外・除外）')

# ② 補間曲線
ax1.plot(smooth_df['date'], smooth_df['ndvi_interp'],
         color='steelblue', lw=1.2, alpha=0.6, linestyle='-', zorder=2,
         label=interp_method)

# ③ SG平滑化曲線
ax1.plot(smooth_df['date'], smooth_df['ndvi_sg'],
         color='crimson', lw=2.2, zorder=3, label=f'SGフィルタ (w={win})')

# 播種日・収穫日の縦線
ax1.axvline(seed_date,    color='green', lw=1.5, linestyle='--', alpha=0.7, label='播種日')
ax1.axvline(harvest_date, color='brown', lw=1.5, linestyle='--', alpha=0.7, label='収穫日')

# ピーク位置にアノテーション
peak_date = smooth_df.loc[smooth_df['ndvi_sg'].idxmax(), 'date']
peak_val  = smooth_df['ndvi_sg'].max()
ax1.annotate(f'Peak={peak_val:.3f}\n{peak_date.strftime("%m/%d")}',
             xy=(peak_date, peak_val),
             xytext=(peak_date + pd.Timedelta(days=7), peak_val - 0.05),
             arrowprops=dict(arrowstyle='->', color='black'),
             fontsize=9)

ax1.set_title(
    f'NDVI 生育曲線  [field_id={TARGET_FIELD_ID}, year={TARGET_YEAR}]  '
    f'実測収量: {actual_yield:.1f} kg/10a',
    fontsize=12, fontweight='bold'
)
ax1.set_ylabel('NDVI')
ax1.set_ylim(-0.15, 1.05)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left', fontsize=8, ncol=2)

# ── 下段：DAP軸（フェノロジー・アライメント後の形）───────────────
ax2 = axes[1]

ax2.scatter(interp_df['DAP'], interp_df['ndvi'],
            color='gray', s=55, zorder=4, edgecolors='black',
            linewidths=0.5, label='観測点（DAP基準）')
ax2.plot(smooth_df['DAP'], smooth_df['ndvi_interp'],
         color='steelblue', lw=1.2, alpha=0.6, label=interp_method)
ax2.plot(smooth_df['DAP'], smooth_df['ndvi_sg'],
         color='crimson', lw=2.2, label=f'SGフィルタ (w={win})')

# ピーク位置
peak_dap = dap_daily[ndvi_sg.argmax()]
ax2.axvline(peak_dap, color='crimson', lw=1.2, linestyle=':', alpha=0.7)
ax2.text(peak_dap + 1, 0.05, f'ピーク\nDAP={peak_dap}日',
         fontsize=8, color='crimson')

ax2.set_title('フェノロジー調整済み（播種日 = DAP0 基準）', fontsize=11)
ax2.set_xlabel('播種後日数（DAP: Days After Planting）')
ax2.set_ylabel('NDVI')
ax2.set_ylim(-0.15, 1.05)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper left', fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.97])
fig.suptitle(
    f'{interp_method} + SGフィルタによるNDVI平滑化  '
    f'[SGwindow={win}, polyorder={SG_POLYORDER}]',
    fontsize=10, color='gray'
)

save_path = f'../../outputs/ndvi_smooth_field{TARGET_FIELD_ID}_{TARGET_YEAR}.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()
print(f'グラフ保存: {save_path}')


# =============================================================================
# [Cell 5] 抽出された主要特徴量の確認
# =============================================================================
peak_idx    = ndvi_sg.argmax()
peak_ndvi   = ndvi_sg[peak_idx]
peak_dap    = int(dap_daily[peak_idx])

# 枯れ下がり速度（ピーク以降の平均的な低下速度: NDVI/日）
post_peak = ndvi_sg[peak_idx:]
if len(post_peak) > 1:
    senescence_rate = (post_peak[0] - post_peak[-1]) / (len(post_peak) - 1)
else:
    senescence_rate = 0.0

# AUC（曲線の積分値：台形近似）
auc = float(np.trapz(ndvi_sg, dap_daily))

# 子実肥大期相当（播種後 60〜90日）の平均NDVI
mask_r5r6 = (dap_daily >= 60) & (dap_daily <= 90)
ndvi_r5r6_mean = float(ndvi_sg[mask_r5r6].mean()) if mask_r5r6.any() else np.nan

print('=' * 45)
print(f'【抽出された特徴量一覧】 field_id={TARGET_FIELD_ID}, year={TARGET_YEAR}')
print('=' * 45)
print(f'  Peak NDVI              : {peak_ndvi:.4f}')
print(f'  ピーク到達日（DAP）     : {peak_dap} 日')
print(f'  AUC（積分値）           : {auc:.2f}')
print(f'  DAP60〜90の平均NDVI     : {ndvi_r5r6_mean:.4f}')
print(f'  枯れ下がり速度          : {senescence_rate:.5f} NDVI/日')
print(f'  使用データ点数          : {len(interp_df)} 点（播種〜収穫）')
print()
print(f'  実測収量                : {actual_yield:.1f} kg/10a')
