"""
03_fetch_all_hls_ndvi.py
========================
全圃場のHLS（Harmonized Landsat Sentinel-2）NDVI時系列データを
一括取得し、ロングフォーマットのCSVに保存するスクリプト。

Jupyter Notebook で使う場合:
  各 [Cell N] のコードを、新しいセルに貼り付けてください。

出力ファイル:
  data/raw/hls_ndvi_timeseries.csv

スキーマ:
  field_id | year | date | ndvi | n_pixels | source

設計方針:
  - 5/1〜12/31 を取得期間とする（播種〜収穫をカバー）
  - チェックポイント保存: 1圃場ごとにCSVへ追記
    → 途中でエラーが起きても処理済みの圃場からやり直し不要
  - リトライ処理: GEEエラー時に最大3回、指数バックオフで再試行
  - tqdm による進行状況バー表示
"""

# =============================================================================
# [Cell 1] GEE 初期化
# =============================================================================
import ee

GCP_PROJECT = 'ndvi-505105'  # <- 自分の GCP プロジェクト ID に変更

ee.Initialize(project=GCP_PROJECT)
print('GEE 接続成功！')
print(f'  使用プロジェクト: {GCP_PROJECT}')


# =============================================================================
# [Cell 2] 圃場データの読み込み
# =============================================================================
import sqlite3
import pandas as pd

FIELD_DB = '../../data/processed/FieldData_fieldid.db'

conn = sqlite3.connect(FIELD_DB)
fields_df = pd.read_sql("""
    SELECT field_id, year, lat, lon
    FROM Questionaire
    WHERE lat IS NOT NULL AND lon IS NOT NULL AND yield IS NOT NULL
    ORDER BY field_id, year
""", conn)
conn.close()

fields_df['field_id'] = fields_df['field_id'].astype(int)
fields_df['year']     = fields_df['year'].astype(int)
fields_df['lat']      = pd.to_numeric(fields_df['lat'], errors='coerce')
fields_df['lon']      = pd.to_numeric(fields_df['lon'], errors='coerce')
fields_df = fields_df.dropna(subset=['lat', 'lon'])

print(f'取得対象: {len(fields_df)} 件（圃場×年の組み合わせ）')
print(f'圃場数: {fields_df["field_id"].nunique()} 圃場')
print(f'対象年: {sorted(fields_df["year"].unique().tolist())}')
fields_df.head()


# =============================================================================
# [Cell 3] HLS NDVI 取得関数（02_hls_ndvi.py と同じ）
#
# 取得期間: 5/1〜12/31（播種〜収穫をカバー）
# 雲マスク: Fmask（Bit1:雲, Bit2:雲の影, Bit3:隣接, Bit4:雪, Bit5:水）
# 解像度  : 30m
# バッファ : 100m
# =============================================================================
import ee
import pandas as pd

def get_ndvi_timeseries_hls(lat, lon, year, buffer_m=100,
                             apply_cloud_mask=True, verbose=False):
    """
    HLS（Harmonized Landsat Sentinel-2）からNDVI時系列を取得する。

    Args:
        lat             (float): 緯度
        lon             (float): 経度
        year            (int)  : 対象年
        buffer_m        (int)  : バッファ半径[m]（デフォルト100m）
        apply_cloud_mask (bool): Fmask 雲マスクを適用するか
        verbose         (bool) : デバッグ情報を表示するか

    Returns:
        pd.DataFrame: date, ndvi, n_pixels, source 列
                      取得できない場合は空の DataFrame
    """
    point = ee.Geometry.Point([lon, lat])
    aoi   = point.buffer(buffer_m)

    # 取得期間: 5/1〜12/31
    start_date = f'{year}-05-01'
    end_date   = f'{year}-12-31'

    # ── Fmask 雲マスク ───────────────────────────────────────
    CLOUD_BITMASK = (1 << 1) | (1 << 2) | (1 << 3)

    def mask_hls_clouds(img):
        fmask = img.select('Fmask')
        mask  = fmask.bitwiseAnd(CLOUD_BITMASK).eq(0)
        return img.updateMask(mask)

    # ── HLSL30: Landsat 8/9（赤=B4, 近赤外=B5）──────────────
    def add_ndvi_l30(img):
        ndvi = img.normalizedDifference(['B5', 'B4']).rename('NDVI')
        return img.addBands(ndvi)

    l30_raw = (
        ee.ImageCollection('NASA/HLS/HLSL30/v002')
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
    )
    l30 = (l30_raw.map(mask_hls_clouds) if apply_cloud_mask else l30_raw)
    l30 = l30.map(add_ndvi_l30).select('NDVI').map(
        lambda img: img.set('source', 'L30')
    )

    # ── HLSS30: Sentinel-2（赤=B4, 近赤外=B8A）──────────────
    def add_ndvi_s30(img):
        ndvi = img.normalizedDifference(['B8A', 'B4']).rename('NDVI')
        return img.addBands(ndvi)

    s30_raw = (
        ee.ImageCollection('NASA/HLS/HLSS30/v002')
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
    )
    s30 = (s30_raw.map(mask_hls_clouds) if apply_cloud_mask else s30_raw)
    s30 = s30.map(add_ndvi_s30).select('NDVI').map(
        lambda img: img.set('source', 'S30')
    )

    # ── 2つのコレクションをマージ ────────────────────────────
    hls_merged = l30.merge(s30)

    # ── 各画像から NDVI 平均・ピクセル数を抽出 ───────────────
    def extract_value(img):
        stats = img.reduceRegion(
            reducer=ee.Reducer.mean().combine(
                ee.Reducer.count(), sharedInputs=True
            ),
            geometry=aoi,
            scale=30,
            bestEffort=True,
        )
        return ee.Feature(None, {
            'date':     ee.Date(img.get('system:time_start')).format('YYYY-MM-dd'),
            'NDVI':     stats.get('NDVI_mean'),
            'n_pixels': stats.get('NDVI_count'),
            'source':   img.get('source'),
        })

    fc = hls_merged.map(extract_value).getInfo()

    rows = []
    for feat in fc['features']:
        p = feat['properties']
        rows.append({
            'date':     p.get('date'),
            'ndvi':     p.get('NDVI'),
            'n_pixels': p.get('n_pixels'),
            'source':   p.get('source'),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df['date']     = pd.to_datetime(df['date'])
    df['ndvi']     = pd.to_numeric(df['ndvi'],     errors='coerce')
    df['n_pixels'] = pd.to_numeric(df['n_pixels'], errors='coerce')

    # 有効ピクセルが 1 以上の行のみ残す
    df = df[df['n_pixels'] > 0]

    if verbose:
        print(f'    有効シーン数: {len(df.dropna(subset=["ndvi"]))}')

    return df.dropna(subset=['ndvi']).sort_values('date').reset_index(drop=True)


print('NDVI取得関数を定義しました。')


# =============================================================================
# [Cell 4] 一括取得メイン処理
#
# 設計:
#   - OUTPUT_CSV に既存データがある場合、処理済みの field_id×year は
#     スキップして未処理分から再開する（チェックポイント）
#   - GEEエラー時は最大 MAX_RETRY 回リトライ（指数バックオフ）
#   - 1圃場ごとに CSV へ追記保存
# =============================================================================
import os
import time
from tqdm.auto import tqdm

OUTPUT_CSV  = '../../data/raw/hls_ndvi_timeseries.csv'
BUFFER_M    = 100
MAX_RETRY   = 3      # リトライ最大回数
RETRY_WAIT  = 10     # 初回リトライ待機秒数（指数バックオフ: 10, 20, 40秒）

os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_CSV)), exist_ok=True)

# ── チェックポイント: 処理済みの (field_id, year) を確認 ─────
done_pairs = set()
if os.path.exists(OUTPUT_CSV):
    existing = pd.read_csv(OUTPUT_CSV, usecols=['field_id', 'year'])
    done_pairs = set(zip(existing['field_id'], existing['year']))
    print(f'既存データを検出: {len(done_pairs)} 件の (field_id, year) が取得済み')
else:
    print('新規取得を開始します。')

# ── 未処理の圃場×年を抽出 ──────────────────────────────────
todo_df = fields_df[
    ~fields_df.apply(lambda r: (int(r['field_id']), int(r['year'])) in done_pairs, axis=1)
].reset_index(drop=True)

print(f'取得対象（未処理）: {len(todo_df)} 件 / 全体 {len(fields_df)} 件')
print()

# ── メインループ ────────────────────────────────────────────
success_count = 0
skip_count    = 0
error_log     = []  # エラーが出た圃場の記録

for idx, row in tqdm(todo_df.iterrows(), total=len(todo_df),
                     desc='NDVI取得', unit='圃場'):
    fid  = int(row['field_id'])
    year = int(row['year'])
    lat  = float(row['lat'])
    lon  = float(row['lon'])

    # リトライループ
    df_result = None
    for attempt in range(1, MAX_RETRY + 1):
        try:
            df_result = get_ndvi_timeseries_hls(
                lat, lon, year,
                buffer_m=BUFFER_M,
                apply_cloud_mask=True,
                verbose=False
            )
            break  # 成功したらリトライループを抜ける

        except Exception as e:
            wait_sec = RETRY_WAIT * (2 ** (attempt - 1))  # 10, 20, 40秒
            if attempt < MAX_RETRY:
                tqdm.write(
                    f'  [RETRY {attempt}/{MAX_RETRY}] field_id={fid}, year={year} '
                    f'エラー: {str(e)[:80]}... {wait_sec}秒後に再試行'
                )
                time.sleep(wait_sec)
            else:
                tqdm.write(
                    f'  [ERROR] field_id={fid}, year={year} '
                    f'最大リトライ超過: {str(e)[:80]}'
                )
                error_log.append({'field_id': fid, 'year': year, 'error': str(e)})

    # 取得できた場合のみ保存
    if df_result is not None and not df_result.empty:
        df_result.insert(0, 'field_id', fid)
        df_result.insert(1, 'year', year)

        # ヘッダーはファイルが存在しない初回のみ書き出す
        write_header = not os.path.exists(OUTPUT_CSV)
        df_result.to_csv(OUTPUT_CSV, mode='a', header=write_header, index=False)
        success_count += 1
    else:
        skip_count += 1
        tqdm.write(f'  [SKIP] field_id={fid}, year={year}: データなし（0件）')

print()
print('=' * 50)
print(f'処理完了')
print(f'  成功: {success_count} 件')
print(f'  データなし: {skip_count} 件')
print(f'  エラー: {len(error_log)} 件')
print(f'  保存先: {os.path.abspath(OUTPUT_CSV)}')

if error_log:
    print()
    print('--- エラーが発生した圃場 ---')
    for e in error_log:
        print(f'  field_id={e["field_id"]}, year={e["year"]}: {e["error"][:80]}')


# =============================================================================
# [Cell 5] 取得結果の確認
# =============================================================================
import pandas as pd

OUTPUT_CSV = '../../data/raw/hls_ndvi_timeseries.csv'

df = pd.read_csv(OUTPUT_CSV, parse_dates=['date'])

print('=== 取得結果サマリー ===')
print(f'総行数 (観測数合計)   : {len(df):,} 行')
print(f'圃場×年の組み合わせ数: {df.groupby(["field_id","year"]).ngroups:,} 件')
print(f'圃場数               : {df["field_id"].nunique():,} 圃場')
print(f'年                   : {sorted(df["year"].unique().tolist())}')
print(f'取得期間             : {df["date"].min().date()} 〜 {df["date"].max().date()}')
print()
print('--- NDVI 基本統計 ---')
print(df['ndvi'].describe().round(3).to_string())
print()
print('--- ソース別件数 ---')
print(df['source'].value_counts().to_string())
print()
print('--- 先頭10行 ---')
print(df.head(10).to_string())
