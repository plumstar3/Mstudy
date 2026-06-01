"""
長期気象データ一括取得スクリプト

使い方:
  - まずTEST_MODE=Trueで動作確認（最初の1地点のみ処理）
  - TEST_MODE=Falseで全地点を処理

依存: PythonWorks.AMD_Tools4 の GetMetData を利用

FieldData.db は読み込み専用で使用し、既存DBへ変更は加えません。
出力は新規SQLiteファイル `weather_database.db` と CSV `all_locations_weather_longterm.csv` に保存します。
"""
from __future__ import annotations
import os
import sys
import sqlite3
import time
from datetime import date
import pandas as pd
import numpy as np
from typing import List
import concurrent.futures

# プロジェクトルート（Mstudy/）を sys.path に追加
# このファイルは src/data/ にあるため、2階層上がプロジェクトルート
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from src.utils import amd_tools as amd

# テストモード: True の場合はユニーク地点リストの最初の1件のみ処理
TEST_MODE = True

# データベース/出力ファイル名（プロジェクトルートからの絶対パス）
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FIELD_DB = os.path.join(_ROOT, "data", "raw", "FieldData.db")
OUT_CSV  = os.path.join(_ROOT, "data", "processed", "all_locations_weather_longterm.csv")
OUT_DB   = os.path.join(_ROOT, "data", "raw", "weather_database.db")

# 期間の定義
START_A = "1981-01-01"  # グループA 開始
START_B = "2008-01-01"  # グループB 開始
END_DATE = date.today().isoformat()  # 取得終了日（今日）

# グループ定義
GROUP_A = [
    "TMP_mea", "TMP_max", "TMP_min", "OPR", "SSD", "GSR",
    "SD", "SWE", "SFW", "APCP"
]

GROUP_B = [
    "APCPRA", "DLR", "RH", "WIND"
]

# GPU オプション: True にすると cupy を使って配列操作を試みます（cupy 未導入時はフォールバック）
USE_GPU = False
GPU_AVAILABLE = False
try:
    if USE_GPU:
        import cupy as cp
        GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False



def read_unique_places(field_db: str) -> pd.DataFrame:
    """FieldData.db の Questionaire テーブルから (place, lat, lon) を取得し重複排除して返す。"""
    if not os.path.exists(field_db):
        raise FileNotFoundError(f"{field_db} が見つかりません。カレントディレクトリを確認してください。")
    conn = sqlite3.connect(field_db)
    df = pd.read_sql_query("SELECT place, lat, lon FROM Questionaire", conn)
    conn.close()
    df = df.dropna(subset=["lat", "lon"])  # 緯度経度が無い行は除外
    df = df.drop_duplicates(subset=["place", "lat", "lon"]).reset_index(drop=True)
    return df


def safe_get_metdata(code: str, itsu: List[str], doko: List[float]):
    """amd.GetMetData をラップしてエラーを吸収し、日付インデックスの DataFrame を返す。
    失敗した場合は空の DataFrame を返す。
    HTTP 502 / 503 などの一時的なエラーはリトライします。
    """
    max_retries = 3
    delay_seconds = 5
    for attempt in range(1, max_retries + 1):
        try:
            res = amd.GetMetData(code, itsu, doko, namuni=True)
            if res is None:
                return pd.DataFrame()
            # AMD_Tools4 の戻り値は (data, tim, lat, lon) や
            # (data, tim, lat, lon, name, unit) のタプルであることが多い
            if isinstance(res, (list, tuple)):
                data = res[0]
                tim = res[1] if len(res) > 1 else None
                if tim is None:
                    return pd.DataFrame()
                try:
                    dates = pd.to_datetime(tim)
                except Exception:
                    return pd.DataFrame()
                arr = np.asarray(data)
                # GPU が使える場合は cupy で squeeze 等を試みる（cuPy が無い場合は numpy を使用）
                try:
                    if GPU_AVAILABLE:
                        arr_cp = cp.asarray(arr)
                        arr_cp = cp.squeeze(arr_cp)
                        arr = cp.asnumpy(arr_cp)
                    else:
                        arr = np.squeeze(arr)
                except Exception:
                    try:
                        arr = np.squeeze(arr)
                    except Exception:
                        pass
                if arr.ndim == 0:
                    # スカラー
                    series = pd.Series([arr] * len(dates), index=dates, name=code)
                    return series.to_frame()
                if arr.shape[0] == len(dates):
                    series = pd.Series(arr, index=dates, name=code)
                    return series.to_frame()
                # もし最初の次元が異なる場合は失敗扱い
                return pd.DataFrame()
            # 期待: res が DataFrame または Series
            if isinstance(res, pd.Series):
                df = res.to_frame()
            elif isinstance(res, pd.DataFrame):
                df = res
            else:
                # 汎用的に DataFrame に変換を試みる
                df = pd.DataFrame(res)
            # 列名が複数ある場合は、列名を code に統一するケースは避ける。
            # ただし単一列なら名前を code にする。
            if df.shape[1] == 1:
                df.columns = [code]
            # インデックスを日付に整える
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception:
                    # もし日付列が 'date' という名で存在すればそれを使う
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        df = df.set_index('date')
                    else:
                        # 失敗したら空 DF
                        return pd.DataFrame()
            return df
        except Exception as e:
            msg = str(e)
            is_transient = "HTTP Error 502" in msg or "HTTP Error 503" in msg or "Bad Gateway" in msg or "Service Unavailable" in msg
            print(f"GetMetData error for {code} (attempt {attempt}/{max_retries}): {msg}")
            if attempt < max_retries and is_transient:
                print(f"  Transient error detected; retrying after {delay_seconds}s...")
                time.sleep(delay_seconds)
                delay_seconds *= 2
                continue
            return pd.DataFrame()


def fetch_group(dcodes: List[str], start: str, end: str, doko: List[float]) -> pd.DataFrame:
    """指定した要素群を並列に取得して横方向に結合（index=日付）。"""
    itsu = [start, end]
    results = {}
    max_workers = min(6, len(dcodes)) if len(dcodes) > 0 else 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_code = {}
        for code in dcodes:
            print(f"  -> fetching {code} ({start} to {end})")
            future = ex.submit(safe_get_metdata, code, itsu, doko)
            future_to_code[future] = code
        for fut in concurrent.futures.as_completed(future_to_code):
            code = future_to_code[fut]
            try:
                dfc = fut.result()
            except Exception as e:
                print(f"  Exception fetching {code}: {e}")
                dfc = pd.DataFrame()
            if dfc.empty:
                print(f"    (no data) {code}")
            else:
                results[code] = dfc

    if not results:
        return pd.DataFrame()
    # 結果を列方向に連結
    base = pd.concat([results[c] for c in results], axis=1)
    base = base.sort_index()
    return base


def process_all_places(field_db: str, test_mode: bool = True) -> pd.DataFrame:
    places = read_unique_places(field_db)
    print(f"Found {len(places)} unique places")
    out_list = []
    if test_mode and len(places) > 0:
        places = places.iloc[[0]]
        print("TEST_MODE active: processing only the first place")

    for idx, r in places.iterrows():
        place = r['place']
        lat = float(r['lat'])
        lon = float(r['lon'])
        print(f"Processing {idx+1}/{len(places)}: {place} ({lat},{lon})")
        # doko の形式: [lat, lat, lon, lon]
        doko = [lat, lat, lon, lon]

        # グループA を取得
        dfA = fetch_group(GROUP_A, START_A, END_DATE, doko)

        # グループB を取得（開始日を2008-01-01に固定）
        dfB = fetch_group(GROUP_B, START_B, END_DATE, doko)

        if dfA.empty and dfB.empty:
            print(f"  No data returned for {place}, skip.")
            continue

        # Outer join on index
        if dfA.empty:
            joined = dfB
        elif dfB.empty:
            joined = dfA
        else:
            joined = pd.merge(dfA, dfB, left_index=True, right_index=True, how='outer')

        joined = joined.sort_index()
        joined = joined.reset_index().rename(columns={'index': 'date'})
        # Ensure date column name
        if 'date' not in joined.columns:
            # try to find datetime-like column
            for c in joined.columns:
                if pd.api.types.is_datetime64_any_dtype(joined[c]):
                    joined = joined.rename(columns={c: 'date'})
                    break
        # Add distinguishing columns: place, lat, lon
        joined['place'] = place
        joined['lat'] = lat
        joined['lon'] = lon
        # Reorder: date, place, lat, lon, ...
        cols = ['date', 'place', 'lat', 'lon'] + [c for c in joined.columns if c not in ['date', 'place', 'lat', 'lon']]
        joined = joined.loc[:, cols]

        out_list.append(joined)

    if not out_list:
        return pd.DataFrame()
    all_df = pd.concat(out_list, ignore_index=True, sort=False)
    # Format date column as ISO string
    all_df['date'] = pd.to_datetime(all_df['date']).dt.strftime('%Y-%m-%d')
    return all_df


def save_outputs(df: pd.DataFrame, csv_path: str, db_path: str):
    if df.empty:
        print("No data to save.")
        return
    print(f"Saving CSV to {csv_path}")
    df.to_csv(csv_path, index=False)
    print(f"Saving SQLite DB to {db_path} (table: weather_data)")
    conn = sqlite3.connect(db_path)
    df.to_sql('weather_data', conn, if_exists='replace', index=False)
    conn.close()


def main():
    print(f"Longterm weather fetch: END_DATE={END_DATE} TEST_MODE={TEST_MODE}")
    all_df = process_all_places(FIELD_DB, test_mode=TEST_MODE)
    if all_df.empty:
        print("No weather data was fetched.")
        return
    save_outputs(all_df, OUT_CSV, OUT_DB)
    print("Done.")


if __name__ == '__main__':
    main()
