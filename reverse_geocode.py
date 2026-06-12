"""
reverse_geocode.py
==================
FieldData_fieldid.db の field_master テーブルから lat/lon を読み込み、
国土地理院 逆ジオコーディングAPI で都道府県・市区町村・町名を取得する。

出力:
  outputs/reverse_geocode/field_addresses.csv         -- 全農地の住所情報
  outputs/reverse_geocode/field_addresses_summary.csv -- 都道府県別集計

API:
  https://mreversegeocoder.gsi.go.jp/reverse-geocoder/LonLatToAddress
  (無料・登録不要・負荷軽減のため 0.6秒/リクエスト)

市区町村コードマスター:
  https://maps.gsi.go.jp/js/muni.js  (国土地理院)
"""

import os
import re
import sys
import time
import sqlite3
import requests
import pandas as pd
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# 設定
# ─────────────────────────────────────────────────────────────────────────────
DB_PATH  = os.path.join(os.path.dirname(__file__), "data", "processed", "FieldData_fieldid.db")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "outputs", "reverse_geocode")
OUT_CSV  = os.path.join(OUT_DIR, "field_addresses.csv")
SUM_CSV  = os.path.join(OUT_DIR, "field_addresses_summary.csv")
os.makedirs(OUT_DIR, exist_ok=True)

GSI_URL      = "https://mreversegeocoder.gsi.go.jp/reverse-geocoder/LonLatToAddress"
MUNI_JS_URL  = "https://maps.gsi.go.jp/js/muni.js"
INTERVAL_S   = 0.6   # リクエスト間隔（秒）
RETRY_MAX    = 3     # 失敗時リトライ回数
RETRY_WAIT   = 3.0   # リトライ待機（秒）


# ─────────────────────────────────────────────────────────────────────────────
# 市区町村コードマスター取得
# ─────────────────────────────────────────────────────────────────────────────
def load_muni_master() -> dict:
    """
    国土地理院の muni.js から市区町村コード辞書を構築する。

    Returns:
        dict: {muniCd(str) -> {"prefecture": ..., "city": ...}}
        APIは先頭ゼロ埋め5桁（例 "02210"）、
        muni.jsは先頭ゼロなし4桁（例 "2210" -> APIの "02210" に対応）
    """
    print("[INFO] 市区町村コードマスターを取得中...")
    r = requests.get(MUNI_JS_URL, timeout=15)
    r.encoding = "utf-8"

    # GSI.MUNI_ARRAY["XXXX"] = 'N,都道府県,コード,市区町村名'; を抽出
    pattern = re.compile(r'GSI\.MUNI_ARRAY\["(\d+)"\]\s*=\s*\'([^\']+)\'')
    muni = {}
    for m in pattern.finditer(r.text):
        key  = m.group(1)   # 4桁キー（先頭ゼロなし）
        val  = m.group(2)   # "N,都道府県,コード,市区町村名"
        parts = val.split(",", 3)
        if len(parts) >= 4:
            pref = parts[1].strip()
            city = parts[3].strip()
        elif len(parts) == 3:
            pref = parts[1].strip()
            city = parts[1].strip()  # 政令市などコード直下
        else:
            continue

        # APIのmuniCdは5桁（先頭ゼロ埋め）なので 0埋め5桁キーも登録
        muni[key]            = {"prefecture": pref, "city": city}
        muni[key.zfill(5)]   = {"prefecture": pref, "city": city}

    print(f"[INFO] 市区町村コード: {len(muni)//2} 件読み込み完了")
    return muni


# ─────────────────────────────────────────────────────────────────────────────
# データ読み込み
# ─────────────────────────────────────────────────────────────────────────────
def load_fields(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(
        "SELECT field_id, place, lat, lon FROM field_master ORDER BY field_id",
        conn
    )
    conn.close()
    print(f"[INFO] ユニーク農地数: {len(df)} 件")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 国土地理院 逆ジオコーディング
# ─────────────────────────────────────────────────────────────────────────────
def reverse_geocode_gsi(lat: float, lon: float, muni: dict) -> dict:
    """
    国土地理院 逆ジオコーディングAPIを呼び出し、住所情報を返す。

    Returns:
        dict: prefecture, city, town, muniCd, error
    """
    params = {"lat": lat, "lon": lon}

    for attempt in range(1, RETRY_MAX + 1):
        try:
            resp = requests.get(GSI_URL, params=params, timeout=10)
            resp.raise_for_status()
            resp.encoding = "utf-8"
            data = resp.json()

            results = data.get("results", {})
            if not results:
                return {"prefecture": "", "city": "", "town": "",
                        "muniCd": "", "error": "empty response"}

            muni_cd  = results.get("muniCd", "")
            town     = results.get("lv01Nm", "")  # 大字・町名

            # muniCdから都道府県・市区町村を引く
            muni_info = muni.get(muni_cd, muni.get(muni_cd.lstrip("0"), {}))
            prefecture = muni_info.get("prefecture", "")
            city       = muni_info.get("city", "")

            return {
                "prefecture": prefecture,
                "city":       city,
                "town":       town,
                "muniCd":     muni_cd,
                "error":      "",
            }

        except requests.exceptions.Timeout:
            err = f"Timeout (attempt {attempt}/{RETRY_MAX})"
        except requests.exceptions.RequestException as e:
            err = f"{type(e).__name__}: {e} (attempt {attempt}/{RETRY_MAX})"
        except Exception as e:
            err = f"UnexpectedError: {e} (attempt {attempt}/{RETRY_MAX})"

        print(f"  [WARN] {err}", file=sys.stderr)
        if attempt < RETRY_MAX:
            time.sleep(RETRY_WAIT)

    return {"prefecture": "", "city": "", "town": "",
            "muniCd": "", "error": err}


# ─────────────────────────────────────────────────────────────────────────────
# バッチ処理（再開対応）
# ─────────────────────────────────────────────────────────────────────────────
def run_geocoding(fields: pd.DataFrame, muni: dict) -> pd.DataFrame:
    """
    全農地に対して逆ジオコーディングを実行する。
    途中で止まっても既存CSVがあればスキップして再開できる。
    """
    # 既存結果を読み込み
    done_ids = set()
    existing_rows = []
    if os.path.exists(OUT_CSV):
        existing = pd.read_csv(OUT_CSV, dtype={"muniCd": str})
        done_ids = set(existing["field_id"].tolist())
        existing_rows = existing.to_dict("records")
        print(f"[INFO] 既存結果 {len(done_ids)} 件をスキップします（再開モード）")

    todo  = fields[~fields["field_id"].isin(done_ids)]
    total = len(todo)
    print(f"[INFO] 残り {total} 件を処理します\n")

    results    = list(existing_rows)
    start_time = time.time()

    for i, (_, row) in enumerate(todo.iterrows(), 1):
        geo = reverse_geocode_gsi(row["lat"], row["lon"], muni)

        result = {
            "field_id":   int(row["field_id"]),
            "place":      row["place"],
            "lat":        row["lat"],
            "lon":        row["lon"],
            "prefecture": geo["prefecture"],
            "city":       geo["city"],
            "town":       geo["town"],
            "muniCd":     geo["muniCd"],
            "error":      geo["error"],
        }
        results.append(result)

        # 進捗表示（文字化け防止のためバイト出力）
        elapsed   = time.time() - start_time
        per_item  = elapsed / i
        remaining = per_item * (total - i)
        line = (
            f"  [{i:4d}/{total}] "
            f"field_id={result['field_id']:4d}  "
            f"{result['prefecture']} {result['city']} {result['town']}  "
            f"(残り約 {remaining/60:.1f} 分)"
        )
        sys.stdout.buffer.write((line + "\n").encode("utf-8", errors="replace"))
        sys.stdout.flush()

        # 50件ごとに中間保存
        if i % 50 == 0:
            _save_csv(results, OUT_CSV)
            sys.stdout.buffer.write(
                f"  [SAVE] 中間保存 ({i} 件完了)\n".encode("utf-8"))
            sys.stdout.flush()

        time.sleep(INTERVAL_S)

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# 保存・集計
# ─────────────────────────────────────────────────────────────────────────────
def _save_csv(records: list, path: str):
    pd.DataFrame(records).to_csv(path, index=False, encoding="utf-8-sig")


def save_results(df: pd.DataFrame):
    """メイン結果 CSV と都道府県別集計 CSV を保存する。"""
    # ① 全件 CSV（UTF-8 BOM付き → Excelで開ける）
    df_out = df.drop(columns=["error"], errors="ignore")
    df_out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    sys.stdout.buffer.write(f"\n[SAVED] 全件: {OUT_CSV}\n".encode("utf-8"))

    # ② 都道府県別集計
    summary = (
        df.groupby("prefecture", dropna=False)
          .agg(
              field_count=("field_id", "count"),
              city_list=("city", lambda s: "・".join(
                  sorted(set(v for v in s if v))))
          )
          .reset_index()
          .sort_values("field_count", ascending=False)
    )
    summary.to_csv(SUM_CSV, index=False, encoding="utf-8-sig")
    sys.stdout.buffer.write(f"[SAVED] 集計: {SUM_CSV}\n".encode("utf-8"))

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────────────────────────────────
def main():
    header = (
        "=" * 60 + "\n"
        "  国土地理院 逆ジオコーディング\n"
        f"  開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        "=" * 60 + "\n"
    )
    sys.stdout.buffer.write(header.encode("utf-8"))
    sys.stdout.flush()

    muni   = load_muni_master()
    fields = load_fields(DB_PATH)
    df     = run_geocoding(fields, muni)

    sys.stdout.buffer.write(
        "\n[INFO] 全件処理完了。結果を保存します...\n".encode("utf-8"))
    summary = save_results(df)

    # 集計表示
    sys.stdout.buffer.write("\n=== 都道府県別 農地数 ===\n".encode("utf-8"))
    for _, r in summary.iterrows():
        pref = r["prefecture"] if r["prefecture"] else "(取得失敗)"
        line = f"  {pref:<8} : {int(r['field_count']):3d} 件  [{r['city_list'][:60]}]\n"
        sys.stdout.buffer.write(line.encode("utf-8", errors="replace"))

    errors = df[df["error"] != ""]
    if not errors.empty:
        sys.stdout.buffer.write(
            f"\n[WARN] エラーが {len(errors)} 件ありました:\n".encode("utf-8"))
        sys.stdout.buffer.write(
            errors[["field_id", "lat", "lon", "error"]].to_string(index=False).encode("utf-8"))

    footer = (
        f"\n\n完了: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"出力先: {OUT_DIR}\n"
    )
    sys.stdout.buffer.write(footer.encode("utf-8"))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
