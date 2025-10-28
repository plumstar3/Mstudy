"""
weather_database.dbの蒸気圧行VP列に気象庁から取得したエクセルファイルの中身を書き込むためのコード。
"""

import pandas as pd
import sqlite3
import os

# --- 設定 ---
excel_filename = r'.\Input\VP\2018\Yamagata_2018.xlsx'
db_filename = '.\Input\weather_database.db'
table_name = 'weather_data'
new_column_name = 'VP'

# 更新対象としたい place と year の組み合わせリスト
targets = [('C01', 2018),
 ('C04', 2018),
 ('C06', 2018)
]


# Excel/CSV側の列名
date_column_excel = '年月日'
vp_column_excel = '平均蒸気圧(hPa)'

# DB側の列名
date_column_db = 'date'
place_column_db = 'place'
year_column_db = 'year'
# -------------

# --- 1. Excelファイルから蒸気圧データを読み込む ---
if not os.path.exists(excel_filename):
    print(f"エラー: Excelファイル '{excel_filename}' が見つかりません。")
else:
    df_vp = None
    try:
        df_vp = pd.read_excel(excel_filename, engine='openpyxl')
        print(f"'{excel_filename}' からデータを読み込みました。")
        df_vp = df_vp[[date_column_excel, vp_column_excel]].copy()
        df_vp.rename(columns={
            date_column_excel: date_column_db,
            vp_column_excel: new_column_name
        }, inplace=True)
        df_vp[date_column_db] = pd.to_datetime(df_vp[date_column_db]).dt.strftime('%Y-%m-%d')
        print("蒸気圧データの準備完了 (先頭5行):")
        print(df_vp.head())

    except Exception as e:
        print(f"Excelファイルの読み込みまたは処理中にエラーが発生しました: {e}")

    # --- 2. データベース操作 ---
    if df_vp is not None:
        conn = None
        try:
            conn = sqlite3.connect(db_filename)
            cursor = conn.cursor()
            print(f"\nデータベース '{db_filename}' に接続しました。")

            try:
                cursor.execute(f'ALTER TABLE "{table_name}" ADD COLUMN "{new_column_name}" REAL;')
                print(f"テーブル '{table_name}' に '{new_column_name}' 列を追加しました。")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e): print(f"'{new_column_name}' 列は既に存在します。")
                else: raise

            total_update_count = 0
            for target_place, target_year in targets:
                print(f"\nデータベースの値を更新しています (対象: place='{target_place}', year={target_year})...")
                
                # --- ▼▼▼ ここからが変更点 ▼▼▼ ---
                # 1. target_year に基づいて動的な期間を定義
                start_date = f"{target_year}-04-01"
                end_date = f"{target_year + 1}-03-24"
                print(f"  -> 適用する期間: {start_date} から {end_date} まで")

                # 2. 読み込んだExcelデータ(df_vp)をその期間でフィルタリング
                df_vp_filtered = df_vp[(df_vp[date_column_db] >= start_date) & (df_vp[date_column_db] <= end_date)]
                
                if df_vp_filtered.empty:
                    print(f"  -> 指定された期間に該当する蒸気圧データがExcelファイル内にありませんでした。スキップします。")
                    continue
                # --- ▲▲▲ ここまでが変更点 ▲▲▲ ---

                update_count_per_target = 0
                # フィルタリングされたデータフレーム (df_vp_filtered) でループ
                for index, row in df_vp_filtered.iterrows():
                    update_query = f'''
                    UPDATE "{table_name}"
                    SET "{new_column_name}" = ?
                    WHERE "{place_column_db}" = ? AND "{year_column_db}" = ? AND "{date_column_db}" = ?;
                    '''
                    cursor.execute(update_query, (
                        row[new_column_name],
                        target_place,
                        target_year,
                        row[date_column_db]
                    ))
                    update_count_per_target += cursor.rowcount

                if update_count_per_target > 0:
                    print(f"  -> place='{target_place}', year={target_year} の条件に一致する {update_count_per_target} 件のデータを反映しました。")
                else:
                    print(f"  -> place='{target_place}', year={target_year} に一致するデータが見つからなかったか、日付が一致しませんでした。")
                total_update_count += update_count_per_target
            
            conn.commit()
            print(f"\n--- 全ての更新が完了しました ---")
            print(f"合計 {total_update_count} 件のデータが '{new_column_name}' 列に反映されました。")

        except sqlite3.Error as e:
            print(f"\nデータベース操作中にエラーが発生しました: {e}")
        finally:
            if conn:
                conn.close()
                print("\nデータベース接続を閉じました。")