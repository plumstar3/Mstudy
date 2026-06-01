"""
.dbファイルの欠損値を確認するためのコード    """
import pandas as pd
import sqlite3
import os

# --- 設定 ---
db_filename = '.\Input\weather_database.db'
table_name = 'weather_data'
# -------------

# --- データベース接続とデータ読み込み ---
conn = None
if not os.path.exists(db_filename):
    print(f"エラー: データベースファイル '{db_filename}' が見つかりません。")
else:
    try:
        conn = sqlite3.connect(db_filename)
        print(f"データベース '{db_filename}' に接続しました。")

        # テーブル全体をDataFrameに読み込む
        query = f"SELECT * FROM {table_name}"
        df_db = pd.read_sql_query(query, conn)

        print(f"テーブル '{table_name}' を読み込みました ({len(df_db)} 行)。")

        # --- 欠損値 (NULL/NaN) の確認 ---
        print("\n--- 欠損値チェック ---")

        # 各列に含まれる欠損値の数をカウント
        nan_counts = df_db.isnull().sum()

        print("[各列の欠損値の数]")
        print(nan_counts)

        total_nans = nan_counts.sum()
        print(f"\nテーブル全体の欠損値の合計: {total_nans}")

        if total_nans == 0:
            print("-> テーブルに欠損値 (NULL) はありませんでした。👍")
        else:
            print("-> テーブルに欠損値 (NULL) が含まれています。")
            # 必要であれば、欠損値のある行を表示
            # print("\n[欠損値を含む行 (一部)]")
            # print(df_db[df_db.isnull().any(axis=1)].head())

    except sqlite3.Error as e:
        print(f"\nデータベース操作中にエラーが発生しました: {e}")
    except Exception as e:
        print(f"\n予期せぬエラーが発生しました: {e}")
    finally:
        if conn:
            conn.close()
            print("\nデータベース接続を閉じました。")