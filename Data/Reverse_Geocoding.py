"""
逆ジオコーティングをするためのコード
"""
import reverse_geocoder as rg
import pandas as pd
import multiprocessing # freeze_supportのために追加
import os

# --- 設定 ---
excel_filename = 'locations_complete.xlsx'
columns_to_read = ['place', 'year', 'lat', 'lon'] # 読み込みたい列を指定
# -------------


# --- ここからメイン処理 ---
# ▼▼▼ この if ブロックで囲む ▼▼▼
if __name__ == '__main__':
    
    # ファイルが存在するか確認
    if not os.path.exists(excel_filename):
        print(f"エラー: ファイル '{excel_filename}' が見つかりません。")
    else:
        try:
            # Excelファイルを読み込み、指定した列だけをDataFrameにする
            locations_df = pd.read_excel(excel_filename, usecols=columns_to_read)

            # 読み込んだデータの先頭を表示して確認
            print(f"'{excel_filename}' からデータを読み込みました。")
            print("--- DataFrame (先頭15行) ---")
            print(locations_df.head(15))

        except FileNotFoundError:
            print(f"エラー: ファイル '{excel_filename}' が見つかりません。")
        except ValueError as e:
            print(f"エラー: ファイル '{excel_filename}' 内に必要な列が見つからない可能性があります。")
            print(f"詳細: {e}")
        except Exception as e:
            print(f"Excelファイルの読み込み中に予期せぬエラーが発生しました: {e}")
    # ▼▼▼ freeze_support() を追加 ▼▼▼
    # Windowsで実行可能ファイル(.exe)にする場合などに必要ですが、
    # multiprocessingのエラーを防ぐおまじないとしても有効です。
    multiprocessing.freeze_support()
    
    if locations_df is not None:
        print(f"--- {len(locations_df)}地点の逆ジオコーディングを開始します ---")

        # 1. 緯度経度のリストを作成
        coordinates = list(zip(locations_df['lat'], locations_df['lon']))

        # 2. 逆ジオコーディングを実行
        results = rg.search(coordinates) # この行が if ブロックの中に入る

        # 3. 結果をDataFrameに追加
        locations_df['city'] = [result['name'] for result in results]
        locations_df['prefecture'] = [result['admin1'] for result in results]

        # --- 4. 結果の表示 ---
        print("\n--- 逆ジオコーディング結果 ---")
        print(locations_df) # head(15) を削除して全件表示（もし必要なら）

        # --- 5. (任意) 結果をExcelファイルに保存 ---
        output_excel_filename = 'locations_with_geocode.xlsx'
        try:
            locations_df.to_excel(output_excel_filename, index=False)
            print(f"\n逆ジオコーディング結果を '{output_excel_filename}' に保存しました。")
        except Exception as e:
            print(f"\nExcelファイルへの保存中にエラーが発生しました: {e}")

    else:
        print("locations_df が見つかりません。")
# ▲▲▲ ここまで if ブロック ▲▲▲