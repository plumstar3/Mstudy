"""
weather_data_weekly_with_yield.xlsxからCRNNに必要となる要素のみを抜き出し、
列名を変更して新しいCSVファイルを作成する。
"""

import pandas as pd
import os
import sys

# --- 設定 ---
input_filename = './Output/weather_data_weekly_with_yield.xlsx'
output_filename = './Input/soybean_data_new.csv'

# 1. 常に保持する列
base_columns = ['place', 'year', 'yield']

# 2. 抽出したい気象データのプレフィックスと、新しいプレフィックスへの対応
# キー: 元のプレフィックス, 値: 新しいプレフィックス
prefix_mapping = {
    'W_4_': 'W_1_',
    'W_7_': 'W_2_',
    'W_12_': 'W_3_',
    'W_2_': 'W_4_',
    'W_3_': 'W_5_',
    'W_14_': 'W_6_'
}
# ----------------

print(f"現在の作業ディレクトリ: {os.getcwd()}")
print(f"'{input_filename}' の読み込みを開始します...")

if not os.path.exists(input_filename):
    print(f"エラー: 入力ファイル '{input_filename}' が見つかりません。パスを確認してください。")
    sys.exit(1) # エラー終了

try:
    # Excelファイルを読み込む
    df = pd.read_excel(input_filename, engine='openpyxl')
    print("Excelファイルの読み込みに成功しました。")
    
    # 元のすべての列名を取得
    all_columns = df.columns.tolist()
    print(f"元の列数: {len(all_columns)}")
    
    # 最終的に保持する列のリストと、列名変更用の辞書を作成
    columns_to_keep = []
    rename_dict = {}
    
    # base_columns の処理
    for col in base_columns:
        if col in all_columns:
            columns_to_keep.append(col)
        else:
            print(f"警告: 基本列 '{col}' が入力ファイルに見つかりません。")

    # 気象データ列の処理
    for old_prefix, new_prefix in prefix_mapping.items():
        # このプレフィックスを持つ列を探す (例: 'W_2_' で始まる列)
        found_cols = [col for col in all_columns if col.startswith(old_prefix)]
        print(f"プレフィックス '{old_prefix}' に一致する列数: {len(found_cols)}")
        
        for col in found_cols:
            columns_to_keep.append(col)
            # 新しい列名を作成 (例: W_2_1 -> W_1_1)
            # replaceの第3引数 1 は、最初の1回だけ置換することを意味します
            new_col_name = col.replace(old_prefix, new_prefix, 1)
            rename_dict[col] = new_col_name

    # 抽出
    df_filtered = df[columns_to_keep].copy()
    
    # 列名の変更
    df_filtered.rename(columns=rename_dict, inplace=True)
    
    print(f"列の抽出と名前変更が完了しました。")
    print(f"最終的な列数: {len(df_filtered.columns)}")
    
    # 出力ディレクトリが存在しない場合は作成
    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"ディレクトリ '{output_dir}' を作成しました。")

    # CSVファイルとして保存
    df_filtered.to_csv(output_filename, index=False, encoding='utf-8')
    
    print(f"--- 成功 ---")
    print(f"処理が完了しました。'{output_filename}' として保存されました。")

except Exception as e:
    print(f"エラーが発生しました: {e}")
    import traceback
    traceback.print_exc()