import pandas as pd
import sqlite3
import os

def load_weather_data(db_filename='./Input/weather_database.db', table_name='weather_data'):
    """weather_database.dbから日別気象データを読み込む"""
    print(f"データベース '{db_filename}' から気象データを読み込んでいます...")
    if not os.path.exists(db_filename):
        print(f"エラー: データベースファイル '{db_filename}' が見つかりません。")
        return None
    
    conn = None
    try:
        conn = sqlite3.connect(db_filename)
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql_query(query, conn)
        print(f"気象データの読み込み完了。{len(df)} 行取得しました。")
        return df
    except Exception as e:
        print(f"気象データの読み込み中にエラーが発生しました: {e}")
        return None
    finally:
        if conn:
            conn.close()

def load_yield_data(db_filename='./Input/FieldData.db', table_name='Questionaire'):
    """FieldData.dbのQuestionaireテーブルからyieldデータを読み込む"""
    print(f"データベース '{db_filename}' から収量データを読み込んでいます...")
    if not os.path.exists(db_filename):
        print(f"エラー: データベースファイル '{db_filename}' が見つかりません。")
        return None
    
    conn = None
    try:
        conn = sqlite3.connect(db_filename)
        # place, year, yield 列のみを読み込み、欠損値を持つ行は除外
        query = "SELECT place, year, yield FROM Questionaire"
        df = pd.read_sql_query(query, conn)
        df.dropna(subset=['place', 'year', 'yield'], inplace=True)
        # yearを整数型に変換
        df['year'] = pd.to_numeric(df['year']).astype(int)
        print(f"収量データの読み込み完了。{len(df)} 行取得しました。")
        return df
    except Exception as e:
        print(f"収量データの読み込み中にエラーが発生しました: {e}")
        return None
    finally:
        if conn:
            conn.close()

def process_weekly_averages(df):
    """
    日別データのDataFrameを週平均データに変換する。
    1行が1地点・1年分となる。
    """
    if df is None:
        return None
        
    print("日別データを週平均データに加工しています...")
    
    try:
        df['date'] = pd.to_datetime(df['date'])
    except Exception as e:
        print(f"日付列の変換に失敗しました: {e}")
        return None
        
    metric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if 'year' in metric_cols:
        metric_cols.remove('year')
    
    print(f"処理対象の気象要素: {metric_cols}")

    processed_rows = [] 

    # groupby実行時にソートしないように sort=False を指定
    for (place, year), group in df.groupby(['place', 'year'], sort=False):
        
        row_data = {'place': place, 'year': int(year)}
        
        start_date = pd.to_datetime(f"{int(year)}-04-01")
        end_date = pd.to_datetime(f"{int(year) + 1}-03-24")
        
        group_filtered = group[(group['date'] >= start_date) & (group['date'] <= end_date)].set_index('date').sort_index()

        if group_filtered.empty:
            print(f"  -> スキップ: {place} ({year}) は対象期間のデータがありません。")
            continue
            
        for m_idx, metric_name in enumerate(metric_cols, start=1):
            if metric_name not in group_filtered.columns:
                continue 
                
            weekly_avg = group_filtered[metric_name].resample('7D').mean()
            
            weekly_avg = weekly_avg.iloc[:52] # 52週に丸める
            
            for w_idx, value in enumerate(weekly_avg, start=1):
                col_name = f"W_{m_idx}_{w_idx}"
                row_data[col_name] = value
        
        processed_rows.append(row_data)

    print("加工が完了しました。")
    return pd.DataFrame(processed_rows)

def save_to_excel(df, output_filename='./Output/weather_data_weekly_with_yield.xlsx'):
    """加工後のDataFrameをExcelファイルに保存する（ソートしない）"""
    if df is None or df.empty:
        print("保存するデータがありません。")
        return
        
    try:
        # 列の順序を制御: [place, year, yield, W_1_1, W_1_2, ...]
        cols = ['place', 'year']
        if 'yield' in df.columns:
            cols.append('yield')
        
        # 'place', 'year', 'yield' 以外の列を元の順序で取得
        other_cols = [col for col in df.columns if col not in cols]
        
        # 順序を結合
        df = df[cols + other_cols]
        
        df.to_excel(output_filename, index=False, engine='openpyxl')
        print(f"\n成功: データを '{output_filename}' に保存しました。")
    except Exception as e:
        print(f"\nエラー: Excelファイルへの保存中にエラーが発生しました: {e}")

# --- メイン実行ブロック ---
if __name__ == "__main__":
    
    # 1. 気象データの読み込み
    daily_data_df = load_weather_data(db_filename='./Input/weather_database.db')
    
    # 2. 収量データの読み込み (FieldData.db から)
    yield_df = load_yield_data(db_filename='./Input/FieldData.db') 

    # 3. 気象データの週平均加工
    weekly_data_df = process_weekly_averages(daily_data_df)
    
    if weekly_data_df is not None and yield_df is not None:
        
        # 4. 収量データを週平均データに結合 (placeとyearをキーに)
        print("週平均データに収量データを結合しています...")
        final_df = pd.merge(weekly_data_df, yield_df, on=['place', 'year'], how='left')
        
        # 5. W_14 (14番目の指標) の値を100倍する
        print("W_14 (14番目の指標、恐らくVP) の値を100倍に変換しています...")
        # m_idx=14 に対応する列 (W_14_1, W_14_2, ...) を探す
        w_14_cols = [col for col in final_df.columns if col.startswith('W_14_')]
        
        if w_14_cols:
            # .loc を使って該当列の値を安全に100倍する
            final_df.loc[:, w_14_cols] = final_df.loc[:, w_14_cols] * 100
            print(f"  -> {len(w_14_cols)} 個の列 (W_14_1 から W_14_52) を100倍しました。")
        else:
            print("  -> 警告: W_14_ で始まる列が見つかりませんでした。")
        
        # 6. データの保存
        print("\n--- 加工後のデータ (先頭5行、一部列) ---")
        cols_to_show = ['place', 'year', 'yield'] + w_14_cols[:3] # yieldとW_14の先頭3列を表示
        print(final_df[cols_to_show].head())
        
        save_to_excel(final_df, output_filename='./Output/weather_data_weekly_with_yield.xlsx')
    else:
        print("データの読み込みまたは加工に失敗したため、処理を中断しました。")