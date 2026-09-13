import sqlite3
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 日本語フォント設定（必要に応じて）
plt.rcParams['font.family'] = 'Meiryo'

DB_PATH = 'data/processed/FieldData_fieldid.db'

def main():
    if not os.path.exists(DB_PATH):
        print(f"Error: {DB_PATH} not found.")
        return

    conn = sqlite3.connect(DB_PATH)
    # 対象年度（2015〜2018）の有効な収量データを取得
    df = pd.read_sql('''
        SELECT yield
        FROM Questionaire
        WHERE yield IS NOT NULL
          AND year BETWEEN 2015 AND 2018
    ''', conn)
    conn.close()

    if df.empty:
        print("No data found.")
        return

    # プロットの作成
    plt.figure(figsize=(10, 6))
    # 収量は0未満にならないため、clip=(0, None) で0以下をカットする
    sns.kdeplot(data=df, x='yield', fill=True, color='forestgreen', alpha=0.5, linewidth=2, clip=(0, None))
    
    # 平均値などの統計線を引く
    mean_val = df['yield'].mean()
    median_val = df['yield'].median()
    
    plt.axvline(mean_val, color='red', linestyle='--', label=f'平均値: {mean_val:.1f} kg/10a')
    plt.axvline(median_val, color='blue', linestyle=':', label=f'中央値: {median_val:.1f} kg/10a')

    plt.title('大豆収量の確率密度関数（2015-2018年）', fontsize=16)
    plt.xlabel('収量 (kg/10a)', fontsize=14)
    plt.ylabel('確率密度', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    
    out_dir = 'outputs'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'yield_pdf.png')
    plt.savefig(out_path, dpi=300)
    print(f"Plot saved to: {out_path}")
    print(f"Sample count: {len(df)}")
    print(f"Mean: {mean_val:.2f}, Std: {df['yield'].std():.2f}")

if __name__ == '__main__':
    main()
