import sqlite3, pandas as pd, numpy as np

conn = sqlite3.connect('data/processed/FieldData_fieldid.db')
harm = pd.read_sql("SELECT * FROM harm WHERE field_id IS NOT NULL", conn)
conn.close()

score_cols = ['bug', 'sick', 'weed', 'wet', 'unripen', 'lay', 'loss']
flag_cols  = ['heavy_rain', 'drought', 'typhoon', 'gale', 'frost', 'hail',
              'cold', 'long_rain', 'few_solar', 'early_snow']

for c in score_cols:
    harm[c] = pd.to_numeric(harm[c], errors='coerce')

print("=== スコア変数（順序・カテゴリ）===")
for c in score_cols:
    vc = harm[c].value_counts(dropna=False).sort_index()
    n  = len(harm)
    print(f"\n[{c}] n={n}")
    for val, cnt in vc.items():
        bar = '#' * int(cnt / n * 40)
        print(f"  {str(val):5s}: {cnt:3d}件 ({cnt/n*100:5.1f}%) {bar}")

print("\n=== フラグ変数（True/False）===")
for c in flag_cols:
    vc = harm[c].value_counts(dropna=False)
    n  = len(harm)
    true_n  = vc.get('TRUE', vc.get(True, 0))
    false_n = vc.get('FALSE', vc.get(False, 0))
    nan_n   = harm[c].isna().sum()
    print(f"  {c:<15}: TRUE={true_n}件({true_n/n*100:.1f}%)  FALSE={false_n}件({false_n/n*100:.1f}%)  NaN={nan_n}件({nan_n/n*100:.1f}%)")
