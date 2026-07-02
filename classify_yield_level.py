"""
classify_yield_level.py
===========================================================
【目的】
  「現在の気象・アンケート特徴量で多収/低収を分類できるか？」の事前検証。
  年偏差特徴量(year_dev=replace) と Harm の組み合わせで LOYO AUC が
  改善するかを確認し、Mixture-of-Experts アンサンブルの実現可能性を評価。

【実験設定（3 × 2 × 2）】
  特徴量セット:
    1. ベースライン  : 気象 135 列
    2. +年偏差       : 気象年偏差 135 列 (year_dev=replace)
    3. +年偏差+Harm  : 気象年偏差 + Harm 5 列 = 140 列
  ラベル方式:
    A. 全体中央値で 多収(1)/低収(0) を定義
    B. 年別中央値で 多収(1)/低収(0) を定義（年内相対比較）
  CV方式: k-fold / LOYO

【判断基準】
  LOYO AUC >= 0.65 → アンサンブルが有効
  LOYO AUC ~= 0.50 → 特徴量では区別できず → 効果薄
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import lightgbm as lgb

warnings.filterwarnings('ignore')

# 日本語フォント
_JP_FONTS = ['Yu Gothic', 'Meiryo', 'MS Gothic']
for _fn in _JP_FONTS:
    if any(_fn.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        plt.rcParams['font.family'] = _fn
        break
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yield_prediction_v3 import build_dataset, FIELD_DB, WEATHER_DB, GDD_CSV

OUT_DIR = os.path.join('outputs', 'yield_pred_v3')
os.makedirs(OUT_DIR, exist_ok=True)

N_SPLITS     = 5
RANDOM_STATE = 42
LGBM_PARAMS  = dict(n_estimators=200, learning_rate=0.05, num_leaves=31,
                    random_state=RANDOM_STATE, verbose=-1)

def make_lr_pipe():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=0.01, max_iter=1000, random_state=RANDOM_STATE))
    ])

# ── データセット定義 ───────────────────────────────────────────────────────────
DATASET_CONFIGS = [
    {'label': 'Baseline\n(気象135列)',         'add_harm': False, 'year_dev': 'none'},
    {'label': '+年偏差\n(気象偏差135列)',       'add_harm': False, 'year_dev': 'replace'},
    {'label': '+年偏差+Harm\n(偏差+Harm140列)', 'add_harm': True,  'year_dev': 'replace'},
]

# ── CV 評価関数 ────────────────────────────────────────────────────────────────
def run_cv(X, y_bin, years_arr, cv_mode='kfold'):
    all_years = sorted(set(years_arr))

    if cv_mode == 'kfold':
        skf   = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        folds = list(skf.split(X, y_bin))
        fold_labels = [f'fold{i+1}' for i in range(N_SPLITS)]
    else:
        folds = [(np.where(years_arr != yr)[0], np.where(years_arr == yr)[0])
                 for yr in all_years]
        fold_labels = [str(yr) for yr in all_years]

    lr_aucs, lgbm_aucs = [], []
    lr_accs, lgbm_accs = [], []
    lr_f1s,  lgbm_f1s  = [], []

    for (train_idx, test_idx), fl in zip(folds, fold_labels):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y_bin[train_idx], y_bin[test_idx]
        has_both   = len(np.unique(y_te)) > 1

        # Logistic Regression
        lr = make_lr_pipe()
        lr.fit(X_tr, y_tr)
        y_pred = lr.predict(X_te)
        y_prob = lr.predict_proba(X_te)[:, 1]
        lr_accs.append(accuracy_score(y_te, y_pred))
        lr_aucs.append(roc_auc_score(y_te, y_prob) if has_both else np.nan)
        lr_f1s.append(f1_score(y_te, y_pred, zero_division=0))

        # LightGBM
        clf = lgb.LGBMClassifier(**LGBM_PARAMS)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1]
        lgbm_accs.append(accuracy_score(y_te, y_pred))
        lgbm_aucs.append(roc_auc_score(y_te, y_prob) if has_both else np.nan)
        lgbm_f1s.append(f1_score(y_te, y_pred, zero_division=0))

    def mn(v): return float(np.nanmean(v)) if v else np.nan

    return {
        'fold_labels': fold_labels,
        'lr'  : {'auc': lr_aucs,   'acc': lr_accs,   'f1': lr_f1s,
                 'auc_mean': mn(lr_aucs), 'acc_mean': mn(lr_accs)},
        'lgbm': {'auc': lgbm_aucs, 'acc': lgbm_accs, 'f1': lgbm_f1s,
                 'auc_mean': mn(lgbm_aucs), 'acc_mean': mn(lgbm_accs)},
    }

# ── 全実験実行 ────────────────────────────────────────────────────────────────
print('=' * 70)
print('  多収/低収 分類事前検証  (年偏差特徴量の効果確認)')
print('=' * 70)

all_results = {}   # key: (ds_label, label_mode, cv_mode)

for cfg in DATASET_CONFIGS:
    ds_label = cfg['label']
    print(f'\n{"="*70}')
    print(f'  データセット: {ds_label.replace(chr(10)," ")}')
    print(f'{"="*70}')

    X_raw, y, geo, meta, feat_cols = build_dataset(
        FIELD_DB, WEATHER_DB, GDD_CSV,
        add_harm=cfg['add_harm'],
        add_spacing=False,
        add_vwc=False,
        add_breed=False,
        year_dev=cfg['year_dev'],
    )

    # NaN 補完
    imp = SimpleImputer(strategy='median')
    X   = imp.fit_transform(X_raw).astype(np.float32)

    years_arr = meta['year'].values

    # 2種のラベル
    THRESHOLD_GLOBAL = float(np.median(y))
    y_bin_global = (y >= THRESHOLD_GLOBAL).astype(int)

    y_bin_yearly = np.zeros(len(y), dtype=int)
    for yr in [2015, 2016, 2017, 2018]:
        mask = years_arr == yr
        thr  = float(np.median(y[mask]))
        y_bin_yearly[mask] = (y[mask] >= thr).astype(int)

    for label_mode, y_bin in [('全体中央値', y_bin_global), ('年別中央値', y_bin_yearly)]:
        print(f'\n  【ラベル: {label_mode}】  高収={y_bin.sum()}件 低収={(1-y_bin).sum()}件')
        for cv_mode in ['kfold', 'loyo']:
            print(f'  --- {cv_mode.upper()} ---')
            res = run_cv(X, y_bin, years_arr, cv_mode=cv_mode)
            all_results[(ds_label, label_mode, cv_mode)] = res

            for fl, lr_auc, lgbm_auc in zip(
                    res['fold_labels'], res['lr']['auc'], res['lgbm']['auc']):
                print(f'    [{fl}]  LR AUC={lr_auc:.3f}  LGBM AUC={lgbm_auc:.3f}')
            print(f'    SUMMARY: LR AUC={res["lr"]["auc_mean"]:.3f}  '
                  f'LGBM AUC={res["lgbm"]["auc_mean"]:.3f}')

# ── 結果サマリ CSV 出力 ─────────────────────────────────────────────────────
rows = []
for (ds_label, label_mode, cv_mode), res in all_results.items():
    rows.append({
        'dataset'   : ds_label.replace('\n', ' '),
        'label_mode': label_mode,
        'cv_mode'   : cv_mode,
        'lr_auc'    : round(res['lr']['auc_mean'], 3),
        'lgbm_auc'  : round(res['lgbm']['auc_mean'], 3),
        'lr_acc'    : round(res['lr']['acc_mean'], 3),
        'lgbm_acc'  : round(res['lgbm']['acc_mean'], 3),
    })
summary_df = pd.DataFrame(rows)
csv_path = os.path.join(OUT_DIR, 'classify_summary.csv')
summary_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f'\n\nサマリCSV: {csv_path}')

# ── 可視化 ────────────────────────────────────────────────────────────────────
DS_LABELS  = [cfg['label'] for cfg in DATASET_CONFIGS]
DS_COLORS  = ['#4C72B0', '#55A868', '#C44E52']
LABEL_MODES = ['全体中央値', '年別中央値']
CV_MODES    = ['kfold', 'loyo']

# 図1: データセット × CV × ラベルモード の LightGBM AUC ヒートマップ的棒グラフ
fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor='#f9f9f9')
fig.suptitle('多収/低収 分類 LGBM AUC: データセット設定と評価方法の比較',
             fontsize=13, fontweight='bold')

for col, label_mode in enumerate(LABEL_MODES):
    for row_i, cv_mode in enumerate(CV_MODES):
        ax = axes[row_i][col]

        # 各データセットの fold ごとの AUC をまとめてプロット
        fold_labels_ref = all_results[(DS_LABELS[0], label_mode, cv_mode)]['fold_labels']
        n_folds = len(fold_labels_ref)
        x = np.arange(n_folds)
        w = 0.25

        for di, (ds_label, color) in enumerate(zip(DS_LABELS, DS_COLORS)):
            aucs = all_results[(ds_label, label_mode, cv_mode)]['lgbm']['auc']
            ax.bar(x + (di - 1) * w, aucs, w,
                   label=ds_label.replace('\n', ' '), color=color, alpha=0.8)

        ax.axhline(0.65, color='green', linestyle='--', lw=1.5,
                   label='AUC=0.65 (目標)', alpha=0.8)
        ax.axhline(0.50, color='gray',  linestyle=':',  lw=1.2,
                   label='AUC=0.50 (ランダム)', alpha=0.7)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x)
        ax.set_xticklabels(fold_labels_ref, fontsize=9)
        ax.set_title(f'{cv_mode.upper()} / {label_mode}', fontsize=11, fontweight='bold')
        ax.set_ylabel('ROC-AUC (LGBM)', fontsize=9)
        ax.set_xlabel('Fold / Test Year', fontsize=9)
        ax.legend(fontsize=8, loc='lower right')
        ax.set_facecolor('#fdfdfd')
        ax.grid(True, alpha=0.3, axis='y')

fig.tight_layout()
fig_path = os.path.join(OUT_DIR, 'classify_auc_comparison.png')
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'AUC比較図: {fig_path}')

# 図2: LOYO AUC のデータセット別平均値まとめ棒グラフ（最重要図）
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6), facecolor='#f9f9f9')
fig2.suptitle('LOYO AUC まとめ: 年偏差特徴量の効果確認\n（値が高いほどアンサンブル手法が有効）',
              fontsize=13, fontweight='bold')

for col, label_mode in enumerate(LABEL_MODES):
    ax = axes2[col]
    ds_names_short = [d.replace('\n', '\n') for d in DS_LABELS]
    lgbm_means = [all_results[(ds, label_mode, 'loyo')]['lgbm']['auc_mean']
                  for ds in DS_LABELS]
    lr_means   = [all_results[(ds, label_mode, 'loyo')]['lr']['auc_mean']
                  for ds in DS_LABELS]

    x = np.arange(len(DS_LABELS))
    w = 0.35
    bars1 = ax.bar(x - w/2, lgbm_means, w, label='LightGBM', color='#3498db', alpha=0.85)
    bars2 = ax.bar(x + w/2, lr_means,   w, label='Logistic Regression', color='#e67e22', alpha=0.85)

    ax.axhline(0.65, color='green', linestyle='--', lw=2, label='AUC=0.65 (目標)')
    ax.axhline(0.50, color='gray',  linestyle=':',  lw=1.5, label='AUC=0.50 (ランダム)')
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(ds_names_short, fontsize=9)
    ax.set_title(f'LOYO 平均AUC / {label_mode}', fontsize=11, fontweight='bold')
    ax.set_ylabel('平均 ROC-AUC', fontsize=10)
    ax.legend(fontsize=9)
    ax.set_facecolor('#fdfdfd')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars1, lgbm_means):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar, val in zip(bars2, lr_means):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

fig2.tight_layout()
fig2_path = os.path.join(OUT_DIR, 'classify_loyo_auc_summary.png')
fig2.savefig(fig2_path, dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f'LOYO AUCサマリ図: {fig2_path}')

# ── 最終判定 ─────────────────────────────────────────────────────────────────
print('\n' + '=' * 70)
print('  最終判定')
print('=' * 70)
for label_mode in LABEL_MODES:
    print(f'\n  [{label_mode}]')
    for ds in DS_LABELS:
        loyo_auc = all_results[(ds, label_mode, 'loyo')]['lgbm']['auc_mean']
        kfold_auc = all_results[(ds, label_mode, 'kfold')]['lgbm']['auc_mean']
        judge = 'GOOD: アンサンブル有効' if loyo_auc >= 0.65 else (
                'MARGINAL: 限定的効果'  if loyo_auc >= 0.58 else
                'POOR: 効果薄')
        ds_short = ds.replace('\n', ' ')
        print(f'    {ds_short:30s} kfold={kfold_auc:.3f}  loyo={loyo_auc:.3f}  -> {judge}')

print('\n完了')
