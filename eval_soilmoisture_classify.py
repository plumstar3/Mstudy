"""
eval_soilmoisture_classify.py
================================================================
[目的]
  土壌水分量（VWC）特徴量のみを使って、
  多収圃場（上位33%）vs 低収圃場（下位33%）の2値分類を LOYO で評価する。
  中間33%は除外して境界の明確な比較を行う。

[特徴量]
  VWC DAP% 5%刻みビン（DAP 30〜80%の11ビン）のみ（breedは除外）

[分類ラベル]
  年ごとに収量を順位付け:
    上位 33% → 1 (多収 High)
    下位 33% → 0 (低収 Low)
    中間 33% → 除外

[モデル]
  LightGBM分類器（Optunaで最適化）
  ロジスティック回帰 Ridge (L2)
  ロジスティック回帰 Lasso (L1)

[評価指標]
  Accuracy / Balanced Accuracy / AUC-ROC / F1(macro)

[LOYO]
  eval_soilmoisture_deviation.py と同一の年次交差分割法
"""

import sqlite3, os, warnings, json, sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from scipy.stats import linregress
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             roc_auc_score, f1_score, confusion_matrix)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

warnings.filterwarnings('ignore')

# ── フォント設定 ──────────────────────────────────────────────
for _fn in ['IPAexGothic', 'IPAGothic', 'Noto Sans CJK JP', 'MS Gothic', 'Yu Gothic']:
    if any(_fn.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        plt.rcParams['font.family'] = _fn
        break
plt.rcParams['axes.unicode_minus'] = False

# ── 設定 ─────────────────────────────────────────────────────
FIELD_DB     = os.path.join('data', 'processed', 'FieldData_fieldid.db')
OUT_DIR      = os.path.join('outputs', 'yield_classify_vwc')
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42
N_TRIALS     = 100
# 上位・下位 何%を多収・低収とするか
TOP_QUANTILE = 0.33   # 上位33% → 多収
BOT_QUANTILE = 0.33   # 下位33% → 低収

# VWC特徴量: DAP 30〜80%（カバレッジ主要帯）の 11ビン
VWC_BIN_EDGES = list(range(30, 85, 5))   # [30, 35, 40, ..., 80]
VWC_BIN_COLS  = [f'vwc_dap{lo:03d}' for lo in VWC_BIN_EDGES[:-1]]   # 11ビン
VWC_FEAT_COLS = VWC_BIN_COLS                                          # 統計量なし（回帰と同条件）
CAT_COLS      = []   # 品種(breed)は入力から除外

# ── 1. 収量・播種・収穫データ取得 ────────────────────────────
print('=' * 60)
print('  VWC特徴量 多収/低収 2値分類モデル（LOYO）')
print('=' * 60)

conn = sqlite3.connect(FIELD_DB)
quest_df = pd.read_sql('''
    SELECT field_id, year, yield, seed_date, harvest_date
    FROM Questionaire
    WHERE field_id IS NOT NULL AND yield IS NOT NULL
      AND seed_date IS NOT NULL AND harvest_date IS NOT NULL
      AND year BETWEEN 2016 AND 2018''', conn)

sm_ids = pd.read_sql(
    'SELECT DISTINCT field_id, year FROM SolidMoisture WHERE field_id IS NOT NULL',
    conn)
conn.close()

quest_df['field_id']     = quest_df['field_id'].astype(int)
quest_df['year']         = quest_df['year'].astype(int)
quest_df['yield']        = quest_df['yield'].astype(float)
quest_df['seed_date']    = pd.to_datetime(quest_df['seed_date'], errors='coerce')
quest_df['harvest_date'] = pd.to_datetime(quest_df['harvest_date'], errors='coerce')
quest_df['growing_days'] = (quest_df['harvest_date'] - quest_df['seed_date']).dt.days
quest_df = quest_df[quest_df['growing_days'] > 0].copy()

sm_ids['field_id'] = pd.to_numeric(sm_ids['field_id'], errors='coerce').dropna().astype(int)
sm_ids['year']     = sm_ids['year'].astype(int)
sm_key = set(zip(sm_ids['field_id'], sm_ids['year']))
quest_df = quest_df[
    quest_df.apply(lambda r: (r['field_id'], r['year']) in sm_key, axis=1)
].dropna(subset=['yield']).reset_index(drop=True)

print(f'質問票データ（SolidMoisture圃場）: {len(quest_df)} 件')
print(f'年別: {quest_df.groupby("year").size().to_dict()}')

# ── 2. 年ごとに収量ラベル付与（上位33% / 下位33% / 中間除外） ─
def assign_label(df, top_q, bot_q):
    labels = pd.Series(index=df.index, dtype='float64')
    for yr, grp in df.groupby('year'):
        hi_thr = grp['yield'].quantile(1 - top_q)
        lo_thr = grp['yield'].quantile(bot_q)
        labels[grp.index[grp['yield'] >= hi_thr]] = 1.0  # 多収
        labels[grp.index[grp['yield'] <= lo_thr]] = 0.0  # 低収
        # 中間はNaN（除外）
    return labels

quest_df['label'] = assign_label(quest_df, TOP_QUANTILE, BOT_QUANTILE)
labeled_df = quest_df.dropna(subset=['label']).copy()
labeled_df['label'] = labeled_df['label'].astype(int)

print(f'\n分類ラベル付与後: {len(labeled_df)} 件（中間除外後）')
for yr, grp in labeled_df.groupby('year'):
    h = (grp['label'] == 1).sum()
    l = (grp['label'] == 0).sum()
    print(f'  {yr}: 多収(1)={h}件  低収(0)={l}件  合計={len(grp)}件')

# ── 3. VWC特徴量エンジニアリング ────────────────────────────
print('\nVWC特徴量生成中...')
conn = sqlite3.connect(FIELD_DB)
sm_all = pd.read_sql(
    'SELECT field_id, year, date, VWC FROM SolidMoisture WHERE field_id IS NOT NULL', conn)
conn.close()
sm_all['field_id'] = pd.to_numeric(sm_all['field_id'], errors='coerce').dropna().astype(int)
sm_all['year']     = sm_all['year'].astype(int)
sm_all['date']     = pd.to_datetime(sm_all['date'], errors='coerce')
sm_all['VWC']      = pd.to_numeric(sm_all['VWC'], errors='coerce')
sm_all = sm_all.dropna(subset=['field_id', 'date', 'VWC'])

seed_info = labeled_df[['field_id', 'year', 'seed_date', 'growing_days']].drop_duplicates()
sm_merged = sm_all.merge(seed_info, on=['field_id', 'year'], how='inner')
sm_merged['DAP_pct'] = (sm_merged['date'] - sm_merged['seed_date']).dt.days / sm_merged['growing_days'] * 100

feat_rows = []
for (fid, yr), grp in sm_merged.groupby(['field_id', 'year']):
    grp = grp.sort_values('DAP_pct')
    vwc = grp['VWC'].values
    dap = grp['DAP_pct'].values
    row = {'field_id': fid, 'year': yr}
    for lo, col in zip(VWC_BIN_EDGES[:-1], VWC_BIN_COLS):
        mask = (dap >= lo) & (dap < lo + 5)
        row[col] = float(vwc[mask].mean()) if mask.sum() > 0 else np.nan
    feat_rows.append(row)

vwc_feat_df = pd.DataFrame(feat_rows)
vwc_mean_col = sm_merged.groupby(['field_id', 'year'])['VWC'].mean().reset_index(name='_vwc_mean')
vwc_feat_df = vwc_feat_df.merge(vwc_mean_col, on=['field_id', 'year'], how='left')
for col in VWC_BIN_COLS:
    nan_mask = vwc_feat_df[col].isna()
    vwc_feat_df.loc[nan_mask, col] = vwc_feat_df.loc[nan_mask, '_vwc_mean']
vwc_feat_df.drop(columns=['_vwc_mean'], inplace=True)
for col in VWC_BIN_COLS:
    vwc_feat_df[col] = vwc_feat_df[col].fillna(vwc_feat_df[col].median())

all_data = labeled_df.merge(vwc_feat_df[['field_id', 'year'] + VWC_BIN_COLS],
                             on=['field_id', 'year'], how='inner')
print(f'特徴量結合後: {len(all_data)} 件')
print(f'特徴量: {VWC_FEAT_COLS + CAT_COLS}')

# ── 4. 準備 ──────────────────────────────────────────────────
X_num_all = all_data[VWC_FEAT_COLS].to_numpy(dtype=np.float64)
X_df      = all_data[VWC_FEAT_COLS + CAT_COLS].copy()
y_cls     = all_data['label'].to_numpy(dtype=int)
year_arr  = all_data['year'].to_numpy(dtype=int)
test_years = sorted(np.unique(year_arr))

print(f'\n特徴量次元数: {len(VWC_FEAT_COLS)}（breedなし）')

# ── 5. LightGBM 分類モデル定義 ───────────────────────────────
def make_lgb_clf(params=None):
    if params is None:
        params = {}
    return lgb.LGBMClassifier(
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
        class_weight='balanced', **params)

def fit_predict_clf(model, X_tr, y_tr, X_va, imp, sc=None):
    X_tr_i = imp.fit_transform(X_tr)
    X_va_i = imp.transform(X_va)
    if sc:
        X_tr_i = sc.fit_transform(X_tr_i)
        X_va_i = sc.transform(X_va_i)
    model.fit(X_tr_i, y_tr)
    prob = model.predict_proba(X_va_i)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return pred, prob

def _lgb_hpo_obj(trial, X_all, y_all, year_all):
    params = {
        'num_leaves'       : trial.suggest_int(  'num_leaves',        10, 100),
        'min_child_samples': trial.suggest_int(  'min_child_samples',  5,  60),
        'learning_rate'    : trial.suggest_float('learning_rate',  0.01, 0.3, log=True),
        'n_estimators'     : trial.suggest_int(  'n_estimators',     50, 400),
        'reg_lambda'       : trial.suggest_float('reg_lambda',      0.0, 20.0),
        'reg_alpha'        : trial.suggest_float('reg_alpha',       0.0, 10.0),
        'subsample'        : trial.suggest_float('subsample',       0.5,  1.0),
        'colsample_bytree' : trial.suggest_float('colsample_bytree',0.4,  1.0),
    }
    auc_list = []
    for ty in sorted(np.unique(year_all)):
        tr = np.where(year_all != ty)[0]
        va = np.where(year_all == ty)[0]
        if len(tr) == 0 or len(va) == 0 or len(np.unique(y_all[tr])) < 2:
            continue
        imp = SimpleImputer(strategy='mean')
        clf = make_lgb_clf(params)
        _, prob = fit_predict_clf(clf, X_all[tr], y_all[tr], X_all[va], imp)
        if len(np.unique(y_all[va])) < 2:
            continue
        auc_list.append(roc_auc_score(y_all[va], prob))
    return -float(np.mean(auc_list))   # 最大化→負値で最小化

# ── 6. LGBM ハイパーパラメータ探索 ──────────────────────────
bp_path = os.path.join(OUT_DIR, 'best_params_classify.json')
if os.path.exists(bp_path):
    with open(bp_path) as f:
        saved = json.load(f)
    best_params_lgb = saved['best_params']
    print(f'\nLGBM パラメータ読み込み: {bp_path}')
else:
    print(f'\nOptuna 探索 ({N_TRIALS} trials)...')
    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(
        lambda trial: _lgb_hpo_obj(trial, X_num_all, y_cls, year_arr),
        n_trials=N_TRIALS, show_progress_bar=False)
    best_params_lgb = study.best_params
    best_auc = -study.best_value
    print(f'  最良 AUC: {best_auc:.4f}')
    with open(bp_path, 'w') as f:
        json.dump({'best_auc': best_auc, 'best_params': best_params_lgb}, f, indent=2)
    print(f'  パラメータ保存: {bp_path}')

# ── 7. 3モデル LOYO ループ ────────────────────────────────────
print(f'\n{"=" * 60}')
print(f'  LOYO 分類評価: LGBM / Logistic-Ridge / Logistic-Lasso')
print(f'  サンプル数: {len(all_data)} 件 / テスト年: {test_years}')
print(f'{"=" * 60}')

LR_Cs   = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]  # 正則化強度（Cは小さいほど強い正則化）

model_results = {m: {'y_true': [], 'y_pred': [], 'y_prob': [], 'years': []}
                 for m in ['LGBM', 'LR-Ridge', 'LR-Lasso']}
per_year_rows = []

for test_year in test_years:
    tr = np.where(year_arr != test_year)[0]
    va = np.where(year_arr == test_year)[0]
    if len(tr) == 0 or len(va) == 0:
        continue
    if len(np.unique(y_cls[tr])) < 2:
        print(f'  [test={test_year}] 訓練データにクラスが1つしかない → スキップ')
        continue

    print(f'\n  [test={test_year}]  train={len(tr)}件 / test={len(va)}件  '
          f'(多収={y_cls[va].sum()}/低収={(y_cls[va]==0).sum()})')

    # ── LGBM ──────────────────────────────────────────────────
    imp_lgb = SimpleImputer(strategy='mean')
    clf_lgb = make_lgb_clf(best_params_lgb)
    pred_lgb, prob_lgb = fit_predict_clf(clf_lgb, X_num_all[tr], y_cls[tr],
                                          X_num_all[va], imp_lgb)

    # ── Logistic-Ridge ────────────────────────────────────────
    imp_r = SimpleImputer(strategy='mean'); sc_r = StandardScaler()
    lr_r  = LogisticRegressionCV(Cs=LR_Cs, penalty='l2', solver='lbfgs',
                                  cv=min(3, len(np.unique(year_arr[tr]))),
                                  class_weight='balanced', max_iter=2000,
                                  random_state=RANDOM_STATE)
    pred_r, prob_r = fit_predict_clf(lr_r, X_num_all[tr], y_cls[tr],
                                      X_num_all[va], imp_r, sc_r)

    # ── Logistic-Lasso ────────────────────────────────────────
    imp_l = SimpleImputer(strategy='mean'); sc_l = StandardScaler()
    lr_l  = LogisticRegressionCV(Cs=LR_Cs, penalty='l1', solver='liblinear',
                                  cv=min(3, len(np.unique(year_arr[tr]))),
                                  class_weight='balanced', max_iter=2000,
                                  random_state=RANDOM_STATE)
    pred_l, prob_l = fit_predict_clf(lr_l, X_num_all[tr], y_cls[tr],
                                      X_num_all[va], imp_l, sc_l)

    # ── 各モデル評価 ──────────────────────────────────────────
    row = {'year': test_year, 'n': len(va)}
    for mname, pred, prob in [('LGBM', pred_lgb, prob_lgb),
                               ('LR-Ridge', pred_r, prob_r),
                               ('LR-Lasso', pred_l, prob_l)]:
        has2 = len(np.unique(y_cls[va])) >= 2
        acc  = accuracy_score(y_cls[va], pred)
        bacc = balanced_accuracy_score(y_cls[va], pred)
        auc  = roc_auc_score(y_cls[va], prob) if has2 else float('nan')
        f1   = f1_score(y_cls[va], pred, average='macro', zero_division=0)
        print(f'    {mname:<12}: Acc={acc:.3f}  BalAcc={bacc:.3f}  AUC={auc:.3f}  F1={f1:.3f}')
        row[f'Acc_{mname}'] = acc
        row[f'BalAcc_{mname}'] = bacc
        row[f'AUC_{mname}'] = auc
        row[f'F1_{mname}'] = f1

        model_results[mname]['y_true'].extend(y_cls[va].tolist())
        model_results[mname]['y_pred'].extend(pred.tolist())
        model_results[mname]['y_prob'].extend(prob.tolist())
        model_results[mname]['years'].extend([test_year] * len(va))
    per_year_rows.append(row)

# ── 8. プール計算サマリ ───────────────────────────────────────
# ランダム分類ベース（balanced accuracy=0.5、AUC=0.5）
print(f'\n{"=" * 60}')
print('  プール計算 サマリ（全年合算）')
print(f'{"=" * 60}')
print(f'  {"モデル":<14} {"Acc":>7} {"BalAcc":>8} {"AUC":>7} {"F1(macro)":>10}')
print(f'  {"-"*48}')
print(f'  {"ランダム基準":<14} {"0.500":>7} {"0.500":>8} {"0.500":>7} {"0.500":>10}')

for mname in ['LGBM', 'LR-Ridge', 'LR-Lasso']:
    yt = np.array(model_results[mname]['y_true'])
    yp = np.array(model_results[mname]['y_pred'])
    yprob = np.array(model_results[mname]['y_prob'])
    has2 = len(np.unique(yt)) >= 2
    acc  = accuracy_score(yt, yp)
    bacc = balanced_accuracy_score(yt, yp)
    auc  = roc_auc_score(yt, yprob) if has2 else float('nan')
    f1   = f1_score(yt, yp, average='macro', zero_division=0)
    print(f'  {mname:<14} {acc:>7.3f} {bacc:>8.3f} {auc:>7.3f} {f1:>10.3f}')

# ── 9. 可視化 ─────────────────────────────────────────────────
YEAR_COLORS = {2016: '#e74c3c', 2017: '#2980b9', 2018: '#27ae60'}

fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='#f8f9fa')
fig.suptitle('VWC特徴量 LOYO 2値分類（多収 vs 低収）\n'
             f'DAP 30〜80% 11ビン  |  上位33%=多収 / 下位33%=低収  |  n={len(all_data)}件',
             fontsize=12, fontweight='bold')

for ax, mname in zip(axes, ['LGBM', 'LR-Ridge', 'LR-Lasso']):
    yt    = np.array(model_results[mname]['y_true'])
    yprob = np.array(model_results[mname]['y_prob'])
    yrs_  = np.array(model_results[mname]['years'])
    has2  = len(np.unique(yt)) >= 2
    auc   = roc_auc_score(yt, yprob) if has2 else float('nan')
    bacc  = balanced_accuracy_score(yt, (yprob >= 0.5).astype(int))

    # 確率プロット（横: 多収/低収ラベル、縦: 多収予測確率）
    jitter = np.random.default_rng(42).uniform(-0.08, 0.08, size=len(yt))
    for yr, col in YEAR_COLORS.items():
        m = yrs_ == yr
        if m.any():
            ax.scatter(yt[m] + jitter[m], yprob[m], alpha=0.6, s=40, c=col,
                       edgecolors='white', lw=0.3, label=f'{yr}(n={m.sum()})')
    ax.axhline(0.5, ls='--', color='#555', lw=1.2, label='threshold=0.5')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['低収(0)', '多収(1)'])
    ax.set_ylabel('多収クラス予測確率', fontsize=9)
    ax.set_title(f'{mname}\nAUC={auc:.3f}  BalAcc={bacc:.3f}', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='upper left'); ax.grid(axis='y', alpha=0.25)

fig.tight_layout()
out_path = os.path.join(OUT_DIR, 'eval_classify_loyo.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)

# 混同行列も出力
fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4), facecolor='#f8f9fa')
fig2.suptitle('混同行列（全年合算）', fontsize=12, fontweight='bold')
for ax, mname in zip(axes2, ['LGBM', 'LR-Ridge', 'LR-Lasso']):
    yt = np.array(model_results[mname]['y_true'])
    yp = np.array(model_results[mname]['y_pred'])
    cm = confusion_matrix(yt, yp)
    im = ax.imshow(cm, cmap='Blues')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() * 0.5 else 'black', fontsize=14)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['予測:低収', '予測:多収'])
    ax.set_yticklabels(['実際:低収', '実際:多収'])
    ax.set_title(mname, fontsize=10, fontweight='bold')

fig2.tight_layout()
out_path2 = os.path.join(OUT_DIR, 'eval_classify_confusion.png')
fig2.savefig(out_path2, dpi=150, bbox_inches='tight')
plt.close(fig2)

# CSV保存
pd.DataFrame(per_year_rows).to_csv(
    os.path.join(OUT_DIR, 'loyo_classify_summary.csv'), index=False, encoding='utf-8-sig')
print(f'\n  予測確率プロット -> {out_path}')
print(f'  混同行列       -> {out_path2}')
print(f'  CSVサマリ      -> {os.path.join(OUT_DIR, "loyo_classify_summary.csv")}')
print('\n完了')
