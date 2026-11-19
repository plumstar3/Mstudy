"""
段階19: 相対収量の回帰で学習し、分類で評価する(EXP-028)。

2つに分けたラベル(label_muniyear_median)では、収量の差の大きさと、ラベルの付かない件(2〜3圃場の集団、奇数の
集団の中央の1件)の情報を捨てている。目的変数を「同じ市町村×年の中での相対収量」にして Ridge 回帰で学習し、
予測値をスコアにして、段階13の median と同じテスト件(123件)で AUC を出す。
仮説: 捨てていた情報を学習に使えば、分類の精度(AUC)が上がる。
参考: ROCKET の特徴量 + Ridge 回帰は時系列外部回帰で使われている(Tan et al., DMKD 2021)。

データ
  学習に使える件: 開花起点 0〜+59 日の窓がそろい、属する市町村×年の集団が2圃場以上(集団の大きさは
    field_year_labeled.csv の全304件で数える)。圃場472・473(2018年、VWC が全日同じ)は除く。想定 235件・51市町村・84集団
  目的変数(主) y_rel = log(yield / 集団の平均収量)。集団の平均は全304件のうちその市町村×年の全件(窓の外も含む)
分割
  段階13の median の分割(make_splits(y, muniCd, 5, 20, 0))をそのまま使う。fold のテスト件の市町村 = テストの市町村。
  学習 = 学習に使える件のうちテストの市町村に属さない全件。評価 = テストの市町村に属する、評価の123件だけ
設定
  R_all(主)    1次差分 → MiniRocket(10000、乱数0〜9)→ StandardScaler(with_mean=False) → RidgeCV(logspace(-3,3,10)×n_train、GCV)
  M_diff       比べる相手。段階13の median・M(outputs/stage13/)。乱数0の先頭10 fold を計算し直して一致を確かめる
  R_123        参考。学習を評価の123件(段階13の学習側)に限った回帰(目的変数 y_rel)
  R_all_rank   参考。R_all の目的変数を集団の中の順位のパーセンタイル((順位 − 0.5) / n、n は集団の全件)にしたもの
  スコア = 予測値。AUC は median のラベル、bacc のしきい値は 0(R_all_rank だけ 0.5 = 集団の中央)
置換検定(R_all、置換 i の乱数は [0, i])
  市町村×年の集団の中で収量を圃場の間で並べ替え(全304件の集団。窓の外の件も含む)、y_rel とテスト件の median の
  ラベル(add_yield_labels.py と同じ規則)を作り直す。分割(市町村単位)は変わらない。テスト件の顔ぶれは変わりうる
判定(事前に固定)
  1. R_all − M_diff(同じテスト件・同じ乱数)の繰り返し平均の 2.5〜97.5%点が0を含まず正の側
  2. R_all の全体の AUC が帰無分布の95%点を上回る
  両方を満たせば R_all を主の設定に採用。満たさなければ M_diff のまま

実行:
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage19\\run_stage19.py
  試運転: ... run_stage19.py --n-seeds 1 --n-perm 20 --n-boot 50
出力: rocket/outputs/stage19/、入力: rocket/data/processed/stage19/
"""
import argparse
import os
import sys
import time
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import linalg
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import RidgeCV
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROCKET_DIR = os.path.dirname(os.path.dirname(STAGE_DIR))
sys.path.insert(0, ROCKET_DIR)
sys.path.insert(0, os.path.join(ROCKET_DIR, 'scripts'))
for d in ['stage6', 'stage7', 'stage11', 'stage12', 'stage13', 'stage14', 'stage16']:
    sys.path.insert(0, os.path.join(ROCKET_DIR, 'analysis', d))
import run_stage6 as s6  # noqa: E402
import run_stage6_local as fast  # noqa: E402
import run_stage7 as s7  # noqa: E402
import run_stage12 as s12  # noqa: E402
import run_stage14 as s14  # noqa: E402
import run_stage16 as s16  # noqa: E402
from add_yield_labels import assign_label  # noqa: E402
from run_stage6 import BASE_ALPHAS, RidgeNTrain  # noqa: E402
from src.evaluation import make_splits, permute_within, split_table  # noqa: E402
from src.windows import extract_windows, first_diff, load_timeseries  # noqa: E402

KEY, LABEL = 'median', 'label_muniyear_median'
PROC = os.path.join(ROCKET_DIR, 'data', 'processed')
DATA_DIR = os.path.join(PROC, 'stage19')
S13_DIR = os.path.join(ROCKET_DIR, 'outputs', 'stage13')
OUT_DIR = os.path.join(ROCKET_DIR, 'outputs', 'stage19')
N_SPLITS, N_REPEATS, SEED0 = s6.N_SPLITS, s6.N_REPEATS, s6.SEED0
N_JOBS = fast.N_JOBS
TIE_RTOL = fast.TIE_RTOL
TYPES = ['W', 'B', 'X']
DROP = [(472, 2018), (473, 2018)]  # VWC が全日同じ(docs/data_issues.md の3)
# 想定: 学習に使える件, 市町村, 集団, 評価の件, 4圃場以上の中央の件, 2〜3圃場の件, 評価外の市町村の件, その市町村数
EXPECT = dict(n=235, munis=51, groups=84, n123=123, center=3, small=109, n_noneval=88, munis_noneval=30)
CFGS = ['M_diff', 'R_123', 'R_all_rank', 'R_all']
CFG_LABEL = {'M_diff': 'M_diff 分類・123件(段階13)', 'R_123': 'R_123 回帰・123件(参考)',
             'R_all_rank': 'R_all_rank 回帰・順位・全件(参考)', 'R_all': 'R_all 回帰・y_rel・全件(主)'}
CFG_COLOR = {'M_diff': '#9a9890', 'R_123': '#8fb8e8', 'R_all_rank': '#eb6834', 'R_all': '#2a78d6'}
THRESH = {'R_all': 0.0, 'R_123': 0.0, 'R_all_rank': 0.5, 'M_diff': 0.0}

matplotlib.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
INK, MUTED = s6.INK, s6.MUTED


# ---------------------------------------------------------------- 回帰器
class RidgeRegNTrain(BaseEstimator, RegressorMixin):
    """RidgeCV(alphas = base_alphas × 学習データの件数、GCV)。"""

    def __init__(self, base_alphas=BASE_ALPHAS):
        self.base_alphas = base_alphas

    def fit(self, X, y):
        self.n_train_ = len(X)
        self.reg_ = RidgeCV(alphas=np.asarray(self.base_alphas) * self.n_train_).fit(X, y)
        self.alpha_ = self.reg_.alpha_
        return self

    def predict(self, X):
        return self.reg_.predict(X)


def batched_ridge_reg(Ftr, Fpr, Ytr):
    """RidgeRegNTrain を Ytr の各行(連続値の目的変数)で学習したときの alpha と予測値をまとめて解く。

    sklearn 1.5.1 の _RidgeGCV の eigen モード(n_samples < n_features、fit_intercept=True、重みなし)と同じ式:
    K = Xc Xc^T + 1 の固有値分解、切片の方向の固有値を0、GCV スコア = −mean((c / diag(G^-1))^2)、同点なら先の alpha。
    run_stage6_local._batched_ridge の回帰版(ラベルを ±1 にしない)。
    戻り値: alpha (P,), pred (P, n_pr), unsure (P,) bool(sklearn で学習し直すべき行)
    """
    n = len(Ftr)
    alphas = np.asarray(BASE_ALPHAS) * n
    X_offset = Ftr.mean(axis=0)
    Xc = Ftr - X_offset
    eigvals, Q = linalg.eigh(Xc @ Xc.T + 1.0)
    intercept_dim = np.argmax(np.abs(np.full(n, 1 / np.sqrt(n)) @ Q))
    Y = np.asarray(Ytr, float).T  # (n, P)
    y_offset = Y.mean(axis=0)
    QT_y = Q.T @ (Y - y_offset)
    scores, cs = [], []
    for a in alphas:
        w = 1.0 / (eigvals + float(a))
        w[intercept_dim] = 0
        c = Q @ (w[:, None] * QT_y)
        G_inverse_diag = (w * Q ** 2).sum(axis=-1)
        scores.append(-((c / G_inverse_diag[:, None]) ** 2).mean(axis=0))
        cs.append(c)
    scores = np.array(scores)  # (n_alphas, P)
    best = np.argmax(scores, axis=0)
    P = Y.shape[1]
    c_best = np.stack(cs)[best, :, np.arange(P)]  # (P, n)
    coef = c_best @ Xc  # (P, n_features) sklearn と同じ coef = dual_coef^T X
    pred = Fpr @ coef.T + (y_offset - coef @ X_offset)  # (n_pr, P)

    s_best = scores[best, np.arange(P)]
    gap = np.abs(scores - s_best)
    gap[best, np.arange(P)] = np.inf
    unsure = gap.min(axis=0) <= TIE_RTOL * np.abs(s_best)
    scale = np.abs(pred).max(axis=0)
    unsure |= (np.abs(pred) <= TIE_RTOL * scale).any(axis=0)  # bacc のしきい値0の付近
    srt = np.sort(pred, axis=0)
    unsure |= (np.diff(srt, axis=0) <= TIE_RTOL * scale).any(axis=0)
    return alphas[best], pred.T, unsure


# ---------------------------------------------------------------- データ
def load_all(log):
    """全304件(収量・集団・ラベル)と、学習に使える件の窓(生の VWC)。"""
    lab = pd.read_csv(os.path.join(PROC, 'field_year_labeled.csv'), encoding='utf-8-sig', dtype={'muniCd': str})
    lab['muniyear'] = lab['muniCd'] + '_' + lab['year'].astype(str)
    lab['group_n_all'] = lab.muniyear.map(lab.groupby('muniyear').size())
    lab['group_mean_yield'] = lab.groupby('muniyear')['yield'].transform('mean')
    lab['y_rel'] = np.log(lab['yield'] / lab['group_mean_yield'])
    rank = lab.sort_values(['yield', 'field_id']).groupby('muniyear').cumcount() + 1  # 同順位は field_id の昇順
    lab['y_rank'] = (rank.reindex(lab.index) - 0.5) / lab['group_n_all']
    assert lab['yield'].notna().all()

    d = lab.copy()
    for c in ['seed_date', 'harvest_date', 'flower_date']:
        d[c] = pd.to_datetime(d[c])
    d['seed_doy'] = d['seed_date'].dt.dayofyear
    d['harvest_dap'] = (d['harvest_date'] - d['seed_date']).dt.days
    d['flower_dap'] = (d['flower_date'] - d['seed_date']).dt.days
    assert d['flower_dap'].notna().all() and d['harvest_dap'].notna().all()
    d['flower_dap'] = d['flower_dap'].astype(int)
    d['lab_pos'] = np.arange(len(d))
    ts = load_timeseries(d)
    X, meta, excluded = extract_windows(d, ts, 'flower', 0, 59)
    log(f'[データ] 全 {len(lab)} 件 → 窓(開花起点 0〜+59 日)を覆う {len(meta)} 件')
    keep = (meta.group_n_all >= 2) & ~pd.Series(list(zip(meta.field_id, meta.year))).isin(DROP).values
    drop_rows = meta[~keep]
    log(f'  窓を覆うが学習に使わない {len(drop_rows)} 件: 1圃場の集団 {int((drop_rows.group_n_all < 2).sum())}、'
        f'圃場472・473(2018年) {int(drop_rows.group_n_all.ge(2).sum())}')
    X, meta = X[keep.values], meta[keep.values].reset_index(drop=True)
    excluded = pd.concat([excluded.assign(reason_stage19='窓を覆わない'),
                          drop_rows.assign(reason_stage19=np.where(drop_rows.group_n_all < 2, '1圃場の集団',
                                                                   'VWC が全日同じ(472・473)'))])
    os.makedirs(DATA_DIR, exist_ok=True)
    np.save(os.path.join(DATA_DIR, 'eligible_X.npy'), X)
    meta.to_csv(os.path.join(DATA_DIR, 'eligible_meta.csv'), index=False, encoding='utf-8-sig')
    excluded.to_csv(os.path.join(DATA_DIR, 'excluded.csv'), index=False, encoding='utf-8-sig')
    return lab, X, meta


def check_counts(lab, meta, X, meta123, X123, log):
    """確認1: 件数と内訳。123件の窓が段階13と同じであること。"""
    k123 = list(zip(meta123.field_id, meta123.year))
    pos = pd.Series(np.arange(len(meta)), index=pd.MultiIndex.from_arrays([meta.field_id, meta.year]))
    i123 = pos.loc[k123].values  # 123件の、学習に使える件の中での位置(段階13の並び順)
    in123 = np.zeros(len(meta), bool)
    in123[i123] = True
    evm = set(meta123.muniCd)
    center = (meta.group_n_all >= 4) & meta[LABEL].isna()
    small = meta.group_n_all <= 3
    noneval = ~meta.muniCd.isin(evm)
    got = dict(n=len(meta), munis=meta.muniCd.nunique(), groups=meta.muniyear.nunique(), n123=int(in123.sum()),
               center=int(center.sum()), small=int(small.sum()), n_noneval=int(noneval.sum()),
               munis_noneval=meta[noneval].muniCd.nunique())
    log(f'  確認1 件数: 学習に使える {got["n"]} 件・{got["munis"]} 市町村・{got["groups"]} 集団。内訳: 評価の件 {got["n123"]}、'
        f'4圃場以上の集団の中央の件 {got["center"]}、2〜3圃場の集団の件 {got["small"]}'
        f'(うち評価の市町村 {int((small & ~noneval).sum())})。評価の {len(evm)} 市町村以外の {got["munis_noneval"]} 市町村の件 '
        f'{got["n_noneval"]}')
    log('    中央の件: ' + '、'.join(f'{r.field_id}({r.muniyear}、{r.group_n_all}圃場)' for _, r in meta[center].iterrows()))
    log(f'    年別: {meta.year.value_counts().sort_index().to_dict()}、開花日 {meta.flower_source.value_counts().to_dict()}')
    assert got == EXPECT, f'件数が想定 {EXPECT} と違う: {got}'
    assert ((center | small) == ~in123).all(), '評価の123件・中央の件・2〜3圃場の件で全件にならない'
    dx = np.abs(X[i123] - X123).max()
    log(f'    123件の窓の生の値が段階13の X と一致(最大差 {dx:.1e})、ラベルも一致 '
        f'{bool((meta[LABEL].values[i123] == meta123.y.values).all())}')
    assert dx == 0 and (meta[LABEL].values[i123] == meta123.y.values).all()
    return i123, in123


# ---------------------------------------------------------------- ラベル・置換
class Relabel:
    """全304件の収量から median のラベルを作る(add_yield_labels.assign_label と同じ規則を numpy で)。"""

    def __init__(self, lab):
        self.fid = lab.field_id.values
        my = lab.muniyear.to_numpy(dtype=str)
        self.groups = [np.flatnonzero(my == g) for g in np.unique(my)]
        self.groups = [g for g in self.groups if len(g) >= 4]

    def __call__(self, yld):
        out = np.full(len(yld), np.nan)
        for idx in self.groups:
            order = idx[np.lexsort((self.fid[idx], yld[idx]))]
            k = len(idx) // 2
            out[order[:k]] = 0
            out[order[-k:]] = 1
        return out


def perm_yields(lab, P):
    """置換 i(乱数 [0, i])で、市町村×年の集団の中で収量を圃場の間で並べ替える。(P, 304)"""
    yld = lab['yield'].values
    my = lab.muniyear.to_numpy(dtype=str)
    return np.array([permute_within(yld, my, np.random.default_rng([0, i])) for i in range(P)]).reshape(P, len(yld))


def check_relabel(lab, relabel, meta, i123, meta123, Yp, log):
    """確認6: 置換なしで作ったラベルが段階13(と CSV の列)と完全に一致。置換20回分で assign_label とも一致。"""
    L0 = relabel(lab['yield'].values)
    ok_csv = np.array_equal(L0, lab[LABEL].astype(float).values, equal_nan=True)
    ok_13 = np.array_equal(L0[meta.lab_pos.values[i123]], meta123.y.values.astype(float))
    ok_p = True
    for i in range(min(20, len(Yp))):
        ref = assign_label(lab.assign(**{'yield': Yp[i]}), ['muniCd', 'year'], 4, split='median').values
        ok_p &= np.array_equal(relabel(Yp[i]), ref, equal_nan=True)
    log(f'  確認6 置換なしで作り直したラベル: field_year_labeled.csv の列と一致 {ok_csv}、段階13の123件のラベルと一致 '
        f'{ok_13}。置換{min(20, len(Yp))}回分で add_yield_labels.assign_label と一致 {ok_p}')
    assert ok_csv and ok_13 and ok_p


# ---------------------------------------------------------------- 分割
def build_folds(meta, i123, meta123, log):
    """段階13の分割を、学習に使える件の位置に直す。[(r, f, tr_all, pr_all, te123, ev_in_pr)]"""
    y123 = meta123.y.values
    sp = make_splits(y123, meta123.muniCd.values, N_SPLITS, N_REPEATS, SEED0)
    # 確認2(a): splits.csv と一致
    st = split_table(sp, y123, meta123.muniCd.values)
    ref = pd.read_csv(os.path.join(S13_DIR, 'splits.csv'), encoding='utf-8-sig')
    ref = ref[ref.label == KEY].reset_index(drop=True)
    cols = ['repeat', 'fold', 'n_train', 'n_test', 'n_test_pos', 'n_test_neg', 'n_test_groups']
    ok_tab = st[cols].reset_index(drop=True).equals(ref[cols])
    muni = meta.muniCd.to_numpy(dtype=str)
    folds, ok_te, ok_leak = [], True, True
    for r, f, tr, te in sp:
        tm = np.unique(meta123.muniCd.values[te])
        in_tm = np.isin(muni, tm)
        tr_all, pr_all = np.flatnonzero(~in_tm), np.flatnonzero(in_tm)
        ev = np.isin(pr_all, i123)
        te_from_pr = np.sort(pr_all[ev])
        ok_te &= np.array_equal(te_from_pr, np.sort(i123[te]))
        ok_leak &= not np.isin(muni[tr_all], tm).any() and not np.isin(i123[te], tr_all).any()
        # 評価の件は段階13のテストの並び順(te)にそろえる
        folds.append((r, f, tr_all, pr_all, te, np.array([np.flatnonzero(pr_all == i)[0] for i in i123[te]])))
    log(f'  確認2 テスト件: 段階13の splits.csv(件数・クラス・集団数)と一致 {ok_tab}、テストの市町村に属する評価の件 = '
        f'段階13のテスト件 {ok_te}')
    log(f'  確認3 学習側にテストの市町村の件が無い(全 {len(sp)} fold): {ok_leak}')
    assert ok_tab and ok_te and ok_leak
    ntr = np.array([len(x[2]) for x in folds])
    npr = np.array([len(x[3]) for x in folds])
    log(f'    R_all の学習件数: 平均 {ntr.mean():.1f}(最小 {ntr.min()}・最大 {ntr.max()})、段階13の学習件数の平均 '
        f'{np.mean([len(tr) for _, _, tr, _ in sp]):.1f}。予測する件(テストの市町村の全件)平均 {npr.mean():.1f}')
    return sp, folds


# ---------------------------------------------------------------- 特徴量と学習(fold ごと)
def _feats(seed, X, tr, pr):
    tf = make_pipeline(s6.Panel('minirocket', random_state=seed), StandardScaler(with_mean=False)).fit(X[tr])
    return tf.transform(X[tr]), tf.transform(X[pr])


def _fold_all(seed, X, tr, pr, Ytr):
    """R_all・R_all_rank と置換。Ytr の行0 = y_rel、行1 = y_rank、行2.. = 置換した y_rel。"""
    Ftr, Fpr = _feats(seed, X, tr, pr)
    alpha, pred, unsure = batched_ridge_reg(Ftr, Fpr, Ytr)
    ok = True
    for j in (0, 1):  # 観測の目的変数は sklearn の結果を使い、まとめて解いた結果と照合する
        ref = RidgeRegNTrain().fit(Ftr, Ytr[j])
        rp = ref.predict(Fpr)
        ok &= bool(np.isclose(alpha[j], ref.alpha_) and np.allclose(pred[j], rp, rtol=1e-7, atol=1e-10))
        alpha[j], pred[j] = ref.alpha_, rp
    redo = np.flatnonzero(unsure[2:]) + 2
    for i in redo:
        ref = RidgeRegNTrain().fit(Ftr, Ytr[i])
        alpha[i], pred[i] = ref.alpha_, ref.predict(Fpr)
    return alpha, pred, ok, len(redo)


def _fold_123(seed, X, tr, pr, y_lab, y_rel):
    """123件で学習: M_diff(分類、段階13と同じ)と R_123(回帰)。"""
    Ftr, Fpr = _feats(seed, X, tr, pr)
    clf = RidgeNTrain().fit(Ftr, y_lab)
    reg = RidgeRegNTrain().fit(Ftr, y_rel)
    return (clf.alpha_, clf.decision_function(Fpr)), (reg.alpha_, reg.predict(Fpr))


def _fold_check(seed, X, tr, pr, Ytr):
    """確認5: まとめて解いた結果と、sklearn の RidgeCV を1行ずつ学習した結果。"""
    Ftr, Fpr = _feats(seed, X, tr, pr)
    a, p, u = batched_ridge_reg(Ftr, Fpr, Ytr)
    ra, rp = [], []
    for yy in Ytr:
        ref = RidgeRegNTrain().fit(Ftr, yy)
        ra.append(ref.alpha_)
        rp.append(ref.predict(Fpr))
    return a, p, u, np.array(ra), np.array(rp)


# ---------------------------------------------------------------- 集計
def masks_for(meta, idx):
    my = meta.muniyear.to_numpy(dtype=str)[idx]
    yr = meta.year.values[idx]
    same_my, same_yr = my[:, None] == my[None, :], yr[:, None] == yr[None, :]
    return {'W': same_my, 'B': same_yr & ~same_my, 'X': ~same_yr}


def pair_counts_lab(dec, L, masks):
    """dec, L: (P, m)。L は 1 / 0 / NaN(ラベルなし)。種類ごとの (一致の数(同点 0.5), 多収 × 低収のペア数)。"""
    d = np.atleast_2d(dec)
    L = np.atleast_2d(L)
    pos, neg = (L == 1).astype(float), (L == 0).astype(float)
    cmp = (d[:, :, None] > d[:, None, :]) + 0.5 * (d[:, :, None] == d[:, None, :])
    pn = pos[:, :, None] * neg[:, None, :]
    out = {}
    for t, T in masks.items():
        w = pn * T
        out[t] = ((cmp * w).sum(axis=(1, 2)), w.sum(axis=(1, 2)))
    return out


def bacc_lab(dec, L, thr):
    pred = np.atleast_2d(dec) > thr
    L = np.atleast_2d(L)
    pos, neg = L == 1, L == 0
    with np.errstate(invalid='ignore', divide='ignore'):
        tpr = (pred & pos).sum(1) / pos.sum(1)
        tnr = (~pred & neg).sum(1) / neg.sum(1)
    return np.nanmean(np.c_[tpr, tnr], axis=1)


def analyse(preds, Lpr, meta, folds, thr, with_null=False):
    """preds: fold ごとの (P, n_pr)、Lpr: fold ごとの (P, n_pr) のラベル。行0 = 観測。

    戻り値 obs(fold の AUC・bacc、繰り返しごとの W・B・X)、null(全体 (P, 2)・W・B、with_null のとき)。
    """
    counts, f_auc, f_bacc, n_auc, n_bacc = [], [], [], [], []
    for (r, f, tr, pr, te, ev), d, L in zip(folds, preds, Lpr):
        m = masks_for(meta, pr)
        m['ALL'] = np.ones((len(pr), len(pr)), bool)
        rows = slice(None) if with_null else slice(0, 1)
        c = pair_counts_lab(d[rows], L[rows], m)
        counts.append(c)
        with np.errstate(invalid='ignore', divide='ignore'):
            auc = np.where(c['ALL'][1] > 0, c['ALL'][0] / c['ALL'][1], np.nan)
        b = bacc_lab(d[rows], L[rows], thr)
        f_auc.append(auc[0])
        f_bacc.append(b[0])
        n_auc.append(auc[1:])
        n_bacc.append(b[1:])
    obs = {'fold_auc': np.array(f_auc), 'fold_bacc': np.array(f_bacc), 'counts': counts}
    obs_counts = [{t: (c[t][0][:1], c[t][1][:1]) for t in c} for c in counts]
    sp = [(r, f, None, None) for r, f, *_ in folds]
    for t in TYPES:
        a, n = s12.repeat_auc(obs_counts, sp, t)
        obs[t] = (a[:, 0], n[:, 0])
    null = None
    if with_null:
        null = {'all': np.c_[np.nanmean(np.array(n_bacc), axis=0), np.nanmean(np.array(n_auc), axis=0)]}
        null_counts = [{t: (c[t][0][1:], c[t][1][1:]) for t in ('W', 'B')} for c in counts]
        for t in ('W', 'B'):
            a, _ = s12.repeat_auc(null_counts, sp, t)
            null[t] = np.nanmean(a, axis=0)
    return obs, null


def check_obs(tag, obs, preds, y123, folds, thr, log):
    """確認7: W・B・X をペア数で重み付けして合わせると fold の AUC。fold の AUC・bacc は sklearn の関数と一致。"""
    auc_ref = np.array([roc_auc_score(y123[te], d[0][ev]) if len(np.unique(y123[te])) == 2 else np.nan
                        for (r, f, tr, pr, te, ev), d in zip(folds, preds)])
    bacc_ref = np.array([balanced_accuracy_score(y123[te], (d[0][ev] > thr).astype(int))
                         for (r, f, tr, pr, te, ev), d in zip(folds, preds)])
    pooled = []
    for c in obs['counts']:
        n = sum(c[t][1][0] for t in TYPES)
        pooled.append(sum(c[t][0][0] for t in TYPES) / n if n else np.nan)
    ok1 = np.allclose(obs['fold_auc'], auc_ref, equal_nan=True, rtol=0, atol=1e-12)
    ok2 = np.allclose(obs['fold_bacc'], bacc_ref, equal_nan=True, rtol=0, atol=1e-12)
    ok3 = np.allclose(np.array(pooled), obs['fold_auc'], equal_nan=True, rtol=0, atol=1e-12)
    log(f'    確認7 {tag}: W+B+X(ペア数で重み付け)= fold の AUC {ok3}、fold の AUC・bacc が sklearn と一致 {ok1}・{ok2}')
    assert ok1 and ok2 and ok3


def within_spearman(pred_by_case, meta, evm_mask):
    """テストの市町村の窓のある全件で、3件以上の集団ごとに予測値と収量のスピアマン相関。pred_by_case: (n,)"""
    rows = []
    sub = meta[evm_mask]
    for g, d in sub.groupby('muniyear'):
        if len(d) < 3:
            continue
        p = pred_by_case[d.index.values]
        rho = spearmanr(p, d['yield'].values).correlation if np.ptp(p) > 0 else np.nan
        rows.append((g, len(d), rho))
    return rows


# ---------------------------------------------------------------- 起動時の確認(4・5)
def precheck_pipeline(Xd, yrel, folds, y123, log):
    """確認4: Pipeline をそのまま回した結果 = 特徴量を使い回した結果(R_all、乱数0、先頭10 fold)。"""
    ok = True
    for r, f, tr, pr, te, ev in folds[:10]:
        pipe = make_pipeline(s6.Panel('minirocket', random_state=0), StandardScaler(with_mean=False),
                             RidgeRegNTrain()).fit(Xd[tr], yrel[tr])
        a = pipe.predict(Xd[pr])
        Ftr, Fpr = _feats(0, Xd, tr, pr)
        b = RidgeRegNTrain().fit(Ftr, yrel[tr]).predict(Fpr)
        ok &= np.allclose(a, b, rtol=0, atol=1e-12) and \
            roc_auc_score(y123[te], a[ev]) == roc_auc_score(y123[te], b[ev])
    log(f'  確認4 Pipeline をそのまま回した結果 = 特徴量を使い回した結果(R_all、乱数0、先頭10 fold): {ok}')
    assert ok


def precheck_batched(Xd, Yrel_tr_rows, folds, Lpr_rows, meta, log, n_perm=20):
    """確認5: 置換 n_perm 回分で、まとめて解いた alpha・予測値 = sklearn の RidgeCV を1回ずつ学習した結果。"""
    t = time.time()
    res = Parallel(n_jobs=N_JOBS)(delayed(_fold_check)(0, Xd, tr, pr, Yrel_tr_rows[k][:n_perm])
                                  for k, (r, f, tr, pr, te, ev) in enumerate(folds))
    n_tot = n_unsure = n_bad = 0
    preds_b, preds_s = [], []
    for a, p, u, ra, rp in res:
        good = ~u
        n_tot += len(a)
        n_unsure += int(u.sum())
        n_bad += int((~(np.isclose(a, ra) & np.isclose(p, rp, rtol=1e-7, atol=1e-10).all(axis=1)) & good).sum())
        pb = p.copy()
        pb[u] = rp[u]  # 本番と同じく、あいまいな行は sklearn で学習し直す
        preds_b.append(pb)
        preds_s.append(rp)
    alpha_same = all(np.array_equal(a[~u], ra[~u]) for a, p, u, ra, rp in res)
    Lp = [L[:n_perm] for L in Lpr_rows]
    # 行0 に意味はないので、観測の行を足さずに帰無分布の部分だけを比べる
    _, na = analyse([np.vstack([pb[:1], pb]) for pb in preds_b], [np.vstack([L[:1], L]) for L in Lp], meta, folds, 0.0, True)
    _, nb = analyse([np.vstack([ps[:1], ps]) for ps in preds_s], [np.vstack([L[:1], L]) for L in Lp], meta, folds, 0.0, True)
    ok_null = all(np.array_equal(na[k], nb[k], equal_nan=True) for k in ['all', 'W', 'B'])
    log(f'  確認5 置換{n_perm}回 × {len(folds)} fold(乱数0): まとめて解いた alpha・予測値 = sklearn RidgeCV を1回ずつ '
        f'(あいまいでない {n_tot - n_unsure} 組のうち不一致 {n_bad}、alpha の一致 {alpha_same}、あいまいで学習し直す組 '
        f'{n_unsure})。学習し直した後の帰無分布(全体・W・B)が完全に一致 {ok_null}  [{time.time() - t:.0f}s]')
    assert n_bad == 0 and alpha_same and ok_null


# ---------------------------------------------------------------- 図
def plot_target(lab, meta, in123):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
    ax = axs[0]
    bins = np.linspace(min(lab.y_rel.min(), -1.2), max(lab.y_rel.max(), 0.8), 40)
    cats = [('評価の123件', in123, '#2a78d6'),
            ('4圃場以上の中央の件', (~in123) & (meta.group_n_all.values >= 4), '#eb6834'),
            ('2〜3圃場の集団の件', (~in123) & (meta.group_n_all.values <= 3), '#9a9890')]
    ax.hist([meta.y_rel.values[m] for _, m, _ in cats], bins=bins, stacked=True, color=[c for *_, c in cats],
            label=[f'{n}({int(m.sum())})' for n, m, _ in cats])
    for fid in [525, 248, 566]:
        for _, r in lab[lab.field_id == fid].iterrows():
            ax.annotate(f'{fid}', (r.y_rel, 0), xytext=(r.y_rel, 3), fontsize=8, color=INK, ha='center',
                        arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.6))
    ax.axvline(0, color=MUTED, lw=0.8, ls=':')
    ax.set_xlabel('y_rel = log(収量 / 集団の平均収量)', color=INK)
    ax.set_ylabel('件数', color=INK)
    ax.set_title(f'学習に使える {len(meta)} 件の y_rel', fontsize=11, color=INK, loc='left')
    ax.legend(frameon=False, fontsize=8.5)
    s6.style(ax)
    ax = axs[1]
    m = meta[in123]
    data = [m.y_rel.values[m[LABEL].values == k] for k in (0, 1)]
    bp = ax.boxplot(data, positions=[0, 1], widths=0.5, patch_artist=True, medianprops=dict(color=INK))
    for b, c in zip(bp['boxes'], ['#eb6834', '#2a78d6']):
        b.set_facecolor(c)
        b.set_alpha(0.55)
    ax.set_xticks([0, 1], ['低収(0)', '多収(1)'])
    ax.axhline(0, color=MUTED, lw=0.8, ls=':')
    ax.set_ylabel('y_rel', color=INK)
    ax.set_title('評価の123件: median のラベルと y_rel', fontsize=11, color=INK, loc='left')
    s6.style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'target_distribution.png'), dpi=150)
    plt.close(fig)


def plot_overview(summary):
    fig, ax = plt.subplots(figsize=(11, 5))
    mets = [('all_auc', '全体(fold の AUC)'), ('W', 'W 地域内'), ('B', 'B 地域間')]
    sm = summary.set_index(['config', 'metric'])
    w = 0.2
    for k, c in enumerate(CFGS):
        xs = np.arange(len(mets)) + (k - 1.5) * w
        vals = np.array([sm.loc[(c, m), 'mean'] for m, _ in mets])
        ax.bar(xs, vals - 0.5, bottom=0.5, width=w * 0.92, color=CFG_COLOR[c], alpha=0.9, label=CFG_LABEL[c])
        for x, (m, _), v in zip(xs, mets, vals):
            q = sm.loc[(c, m), 'null_q95']
            if pd.notna(q):
                ax.hlines(q, x - w / 2, x + w / 2, color=s6.C_RED, lw=2, ls='--')
            ax.text(x, max(v, 0.5) + 0.006, f'{v:.3f}', ha='center', fontsize=8, color=INK)
    ax.axhline(0.5, color=MUTED, lw=1, ls=':')
    ax.plot([], [], color=s6.C_RED, lw=2, ls='--', label='帰無分布の95%点(M_diff・R_all)')
    ax.set_xticks(range(len(mets)), [n for _, n in mets])
    ax.set_ylim(min(0.4, summary['mean'].min() - 0.03), max(0.72, summary['mean'].max() + 0.06))
    ax.set_ylabel('AUC', color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc='upper left', ncol=2)
    ax.set_title('段階19: 相対収量の回帰と分類の比較(label_muniyear_median の123件で評価)', fontsize=12, color=INK,
                 loc='left')
    s6.style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'stage19_overview.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-perm', type=int, default=1000)
    ap.add_argument('--n-seeds', type=int, default=10, help='試運転用。本番は10')
    ap.add_argument('--n-boot', type=int, default=2000)
    args = ap.parse_args()
    seeds = list(range(args.n_seeds))
    P = args.n_perm
    s6.N_JOBS = N_JOBS
    os.makedirs(OUT_DIR, exist_ok=True)
    logf = open(os.path.join(OUT_DIR, 'run_log.txt'), 'w', encoding='utf-8')

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()

    t0 = time.time()
    log(f'引数: {vars(args)}  並列数 {N_JOBS}')
    lab, X, meta = load_all(log)
    X123, meta123 = s14.load()
    y123 = meta123.y.values

    log('\n[起動時の確認]')
    i123, in123 = check_counts(lab, meta, X, meta123, X123, log)

    # ---- 目的変数
    q = [0, .01, .05, .25, .5, .75, .95, .99, 1]
    for name, v in [('学習に使える235件', meta.y_rel), ('全304件', lab.y_rel)]:
        log(f'  y_rel の分布({name}): ' + '、'.join(f'{int(p * 100)}% {x:+.3f}' for p, x in zip(q, v.quantile(q).values))
            + f'(平均 {v.mean():+.3f}、標準偏差 {v.std():.3f})')
    log('  外れ値の候補: ' + '、'.join(f'{r.field_id}({r.year}、{r.muniyear}) 収量 {r["yield"]:.0f}、集団平均 '
                                  f'{r.group_mean_yield:.0f}、y_rel {r.y_rel:+.3f}'
                                  for _, r in lab[lab.field_id.isin([525, 248, 566])].iterrows()))
    m = meta[in123]
    log(f'  評価の123件の y_rel: 多収 平均 {m.y_rel[m[LABEL] == 1].mean():+.3f}、低収 平均 {m.y_rel[m[LABEL] == 0].mean():+.3f}。'
        f'y_rel > 0 と多収の一致率 {((m.y_rel > 0) == (m[LABEL] == 1)).mean():.3f}(y_rel をそのままスコアにした AUC '
        f'{roc_auc_score(m[LABEL].astype(int), m.y_rel):.3f})')
    plot_target(lab, meta, in123)

    sp, folds = build_folds(meta, i123, meta123, log)
    relabel = Relabel(lab)
    Yp = perm_yields(lab, P)  # (P, 304)
    check_relabel(lab, relabel, meta, i123, meta123, Yp, log)
    gm = lab.group_mean_yield.values
    lpos = meta.lab_pos.values
    Yrel_p = np.log(Yp[:, lpos] / gm[lpos])  # (P, 235) 集団の平均は並べ替えで変わらない
    Lab_p = np.array([relabel(yy) for yy in Yp])[:, lpos]  # (P, 235)
    L_obs = relabel(lab['yield'].values)[lpos]
    yrel, yrank = meta.y_rel.values, meta.y_rank.values
    assert np.allclose(np.log(lab['yield'].values[lpos] / gm[lpos]), yrel)
    evm_mask = meta.muniCd.isin(set(meta123.muniCd)).values
    n_test_p = np.array([[np.isin(np.flatnonzero(~np.isnan(L)), np.flatnonzero(evm_mask)).sum(),
                          int(((L == 1) & evm_mask).sum()), int(((L == 0) & evm_mask).sum()),
                          int((~np.isnan(L) & evm_mask & ~in123).sum()), int((np.isnan(L) & in123).sum())]
                         for L in Lab_p])
    ntc = pd.DataFrame(n_test_p, columns=['n_test', 'n_high', 'n_low', 'n_added', 'n_dropped']).rename_axis('perm')
    ntc.to_csv(os.path.join(OUT_DIR, 'null_test_counts.csv'), encoding='utf-8-sig')
    vc = ntc.n_test.value_counts().sort_index()
    log(f'  置換ごとのテスト件の数(評価の市町村でラベルが付く窓のある件): ' + '、'.join(f'{k}件 {v}回' for k, v in vc.items())
        + f'。123件と顔ぶれが同じ置換 {int(((ntc.n_added == 0) & (ntc.n_dropped == 0)).sum())} / {P}')
    Ytr_rows = [np.vstack([yrel[tr], yrank[tr], Yrel_p[:, tr]]) for r, f, tr, pr, te, ev in folds]
    Lpr_rows = [np.vstack([L_obs[pr], L_obs[pr], Lab_p[:, pr]]) for r, f, tr, pr, te, ev in folds]
    for (r, f, tr, pr, te, ev), L in zip(folds, Lpr_rows):
        assert np.array_equal(np.flatnonzero(~np.isnan(L[0])), np.sort(ev)) and np.array_equal(L[0][ev], y123[te])

    Xd, Xd123 = first_diff(X), first_diff(X123)
    precheck_pipeline(Xd, yrel, folds, y123, log)
    precheck_batched(Xd, [Y[2:] for Y in Ytr_rows], folds, [L[2:] for L in Lpr_rows], meta, log, n_perm=min(20, P))
    mfolds, S_m, seeds13, summ_m, pa_m = s16.m_diff_from_stage13(meta123, sp, Xd123, log)  # M_diff 乱数0・先頭10 fold
    summ_m = [{k: v for k, v in r.items() if k != 'source'} for r in summ_m]

    # ---- 本番
    out = {c: {'folds': [], 'pair': [], 'seed': [], 'scores': [], 'spear': []} for c in CFGS}
    S = {c: np.full((len(seeds), N_REPEATS, len(meta123)), np.nan) for c in CFGS if c != 'M_diff'}
    nulls = {'all': [], 'W': [], 'B': []}
    m13_maxdiff = 0.0
    for k, seed in enumerate(seeds):
        t1 = time.time()
        ra = Parallel(n_jobs=N_JOBS)(delayed(_fold_all)(seed, Xd, tr, pr, Ytr_rows[j])
                                     for j, (r, f, tr, pr, te, ev) in enumerate(folds))
        assert all(ok for _, _, ok, _ in ra), '観測の目的変数で、まとめて解いた結果が sklearn と一致しない fold がある'
        n_redo = sum(x[3] for x in ra)
        rb = Parallel(n_jobs=N_JOBS)(delayed(_fold_123)(seed, Xd, i123[tr13], pr, y123[tr13], yrel[i123[tr13]])
                                     for (r, f, tr13, te13), (_, _, _, pr, _, _) in zip(sp, folds))
        res = {
            'R_all': ([a[:, None] for a, *_ in ra], [p[[0]] for _, p, *_ in ra]),
            'R_all_rank': ([a[1:2, None] for a, *_ in ra], [p[[1]] for _, p, *_ in ra]),
            'R_123': ([np.array([[x[1][0]]]) for x in rb], [x[1][1][None] for x in rb]),
            'M_diff': ([np.array([[x[0][0]]]) for x in rb], [x[0][1][None] for x in rb]),
        }
        # M_diff を計算し直した結果が段階13の oof と一致すること(全 fold)
        if seed in seeds13:
            ks = seeds13.index(seed)
            for (r, f, tr, pr, te, ev), d in zip(folds, res['M_diff'][1]):
                m13_maxdiff = max(m13_maxdiff, np.abs(d[0][ev] - S_m[ks, r, te]).max())
        for c in CFGS:
            alphas, preds = res[c]
            Lrows = [L[:1] for L in Lpr_rows]
            if c == 'R_all':
                obs, null = analyse([np.vstack([p0, p[2:]]) for p0, (_, p, *_) in zip(preds, ra)],
                                    [np.vstack([L[:1], L[2:]]) for L in Lpr_rows], meta, folds, THRESH[c], True)
                for kk in nulls:
                    nulls[kk].append(null[kk])
            else:
                obs, _ = analyse(preds, Lrows, meta, folds, THRESH[c])
            check_obs(f'{c} 乱数{seed}', obs, preds, y123, folds, THRESH[c], log)
            ntr = np.array([len(tr) if c in ('R_all', 'R_all_rank') else len(tr13)
                            for (_, _, tr, *_), (_, _, tr13, _) in zip(folds, sp)])
            al = np.array([a[0, 0] for a in alphas]) / ntr
            fd = pd.DataFrame({'config': c, 'seed': seed, 'repeat': [x[0] for x in folds], 'fold': [x[1] for x in folds],
                               'n_train': ntr, 'n_test': [len(x[4]) for x in folds], 'alpha_over_n': al,
                               'auc': obs['fold_auc'], 'bacc': obs['fold_bacc']})
            out[c]['folds'].append(fd)
            sc_all = np.full((N_REPEATS, len(meta)), np.nan)
            for (r, f, tr, pr, te, ev), p in zip(folds, preds):
                sc_all[r, pr] = p[0]
                if c != 'M_diff':
                    S[c][k, r, te] = p[0][ev]
                    mm = meta123.iloc[te]
                    out[c]['scores'].append(pd.DataFrame({
                        'config': c, 'seed': seed, 'repeat': r, 'fold': f, 'field_id': mm.field_id.values,
                        'year': mm.year.values, 'muniCd': mm.muniCd.values, 'muniyear': mm.muniyear.values,
                        'y': mm.y.values, 'y_rel': yrel[i123[te]], 'score': p[0][ev]}))
            sp_rows = []
            for r in range(N_REPEATS):
                for g, n, rho in within_spearman(sc_all[r], meta, evm_mask):
                    sp_rows.append({'config': c, 'seed': seed, 'repeat': r, 'muniyear': g, 'n': n, 'rho': rho})
            spd = pd.DataFrame(sp_rows)
            out[c]['spear'].append(spd)
            wm = spd.dropna().groupby('repeat').apply(lambda d: np.average(d.rho, weights=d.n), include_groups=False)
            row = {'config': c, 'seed': seed, 'all_auc': fd.auc.mean(), 'all_bacc': fd.bacc.mean(),
                   'alpha_at_max': np.isclose(al, BASE_ALPHAS[-1]).mean(),
                   'alpha_at_min': np.isclose(al, BASE_ALPHAS[0]).mean(), 'spearman_w': wm.mean(),
                   'n_redo': n_redo if c == 'R_all' else 0}
            for t in TYPES:
                a, n = obs[t]
                row[t] = np.nanmean(a)
                out[c]['pair'].append(pd.DataFrame({'config': c, 'type': t, 'seed': seed, 'repeat': range(len(a)),
                                                    'auc': a, 'n_pairs': n}))
            out[c]['seed'].append(row)
            log(f'    {c:10s} 乱数{seed}: 全体 AUC {row["all_auc"]:.3f} bacc {row["all_bacc"]:.3f}  '
                + '  '.join(f'{t} {row[t]:.3f}' for t in TYPES)
                + f'  順位相関 {row["spearman_w"]:+.3f}  alpha 上限 {row["alpha_at_max"]:.2f}・下限 {row["alpha_at_min"]:.2f}')
        log(f'  乱数{seed} 終了(R_all の置換で sklearn で学習し直した組 {n_redo})[{time.time() - t1:.0f}s]')
    log(f'  M_diff を計算し直した予測値と段階13の oof_scores.csv の最大差(全 fold、乱数 {len(seeds)} 通り): {m13_maxdiff:.1e}')
    assert m13_maxdiff < 1e-9

    # ---- まとめ
    null_all = np.mean(np.stack(nulls['all']), axis=0)
    null_W, null_B = np.mean(np.stack(nulls['W']), axis=0), np.mean(np.stack(nulls['B']), axis=0)
    np.save(os.path.join(OUT_DIR, 'null_R_all.npy'), null_all)
    np.save(os.path.join(OUT_DIR, 'null_R_all_W.npy'), null_W)
    np.save(os.path.join(OUT_DIR, 'null_R_all_B.npy'), null_B)
    folds_df = pd.concat([f for c in CFGS for f in out[c]['folds']])
    per_seed = pd.DataFrame([r for c in CFGS for r in out[c]['seed']])
    pair_auc = pd.concat([p for c in CFGS for p in out[c]['pair']])
    spear = pd.concat([s for c in CFGS for s in out[c]['spear']])
    summary = []
    m13 = pd.DataFrame(summ_m).set_index('metric')
    for c in CFGS:
        ps = per_seed[per_seed.config == c]
        fd = folds_df[folds_df.config == c]
        base = {'config': c, 'n_eval': len(meta123), 'n_seeds': len(ps), 'alpha_at_max': ps.alpha_at_max.mean(),
                'alpha_at_min': ps.alpha_at_min.mean(),
                'alpha_stuck': ps.alpha_at_max.mean() + ps.alpha_at_min.mean(),
                'n_train_mean': fd.groupby(['repeat', 'fold']).n_train.first().mean(),
                'n_train_min': fd.n_train.min(), 'n_train_max': fd.n_train.max()}
        for met in ['all_auc', 'all_bacc', 'W', 'B', 'X', 'spearman_w']:
            v = ps[met]
            r = {**base, 'metric': met, 'mean': v.mean(), 'seed_min': v.min(), 'seed_max': v.max()}
            if met != 'spearman_w':
                r['n_pairs_per_repeat'] = (pair_auc[(pair_auc.config == c) & (pair_auc.type == met)]
                                           .groupby('repeat').n_pairs.first().mean() if met in TYPES else np.nan)
            nl = None
            if c == 'R_all':
                nl = {'all_auc': null_all[:, 1], 'all_bacc': null_all[:, 0], 'W': null_W, 'B': null_B}.get(met)
                if nl is not None:
                    r.update({'n_perm': P, 'null_mean': np.nanmean(nl), 'null_q95': np.nanquantile(nl, .95),
                              'null_sd': np.nanstd(nl), 'p': (1 + np.sum(nl >= v.mean())) / (1 + P)})
            if c == 'M_diff' and met in m13.index and met != 'X':
                mr = m13.loc[met]
                r.update({'n_perm': mr.n_perm, 'null_mean': mr.null_mean, 'null_q95': mr.null_q95,
                          'null_sd': mr.null_sd, 'p': mr.p, 'stage13_mean': mr['mean']})
            summary.append(r)
    summary = pd.DataFrame(summary)
    summary['above_q95'] = (summary['mean'] > summary.null_q95).astype(object).where(summary.null_q95.notna(), None)
    summary.to_csv(os.path.join(OUT_DIR, 'summary.csv'), index=False, encoding='utf-8-sig')
    folds_df.to_csv(os.path.join(OUT_DIR, 'folds.csv'), index=False, encoding='utf-8-sig')
    per_seed.to_csv(os.path.join(OUT_DIR, 'per_seed.csv'), index=False, encoding='utf-8-sig')
    pd.concat([s for c in CFGS for s in out[c]['scores']]).to_csv(os.path.join(OUT_DIR, 'oof_scores.csv'),
                                                                   index=False, encoding='utf-8-sig')
    pair_auc.to_csv(os.path.join(OUT_DIR, 'pair_auc.csv'), index=False, encoding='utf-8-sig')

    # 順位相関(集団ごと + 全体)
    sg = spear.groupby(['config', 'muniyear']).agg(n=('n', 'first'), rho_mean=('rho', 'mean'), rho_sd=('rho', 'std'),
                                                  n_nan=('rho', lambda s: int(s.isna().sum()))).reset_index()
    tot = per_seed.groupby('config').spearman_w.agg(['mean', 'min', 'max']).reset_index()
    tot = tot.rename(columns={'mean': 'rho_mean', 'min': 'rho_seed_min', 'max': 'rho_seed_max'})
    tot['muniyear'] = '(全体: 件数で重み付けした平均)'
    tot['n'] = sg.groupby('config').n.sum().reindex(tot.config).values
    pd.concat([tot, sg]).to_csv(os.path.join(OUT_DIR, 'within_group_spearman.csv'), index=False, encoding='utf-8-sig')
    ng = sg[sg.config == 'R_all']
    log(f'\n[集団の中の順位相関] 評価の市町村の窓のある全件のうち、3件以上の集団 {len(ng)}({int(ng.n.sum())} 件)')

    # 対ごとの差
    fM = mfolds[mfolds.seed.isin(seeds)][['seed', 'repeat', 'fold', 'auc', 'bacc']]
    fc = {c: folds_df[folds_df.config == c][['seed', 'repeat', 'fold', 'auc', 'bacc']] for c in CFGS}
    assert np.allclose(fc['M_diff'].sort_values(['seed', 'repeat', 'fold']).auc.values,
                       fM.sort_values(['seed', 'repeat', 'fold']).auc.values, equal_nan=True, rtol=0, atol=1e-12)
    pdiff = pd.concat([s7.paired(fc['R_all'], fM, 'R_all − M_diff'), s7.paired(fc['R_123'], fM, 'R_123 − M_diff'),
                       s7.paired(fc['R_all'], fc['R_123'], 'R_all − R_123'),
                       s7.paired(fc['R_all_rank'], fM, 'R_all_rank − M_diff'),
                       s7.paired(fc['R_all'], fc['R_all_rank'], 'R_all − R_all_rank')])
    pdiff.to_csv(os.path.join(OUT_DIR, 'paired_diff_folds.csv'), index=False, encoding='utf-8-sig')
    pds = s7.paired_summary(pdiff)
    pds['positive_ci'] = pds.rep_q025 > 0
    pds.to_csv(os.path.join(OUT_DIR, 'paired_diff_summary.csv'), index=False, encoding='utf-8-sig')
    log('\n[対ごとの差(同じテスト件・同じ乱数)]')
    for _, r in pds.iterrows():
        log(f'  {r.comparison} {r.metric}: 平均 {r.mean_diff:+.3f}、繰り返し平均の2.5〜97.5%点 [{r.rep_q025:+.3f}, '
            f'{r.rep_q975:+.3f}]、正の割合 {r.rep_frac_pos:.2f}')

    # W の差のブートストラップ
    k_m = [seeds13.index(s) for s in seeds]
    boot_rows = []
    for name, Sa, Sb in [('R_all − M_diff', S['R_all'], S_m[k_m]), ('R_123 − M_diff', S['R_123'], S_m[k_m]),
                         ('R_all − R_123', S['R_all'], S['R_123'])]:
        b, pt = s16.w_diff_bootstrap(Sa, Sb, meta123, sp, args.n_boot)
        b = b.rename(columns={'W_GC_date': 'W_a', 'W_M_diff': 'W_b'}).assign(comparison=name)
        boot_rows.append(b)
        bq = b.W_diff.quantile([.025, .975]).values
        log(f'  W の差 {name}(市町村のブートストラップ {args.n_boot}回): 点推定 {pt["a"] - pt["b"]:+.3f}'
            f'({pt["a"]:.3f} − {pt["b"]:.3f})、95%区間 [{bq[0]:+.3f}, {bq[1]:+.3f}]、正の割合 {(b.W_diff > 0).mean():.3f}')
    pd.concat(boot_rows).to_csv(os.path.join(OUT_DIR, 'w_diff_bootstrap.csv'), index=False, encoding='utf-8-sig')

    plot_overview(summary)
    show = ['config', 'metric', 'mean', 'seed_min', 'seed_max', 'null_mean', 'null_q95', 'p', 'above_q95',
            'alpha_at_max', 'alpha_at_min', 'n_train_mean']
    log('\n[まとめ]')
    log(summary.reindex(columns=show).round(3).to_string(index=False))
    r = summary[(summary.config == 'R_all') & (summary.metric == 'all_auc')].iloc[0]
    d = pds[(pds.comparison == 'R_all − M_diff') & (pds.metric == 'auc')].iloc[0]
    c1, c2 = bool(d.rep_q025 > 0), bool(r['mean'] > r.null_q95)
    log(f'\n[採用の判定] 1. R_all − M_diff の繰り返し平均の 2.5〜97.5%点 [{d.rep_q025:+.3f}, {d.rep_q975:+.3f}] が正の側: {c1}  '
        f'2. R_all の全体 AUC {r["mean"]:.3f} > 95%点 {r.null_q95:.3f}(p={r.p:.3f}): {c2} → '
        + ('採用する(R_all を主の設定にする)' if c1 and c2 else '採用しない(主の設定は M_diff のまま)'))
    log(f'\n全体 {time.time() - t0:.0f} 秒。出力: {OUT_DIR}')
    logf.close()


if __name__ == '__main__':
    main()
