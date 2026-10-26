"""
多収/低収分類の評価の仕組み(段階2で作成、段階5の ROCKET でも再利用する)。
`sys.path` に rocket/ を入れて `from src.evaluation import ...` で使う。

- 分割: muniCd をグループにした StratifiedGroupKFold(n_splits=5, shuffle=True)を
  乱数シードを変えて n_repeats 回繰り返す。分割は y と groups だけで決まるので、
  一度作ったものを特徴量セット間で使い回す(同じ分割で比べるため)。
- 指標: balanced accuracy と ROC-AUC(各 fold で計算)。
- 前処理は estimator(Pipeline)の中に入れ、学習データだけで fit する。
- 置換検定: ラベルを perm_groups(市町村×年)の中で並べ替える。グループ内のクラス数が
  保たれるので StratifiedGroupKFold の分割は並べ替え前と同じになり、分割も使い回せる。

X は numpy 配列でも、長さの揃わない系列のリストでもよい(行番号で取り出す)。
"""
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


def _take(X, idx):
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return X.iloc[idx]
    if isinstance(X, list):
        return [X[i] for i in idx]
    return X[idx]


def _score(est, X):
    """AUC 用の連続値スコア。"""
    if hasattr(est, 'decision_function'):
        return est.decision_function(X)
    return est.predict_proba(X)[:, 1]


def make_splits(y, groups, n_splits=5, n_repeats=20, seed0=0):
    """[(repeat, fold, train_idx, test_idx), ...] を返す。"""
    y = np.asarray(y)
    groups = np.asarray(groups)
    splits = []
    for r in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed0 + r)
        for f, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y, groups)):
            splits.append((r, f, tr, te))
    return splits


def split_table(splits, y, groups):
    """各分割のテスト側の件数・クラス比・グループ数。"""
    y = np.asarray(y)
    groups = np.asarray(groups)
    rows = []
    for r, f, tr, te in splits:
        rows.append({'repeat': r, 'fold': f, 'n_train': len(tr), 'n_test': len(te),
                     'n_test_pos': int(y[te].sum()), 'n_test_neg': int((1 - y[te]).sum()),
                     'test_pos_ratio': y[te].mean(), 'n_test_groups': len(np.unique(groups[te]))})
    return pd.DataFrame(rows)


def evaluate(estimator, X, y, splits):
    """各分割で学習・評価し、fold ごとの指標を DataFrame で返す。

    テスト側が1クラスだけの fold は AUC を NaN にする。
    """
    y = np.asarray(y)
    rows = []
    for r, f, tr, te in splits:
        with warnings.catch_warnings():
            # sklearn 1.5.1 と scipy 1.18 の組み合わせで lbfgs が出す無害な警告(iprint)
            warnings.filterwarnings('ignore', message='Unknown solver options')
            est = clone(estimator).fit(_take(X, tr), y[tr])
        Xte = _take(X, te)
        pred = est.predict(Xte)
        auc = np.nan
        if len(np.unique(y[te])) == 2:
            auc = roc_auc_score(y[te], _score(est, Xte))
        rows.append({'repeat': r, 'fold': f, 'n_test': len(te),
                     'bacc': balanced_accuracy_score(y[te], pred), 'auc': auc})
    return pd.DataFrame(rows)


def summarize(fold_df):
    """fold 単位(全 fold)の平均・標準偏差と、繰り返しごとの平均の標準偏差。"""
    rep = fold_df.groupby('repeat')[['bacc', 'auc']].mean()
    return {
        'bacc_mean': fold_df.bacc.mean(), 'bacc_sd_fold': fold_df.bacc.std(),
        'bacc_sd_repeat': rep.bacc.std(),
        'auc_mean': fold_df.auc.mean(), 'auc_sd_fold': fold_df.auc.std(),
        'auc_sd_repeat': rep.auc.std(),
        'n_auc_nan': int(fold_df.auc.isna().sum()),
    }


def permute_within(y, perm_groups, rng):
    """perm_groups の中だけでラベルを並べ替える。"""
    y = np.asarray(y).copy()
    perm_groups = np.asarray(perm_groups)
    for g in np.unique(perm_groups):
        idx = np.flatnonzero(perm_groups == g)
        y[idx] = y[rng.permutation(idx)]
    return y


def permutation_test(estimator, X, y, splits, perm_groups, n_perm=200, seed=0, n_jobs=-1):
    """置換検定。観測値と、並べ替えたラベルでの平均指標の分布から p 値を出す。

    p = (1 + #{null >= observed}) / (1 + n_perm)
    """
    observed = summarize(evaluate(estimator, X, y, splits))
    rngs = [np.random.default_rng([seed, i]) for i in range(n_perm)]

    def one(rng):
        yp = permute_within(y, perm_groups, rng)
        s = summarize(evaluate(estimator, X, yp, splits))
        return s['bacc_mean'], s['auc_mean']

    null = np.array(Parallel(n_jobs=n_jobs)(delayed(one)(r) for r in rngs))
    out = dict(observed)
    for j, m in enumerate(['bacc', 'auc']):
        out[f'{m}_null_mean'] = np.nanmean(null[:, j])
        out[f'{m}_null_q95'] = np.nanquantile(null[:, j], 0.95)
        out[f'{m}_p'] = (1 + np.sum(null[:, j] >= observed[f'{m}_mean'])) / (1 + n_perm)
    return out, null
