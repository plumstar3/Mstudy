"""
段階6の事前確認(本番の前に行う)。

  0-1. 旧 scikit-learn の RidgeClassifierCV(normalize=True) を手で再現したものと、
       StandardScaler(with_mean=False) + alpha × n_train が等価か。
       段階5の main・diff・random_state=0 の MiniRocket 特徴量で、いくつかの fold と alpha を比べる。
  0-2. sktime の Rocket / MiniRocket の get_params()(normalise の既定値)。

実行:
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage6\\precheck.py
出力: rocket/outputs/stage6/precheck_*.csv
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROCKET_DIR = os.path.dirname(os.path.dirname(STAGE_DIR))
sys.path.insert(0, ROCKET_DIR)
sys.path.insert(0, os.path.join(ROCKET_DIR, 'analysis', 'stage5'))
from run_stage5 import SktimePanel, load  # noqa: E402
from src.evaluation import make_splits  # noqa: E402
from src.windows import first_diff  # noqa: E402

OUT_DIR = os.path.join(ROCKET_DIR, 'outputs', 'stage6')
ALPHAS = np.logspace(-3, 3, 10)


def old_normalize(Ftr, Fte):
    """旧 normalize=True: 学習データで列を中心化し、中心化した列の L2 ノルムで割る(定数列は1)。"""
    mu = Ftr.mean(axis=0)
    norm = np.sqrt(((Ftr - mu) ** 2).sum(axis=0))
    norm[norm == 0] = 1.0
    return (Ftr - mu) / norm, (Fte - mu) / norm


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    X, m = load('main')
    Xd, y = first_diff(X), m.y.values
    splits = make_splits(y, m.muniCd.values, 5, 20, 0)
    picks = [splits[i] for i in (0, 1, 2, 3, 4, 37, 73, 99)]  # 繰り返し0の5 fold と、ほかの繰り返しから3 fold

    rows, rows_fixed = [], []
    for r, f, tr, te in picks:
        tf = SktimePanel('minirocket', random_state=0).fit(Xd[tr])
        Ftr, Fte = tf.transform(Xd[tr]), tf.transform(Xd[te])
        n = len(tr)
        # 旧 normalize=True の再現 + 原論文の alpha
        Ntr, Nte = old_normalize(Ftr, Fte)
        a = RidgeClassifierCV(alphas=ALPHAS).fit(Ntr, y[tr])
        # StandardScaler(with_mean=False) + alpha × n_train
        sc = StandardScaler(with_mean=False).fit(Ftr)
        b = RidgeClassifierCV(alphas=ALPHAS * n).fit(sc.transform(Ftr), y[tr])
        da, db = a.decision_function(Nte), b.decision_function(sc.transform(Fte))
        rows.append({'repeat': r, 'fold': f, 'n_train': n, 'n_features': Ftr.shape[1],
                     'n_constant_features': int((Ftr.std(axis=0) == 0).sum()),
                     'alpha_old': a.alpha_, 'alpha_new': b.alpha_, 'alpha_new_over_n': b.alpha_ / n,
                     'same_alpha_index': bool(np.isclose(a.alpha_ * n, b.alpha_)),
                     'max_abs_diff_decision': float(np.abs(da - db).max()),
                     'max_abs_decision': float(np.abs(da).max()),
                     'same_predictions': bool((a.predict(Nte) == b.predict(sc.transform(Fte))).all())})
        # alpha を固定した RidgeClassifier でも比べる
        for al in (1e-3, 1e-1, 1.0, 10.0, 1e3):
            c = RidgeClassifier(alpha=al).fit(Ntr, y[tr])
            e = RidgeClassifier(alpha=al * n).fit(sc.transform(Ftr), y[tr])
            dc, de = c.decision_function(Nte), e.decision_function(sc.transform(Fte))
            rows_fixed.append({'repeat': r, 'fold': f, 'alpha_old': al, 'alpha_new': al * n,
                               'max_abs_diff_decision': float(np.abs(dc - de).max()),
                               'max_abs_decision': float(np.abs(dc).max())})
        # 参考: 段階5の設定(alpha × n_train なし)で選ばれた alpha
        rows[-1]['alpha_stage5_setting'] = RidgeClassifierCV(alphas=ALPHAS).fit(sc.transform(Ftr), y[tr]).alpha_

    cv = pd.DataFrame(rows)
    fx = pd.DataFrame(rows_fixed)
    cv.to_csv(os.path.join(OUT_DIR, 'precheck_alpha_equivalence_cv.csv'), index=False, encoding='utf-8-sig')
    fx.to_csv(os.path.join(OUT_DIR, 'precheck_alpha_equivalence_fixed.csv'), index=False, encoding='utf-8-sig')
    print('[0-1] RidgeClassifierCV: 旧 normalize の再現 vs StandardScaler + alpha × n_train')
    print(cv.to_string(index=False))
    print('\n[0-1] RidgeClassifier(alpha 固定)')
    print(fx.groupby('alpha_old')[['max_abs_diff_decision', 'max_abs_decision']].max().to_string())
    ok = cv.same_alpha_index.all() and cv.same_predictions.all() and \
        (cv.max_abs_diff_decision <= 1e-6 * np.maximum(1, cv.max_abs_decision)).all() and \
        (fx.max_abs_diff_decision <= 1e-6 * np.maximum(1, fx.max_abs_decision)).all()
    print('等価:', ok)

    print('\n[0-2] sktime の get_params()')
    from sktime.transformations.panel.rocket import MiniRocket, Rocket
    p = []
    for cls in (Rocket, MiniRocket):
        params = cls().get_params()
        print(f'  {cls.__name__}: {params}')
        p.append({'transformer': cls.__name__, **{k: repr(v) for k, v in params.items()}})
    pd.DataFrame(p).to_csv(os.path.join(OUT_DIR, 'precheck_sktime_params.csv'), index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()
