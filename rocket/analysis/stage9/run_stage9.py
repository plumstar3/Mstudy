"""
段階9: 開花日のずれに対する ROCKET(MiniRocket)の感度分析。

予測した開花日を窓(開花日 0〜+59 日)の起点に使っても、多収/低収の分類精度が保てるかを確かめる。
窓の起点だけを変え、ほかの設定は段階6の実験D(確認用の解析)と同じにする。

  base   実測の開花日
  m6/m3/p3/p6  実測の開花日を −6 / −3 / +3 / +6 日ずらす
  pred   播種日回帰の予測値(EXP-016 の GroupKFold(5, 地域×年)、D1〜D4・2015 を含める、の
         out-of-fold 予測 pred_seed_reg。outputs/flowering_raw_seed/cv_predictions.csv)。四捨五入して整数の日にする
         (predict_flowering_targets.py の pred_flower_date と同じ丸め方)

データ: main(82件)のうち実測の開花日がある60件。起点をずらすと窓が観測期間からはみ出す件があるので、
  各条件は「基準と、その条件の両方で窓がそろう件」(対ごとの共通集合)で、基準と同じ件・同じ分割で比べる。
  参考として基準を60件全体でも回す。
モデル・評価は段階6の実験Dと同じ: 1次差分(59点)、sktime 1.2.0 の MiniRocket(num_kernels=10000、乱数0〜9)
  → StandardScaler(with_mean=False) → RidgeClassifierCV(alphas = logspace(−3, 3, 10) × n_train)、
  StratifiedGroupKFold(5, groups=muniCd)×20回(分割の乱数0〜19)。指標は ROC-AUC と balanced accuracy。
  置換検定はしない。差は段階7の paired_summary(20回の繰り返しの95%以上で符号がそろえば差あり)。

実行:
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage9\\run_stage9.py
  試運転: ... run_stage9.py --n-seeds 1
出力: rocket/outputs/stage9/
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROCKET_DIR = os.path.dirname(os.path.dirname(STAGE_DIR))
sys.path.insert(0, ROCKET_DIR)
sys.path.insert(0, os.path.join(ROCKET_DIR, 'analysis', 'stage6'))
sys.path.insert(0, os.path.join(ROCKET_DIR, 'analysis', 'stage7'))
import run_stage6 as s6  # noqa: E402
import run_stage7 as s7  # noqa: E402
from src.windows import extract_windows, first_diff, load_cases, load_timeseries  # noqa: E402

OUT_DIR = os.path.join(ROCKET_DIR, 'outputs', 'stage9')
PROC = os.path.join(ROCKET_DIR, 'data', 'processed')
CV_PRED = os.path.join(ROCKET_DIR, 'outputs', 'flowering_raw_seed', 'cv_predictions.csv')
KEY = ['field_id', 'year']
SHIFTS = {'m6': -6, 'm3': -3, 'p3': 3, 'p6': 6}
LABELS = {'base': '基準(実測)', 'm6': '−6日', 'm3': '−3日', 'p3': '+3日', 'p6': '+6日', 'pred': '予測値(OOF)'}


def load_obs_cases():
    """main の実測開花日の60件。flower_dap は実測、pred_dap は EXP-016 の OOF 予測(四捨五入)。"""
    main = pd.read_csv(os.path.join(PROC, 'stage4', 'main_meta.csv'), encoding='utf-8-sig', dtype={'muniCd': str})
    main = main[~main.flower_estimated]
    c = load_cases().merge(main[KEY + ['flower_dap']], on=KEY)
    assert (c.flower_dap == c.flower_dap_obs).all(), '段階4の main と実測の開花日が違う'
    cv = pd.read_csv(CV_PRED, encoding='utf-8-sig')
    cv = cv[(cv.variant == 'incl_D') & (cv.scheme == 'group')]
    c = c.merge(cv[KEY + ['d2f', 'pred_seed_reg', 'fold']].rename(columns={'fold': 'flower_cv_fold'}), on=KEY, how='left')
    assert c.pred_seed_reg.notna().all() and (c.d2f == c.flower_dap).all(), 'OOF 予測が無い件か、d2f の食い違いがある'
    c['pred_dap'] = c.pred_seed_reg.round().astype(int)
    c['pred_err'] = c.pred_dap - c.flower_dap
    return c.drop(columns=['d2f']).reset_index(drop=True)


def windows_for(cases, ts, cond):
    d = cases.copy()
    d['flower_dap'] = d['pred_dap'] if cond == 'pred' else d['flower_dap'] + SHIFTS.get(cond, 0)
    X, meta, excl = extract_windows(d, ts, 'flower', 0, 59)
    return X, meta, excl.assign(condition=cond)


def subset(X, meta, keys):
    k = list(zip(meta.field_id, meta.year))
    keep = np.array([t in keys for t in k])
    return X[keep], meta[keep].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-seeds', type=int, default=10, help='試運転用。本番は10')
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    logf = open(os.path.join(OUT_DIR, 'run_log.txt'), 'w', encoding='utf-8')

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()

    t0 = time.time()
    s6.OUT_DIR = OUT_DIR
    s6.SEEDS = list(range(args.n_seeds))
    log(f'並列数 {s6.N_JOBS}、乱数 {s6.SEEDS[0]}〜{s6.SEEDS[-1]}、置換なし')

    cases = load_obs_cases()
    ts = load_timeseries(cases)
    e = cases.pred_err
    log(f'実測60件の OOF 予測の誤差(予測 − 実測、日): MAE {e.abs().mean():.2f}、偏り {e.mean():+.2f}、'
        f'範囲 {e.min()}〜{e.max()}、|誤差|≤3 {int((e.abs() <= 3).sum())} 件、≥6 {int((e.abs() >= 6).sum())} 件')

    W = {c: windows_for(cases, ts, c) for c in ['base', *SHIFTS, 'pred']}
    excl = pd.concat([w[2] for w in W.values()], ignore_index=True)
    excl[['condition', 'field_id', 'year', 'muniCd', 'y', 'flower_dap', 'pred_dap', 'reason']].to_csv(
        os.path.join(OUT_DIR, 'excluded.csv'), index=False, encoding='utf-8-sig')
    assert len(W['base'][1]) == 60

    # 件ごとの開花日と、どの条件で窓がそろうか
    cov = cases[KEY + ['muniCd', 'muniyear', 'y', 'flower_dap', 'pred_dap', 'pred_err', 'flower_cv_fold']].copy()
    for c, (_, m, _) in W.items():
        keys = set(zip(m.field_id, m.year))
        cov[f'ok_{c}'] = [t in keys for t in zip(cov.field_id, cov.year)]
    cov.to_csv(os.path.join(OUT_DIR, 'cases.csv'), index=False, encoding='utf-8-sig')

    rows, folds_all, seed_all, splits_all, pair_rows = [], [], [], [], []

    def run(config, cond, comparison, X, meta):
        y = meta.y.values
        r, f, ps, st = s6.run_config(config, LABELS[cond], comparison, cond, first_diff(X), y, meta,
                                     'minirocket', 0, log)
        r.update({'condition': cond, 'comparison': comparison, 'n_regions': meta.muniCd.nunique(),
                  'auc_seed_sd': ps.auc_mean.std(), 'bacc_seed_sd': ps.bacc_mean.std()})
        rows.append(r)
        folds_all.append(f.assign(condition=cond, comparison=comparison))
        seed_all.append(ps.assign(condition=cond, comparison=comparison))
        splits_all.append(st.assign(comparison=comparison))
        return f

    # 参考: 基準を60件全体で
    run('base_all60', 'base', 'all60', *W['base'][:2])

    # 対ごとの共通集合で、基準と各条件
    base_keys = set(zip(W['base'][1].field_id, W['base'][1].year))
    for c in [*SHIFTS, 'pred']:
        keys = base_keys & set(zip(W[c][1].field_id, W[c][1].year))
        Xb, mb = subset(*W['base'][:2], keys)
        Xc, mc = subset(*W[c][:2], keys)
        assert (mb[KEY].values == mc[KEY].values).all() and (mb.y.values == mc.y.values).all()
        log(f'[{LABELS[c]}] 共通 {len(keys)} 件(多収 {int(mb.y.sum())} / 低収 {int((1 - mb.y).sum())}、'
            f'{mb.muniCd.nunique()} 地域)')
        fb = run(f'base_vs_{c}', 'base', c, Xb, mb)
        fc = run(f'{c}', c, c, Xc, mc)
        pair_rows.append(s7.paired(fc, fb, f'{LABELS[c]} − 基準'))

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(OUT_DIR, 'summary.csv'), index=False, encoding='utf-8-sig')
    pd.concat(folds_all).to_csv(os.path.join(OUT_DIR, 'folds.csv'), index=False, encoding='utf-8-sig')
    pd.concat(seed_all).to_csv(os.path.join(OUT_DIR, 'per_seed.csv'), index=False, encoding='utf-8-sig')
    pd.concat(splits_all).to_csv(os.path.join(OUT_DIR, 'splits.csv'), index=False, encoding='utf-8-sig')
    pairs = pd.concat(pair_rows, ignore_index=True)
    pairs.to_csv(os.path.join(OUT_DIR, 'paired_diff_folds.csv'), index=False, encoding='utf-8-sig')
    ps = s7.paired_summary(pairs)
    ps.to_csv(os.path.join(OUT_DIR, 'paired_diff_summary.csv'), index=False, encoding='utf-8-sig')

    pd.set_option('display.width', 250)
    cols = ['config', 'n', 'n_high', 'n_low', 'n_regions', 'auc_mean', 'auc_seed_sd', 'auc_seed_min', 'auc_seed_max',
            'auc_sd_repeat', 'bacc_mean', 'bacc_seed_sd', 'bacc_sd_repeat', 'alpha_at_max']
    log('\n[まとめ]\n' + summary[cols].round(3).to_string(index=False))
    log('\n[基準との fold ごとの差(10通りの乱数で平均)]\n' + ps.round(3).to_string(index=False))
    log(f'\n全体 {time.time() - t0:.0f} 秒。出力: {OUT_DIR}')


if __name__ == '__main__':
    main()
