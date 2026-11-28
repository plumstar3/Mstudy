"""
段階30: 段階27・29の設定どうしの AUC の差の、Holm 補正前の両側 p 値(EXP-039)。

モデルの学習・置換検定は回し直さず、保存済みの結果だけを使う。
  ブートストラップ: outputs/stage27/bootstrap_auc.csv、outputs/stage29/bootstrap_auc.csv
    (市町村単位の対応のあるクラスターブートストラップ、段階14と同じ重み付け、2000回、乱数 default_rng(0))。
    同じリサンプル番号(boot)どうしで差を取る。
  点推定: fold の AUC の平均の差(段階27・29の report と同じ)。段階27は outputs/stage27/folds.csv(D・P・N)、
    段階29は outputs/stage29/folds.csv(R・N_dry・N_post)と段階27の N。
比べる組(この7組だけ、差の向きもこのとおり):
  段階27  D − P、D − N、P − N(P − N は段階27の結果を見てから決めた事後の比較 = 探索的)
  段階29  N_dry − N_post、N − N_post、R − N、N_dry − N
両側 p = min(1, 2 × (1 + min(#{差 ≤ 0}, #{差 ≥ 0})) / (B + 1))、B = 2000。95%区間はパーセンタイル法。Holm 補正はしない。
開始時の確認: 各組の点推定と95%区間が、段階27・28・29の report の値(向きをそろえて符号を反転したもの)と一致する。

実行:
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage30\\run_stage30.py
出力: rocket/outputs/stage30/
"""
import os

import numpy as np
import pandas as pd

STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROCKET_DIR = os.path.dirname(os.path.dirname(STAGE_DIR))
OUT = os.path.join(ROCKET_DIR, 'outputs')
OUT_DIR = os.path.join(OUT, 'stage30')

# (段階, a, b, report の組と向き, report の点推定・下限・上限(report の向き), 反転するか, 注記)
PAIRS = [
    (27, 'D', 'P', 'P − D(段階27)', (-0.029, -0.088, 0.016), True, ''),
    (27, 'D', 'N', 'N − D(段階27)', (0.019, -0.040, 0.074), True, ''),
    (27, 'P', 'N', 'N − P(段階28 部分A)', (0.048, -0.031, 0.140), True, '事後の比較(探索的)'),
    (29, 'N_dry', 'N_post', 'N_dry − N_post(段階29)', (-0.194, -0.301, -0.088), False, ''),
    (29, 'N', 'N_post', 'N_post − N(段階29)', (0.041, -0.000, 0.080), True, '区間の端がほぼ0(境目)'),
    (29, 'R', 'N', 'R − N(段階29)', (-0.197, -0.329, -0.051), False, ''),
    (29, 'N_dry', 'N', 'N_dry − N(段階29)', (-0.152, -0.257, -0.050), False, ''),
]


def two_sided_p(d):
    B = len(d)
    le, ge = int((d <= 0).sum()), int((d >= 0).sum())
    return min(1.0, 2 * (1 + min(le, ge)) / (B + 1)), le, ge


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    logf = open(os.path.join(OUT_DIR, 'run_log.txt'), 'w', encoding='utf-8')

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + '\n')

    docs = open(os.path.join(ROCKET_DIR, 'docs', 'experiments.md'), encoding='utf-8').read()
    log(f'確認 段階30・EXP-039 が未使用(docs に EXP-039 なし): {"EXP-039" not in docs}')
    b27 = pd.read_csv(os.path.join(OUT, 'stage27', 'bootstrap_auc.csv'), encoding='utf-8-sig')
    b29 = pd.read_csv(os.path.join(OUT, 'stage29', 'bootstrap_auc.csv'), encoding='utf-8-sig')
    ok_b = (len(b27) == len(b29) == 2000 and b27.boot.is_unique and b29.boot.is_unique
            and np.array_equal(b27.boot, b29.boot) and np.allclose(b27.N, b29.N, rtol=0, atol=1e-12))
    log(f'確認 ブートストラップの値がリサンプル番号で対応づけて保存されている(各2000行)、段階29の N が段階27の N と'
        f'全リサンプルで一致(同じリサンプル): {ok_b}')
    assert ok_b
    f27 = pd.read_csv(os.path.join(OUT, 'stage27', 'folds.csv'), encoding='utf-8-sig')
    f29 = pd.read_csv(os.path.join(OUT, 'stage29', 'folds.csv'), encoding='utf-8-sig')
    fm = {c: f27[f27.config == c].auc.mean() for c in ['D', 'P', 'N']}
    fm.update({c: f29[f29.config == c].auc.mean() for c in ['R', 'N_dry', 'N_post']})
    nf = {c: int((f27.config == c).sum()) for c in ['D', 'P', 'N']}
    nf.update({c: int((f29.config == c).sum()) for c in ['R', 'N_dry', 'N_post']})
    log(f'  fold の数(乱数10 × 繰り返し20 × 5 fold = 1000): {nf}')
    assert all(v == 1000 for v in nf.values())

    rows, ok_all = [], True
    for st, a, b, rep, (rp, rlo, rhi), flip, note in PAIRS:
        bt = b27 if st == 27 else b29
        d = (bt[a] - bt[b]).values
        point = fm[a] - fm[b]
        lo, hi = np.quantile(d, .025), np.quantile(d, .975)
        p, le, ge = two_sided_p(d)
        s = -1 if flip else 1
        exp_p, exp_lo, exp_hi = (s * rp, s * rhi, s * rlo) if flip else (rp, rlo, rhi)
        # report は小数第3位に丸めた値なので、丸めの幅(±0.0005、浮動小数の誤差を含めて 1e-6 の余裕)で比べる
        ok = all(abs(x - y) <= 0.0005 + 1e-6 for x, y in [(point, exp_p), (lo, exp_lo), (hi, exp_hi)])
        ok_all &= ok
        log(f'確認 {a} − {b}: 点推定 {point:+.4f} [{lo:+.4f}, {hi:+.4f}] / report {rep} を反転 {flip} → '
            f'{exp_p:+.3f} [{exp_lo:+.3f}, {exp_hi:+.3f}]: {ok}')
        rows.append({'stage': st, 'pair': f'{a} − {b}', 'point_fold_mean': point, 'ci_low': lo, 'ci_high': hi,
                     'p_two_sided_uncorrected': p, 'n_diff_le_0': le, 'n_diff_ge_0': ge, 'B': len(d),
                     'report_pair': rep, 'sign_flipped_from_report': flip, 'note': note})
    log(f'開始時の確認(点推定と区間が report と一致、7組): {ok_all}')
    assert ok_all, 'report の値と一致しない組がある。本番に進まず報告する'
    res = pd.DataFrame(rows)
    res['reading'] = np.where(res.p_two_sided_uncorrected < 0.05, '差あり(Holm 補正前)', '差は検出できない')
    res.to_csv(os.path.join(OUT_DIR, 'pairwise_p.csv'), index=False, encoding='utf-8-sig')
    log('\n[結果]')
    log(res[['stage', 'pair', 'point_fold_mean', 'ci_low', 'ci_high', 'p_two_sided_uncorrected', 'n_diff_le_0',
             'n_diff_ge_0', 'B', 'reading']].round(4).to_string(index=False))
    logf.close()


if __name__ == '__main__':
    main()
