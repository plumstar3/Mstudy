"""
段階37: 段階36の検定(max-T・単独の検定)の検出力をシミュレーションで見積もる(EXP-046、D-017)。

段階36の123件・27集団と実データの特徴量22個(+ 対照 P)をそのまま使い、目的変数だけを作る。
  層ごとに信号の特徴量の順位を正規スコア x′ = Φ⁻¹((rank − 0.5)/n) にし、y = −r·x′ + √(1 − r²)·ε。
  r は層内 Spearman の目標値 ρ_s = (6/π)·arcsin(r/2) から逆算(向きは段階36の観測と同じ負)。
  Y2 は y で、ラベル付き120件の中で各層の多収・低収の件数を保ってラベルを付け直す。
帰無分布は同点の無い y(各層の順位)で層内置換10,000回を作り直す(Y1: y の並べ替え、Y2: ラベル付きの件の中でラベルの並べ替え)。
検出: max-T = 族の選択込みの p < 0.05、単独の検定 = その組の補正前の両側 p < 0.05。各条件2000回、乱数 default_rng([37, 条件, i])。
信号は1つの特徴量だけ・全層で同じ強さと仮定するので、検出力は楽観的な見積もり(上限)。

実行: rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage37\\run_stage37.py
出力: rocket/outputs/stage37/
"""
import os
import sys
import time
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')
STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROCKET_DIR = os.path.dirname(os.path.dirname(STAGE_DIR))
sys.path.insert(0, ROCKET_DIR)
sys.path.insert(0, os.path.join(ROCKET_DIR, 'analysis', 'stage36'))
import run_stage36 as s36  # noqa: E402
from src.evaluation import permute_within  # noqa: E402

D36 = os.path.join(ROCKET_DIR, 'data', 'processed', 'stage36')
O36 = os.path.join(ROCKET_DIR, 'outputs', 'stage36')
OUT = os.path.join(ROCKET_DIR, 'outputs', 'stage37')
LABEL = s36.LABEL
RHOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
TARGETS = [('N_post_k3|θ0', '主'), ('rate_k3|θ0', '参考')]
N_SIM = 2000
B = 10000
ALPHA = 0.05
INK, MUTED, GRID = s36.INK, s36.MUTED, s36.GRID
matplotlib.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return c - h, c + h


def within_rank(v, strata):
    r = np.empty(len(v))
    for g in np.unique(strata):
        idx = np.flatnonzero(strata == g)
        r[idx] = stats.rankdata(v[idx])
    return r


def relabel(Y, lab, strata):
    """Y (R, n) から、ラベル付きの件の中で各層の多収・低収の件数を保ってラベルを付け直す。"""
    L = np.full(Y.shape, np.nan)
    for g in np.unique(strata):
        idx = np.flatnonzero((strata == g) & np.isfinite(lab))
        if len(idx) == 0:
            continue
        n_hi = int((lab[idx] == 1).sum())
        order = np.argsort(np.argsort(Y[:, idx], axis=1), axis=1)     # 0 = 最小
        L[:, idx] = (order >= len(idx) - n_hi).astype(float)
    return L


def stats_for(F, spec, Y, L, strata, which):
    """which = 'Y1' なら S1・S2、'Y2' なら S3。戻り値 (組の表, (R, 組数))。"""
    rows, T = [], []
    for j, r in spec.iterrows():
        sts = [('S1', s36.stat_S1), ('S2', s36.stat_S2)] if which == 'Y1' else [('S3', s36.stat_S3)]
        for st, fn in sts:
            rows.append({'fname': r.fname, 'stat': st, 'control': r.feature == 'P'})
            T.append(fn(F[:, j], Y if which == 'Y1' else L, strata))
    return pd.DataFrame(rows), np.column_stack(T)


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    os.makedirs(OUT, exist_ok=True)
    logf = open(os.path.join(OUT, 'run_log.txt'), 'w', encoding='utf-8')

    def log(m):
        print(m)
        logf.write(m + '\n')
        logf.flush()

    t0 = time.time()
    log(f'段階37(D-017)。各条件 {N_SIM} 回、帰無分布の置換 {B} 回')
    meta = pd.read_csv(os.path.join(D36, 'meta.csv'), encoding='utf-8-sig', dtype={'muniCd': str})
    TH = np.load(os.path.join(D36, 'theta.npy'))
    F, spec, _ = s36.features(TH)
    fchk = pd.read_csv(os.path.join(O36, 'features.csv'), encoding='utf-8-sig')
    log(f'特徴量は段階36の features.csv と一致: {np.allclose(F, fchk[spec.fname].to_numpy(), equal_nan=True)}')
    strata = meta.muniyear.to_numpy()
    lab = meta[LABEL].to_numpy(float)
    n = len(meta)
    log(f'{n} 件・{len(np.unique(strata))} 集団、ラベル付き {int(np.isfinite(lab).sum())}')

    # 帰無分布(同点の無い y = 各層の順位)
    y0 = within_rank(np.arange(n, dtype=float), strata)
    perm = np.empty((B + 1, n), dtype=int)
    perm[0] = np.arange(n)
    lab_idx = np.flatnonzero(np.isfinite(lab))
    permL = np.tile(lab, (B + 1, 1))
    for i in range(B):
        perm[i + 1] = permute_within(np.arange(n), strata, np.random.default_rng([0, i]))
        pl = permute_within(lab_idx, strata[lab_idx], np.random.default_rng([1, i]))
        permL[i + 1, lab_idx] = lab[pl]
    c1, N1 = stats_for(F, spec, y0[perm][1:], None, strata, 'Y1')
    c2, N2 = stats_for(F, spec, None, permL[1:], strata, 'Y2')
    nulls = {}
    for fam, c, N in [('Y1', c1, N1), ('Y2', c2, N2)]:
        mu, sd = N.mean(0), N.std(0, ddof=1)
        Za = np.abs((N - mu) / sd)
        fam_m = ~c.control.to_numpy()
        mx = Za[:, fam_m].max(1)
        nulls[fam] = dict(c=c, mu=mu, sd=sd, Za=Za, fam_m=fam_m, mx=mx)
        log(f'[帰無分布 {fam}] 族 {int(fam_m.sum())} 組、族内最大 |z| の95%点 {np.quantile(mx, 0.95):.3f}'
            f'(段階36 は Y1 2.892 / Y2 2.841)')
    log(f'帰無分布の計算 {time.time() - t0:.1f} 秒')

    def p_sel(fam, Zabs):          # Zabs (R, 組数) → 族の選択込みの p の最小 (R,)
        nl = nulls[fam]
        mx_obs = Zabs[:, nl['fam_m']].max(1)
        srt = np.sort(nl['mx'])
        cnt = len(srt) - np.searchsorted(srt, mx_obs - 1e-12, side='left')
        return (cnt + 1) / (B + 1), Zabs[:, nl['fam_m']].argmax(1)

    def p_raw(fam, j, zabs):
        srt = np.sort(nulls[fam]['Za'][:, j])
        cnt = len(srt) - np.searchsorted(srt, zabs - 1e-12, side='left')
        return (cnt + 1) / (B + 1)

    rows, wrows = [], []
    for ti, (tname, trole) in enumerate(TARGETS):
        x = F[:, spec.index[spec.fname == tname][0]]
        u = np.empty(n)
        for g in np.unique(strata):
            idx = np.flatnonzero(strata == g)
            u[idx] = (stats.rankdata(x[idx]) - 0.5) / len(idx)
        xp = stats.norm.ppf(u)
        for ri, rho in enumerate(RHOS):
            r = 2 * np.sin(np.pi * rho / 6)
            rng = np.random.default_rng([37, ti * 10 + ri])
            E = rng.standard_normal((N_SIM, n))
            Y = -r * xp[None, :] + np.sqrt(1 - r ** 2) * E
            L = relabel(Y, lab, strata)
            res = {}
            for fam, Yf, Lf in [('Y1', Y, None), ('Y2', None, L)]:
                nl = nulls[fam]
                _, T = stats_for(F, spec, Yf, Lf, strata, fam)
                Zabs = np.abs((T - nl['mu']) / nl['sd'])
                ps, top = p_sel(fam, Zabs)
                famnames = nl['c'][nl['fam_m']].fname.to_numpy()
                res[fam] = dict(T=T, Zabs=Zabs, ps=ps, top_is_target=(famnames[top] == tname))
            out = {'target': tname, 'role': trole, 'rho': rho, 'r_normal': r}
            for fam in ['Y1', 'Y2']:
                k = int((res[fam]['ps'] < ALPHA).sum())
                lo, hi = wilson(k, N_SIM)
                out[f'{fam}_maxT'] = k / N_SIM
                out[f'{fam}_maxT_lo'], out[f'{fam}_maxT_hi'] = lo, hi
                out[f'{fam}_top_is_target'] = res[fam]['top_is_target'].mean()
            for fam, st in [('Y1', 'S1'), ('Y1', 'S2'), ('Y2', 'S3')]:
                c = nulls[fam]['c']
                j = c.index[(c.fname == tname) & (c.stat == st)][0]
                pr = p_raw(fam, j, res[fam]['Zabs'][:, j])
                k = int((pr < ALPHA).sum())
                lo, hi = wilson(k, N_SIM)
                out[f'single_{st}'] = k / N_SIM
                out[f'single_{st}_lo'], out[f'single_{st}_hi'] = lo, hi
                v = res[fam]['T'][:, j]
                out[f'realized_{st}_mean'] = np.nanmean(v)
                if st == 'S3':
                    w = 1 - v                                  # 信号の向き(負)にそろえた W
                    wrows.append({'target': tname, 'role': trole, 'rho': rho, 'W_mean': w.mean(),
                                  'W_q025': np.quantile(w, 0.025), 'W_q975': np.quantile(w, 0.975),
                                  'W_min': w.min(), 'W_max': w.max(), 'S1_mean': out['realized_S1_mean'],
                                  'single_S3_power': out['single_S3'], 'Y2_maxT_power': out['Y2_maxT']})
            rows.append(out)
            log(f'{tname} ρ = {rho:.1f}: 実現 S1 {out["realized_S1_mean"]:+.3f}、max-T Y1 {out["Y1_maxT"]:.3f}・Y2 '
                f'{out["Y2_maxT"]:.3f}、単独 S1 {out["single_S1"]:.3f}・S2 {out["single_S2"]:.3f}・S3 {out["single_S3"]:.3f}、'
                f'W 平均 {wrows[-1]["W_mean"]:.3f} [{wrows[-1]["W_q025"]:.3f}, {wrows[-1]["W_q975"]:.3f}]')
    pw = pd.DataFrame(rows)
    wt = pd.DataFrame(wrows)
    pw.to_csv(os.path.join(OUT, 'power.csv'), index=False, encoding='utf-8-sig')
    wt.to_csv(os.path.join(OUT, 'rho_W_table.csv'), index=False, encoding='utf-8-sig')

    # 検出力80%に必要な ρ(線形補間)
    need = []
    for tname, trole in TARGETS:
        d = pw[pw.target == tname].sort_values('rho')
        for col in ['Y1_maxT', 'single_S1', 'single_S2', 'Y2_maxT', 'single_S3']:
            yv, xv = d[col].to_numpy(), d.rho.to_numpy()
            k = np.flatnonzero(yv >= 0.8)
            if len(k) == 0:
                val = np.nan
            else:
                k = k[0]
                val = xv[k] if k == 0 else xv[k - 1] + (0.8 - yv[k - 1]) * (xv[k] - xv[k - 1]) / (yv[k] - yv[k - 1])
            need.append({'target': tname, 'role': trole, 'test': col, 'rho_for_80pct': val})
    need = pd.DataFrame(need)
    need.to_csv(os.path.join(OUT, 'rho_for_80pct.csv'), index=False, encoding='utf-8-sig')
    log('\n[検出力80%に必要な ρ(線形補間、0.5 までに届かなければ NaN)]\n' + need.round(3).to_string(index=False))
    log('\n[ρ と W の対応(Y2、S3、1 − AUC)]\n' + wt.round(3).to_string(index=False))
    log('\n[検出力の表]\n' + pw[['target', 'rho', 'realized_S1_mean', 'Y1_maxT', 'single_S1', 'single_S2', 'Y2_maxT',
                               'single_S3', 'Y1_top_is_target']].round(3).to_string(index=False))

    # 図
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    lines = [('Y1_maxT', 'max-T Y1(44組、段階36の主)', '#2a78d6', '-'), ('single_S1', '単独 S1', '#2a78d6', '--'),
             ('single_S2', '単独 S2', '#7fb0ea', ':'), ('Y2_maxT', 'max-T Y2(22組)', '#1f9e74', '-'),
             ('single_S3', '単独 S3', '#1f9e74', '--')]
    for ax, (tname, trole) in zip(axes, TARGETS):
        d = pw[pw.target == tname].sort_values('rho')
        for col, nm, c, ls in lines:
            ax.fill_between(d.rho, d[f'{col}_lo'], d[f'{col}_hi'], color=c, alpha=0.12, lw=0)
            ax.plot(d.rho, d[col], color=c, ls=ls, lw=2, marker='o', ms=4, label=nm)
        ax.axhline(0.8, color=MUTED, lw=0.8, ls='--')
        ax.axhline(0.05, color=MUTED, lw=0.6, ls=':')
        ax.axvspan(0.0, 0.316, color=GRID, alpha=0.35, lw=0)
        ax.set_title(f'信号の特徴量: {tname}({trole})', loc='left', fontsize=10, color=INK)
        ax.set_xlabel('層内 Spearman ρ の大きさ(シミュレーションの設定)', color=INK)
        s36.style(ax)
    axes[0].set_ylabel('検出力', color=INK)
    axes[0].legend(frameon=False, fontsize=8, loc='upper left')
    fig.text(0.01, 0.005, '帯は Wilson の95%区間。灰色の範囲は段階36の最良の組の ρ の95%区間の上端(|ρ| ≤ 0.316)まで。'
             '信号は1つの特徴量・全層で同じ強さと仮定(楽観的な見積もり)', fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(OUT, 'fig_power.png'), dpi=150)
    plt.close(fig)
    log(f'\n所要時間 {time.time() - t0:.1f} 秒')
    logf.close()


if __name__ == '__main__':
    main()
