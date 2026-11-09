"""
段階14: label_muniyear_median の結果の確認と、交絡の候補の基準線(EXP-023)。

段階13(EXP-022)の median(123件)で、M(MiniRocket・1次差分)は全体 AUC 0.607、W 0.536、B 0.640 だった。
  A1  B − W は偶然で説明できる範囲か。段階13の oof_scores.csv から(再学習なし)
      1. 繰り返しごとの差(乱数で平均)  2. 市町村を単位にしたクラスターブートストラップ(主、2000回、乱数0)
  A2  クラスが同数の集団どうしのペアだけでの B(と W)。モデルは全123件で学習したまま。
      置換(市町村×年の中、1000回)ごとの決定関数は _batched_ridge でまとめて解く
  B3  交絡の候補だけの基準線: 品種の one-hot(学習側に無い品種は無視、欠測は「不明」)+ 播種日の通日
      → StandardScaler(数値の列だけ)→ LR(C=1)。段階13と同じ評価・分解・置換と、M − B3 の対ごとの差

データ・分割・M の構成・置換は段階13の median と同じ(data/processed/stage13/label_muniyear_median_*)。

実行:
  rocket\\venv_rocket\\Scripts\\python rocket\\analysis\\stage14\\run_stage14.py
  試運転: ... run_stage14.py --n-seeds 1 --n-perm 20 --n-boot 50
出力: rocket/outputs/stage14/
"""
import argparse
import os
import sqlite3
import sys
import time
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings('ignore')

STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROCKET_DIR = os.path.dirname(os.path.dirname(STAGE_DIR))
sys.path.insert(0, ROCKET_DIR)
for d in ['stage6', 'stage7', 'stage11', 'stage12', 'stage13']:
    sys.path.insert(0, os.path.join(ROCKET_DIR, 'analysis', d))
import run_stage6 as s6  # noqa: E402
import run_stage6_local as fast  # noqa: E402
import run_stage7 as s7  # noqa: E402
import run_stage12 as s12  # noqa: E402
import run_stage13 as s13  # noqa: E402
from src.evaluation import evaluate, make_splits  # noqa: E402
from src.windows import first_diff  # noqa: E402

KEY, LABEL = 'median', 'label_muniyear_median'
DATA_DIR = os.path.join(ROCKET_DIR, 'data', 'processed', 'stage13')
RAW_DB = os.path.join(ROCKET_DIR, 'data', 'raw', 'FieldData_fieldid.db')
S13_DIR = os.path.join(ROCKET_DIR, 'outputs', 'stage13')
OUT_DIR = os.path.join(ROCKET_DIR, 'outputs', 'stage14')
N_SPLITS, N_REPEATS, SEED0 = s6.N_SPLITS, s6.N_REPEATS, s6.SEED0
N_JOBS = fast.N_JOBS
TYPES = ['W', 'B', 'X']
# 事前の集計: (集団数, 件数) 同数 / 両クラスで数が違う / 片方のクラスだけ
EXPECT_BAL = {'balanced': (20, 96), 'unbalanced': (5, 22), 'single': (3, 5)}
BAL_LABEL = {'balanced': '多収と低収が同数', 'unbalanced': '両クラスあるが数が違う', 'single': '片方のクラスだけ'}

matplotlib.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
INK, MUTED = s6.INK, s6.MUTED
C_W, C_B, C_ALL = s13.TYPE_COLOR['W'], s13.TYPE_COLOR['B'], s13.TYPE_COLOR['all']


# ---------------------------------------------------------------- データ
def load():
    X = np.load(os.path.join(DATA_DIR, f'{LABEL}_X.npy'))
    meta = pd.read_csv(os.path.join(DATA_DIR, f'{LABEL}_meta.csv'), encoding='utf-8-sig', dtype={'muniCd': str})
    return X, meta


def load_stage13(meta, sp, log):
    """段階13の median・M の oof スコアを (乱数, 繰り返し, n) にし、fold の AUC が folds.csv と一致するか確かめる。"""
    oof = pd.read_csv(os.path.join(S13_DIR, 'oof_scores.csv'), encoding='utf-8-sig', dtype={'muniCd': str})
    oof = oof[(oof.label == KEY) & (oof.config == 'M')]
    folds = pd.read_csv(os.path.join(S13_DIR, 'folds.csv'), encoding='utf-8-sig')
    folds = folds[(folds.label == KEY) & (folds.config == 'M')]
    pos = pd.Series(np.arange(len(meta)), index=pd.MultiIndex.from_arrays([meta.field_id, meta.year]))
    oof = oof.assign(i=pos.loc[list(zip(oof.field_id, oof.year))].values)
    seeds = sorted(oof.seed.unique().astype(int))
    S = np.full((len(seeds), N_REPEATS, len(meta)), np.nan)
    y = meta.y.values
    ok_te, auc, ref = True, [], []
    for k, s in enumerate(seeds):
        o = oof[oof.seed == s]
        fr = folds[folds.seed == s].set_index(['repeat', 'fold']).auc
        for r, f, tr, te in sp:
            g = o[(o.repeat == r) & (o.fold == f)]
            ok_te &= np.array_equal(np.sort(g.i.values), np.sort(te))
            S[k, r, g.i.values] = g.score.values
            auc.append(roc_auc_score(y[te], S[k, r, te]) if len(np.unique(y[te])) == 2 else np.nan)
            ref.append(fr.loc[(r, f)])
    ok_auc = np.allclose(auc, ref, equal_nan=True, rtol=0, atol=1e-12)
    log(f'確認 段階13の oof_scores.csv(median・M、乱数 {len(seeds)} 通り): テスト件が分割と一致 {ok_te}、'
        f'計算し直した fold の AUC が folds.csv と一致 {ok_auc}(最大差 {np.nanmax(np.abs(np.array(auc) - ref)):.1e})')
    assert ok_te and ok_auc, '段階13の oof_scores.csv と folds.csv が一致しない。止めて報告する'
    assert not np.isnan(S).any()
    return S, seeds, folds


def full_masks(meta):
    """全件 (n, n) のペアの種類と、i 多収 × j 低収。"""
    my = meta.muniyear.to_numpy(dtype=str)
    yr = meta.year.values
    y = meta.y.values
    same_my, same_yr = my[:, None] == my[None, :], yr[:, None] == yr[None, :]
    return {'W': same_my, 'B': same_yr & ~same_my, 'X': ~same_yr}, (y[:, None] == 1) & (y[None, :] == 0)


# ---------------------------------------------------------------- A1
def a1_repeat(S, meta, sp):
    """繰り返しごとの W・B(乱数ごとに fold をまたいで合算し、乱数で平均)と、その差。"""
    y = meta.y.values
    counts = [s12.pair_counts(S[:, r, te], np.tile(y[te], (S.shape[0], 1)), s12.pair_masks(meta, te))
              for r, f, tr, te in sp]
    out = {}
    for t in TYPES:
        a, n = s12.repeat_auc(counts, sp, t)  # (繰り返し, 乱数)
        out[t] = a.mean(axis=1)
        out[f'n_{t}'] = n[:, 0]
    d = pd.DataFrame({'repeat': range(N_REPEATS), 'W': out['W'], 'B': out['B'], 'X': out['X'],
                      'n_pairs_W': out['n_W'], 'n_pairs_B': out['n_B']})
    d['B_minus_W'] = d.B - d.W
    return d


def a1_pair_sums(S, meta, sp):
    """全乱数・全繰り返しで合算した、ペア (i, j) ごとの一致の和 C と、採点された回数 N((n, n)、i 多収・j 低収)。"""
    n = len(meta)
    y = meta.y.values.astype(float)
    C, N = np.zeros((n, n)), np.zeros((n, n))
    for r, f, tr, te in sp:
        d = S[:, r, te]
        cmp = ((d[:, :, None] > d[:, None, :]) + 0.5 * (d[:, :, None] == d[:, None, :])).sum(axis=0)
        pn = y[te][:, None] * (1 - y[te][None, :])
        C[np.ix_(te, te)] += cmp * pn
        N[np.ix_(te, te)] += S.shape[0] * pn
    return C, N


def a1_bootstrap(C, N, masks, meta, n_boot, n_scored):
    """市町村を単位にしたクラスターブートストラップ。

    市町村 c が k_c 回選ばれたら、その件を k_c 回数える。B のペア (i, j) は違う市町村どうしなので重み k_ci × k_cj。
    W のペアは同じ市町村の中で、選ばれた1回ごとに1組として数えるので重み k_c(同じ市町村の別の回の件どうしの組は作らない)。
    n_scored(乱数 × 繰り返し)で割って、ペア数を「乱数・繰り返し1組あたり」にする。
    """
    cd = meta.muniCd.to_numpy(dtype=str)
    units, cidx = np.unique(cd, return_inverse=True)
    CW, NW = C * masks['W'], N * masks['W']
    CB, NB = C * masks['B'], N * masks['B']
    cw_i, nw_i = CW.sum(axis=1), NW.sum(axis=1)  # W は i(多収)の行で集計。i と j は同じ市町村
    rng = np.random.default_rng(0)
    rows = []
    for b in range(n_boot):
        k = np.bincount(rng.integers(0, len(units), len(units)), minlength=len(units))
        w = k[cidx].astype(float)
        nw, nb = w @ nw_i, w @ NB @ w
        W = (w @ cw_i) / nw if nw > 0 else np.nan
        B = (w @ CB @ w) / nb if nb > 0 else np.nan
        rows.append({'boot': b, 'n_units_drawn': int((k > 0).sum()), 'W': W, 'B': B, 'B_minus_W': B - W,
                     'n_pairs_W': nw / n_scored, 'n_pairs_B': nb / n_scored})
    return pd.DataFrame(rows)


def qstats(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return {'mean': v.mean(), 'q025': np.quantile(v, .025), 'q975': np.quantile(v, .975), 'frac_pos': (v > 0).mean(),
            'n_valid': len(v)}


# ---------------------------------------------------------------- A2
def group_balance(meta, log):
    g = meta.groupby('muniyear').agg(muniCd=('muniCd', 'first'), year=('year', 'first'), n=('y', 'size'),
                                     n_high=('y', 'sum'), group_n_all=('group_n_all', 'first'))
    g['n_low'] = g.n - g.n_high
    g['type'] = np.select([g.n_high == g.n_low, (g.n_high > 0) & (g.n_low > 0)], ['balanced', 'unbalanced'], 'single')
    log('\n[A2 窓で絞った後の集団のクラス構成]')
    got = {}
    for t in ['balanced', 'unbalanced', 'single']:
        s = g[g.type == t]
        got[t] = (len(s), int(s.n.sum()))
        log(f'  {BAL_LABEL[t]}: {got[t][0]} 集団・{got[t][1]} 件'
            + ('' if t == 'balanced' else '(' + '、'.join(f'{i} 多{r.n_high}/低{r.n_low}' for i, r in s.iterrows()) + ')'))
    assert got == EXPECT_BAL, f'集団のクラス構成が想定 {EXPECT_BAL} と違う: {got}'
    return g.reset_index()


def a2_seed(decs, meta, sp, Yp, bal):
    """クラスが同数の集団どうしのペアだけで、観測と置換の W・B(繰り返しで平均)。"""
    y = meta.y.values
    obs_c, null_c = [], []
    for (r, f, tr, te), dec in zip(sp, decs):
        m = s12.pair_masks(meta, te)
        bb = bal[te][:, None] & bal[te][None, :]
        mm = {'W': m['W'] & bb, 'B': m['B'] & bb}
        obs_c.append(s12.pair_counts(dec[0], y[te], mm))
        null_c.append(s12.pair_counts(dec[1:], Yp[:, te], mm))
    out = {}
    for t in ['W', 'B']:
        a, n = s12.repeat_auc(obs_c, sp, t)
        out[t] = np.nanmean(a[:, 0])
        out[f'n_{t}'] = n[:, 0].mean()
        a, _ = s12.repeat_auc(null_c, sp, t)
        out[f'null_{t}'] = np.nanmean(a, axis=0)
    return out


# ---------------------------------------------------------------- B3
def b3_features(meta):
    return pd.DataFrame({'breed': meta.breed.fillna('不明').astype(str).values,
                         'seed_doy': meta.seed_doy.astype(float).values})


def b3_pipeline():
    ct = ColumnTransformer([('breed', OneHotEncoder(handle_unknown='ignore'), ['breed']),
                            ('num', StandardScaler(), ['seed_doy'])])
    return make_pipeline(ct, LogisticRegression(C=1.0, max_iter=1000))


def _b3_fold(F, tr, te, Y):
    warnings.filterwarnings('ignore')
    return np.array([clone(b3_pipeline()).fit(F.iloc[tr], yy[tr]).decision_function(F.iloc[te]) for yy in Y])


def confounder_check(meta, log):
    """品種の構成(集団の中に品種が2つ以上あるか)、補完した播種日の件数、播種日の単変量 AUC。"""
    rows = []
    br = meta.breed.fillna('不明').astype(str)
    g = meta.assign(breed_f=br).groupby('muniyear')
    nb = g.breed_f.nunique()
    nb_known = meta.dropna(subset=['breed']).groupby('muniyear').breed.nunique().reindex(nb.index).fillna(0)
    multi = nb[nb >= 2].index
    multi_known = nb_known[nb_known >= 2].index
    rows += [{'item': '市町村×年の集団', 'value': len(nb), 'n_cases': len(meta)},
             {'item': '品種が2つ以上ある集団(欠測は「不明」として数える)', 'value': len(multi),
              'n_cases': int(meta.muniyear.isin(multi).sum())},
             {'item': '品種が2つ以上ある集団(欠測を除いて数える)', 'value': len(multi_known),
              'n_cases': int(meta.muniyear.isin(multi_known).sum())},
             {'item': '品種が欠測の件', 'value': int(meta.breed.isna().sum()), 'n_cases': int(meta.breed.isna().sum())}]
    masks, pn = full_masks(meta)
    b = br.to_numpy()
    diff = b[:, None] != b[None, :]
    rows.append({'item': 'W のペアのうち品種が違うペア', 'value': int((pn & masks['W'] & diff).sum()),
                 'n_cases': int((pn & masks['W']).sum())})
    for v, c in br.value_counts().items():
        rows.append({'item': f'品種の件数: {v}', 'value': int(c), 'n_cases': int(c)})
    # 補完した播種日(元データの seed_date_raw が NULL)
    con = sqlite3.connect(f'file:{RAW_DB}?mode=ro', uri=True)
    q = pd.read_sql('SELECT field_id, year, seed_date, seed_date_raw FROM Questionaire', con)
    con.close()
    m = meta[['field_id', 'year', 'seed_date']].merge(q, on=['field_id', 'year'], how='left', suffixes=('', '_db'))
    assert len(m) == len(meta)
    assert (pd.to_datetime(m.seed_date).dt.date == pd.to_datetime(m.seed_date_db).dt.date).all(), \
        'meta の seed_date が DB と一致しない'
    imp = m.seed_date_raw.isna().values
    rows.append({'item': '播種日が補完値の件(seed_date_raw が NULL)', 'value': int(imp.sum()), 'n_cases': len(meta)})
    rows.append({'item': '播種日が補完値の件を含む市町村×年の集団', 'value': int(meta.muniyear[imp].nunique()),
                 'n_cases': int(meta.muniyear.isin(meta.muniyear[imp]).sum())})
    sd = meta.seed_doy.values
    rows.append({'item': 'W のペアのうち播種日が違うペア', 'value': int((pn & masks['W'] & (sd[:, None] != sd[None, :])).sum()),
                 'n_cases': int((pn & masks['W']).sum())})
    uv = s13.univariate_pairs(meta, sd.astype(float))
    for r in uv:
        rows.append({'item': f'播種日の通日の単変量 AUC: {r["type"]}', 'value': r['auc'], 'n_cases': r['n_pairs'],
                     'tie_rate': r['tie_rate']})
    t = pd.DataFrame(rows)
    log('\n[B3 交絡の候補の記述(検定しない)]')
    for _, r in t.iterrows():
        if not r['item'].startswith('品種の件数'):
            v = f'{r.value:.3f}' if 'AUC' in r['item'] else f'{int(r.value)}'
            log(f'  {r["item"]}: {v}(件数・ペア数 {int(r.n_cases)}'
                + (f'、同点 {r.tie_rate:.0%}' if pd.notna(r.get('tie_rate')) else '') + ')')
    log('  品種の構成: ' + '、'.join(f'{v} {c}' for v, c in br.value_counts().items()))
    return t


# ---------------------------------------------------------------- 図
def plot(boot, a1s, a2, a2_null, b3s, m13):
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    # 1. B − W のブートストラップ
    ax = axs[0]
    v = boot.B_minus_W.dropna().values
    ax.hist(v, bins=40, color=C_B, alpha=0.55)
    r = a1s.set_index('method').loc['cluster_bootstrap']
    for x in [r.q025, r.q975]:
        ax.axvline(x, color=s6.C_RED, ls='--', lw=1.5)
    ax.axvline(r.point, color=INK, lw=2)
    ax.axvline(0, color=MUTED, lw=1, ls=':')
    ax.set_title(f'A1 B − W(市町村のブートストラップ {len(v)}回)\n点推定 {r.point:.3f}、95%区間 [{r.q025:.3f}, {r.q975:.3f}]',
                 fontsize=10, color=INK, loc='left')
    ax.set_xlabel('B − W の AUC', color=INK)
    ax.set_ylabel('回数', color=INK)
    # 2. A2 の B と帰無分布
    ax = axs[1]
    ax.hist(a2_null, bins=40, color=MUTED, alpha=0.6)
    q = np.nanquantile(a2_null, .95)
    ax.axvline(q, color=s6.C_RED, ls='--', lw=1.5, label=f'95%点 {q:.3f}')
    ax.axvline(a2['B'], color=C_B, lw=2.5, label=f'観測 {a2["B"]:.3f}')
    ax.axvline(0.5, color=MUTED, lw=1, ls=':')
    ax.legend(frameon=False, fontsize=9)
    ax.set_title('A2 同数の集団どうしのペアだけの B と帰無分布', fontsize=10, color=INK, loc='left')
    ax.set_xlabel('AUC', color=INK)
    # 3. M と B3
    ax = axs[2]
    mets = [('all_auc', '全体'), ('W', 'W 地域内'), ('B', 'B 地域間')]
    for k, (cfg, d, col) in enumerate([('M', m13, '#2a78d6'), ('B3', b3s, '#9a9890')]):
        dd = d.set_index('metric')
        xs = np.arange(len(mets)) + (k - 0.5) * 0.36
        vals = [dd.loc[m, 'mean'] for m, _ in mets]
        ax.bar(xs, np.array(vals) - 0.5, bottom=0.5, width=0.34, color=col, alpha=0.8,
               label='M MiniRocket・1次差分(段階13)' if cfg == 'M' else 'B3 品種 + 播種日')
        for x, (m, _) in zip(xs, mets):
            ax.hlines(dd.loc[m, 'null_q95'], x - 0.17, x + 0.17, color=s6.C_RED, lw=2, ls='--')
            ax.text(x, max(dd.loc[m, 'mean'], 0.5) + 0.005, f'{dd.loc[m, "mean"]:.3f}', ha='center', fontsize=8, color=INK)
    ax.axhline(0.5, color=MUTED, lw=0.8)
    ax.set_xticks(range(len(mets)), [n for _, n in mets])
    ax.set_ylim(0.35, 0.7)
    ax.set_ylabel('AUC', color=INK)
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    ax.set_title('B3 と M(赤の破線 = 帰無分布の95%点)', fontsize=10, color=INK, loc='left')
    for a in axs:
        s6.style(a)
    fig.suptitle('段階14: label_muniyear_median(123件)の確認', fontsize=13, color=INK, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT_DIR, 'stage14_overview.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-perm', type=int, default=1000)
    ap.add_argument('--n-seeds', type=int, default=10, help='試運転用。本番は10')
    ap.add_argument('--n-boot', type=int, default=2000)
    args = ap.parse_args()
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
    X, meta = load()
    y = meta.y.values
    Xd = first_diff(X)
    sp = make_splits(y, meta.muniCd.values, N_SPLITS, N_REPEATS, SEED0)
    log(f'データ: {LABEL} {len(meta)} 件(多収 {y.sum()} / 低収 {(1 - y).sum()})、地域 {meta.muniCd.nunique()}、'
        f'市町村×年 {meta.muniyear.nunique()}')
    S, seeds13, folds13 = load_stage13(meta, sp, log)
    seeds = list(range(args.n_seeds))
    summary = []

    # ---- A1(段階13の乱数10通りすべてを使う。再学習なし)
    log('\n[A1 B − W(段階13の oof スコア、乱数 %d 通り)]' % len(seeds13))
    rep = a1_repeat(S, meta, sp)
    s13sum = pd.read_csv(os.path.join(S13_DIR, 'summary.csv'), encoding='utf-8-sig')
    m13 = s13sum[(s13sum.label == KEY) & (s13sum.config == 'M')]
    ok = all(np.isclose(rep[t].mean(), m13.set_index('metric').loc[t, 'mean']) for t in TYPES)
    log(f'  確認 繰り返しごとの W・B・X の平均が段階13の summary.csv と一致: {ok}')
    assert ok
    rep.to_csv(os.path.join(OUT_DIR, 'a1_b_minus_w.csv'), index=False, encoding='utf-8-sig')
    masks, pn = full_masks(meta)
    C, N = a1_pair_sums(S, meta, sp)
    point = {t: (C * masks[t]).sum() / (N * masks[t]).sum() for t in TYPES}
    log(f'  全乱数・全繰り返しで合算した点推定: W {point["W"]:.4f}、B {point["B"]:.4f}、B − W {point["B"] - point["W"]:.4f}'
        f'(W のペア {int((pn & masks["W"]).sum())} 組はどの繰り返しでも同じ)')
    t = time.time()
    boot = a1_bootstrap(C, N, masks, meta, args.n_boot, len(seeds13) * N_REPEATS)
    boot.to_csv(os.path.join(OUT_DIR, 'a1_bootstrap.csv'), index=False, encoding='utf-8-sig')
    a1 = []
    st = qstats(rep.B_minus_W)
    a1.append({'method': 'repeat', 'point': rep.B_minus_W.mean(), **st, 'W': rep.W.mean(), 'B': rep.B.mean()})
    st = qstats(boot.B_minus_W)
    a1.append({'method': 'cluster_bootstrap', 'point': point['B'] - point['W'], **st, 'W': point['W'], 'B': point['B'],
               'W_q025': boot.W.quantile(.025), 'W_q975': boot.W.quantile(.975),
               'B_q025': boot.B.quantile(.025), 'B_q975': boot.B.quantile(.975),
               'n_boot': args.n_boot, 'n_units': meta.muniCd.nunique()})
    a1s = pd.DataFrame(a1)
    a1s.to_csv(os.path.join(OUT_DIR, 'a1_summary.csv'), index=False, encoding='utf-8-sig')
    for _, r in a1s.iterrows():
        log(f'  {r.method}: B − W 点推定 {r.point:.3f}、平均 {r["mean"]:.3f}、95%区間 [{r.q025:.3f}, {r.q975:.3f}]、'
            f'正の割合 {r.frac_pos:.3f}(有効 {int(r.n_valid)})')
    rb = a1s.iloc[1]
    log(f'  ブートストラップの W 95%区間 [{rb.W_q025:.3f}, {rb.W_q975:.3f}]、B [{rb.B_q025:.3f}, {rb.B_q975:.3f}]'
        f'({time.time() - t:.1f}s)')
    a1_detect = not (rb.q025 <= 0 <= rb.q975)
    log('  [A1 判定] ' + ('95%区間に0を含まない → W と B の差を検出' if a1_detect
                         else '95%区間に0を含む → W と B の差は検出できない(W が低いのは検出力不足の可能性がある)'))
    summary += [{'section': 'A1', 'item': 'B − W(ブートストラップ、主)', 'value': rb.point, 'ci_low': rb.q025,
                 'ci_high': rb.q975, 'frac_pos': rb.frac_pos, 'judgement': '差を検出' if a1_detect else '差は検出できない'},
                {'section': 'A1', 'item': 'B − W(繰り返しごと)', 'value': a1s.iloc[0].point, 'ci_low': a1s.iloc[0].q025,
                 'ci_high': a1s.iloc[0].q975, 'frac_pos': a1s.iloc[0].frac_pos}]

    # ---- A2
    gb = group_balance(meta, log)
    gb.to_csv(os.path.join(OUT_DIR, 'a2_group_balance.csv'), index=False, encoding='utf-8-sig')
    bal = meta.muniyear.map(gb.set_index('muniyear').type).eq('balanced').values
    ps = s13.perm_sets(KEY, meta, P)
    Ym = ps[0]
    # 置換しても集団のクラス構成(同数かどうか)は変わらない
    assert all(np.array_equal(pd.Series(Ym[i]).groupby(meta.muniyear.values).sum().values,
                              pd.Series(y).groupby(meta.muniyear.values).sum().values) for i in range(min(P, 20)))
    log(f'  同数の集団どうしのペア: W {int((pn & masks["W"] & bal[:, None] & bal[None, :]).sum())} 組(全件)')
    a2_rows, a2_nullB, a2_nullW = [], [], []
    t = time.time()
    for k, seed in enumerate(seeds):
        feats = s6.fold_features('minirocket', seed, Xd, sp)
        decs, n_redo = s12.m_decisions(feats, y, sp, Ym)
        del feats
        # 観測ラベルの決定関数が段階13の oof スコアと一致すること
        dd = max(np.max(np.abs(dec[0] - S[seeds13.index(seed), r, te])) for (r, f, tr, te), dec in zip(sp, decs))
        assert dd < 1e-8, f'乱数{seed}: 決定関数が段階13の oof スコアと一致しない(最大差 {dd:.1e})'
        o = a2_seed(decs, meta, sp, Ym, bal)
        a2_rows.append({'seed': seed, 'B': o['B'], 'W': o['W'], 'n_pairs_B': o['n_B'], 'n_pairs_W': o['n_W'],
                        'n_redo': n_redo, 'max_diff_vs_stage13': dd})
        a2_nullB.append(o['null_B'])
        a2_nullW.append(o['null_W'])
        log(f'  M 乱数{seed}: 同数の集団だけで B {o["B"]:.3f}、W {o["W"]:.3f}(段階13との最大差 {dd:.1e}、学習し直した組 {n_redo})')
    nB, nW = np.mean(a2_nullB, axis=0), np.mean(a2_nullW, axis=0)
    np.save(os.path.join(OUT_DIR, 'null_a2_B.npy'), nB)
    a2s = pd.DataFrame(a2_rows)
    a2 = {'B': a2s.B.mean(), 'W': a2s.W.mean()}
    rows = []
    for tt, nl in [('B', nB), ('W', nW)]:
        rows.append({'metric': tt, 'mean': a2[tt], 'seed_min': a2s[tt].min(), 'seed_max': a2s[tt].max(),
                     'n_pairs_per_repeat': a2s[f'n_pairs_{tt}'].mean(), 'null_mean': np.nanmean(nl),
                     'null_q95': np.nanquantile(nl, .95), 'null_sd': np.nanstd(nl),
                     'p': (1 + np.sum(nl >= a2[tt])) / (1 + P), 'n_seeds': len(seeds), 'n_perm': P,
                     'all_pairs_stage13': m13.set_index('metric').loc[tt, 'mean']})
    a2sum = pd.DataFrame(rows)
    a2sum.to_csv(os.path.join(OUT_DIR, 'a2_summary.csv'), index=False, encoding='utf-8-sig')
    rB = a2sum.iloc[0]
    log(f'  同数の集団だけ: B {rB["mean"]:.3f}(q95 {rB.null_q95:.3f}、p={rB.p:.3f}、ペア/繰り返し {rB.n_pairs_per_repeat:.0f})、'
        f'W {a2sum.iloc[1]["mean"]:.3f}(q95 {a2sum.iloc[1].null_q95:.3f}、p={a2sum.iloc[1].p:.3f})  [{time.time() - t:.0f}s]')
    a2_hit = rB['mean'] > rB.null_q95
    log('  [A2 判定] ' + ('95%点を上回る → B の高さは集団の偏りによるものではない' if a2_hit
                         else '95%点を上回らない → 集団の偏りの影響を否定できない'))
    summary += [{'section': 'A2', 'item': '同数の集団どうしの B', 'value': rB['mean'], 'null_q95': rB.null_q95, 'p': rB.p,
                 'judgement': '上回る' if a2_hit else '上回らない'},
                {'section': 'A2', 'item': '同数の集団どうしの W', 'value': a2sum.iloc[1]['mean'],
                 'null_q95': a2sum.iloc[1].null_q95, 'p': a2sum.iloc[1].p}]

    # ---- B3
    log('\n[B3 品種 + 播種日の基準線]')
    F = b3_features(meta)
    t = time.time()
    ref = evaluate(b3_pipeline(), F, y, sp)
    Y = np.vstack([y, ps[2]])
    decs = Parallel(n_jobs=N_JOBS)(delayed(_b3_fold)(F, tr, te, Y) for r, f, tr, te in sp)
    obs, null = s13.analyse(decs, meta, sp, ps)
    s13.check_obs('B3', obs, ref, log)
    ref.assign(config='B3').to_csv(os.path.join(OUT_DIR, 'b3_folds.csv'), index=False, encoding='utf-8-sig')
    pa = pd.concat([pd.DataFrame({'config': 'B3', 'type': tt, 'repeat': range(N_REPEATS), 'auc': obs[tt][0],
                                  'n_pairs': obs[tt][1]}) for tt in TYPES])
    pa.to_csv(os.path.join(OUT_DIR, 'b3_pair_auc.csv'), index=False, encoding='utf-8-sig')
    b3 = []
    for met, val, nl in [('all_auc', ref.auc.mean(), null['all'][:, 1]), ('all_bacc', ref.bacc.mean(), null['all'][:, 0]),
                         ('W', np.nanmean(obs['W'][0]), null['W']), ('B', np.nanmean(obs['B'][0]), null['B']),
                         ('X', np.nanmean(obs['X'][0]), None)]:
        r = {'config': 'B3', 'metric': met, 'mean': val, 'n_perm': P}
        if nl is not None:
            np.save(os.path.join(OUT_DIR, f'null_b3_{met}.npy'), nl)
            r.update({'null_mean': np.nanmean(nl), 'null_q95': np.nanquantile(nl, .95), 'null_sd': np.nanstd(nl),
                      'p': (1 + np.sum(nl >= val)) / (1 + P)})
        mr = m13.set_index('metric').loc[met]
        r.update({'M_mean': mr['mean'], 'M_null_q95': mr.null_q95})
        b3.append(r)
    b3s = pd.DataFrame(b3)
    b3s.to_csv(os.path.join(OUT_DIR, 'b3_summary.csv'), index=False, encoding='utf-8-sig')
    log('  ' + '  '.join(f'{r.metric} {r["mean"]:.3f}' + (f'(q95 {r.null_q95:.3f}, p={r.p:.3f})' if pd.notna(r.get('null_q95'))
                                                           else '') for _, r in b3s.iterrows()) + f'  [{time.time() - t:.0f}s]')
    fM = folds13[['seed', 'repeat', 'fold', 'auc', 'bacc']]
    fb = pd.concat([ref.assign(seed=s) for s in seeds13])
    pdiff = s7.paired(fM, fb, 'M − B3')
    pds = s7.paired_summary(pdiff)
    pds.to_csv(os.path.join(OUT_DIR, 'b3_paired_diff_summary.csv'), index=False, encoding='utf-8-sig')
    for _, r in pds.iterrows():
        log(f'  M − B3 {r.metric}: 平均 {r.mean_diff:+.3f}、繰り返しの2.5〜97.5%点 [{r.rep_q025:+.3f}, {r.rep_q975:+.3f}]、'
            f'正の割合 {r.rep_frac_pos:.2f}、差を検出 {r.detected}')
    conf = confounder_check(meta, log)
    conf.to_csv(os.path.join(OUT_DIR, 'b3_confounder_check.csv'), index=False, encoding='utf-8-sig')
    ra = b3s.set_index('metric').loc['all_auc']
    b3_chance = ra['mean'] <= ra.null_q95
    b3_below = ra['mean'] < ra.M_mean
    log('  [B3 読み方] 全体 AUC ' + ('偶然水準(95%点以下)' if b3_chance else '95%点を上回る') + '、'
        + ('M を下回る' if b3_below else 'M 以上') + ' → '
        + ('M の結果は品種・播種日では説明できない' if b3_chance and b3_below else '品種・播種日で説明できる可能性がある'))
    pa_auc = pds[pds.metric == 'auc'].iloc[0]
    for _, r in b3s.iterrows():
        summary.append({'section': 'B3', 'item': f'B3 {r.metric}', 'value': r['mean'], 'null_q95': r.get('null_q95'),
                        'p': r.get('p'), 'M_value': r.M_mean})
    summary.append({'section': 'B3', 'item': 'M − B3 AUC(対ごとの差)', 'value': pa_auc.mean_diff, 'ci_low': pa_auc.rep_q025,
                    'ci_high': pa_auc.rep_q975, 'frac_pos': pa_auc.rep_frac_pos,
                    'judgement': '差を検出' if pa_auc.detected else '差は検出できない'})
    pd.DataFrame(summary).to_csv(os.path.join(OUT_DIR, 'summary.csv'), index=False, encoding='utf-8-sig')

    plot(boot, a1s, a2, nB, b3s, m13)
    log('\n[まとめ]')
    log(pd.DataFrame(summary).round(3).to_string(index=False))
    log(f'\n全体 {time.time() - t0:.0f} 秒。出力: {OUT_DIR}')
    logf.close()


if __name__ == '__main__':
    main()
