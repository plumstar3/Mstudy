"""
生育ステージ(開花日)の推定に使う部品(段階3で作成)。

- daylength(): 緯度と通日から日長(時間)を天文学的に計算する。
  Forsythe et al. (1995) Ecological Modelling 80:87-95 の CBM モデル。
  p は太陽が地平線のどこにあるときを日の出・日の入りとするか(度)。
  0.8333 = 太陽の上端+大気差(気象庁の日の出・日の入りと同じ定義、既定)、0 = 太陽の中心、6 = 常用薄明。
- 開花日の推定器: fit(train) → predict(cases) で、播種後日数(dap)の整数を返す。
  cases は1行=1圃場×年の DataFrame(列: key, seed_doy, lat, ほかに fit で flower_dap)。
  GDD を使う推定器には、key → 播種日(dap 0)からの日ごとの累積GDD配列を渡す。
"""
import numpy as np


def daylength(lat_deg, doy, p=0.8333):
    """日長[時間]。lat_deg, doy は配列でもよい。"""
    doy = np.asarray(doy, dtype=float)
    theta = 0.2163108 + 2 * np.arctan(0.9671396 * np.tan(0.00860 * (doy - 186)))
    phi = np.arcsin(0.39795 * np.cos(theta))
    lat = np.radians(lat_deg)
    x = (np.sin(np.radians(p)) + np.sin(lat) * np.sin(phi)) / (np.cos(lat) * np.cos(phi))
    return 24 - (24 / np.pi) * np.arccos(np.clip(x, -1, 1))


def first_reach(cum, threshold):
    """累積値が threshold に初めて達する dap(達しなければ NaN)。"""
    idx = np.flatnonzero(np.asarray(cum) >= threshold)
    return float(idx[0]) if len(idx) else np.nan


class MeanDap:
    """(i) 開花 dap の平均値を全件に当てはめる。"""
    name = 'i_mean_dap'

    def fit(self, tr, **kw):
        self.m = tr['flower_dap'].mean()
        return self

    def predict(self, cases, **kw):
        return np.full(len(cases), np.round(self.m))


class SeedDoyRegression:
    """(ii) 開花 dap を播種日の通日で線形回帰する。"""
    name = 'ii_seed_doy_reg'

    def fit(self, tr, **kw):
        self.b, self.a = np.polyfit(tr['seed_doy'], tr['flower_dap'], 1)
        return self

    def predict(self, cases, **kw):
        return np.round(self.a + self.b * cases['seed_doy'].values)


class GddThreshold:
    """(iii) 積算GDDが学習データの開花時GDDの平均に達した日。"""
    name = 'iii_gdd'

    def fit(self, tr, cum_gdd, **kw):
        self.th = np.mean([cum_gdd[k][int(d)] for k, d in zip(tr['key'], tr['flower_dap'])])
        return self

    def predict(self, cases, cum_gdd, **kw):
        return np.array([first_reach(cum_gdd[k], self.th) for k in cases['key']])


class GddDaylength:
    """(iv) 開花に必要な積算GDD = a + b × 日長 とし、その値に達した日。

    dl='seed': 播種日の日長。
    dl='mean': 播種日から当日までの平均日長。学習では播種〜実際の開花日の平均を使い、
               予測では日ごとに「その日までの平均日長で決まる必要GDD」に達したかを判定する。
    """

    def __init__(self, dl='mean', p=0.8333):
        self.dl, self.p = dl, p
        self.name = f'iv_gdd_daylength_{dl}'

    def _dl_series(self, lat, seed_doy, n):
        dl = daylength(lat, seed_doy + np.arange(n), self.p)
        if self.dl == 'seed':
            return np.full(n, dl[0])
        return np.cumsum(dl) / np.arange(1, n + 1)

    def fit(self, tr, cum_gdd, **kw):
        x, y = [], []
        for k, lat, sd, d in zip(tr['key'], tr['lat'], tr['seed_doy'], tr['flower_dap']):
            d = int(d)
            x.append(self._dl_series(lat, sd, d + 1)[d])
            y.append(cum_gdd[k][d])
        self.b, self.a = np.polyfit(x, y, 1)
        return self

    def predict(self, cases, cum_gdd, **kw):
        out = []
        for k, lat, sd in zip(cases['key'], cases['lat'], cases['seed_doy']):
            cum = cum_gdd[k]
            need = self.a + self.b * self._dl_series(lat, sd, len(cum))
            idx = np.flatnonzero(cum >= need)
            out.append(float(idx[0]) if len(idx) else np.nan)
        return np.array(out)
