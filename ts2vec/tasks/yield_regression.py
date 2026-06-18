import numpy as np
import time
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold


# ── Ridge ──────────────────────────────────────────────────────────────────

def _fit_ridge_regression(train_features, train_y, val_features, val_y, MAX_SAMPLES=100000):
    '''Fit Ridge regression with alpha hyperparameter tuned on validation set.'''
    if train_features.shape[0] > MAX_SAMPLES:
        split = train_test_split(train_features, train_y, train_size=MAX_SAMPLES, random_state=0)
        train_features, train_y = split[0], split[2]

    alphas = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    val_scores = []
    for alpha in alphas:
        lr = Ridge(alpha=alpha).fit(train_features, train_y)
        val_pred = lr.predict(val_features)
        score = (np.sqrt(((val_pred - val_y) ** 2).mean()) +
                 np.abs(val_pred - val_y).mean())
        val_scores.append(score)

    best_alpha = alphas[int(np.argmin(val_scores))]
    lr = Ridge(alpha=best_alpha).fit(train_features, train_y)
    return lr, {'best_alpha': best_alpha}


# ── Random Forest ──────────────────────────────────────────────────────────

def _fit_rf_regression(train_features, train_y, val_features, val_y, MAX_SAMPLES=100000):
    '''Fit RandomForestRegressor with hyperparameters tuned on validation set.'''
    if train_features.shape[0] > MAX_SAMPLES:
        split = train_test_split(train_features, train_y, train_size=MAX_SAMPLES, random_state=0)
        train_features, train_y = split[0], split[2]

    n_estimators_list = [50, 100, 200]
    max_features_list = ['sqrt', 'log2', 1.0]

    best_score = float('inf')
    best_params = {'n_estimators': 100, 'max_features': 'sqrt'}
    best_model = None

    for n_est in n_estimators_list:
        for mf in max_features_list:
            rf = RandomForestRegressor(
                n_estimators=n_est,
                max_features=mf,
                random_state=0,
                n_jobs=-1
            ).fit(train_features, train_y)
            val_pred = rf.predict(val_features)
            score = (np.sqrt(((val_pred - val_y) ** 2).mean()) +
                     np.abs(val_pred - val_y).mean())
            if score < best_score:
                best_score = score
                best_params = {'n_estimators': n_est, 'max_features': mf}
                best_model = rf

    return best_model, best_params


# ── Metrics ────────────────────────────────────────────────────────────────

def _cal_metrics(pred, target):
    '''Calculate regression metrics including MAPE.

    MAPE (Mean Absolute Percentage Error) is computed only over samples
    where |target| > 0 to avoid division by zero.
    '''
    rmse = float(np.sqrt(((pred - target) ** 2).mean()))
    mae  = float(np.abs(pred - target).mean())
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    r2   = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    # MAPE: ゼロターゲットを除外してから計算
    nonzero_mask = np.abs(target) > 0
    if nonzero_mask.any():
        mape = float(np.mean(np.abs((pred[nonzero_mask] - target[nonzero_mask]) /
                                     target[nonzero_mask])) * 100)
    else:
        mape = float('nan')
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape}


# ── Main evaluation ────────────────────────────────────────────────────────

def eval_yield_regression(model,
                          train_data, train_labels,
                          val_data,   val_labels,
                          test_data,  test_labels,
                          combine_train_val=False,
                          regressor='ridge'):
    '''Evaluate yield regression using TS2Vec representations + a regressor.

    Workflow:
      1. Encode each split using encoding_window='full_series'
         → instance-level representation (N, repr_dims)
      2. Fit regressor on train representations, tune hyperparams on val
         (if combine_train_val=True, train+val are merged for final fitting)
      3. Predict and evaluate on train / val / test

    Args:
        model: Trained TS2Vec model.
        train_data   (np.ndarray): (N_train, T, F)
        train_labels (np.ndarray): (N_train,)  yield values
        val_data     (np.ndarray): (N_val, T, F)
        val_labels   (np.ndarray): (N_val,)
        test_data    (np.ndarray): (N_test, T, F)
        test_labels  (np.ndarray): (N_test,)
        combine_train_val (bool): If True, merge train+val for final model fit.
        regressor (str): 'ridge' or 'rf' (random forest).

    Returns:
        out      (dict): encoded representations and predictions
        eval_res (dict): evaluation metrics and best hyperparameters
    '''
    if regressor not in ('ridge', 'rf'):
        raise ValueError(f"Unknown regressor '{regressor}'. Choose 'ridge' or 'rf'.")

    # ── Step 1: エンコード ────────────────────────────────────────────────
    t0 = time.time()
    train_repr = model.encode(train_data, encoding_window='full_series')  # (N_train, D)
    val_repr   = model.encode(val_data,   encoding_window='full_series')  # (N_val, D)
    test_repr  = model.encode(test_data,  encoding_window='full_series')  # (N_test, D)
    encode_time = time.time() - t0

    # ── Step 2: ハイパーパラメータ選択（val で）→ 必要なら train+val で再学習 ──
    t1 = time.time()
    if regressor == 'ridge':
        _, best_params = _fit_ridge_regression(train_repr, train_labels, val_repr, val_labels)
        if combine_train_val:
            fit_X = np.concatenate([train_repr, val_repr], axis=0)
            fit_y = np.concatenate([train_labels, val_labels], axis=0)
            final_model = Ridge(alpha=best_params['best_alpha']).fit(fit_X, fit_y)
        else:
            final_model = Ridge(alpha=best_params['best_alpha']).fit(train_repr, train_labels)
    else:  # rf
        _, best_params = _fit_rf_regression(train_repr, train_labels, val_repr, val_labels)
        if combine_train_val:
            fit_X = np.concatenate([train_repr, val_repr], axis=0)
            fit_y = np.concatenate([train_labels, val_labels], axis=0)
        else:
            fit_X, fit_y = train_repr, train_labels
        final_model = RandomForestRegressor(
            n_estimators=best_params['n_estimators'],
            max_features=best_params['max_features'],
            random_state=0,
            n_jobs=-1
        ).fit(fit_X, fit_y)
    reg_train_time = time.time() - t1

    # ── Step 3: 予測・評価 ────────────────────────────────────────────────
    t2 = time.time()
    train_pred = final_model.predict(train_repr)
    val_pred   = final_model.predict(val_repr)
    test_pred  = final_model.predict(test_repr)
    reg_infer_time = time.time() - t2

    metrics = {
        'train': _cal_metrics(train_pred, train_labels),
        'val':   _cal_metrics(val_pred,   val_labels),
        'test':  _cal_metrics(test_pred,  test_labels),
    }

    out = {
        'train_repr':   train_repr,
        'val_repr':     val_repr,
        'test_repr':    test_repr,
        'train_pred':   train_pred,
        'val_pred':     val_pred,
        'test_pred':    test_pred,
        'train_labels': train_labels,
        'val_labels':   val_labels,
        'test_labels':  test_labels,
    }

    eval_res = {
        'regressor':       regressor,
        'best_params':     best_params,
        'metrics':         metrics,
        'encode_time':     encode_time,
        'reg_train_time':  reg_train_time,
        'reg_infer_time':  reg_infer_time,
        # 後方互換のため Ridge の alpha もキーとして残す
        'best_ridge_alpha': best_params.get('best_alpha', None),
    }

    return out, eval_res


# ── CV 対応評価（LOYO / 5-Fold CV） ────────────────────────────────────────

def _normalize_per_fold(X_train_raw, X_val_raw, X_test_raw):
    '''Compute normalization stats from train, apply to all splits.

    NaN values are filled with 0 after normalization (= imputed to mean).

    Args:
        X_train_raw (np.ndarray): shape (N_train, T, F)
        X_val_raw   (np.ndarray): shape (N_val,   T, F)
        X_test_raw  (np.ndarray): shape (N_test,  T, F)

    Returns:
        X_train, X_val, X_test: normalized arrays of the same shapes.
    '''
    X2d = X_train_raw.reshape(-1, X_train_raw.shape[-1])
    mean = np.nanmean(X2d, axis=0)
    std  = np.nanstd(X2d, axis=0)
    std[std < 1e-8] = 1.0  # ゼロ除算防止

    def _apply(X):
        shape = X.shape
        X2 = (X.reshape(-1, shape[-1]) - mean) / std
        X2 = np.nan_to_num(X2, nan=0.0)
        return X2.reshape(shape).astype(np.float32)

    return _apply(X_train_raw), _apply(X_val_raw), _apply(X_test_raw)


def eval_yield_regression_cv(model, X_raw, y, years,
                              cv_mode='loyo',
                              n_splits=5,
                              val_ratio=0.2,
                              random_state=42,
                              regressor='ridge'):
    '''Cross-validated yield regression using TS2Vec representations.

    For each fold:
      1. Split into test / non-test (LOYO) or test / train-val (KFold)
      2. Randomly split non-test (or train-val) into train (1-val_ratio)
         and val (val_ratio) — val is used only for hyperparameter selection
      3. Normalize using train-fold statistics only (no leakage)
      4. Encode with TS2Vec (encoding_window="full_series")
      5. Fit regressor (Ridge alpha or RF params tuned on val)
      6. Evaluate on test

    Args:
        model       : Trained / frozen TS2Vec model.
        X_raw       (np.ndarray): shape (N, T, F) — raw unnormalized data.
        y           (np.ndarray): shape (N,)       — yield values.
        years       (np.ndarray): shape (N,)       — year per sample (for LOYO).
        cv_mode     (str):  'loyo' or 'kfold'.
        n_splits    (int):  Number of folds for kfold (default 5).
        val_ratio   (float): Fraction of non-test data used as val (default 0.2).
        random_state(int):  Random seed for val split and KFold shuffle.
        regressor   (str):  'ridge' or 'rf'.

    Returns:
        fold_results (list[dict]): Per-fold metrics + metadata.
        summary      (dict):       mean/std over all folds.
    '''
    if cv_mode not in ('loyo', 'kfold'):
        raise ValueError(f"cv_mode must be 'loyo' or 'kfold', got '{cv_mode}'")

    # ── fold インデックスのリストを生成 ─────────────────────────────────
    # 各要素: (fold_label, train_idx, val_idx, test_idx)
    folds = []

    if cv_mode == 'loyo':
        unique_years = sorted(set(years.tolist()))
        for test_year in unique_years:
            test_idx     = np.where(years == test_year)[0]
            non_test_idx = np.where(years != test_year)[0]
            # non-test を train/val にランダム分割
            tr_idx, va_idx = train_test_split(
                non_test_idx,
                test_size=val_ratio,
                random_state=random_state,
            )
            folds.append((f'test_year={test_year}', tr_idx, va_idx, test_idx))

    else:  # kfold
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for fold_i, (train_val_idx, test_idx) in enumerate(kf.split(X_raw)):
            tr_idx, va_idx = train_test_split(
                train_val_idx,
                test_size=val_ratio,
                random_state=random_state,
            )
            folds.append((f'fold={fold_i + 1}', tr_idx, va_idx, test_idx))

    # ── fold ループ ───────────────────────────────────────────────────────
    fold_results = []

    print(f'\n{"─" * 60}')
    print(f'  CV mode : {cv_mode.upper()}  '
          f'({len(folds)} folds)  regressor={regressor.upper()}')
    print(f'  val_ratio={val_ratio}  random_state={random_state}')
    print(f'{"─" * 60}')

    for fold_label, tr_idx, va_idx, te_idx in folds:
        t_fold = time.time()

        # ── 正規化（train統計のみ使用） ─────────────────────────────────
        X_tr, X_va, X_te = _normalize_per_fold(
            X_raw[tr_idx], X_raw[va_idx], X_raw[te_idx]
        )
        y_tr = y[tr_idx]
        y_va = y[va_idx]
        y_te = y[te_idx]

        # ── TS2Vec エンコード ────────────────────────────────────────────
        tr_repr = model.encode(X_tr, encoding_window='full_series')
        va_repr = model.encode(X_va, encoding_window='full_series')
        te_repr = model.encode(X_te, encoding_window='full_series')

        # ── 回帰モデル学習（val でハイパーパラメータ選択） ────────────────
        if regressor == 'ridge':
            reg_model, best_params = _fit_ridge_regression(
                tr_repr, y_tr, va_repr, y_va
            )
        else:
            reg_model, best_params = _fit_rf_regression(
                tr_repr, y_tr, va_repr, y_va
            )

        # ── テスト評価 ───────────────────────────────────────────────────
        te_pred = reg_model.predict(te_repr)
        m = _cal_metrics(te_pred, y_te)

        elapsed = time.time() - t_fold
        param_str = (f'alpha={best_params["best_alpha"]}'
                     if regressor == 'ridge'
                     else f'n_est={best_params["n_estimators"]} '
                          f'max_feat={best_params["max_features"]}')

        print(f'  [{fold_label}]  '
              f'train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)}  '
              f'| RMSE={m["RMSE"]:7.3f}  MAE={m["MAE"]:7.3f}  '
              f'MAPE={m["MAPE"]:6.2f}%  R2={m["R2"]:7.4f}  '
              f'({elapsed:.1f}s)  [{param_str}]')

        fold_results.append({
            'fold':      fold_label,
            'n_train':   len(tr_idx),
            'n_val':     len(va_idx),
            'n_test':    len(te_idx),
            'best_params': best_params,
            **m,
        })

    # ── サマリー（mean ± std） ────────────────────────────────────────────
    summary = {}
    print(f'\n{"=" * 60}')
    print(f'  SUMMARY  [{cv_mode.upper()} / {regressor.upper()}]  mean ± std')
    print(f'{"=" * 60}')
    for key in ('RMSE', 'MAE', 'MAPE', 'R2'):
        vals = [r[key] for r in fold_results]
        summary[f'{key}_mean'] = float(np.mean(vals))
        summary[f'{key}_std']  = float(np.std(vals))
        print(f'  {key:<6} : {np.mean(vals):8.3f} ± {np.std(vals):.3f}')
    print(f'{"=" * 60}')

    return fold_results, summary
