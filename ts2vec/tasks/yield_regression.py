import numpy as np
import time
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


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
