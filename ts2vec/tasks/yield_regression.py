import numpy as np
import time
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split


def _fit_ridge_regression(train_features, train_y, val_features, val_y, MAX_SAMPLES=100000):
    '''Fit Ridge regression with alpha hyperparameter tuned on validation set.'''
    # サブサンプリング（大規模データ対策）
    if train_features.shape[0] > MAX_SAMPLES:
        split = train_test_split(train_features, train_y, train_size=MAX_SAMPLES, random_state=0)
        train_features, train_y = split[0], split[2]

    alphas = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    val_scores = []
    for alpha in alphas:
        lr = Ridge(alpha=alpha).fit(train_features, train_y)
        val_pred = lr.predict(val_features)
        # RMSE + MAE の複合スコアで alpha を選択
        score = (np.sqrt(((val_pred - val_y) ** 2).mean()) +
                 np.abs(val_pred - val_y).mean())
        val_scores.append(score)

    best_alpha = alphas[int(np.argmin(val_scores))]
    lr = Ridge(alpha=best_alpha).fit(train_features, train_y)
    return lr, best_alpha


def _cal_metrics(pred, target):
    '''Calculate regression metrics.'''
    rmse = float(np.sqrt(((pred - target) ** 2).mean()))
    mae  = float(np.abs(pred - target).mean())
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    r2   = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}


def eval_yield_regression(model,
                          train_data, train_labels,
                          val_data,   val_labels,
                          test_data,  test_labels):
    '''Evaluate yield regression using TS2Vec representations + Ridge regression.

    Workflow:
      1. Encode each split using encoding_window='full_series'
         → instance-level representation (N, repr_dims)
      2. Fit Ridge regressor on train representations, tune alpha on val
      3. Predict and evaluate on test

    Args:
        model: Trained TS2Vec model.
        train_data   (np.ndarray): (N_train, T, F)
        train_labels (np.ndarray): (N_train,)  yield values
        val_data     (np.ndarray): (N_val, T, F)
        val_labels   (np.ndarray): (N_val,)
        test_data    (np.ndarray): (N_test, T, F)
        test_labels  (np.ndarray): (N_test,)

    Returns:
        out      (dict): encoded representations and predictions
        eval_res (dict): evaluation metrics
    '''
    # ── Step 1: エンコード（全系列 → 1インスタンスベクトル）──────────────
    t0 = time.time()
    train_repr = model.encode(train_data, encoding_window='full_series')  # (N_train, D)
    val_repr   = model.encode(val_data,   encoding_window='full_series')  # (N_val, D)
    test_repr  = model.encode(test_data,  encoding_window='full_series')  # (N_test, D)
    encode_time = time.time() - t0

    # ── Step 2: Ridge 回帰 ──────────────────────────────────────────────
    t1 = time.time()
    lr, best_alpha = _fit_ridge_regression(train_repr, train_labels, val_repr, val_labels)
    ridge_train_time = time.time() - t1

    # ── Step 3: 予測・評価 ────────────────────────────────────────────────
    t2 = time.time()
    test_pred = lr.predict(test_repr)
    ridge_infer_time = time.time() - t2

    # 訓練セットでの評価（過学習チェック用）
    train_pred = lr.predict(train_repr)
    val_pred   = lr.predict(val_repr)

    metrics = {
        'train': _cal_metrics(train_pred, train_labels),
        'val':   _cal_metrics(val_pred,   val_labels),
        'test':  _cal_metrics(test_pred,  test_labels),
    }

    out = {
        'train_repr':  train_repr,
        'val_repr':    val_repr,
        'test_repr':   test_repr,
        'train_pred':  train_pred,
        'val_pred':    val_pred,
        'test_pred':   test_pred,
        'train_labels': train_labels,
        'val_labels':   val_labels,
        'test_labels':  test_labels,
    }

    eval_res = {
        'metrics':         metrics,
        'best_ridge_alpha': best_alpha,
        'encode_time':     encode_time,
        'ridge_train_time': ridge_train_time,
        'ridge_infer_time': ridge_infer_time,
    }

    return out, eval_res
