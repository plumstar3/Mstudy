import torch
import numpy as np
import argparse
import os
import sys
import time
import datetime
from ts2vec import TS2Vec
import tasks
import datautils
from utils import init_dl_program, name_with_datetime, pkl_save, data_dropout

def save_checkpoint_callback(
    save_every=1,
    unit='epoch'
):
    assert unit in ('epoch', 'iter')
    def callback(model, loss):
        n = model.n_epochs if unit == 'epoch' else model.n_iters
        if n % save_every == 0:
            model.save(f'{run_dir}/model_{n}.pkl')
    return callback

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='The dataset name or path to the soybean dataset directory')
    parser.add_argument('run_name', help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
    parser.add_argument('--loader', type=str, required=True, help='The data loader used to load the experimental data. This can be set to UCR, UEA, forecast_csv, forecast_csv_univar, anomaly, anomaly_coldstart, soybean_pretrain, or soybean_finetune')
    parser.add_argument('--gpu', type=str, default='0', help='The gpu no. used for training and inference (defaults to 0), or "cpu" to use CPU')
    parser.add_argument('--batch-size', type=int, default=8, help='The batch size (defaults to 8)')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('--repr-dims', type=int, default=320, help='The representation dimension (defaults to 320)')
    parser.add_argument('--max-train-length', type=int, default=3000, help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
    parser.add_argument('--epochs', type=int, default=None, help='The number of epochs')
    parser.add_argument('--save-every', type=int, default=None, help='Save the checkpoint every <save_every> iterations/epochs')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--max-threads', type=int, default=None, help='The maximum allowed number of threads used by this process')
    parser.add_argument('--eval', action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--irregular', type=float, default=0, help='The ratio of missing observations (defaults to 0)')
    parser.add_argument('--pretrained-model', type=str, default=None, help='Path to a pretrained TS2Vec model (.pkl) to load before training (used with soybean_finetune loader)')
    args = parser.parse_args()
    
    print("Dataset:", args.dataset)
    print("Arguments:", str(args))
    
    gpu_arg = args.gpu if args.gpu == 'cpu' else int(args.gpu)
    device = init_dl_program(gpu_arg, seed=args.seed, max_threads=args.max_threads)
    
    print('Loading data... ', end='')
    if args.loader == 'UCR':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datautils.load_UCR(args.dataset)
        
    elif args.loader == 'UEA':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datautils.load_UEA(args.dataset)
        
    elif args.loader == 'forecast_csv':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(args.dataset)
        train_data = data[:, train_slice]
        
    elif args.loader == 'forecast_csv_univar':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(args.dataset, univar=True)
        train_data = data[:, train_slice]
        
    elif args.loader == 'forecast_npy':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_npy(args.dataset)
        train_data = data[:, train_slice]
        
    elif args.loader == 'forecast_npy_univar':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_npy(args.dataset, univar=True)
        train_data = data[:, train_slice]
        
    elif args.loader == 'anomaly':
        task_type = 'anomaly_detection'
        all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(args.dataset)
        train_data = datautils.gen_ano_train_data(all_train_data)
        
    elif args.loader == 'anomaly_coldstart':
        task_type = 'anomaly_detection_coldstart'
        all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(args.dataset)
        train_data, _, _, _ = datautils.load_UCR('FordA')

    elif args.loader == 'soybean_pretrain':
        task_type = 'soybean_pretrain'
        # pretrain_X.npy を読み込み、TS2Vec の自己教師あり学習に使用
        train_data = datautils.load_soybean_pretrain(args.dataset)  # (N, T, 9)

    elif args.loader == 'soybean_finetune':
        task_type = 'soybean_finetune'
        # X.npy + y.npy を年度別分割でロード（train:2015-2016, val:2017, test:2018）
        train_data, train_labels, val_data, val_labels, test_data, test_labels = \
            datautils.load_soybean_finetune(args.dataset)

    else:
        raise ValueError(f"Unknown loader {args.loader}.")
        
        
    if args.irregular > 0:
        if task_type == 'classification':
            train_data = data_dropout(train_data, args.irregular)
            test_data = data_dropout(test_data, args.irregular)
        else:
            raise ValueError(f"Task type {task_type} is not supported when irregular>0.")
    print('done')
    
    config = dict(
        batch_size=args.batch_size,
        lr=args.lr,
        output_dims=args.repr_dims,
        max_train_length=args.max_train_length
    )
    
    if args.save_every is not None:
        unit = 'epoch' if args.epochs is not None else 'iter'
        config[f'after_{unit}_callback'] = save_checkpoint_callback(args.save_every, unit)

    run_dir = 'training/' + args.run_name + '__' + name_with_datetime(args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    t = time.time()

    model = TS2Vec(
        input_dims=train_data.shape[-1],
        device=device,
        **config
    )

    # 事前学習済みモデルが指定されていればウェイトをロード（soybean_finetune 用）
    if args.pretrained_model is not None:
        print(f'Loading pretrained model from: {args.pretrained_model}')
        model.load(args.pretrained_model)

    loss_log = model.fit(
        train_data,
        n_epochs=args.epochs,
        n_iters=args.iters,
        verbose=True
    )
    model.save(f'{run_dir}/model.pkl')

    t = time.time() - t
    print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")

    if args.eval:
        if task_type == 'classification':
            out, eval_res = tasks.eval_classification(model, train_data, train_labels, test_data, test_labels, eval_protocol='svm')
        elif task_type == 'forecasting':
            out, eval_res = tasks.eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols)
        elif task_type == 'anomaly_detection':
            out, eval_res = tasks.eval_anomaly_detection(model, all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay)
        elif task_type == 'anomaly_detection_coldstart':
            out, eval_res = tasks.eval_anomaly_detection_coldstart(model, all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay)
        elif task_type == 'soybean_pretrain':
            # 事前学習のみ → 評価なし（モデル保存で完了）
            print('Pre-training complete. Model saved. No downstream evaluation in pretrain mode.')
            out, eval_res = None, None
        elif task_type == 'soybean_finetune':
            out, eval_res = tasks.eval_yield_regression(
                model,
                train_data, train_labels,
                val_data,   val_labels,
                test_data,  test_labels
            )
            m = eval_res['metrics']
            print(f"\n=== Yield Regression Results ===")
            print(f"  Train │ RMSE={m['train']['RMSE']:.2f}  MAE={m['train']['MAE']:.2f}  R²={m['train']['R2']:.4f}")
            print(f"  Val   │ RMSE={m['val']['RMSE']:.2f}  MAE={m['val']['MAE']:.2f}  R²={m['val']['R2']:.4f}")
            print(f"  Test  │ RMSE={m['test']['RMSE']:.2f}  MAE={m['test']['MAE']:.2f}  R²={m['test']['R2']:.4f}")
            print(f"  Best Ridge alpha: {eval_res['best_ridge_alpha']}")
        else:
            assert False
        if out is not None:
            pkl_save(f'{run_dir}/out.pkl', out)
        if eval_res is not None:
            pkl_save(f'{run_dir}/eval_res.pkl', eval_res)
            print('Evaluation result:', eval_res)

    print("Finished.")
