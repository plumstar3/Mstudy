# -*- coding: utf-8 -*-
"""
データ不足対応版：転移学習用スクリプト
修正点：
1. 重み移植時のレイヤー階層アクセスエラーを解消
2. 入力ファイルパスを .csv に変更
3. ★パディングを「平均値」から「0固定(Zero Padding)」に変更
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, callbacks
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

# --- 定数設定 ---
TIME_STEPS = 2  # 2年 (前年 + 当年)
SOURCE_MODEL_PATH = "soybean_yield_model_E_Ybar_only.keras"
# ----------------

# 1. データローダー
def load_and_preprocess_data(path='./Data/Input/soybean_data_new.csv'):
    if not os.path.exists(path):
        print(f"エラー: データファイル '{path}' が見つかりません。")
        return None
    try:
        if path.endswith('.csv'):
            df = pd.read_csv(path)
        elif path.endswith('.xlsx') or path.endswith('.xls'):
            df = pd.read_excel(path)
        else:
            return None
    except Exception as e:
        print(f"読み込みエラー: {e}")
        return None
    
    feature_cols = df.columns[3:] 
    mean = df[feature_cols].mean()
    std = df[feature_cols].std()
    std[std == 0] = 1.0
    df[feature_cols] = (df[feature_cols] - mean) / std
    df = df.fillna(0)
    return df

def create_year_loc_dict_and_avg(df):
    loc_year_dict = { (row.place, int(row.year)): row for index, row in df.iterrows() }
    avg_yield_by_year = df.groupby('year')['yield'].mean()
    mean_yield = avg_yield_by_year.mean()
    std_yield = avg_yield_by_year.std()
    avg_dict = (avg_yield_by_year - mean_yield) / std_yield
    return loc_year_dict, {str(k): v for k, v in avg_dict.to_dict().items()}

class SoybeanDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, loc_year_dict, avg_dict, batch_size, is_training=True):
        super().__init__()
        self.loc_year_dict = loc_year_dict
        self.avg_dict = avg_dict
        self.batch_size = batch_size
        self.is_training = is_training
        self.sequences = []
        
        # 平均値パディング用の計算は削除 (Zero Paddingのため不要)
        # self.feature_mean = df.iloc[:, 3:].mean().values
        
        self.target_samples = []
        for index, row in df.iterrows():
            self.target_samples.append({
                'loc_id': row['place'],
                'year': int(row['year'])
            })

        if is_training:
            self.sequences = [s for s in self.target_samples if s['year'] != 2018]
            print(f"訓練ジェネレータ: {len(self.sequences)}件のデータを0パディング込みで使用します。")
        else:
            self.sequences = [s for s in self.target_samples if s['year'] == 2018]
            print(f"検証ジェネレータ: {len(self.sequences)}件のデータを0パディング込みで使用します。")
        
        self.indices = np.arange(len(self.sequences))
        self.on_epoch_end()

    def __len__(self):
        if len(self.sequences) == 0: return 0
        return int(np.floor(len(self.sequences) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        actual_batch_size = len(batch_indices)
        
        # np.zerosで初期化されているため、何もしなければ0パディングになります
        X_dict = {
            'e_input': np.zeros((actual_batch_size, TIME_STEPS, 312)),
            'ybar_input': np.zeros((actual_batch_size, TIME_STEPS, 1))
        }
        Y_batch = np.zeros((actual_batch_size, 1))

        batch_samples = [self.sequences[i] for i in batch_indices]
        
        for i, sample in enumerate(batch_samples):
            target_loc = sample['loc_id']
            target_year = sample['year']
            
            years_to_fetch = [target_year - 1, target_year]
            
            for t_idx, year in enumerate(years_to_fetch):
                if (target_loc, year) in self.loc_year_dict:
                    # データがある場合: そのデータを使用
                    row_data = self.loc_year_dict[(target_loc, year)]
                    features = row_data.iloc[3:].values
                    X_dict['e_input'][i, t_idx, :] = features[0:312]
                    X_dict['ybar_input'][i, t_idx, 0] = self.avg_dict.get(str(year), 0)
                else:
                    # ★変更: データがない場合は0のままにする (Zero Padding)
                    # 明示的に書くなら pass ですが、初期値が0なのでそのままでOK
                    pass
            
            Y_batch[i] = self.loc_year_dict[(target_loc, target_year)]['yield']

        return X_dict, Y_batch

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

# 2. モデル定義
def build_target_model():
    e_input = layers.Input(shape=(TIME_STEPS, 312), name="e_input")
    ybar_input = layers.Input(shape=(TIME_STEPS, 1), name="ybar_input")

    e_cnn_input = layers.Input(shape=(52, 1), name="e_cnn_input")
    x = layers.Conv1D(8, 9, activation='relu', padding='valid', name="conv1d_1")(e_cnn_input)
    x = layers.AveragePooling1D(2)(x)
    x = layers.Conv1D(12, 3, activation='relu', padding='valid', name="conv1d_2")(x)
    x = layers.AveragePooling1D(2)(x)
    e_cnn_output = layers.Flatten()(x)
    
    shared_e_cnn = models.Model(inputs=e_cnn_input, outputs=e_cnn_output, name="Shared_E_CNN")
    
    e_proc_input = layers.Input(shape=(312,), name="e_proc_input")
    e_reshaped = layers.Reshape((6, 52, 1))(e_proc_input)
    e_sub_outputs = [shared_e_cnn(e_reshaped[:, i]) for i in range(6)]
    e_proc_output = layers.Concatenate()(e_sub_outputs)
    e_processor = models.Model(inputs=e_proc_input, outputs=e_proc_output, name="E_Processor")

    e_processed = layers.TimeDistributed(e_processor, name="TDD_E_Processor")(e_input)

    merged = layers.Concatenate()([e_processed, ybar_input])
    x = layers.Dense(128, activation='relu')(merged)
    x = layers.LSTM(64, return_sequences=True, dropout=0.2)(x)
    output = layers.TimeDistributed(layers.Dense(1))(x)
    
    Yhat1 = output[:, -1, :]
    Yhat1 = layers.Identity(name='Yhat1')(Yhat1)
    
    model = models.Model(inputs=[e_input, ybar_input], outputs=Yhat1)
    return model

# ★修正した転移学習関数★
def transfer_weights(target_model, source_model_path):
    print(f"\n[転移学習] '{source_model_path}' から重みを読み込んでいます...")
    if not os.path.exists(source_model_path):
        print("エラー: ソースモデルが見つかりません。")
        return target_model, False
    
    try:
        # 1. ソースモデルをロード
        source_model = models.load_model(source_model_path)
        
        # 2. 階層を掘り下げてSourceのCNN層を探す
        source_tdd = source_model.get_layer("TDD_E_Processor")
        source_processor = source_tdd.layer
        source_cnn = source_processor.get_layer("Shared_E_CNN")
        
        # 重みを取得
        weights = source_cnn.get_weights()
        
        # 3. Targetモデルの同じ階層へ移植
        target_tdd = target_model.get_layer("TDD_E_Processor")
        target_processor = target_tdd.layer
        target_cnn = target_processor.get_layer("Shared_E_CNN")
        
        target_cnn.set_weights(weights)
        target_cnn.trainable = False
        
        print(" - Shared_E_CNN の重みを正常に移植し、固定(Freeze)しました。")
        return target_model, True
        
    except Exception as e:
        print(f"重み移植失敗: {e}")
        try:
            print(f"  (参考) Source model layers: {[l.name for l in source_model.layers]}")
        except: pass
        return target_model, False

# 3. 実行ブロック
def run_transfer_learning():
    print("\n[Target] データ読み込み...")
    df = load_and_preprocess_data(path='./Data/Input/soybean_data_new.csv')
    if df is None: return

    loc_year_dict, avg_dict = create_year_loc_dict_and_avg(df)
    
    batch_size = 4 
    print(f"[Target] データジェネレータ作成 (TIME_STEPS={TIME_STEPS}, Batch={batch_size})...")
    
    train_generator = SoybeanDataGenerator(df, loc_year_dict, avg_dict, batch_size=batch_size, is_training=True)
    val_generator = SoybeanDataGenerator(df, loc_year_dict, avg_dict, batch_size=batch_size, is_training=False)

    print("[Target] モデル構築...")
    model = build_target_model()
    model, success = transfer_weights(model, SOURCE_MODEL_PATH)

    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss='huber',
                  metrics=['mae'])
    
    print("\n[Target] 訓練開始...")
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

    if len(train_generator) > 0:
        model.fit(train_generator, validation_data=val_generator, epochs=500, callbacks=[early_stop], verbose=2)
    
    model.save("soybean_yield_model_transfer_padded.keras")
    
    if len(val_generator) > 0:
        print("\n[Target] 評価中...")
        val_generator.on_epoch_end = lambda: None
        predictions = model.predict(val_generator)
        
        Y1_pred = predictions.reshape(-1, 1)
        Y1_test_true = np.concatenate([val_generator[i][1] for i in range(len(val_generator))]).reshape(-1, 1)

        rmse = np.sqrt(mean_squared_error(Y1_test_true, Y1_pred))
        print(f"\n Test RMSE: {rmse:.4f}")
        
        mask = Y1_test_true != 0
        if np.any(mask):
            relative_errors = np.abs(Y1_pred[mask] - Y1_test_true[mask]) / np.abs(Y1_test_true[mask])
            print(f" Mean Relative Error: {np.mean(relative_errors):.4f}")
        
        if len(Y1_test_true) >= 2:
            corr, _ = pearsonr(Y1_test_true.flatten(), Y1_pred.flatten())
            print(f" Correlation: {corr:.4f}")

            r2 = r2_score(Y1_test_true, Y1_pred) 
            print(f" R2 Score: {r2:.4f}") 
            
        np.savez("prediction_result_padded.npz", Y1_true=Y1_test_true, Y1_pred=Y1_pred)
    else:
        print("検証用データが生成されませんでした（2018年のデータがありません）。")

if __name__ == "__main__":
    SEED = 42
    os.environ['PYTHONHASHSEED'] = str(SEED)
    tf.keras.utils.set_random_seed(SEED)
    print(f" 大豆収量予測モデル (0パディング・CNN固定版)")
    run_transfer_learning()