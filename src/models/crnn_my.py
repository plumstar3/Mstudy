# -*- coding: utf-8 -*-
"""
気象データ(E)と平均収量(Ybar)のみを使用する大豆収量予測モデル。
データ不足に対応するため、シーケンス長を「3年」に変更。
データローダーをExcel(.xlsx)対応版に変更。
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, callbacks
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

# --- 定数設定 ---
TIME_STEPS = 3  # シーケンス長を3年に設定
# ----------------

# 1. データローダー関連
# ★修正点★: 引数のデフォルトファイル名を.xlsxに変更し、pd.read_excelを使用
def load_and_preprocess_data(path='./Data/Input/soybean_data_new.csv'):
    """CSVファイルを読み込み、基本的な前処理（標準化など）を行う。"""
    if not os.path.exists(path):
        print(f"エラー: データファイル '{path}' が見つかりません。プログラムを終了します。")
        return None
    df = pd.read_csv(path)
    
    # 1列目:place, 2列目:year, 3列目:yield, 4列目以降:気象データ
    feature_cols = df.columns[3:] 
    
    # 2017年までのデータで標準化のパラメータを計算
    train_df = df[df['year'] <= 2017]
    mean = train_df[feature_cols].mean()
    std = train_df[feature_cols].std()
    std[std == 0] = 1.0
    
    df[feature_cols] = (df[feature_cols] - mean) / std
    df = df.fillna(0)
    return df

def create_year_loc_dict_and_avg(df):
    """データ辞書と平均収量辞書を作成する。"""
    loc_year_dict = { (row.place, int(row.year)): row for index, row in df.iterrows() }
    
    avg_yield_by_year_raw = df.groupby('year')['yield'].mean().to_dict()
    
    avg_yield_by_year = df.groupby('year')['yield'].mean()
    mean_yield = avg_yield_by_year.mean()
    std_yield = avg_yield_by_year.std()
    avg_dict = (avg_yield_by_year - mean_yield) / std_yield
    
    # 2018年のデータがない場合のフォールバック
    if 2018 not in avg_dict.index and 2017 in avg_dict.index:
        avg_dict[2018] = avg_dict.get(2017, 0)
        avg_yield_by_year_raw[2018] = avg_yield_by_year_raw.get(2017, 0)
        
    return loc_year_dict, {str(k): v for k, v in avg_dict.to_dict().items()}, avg_yield_by_year_raw

class SoybeanDataGenerator(tf.keras.utils.Sequence):
    """Kerasモデルのためのカスタムデータジェネレータ (3年シーケンス版)"""
    def __init__(self, df, loc_year_dict, avg_dict, batch_size, is_training=True, 
                 mean_last_features=None, avg_yield_raw=None):
        self.loc_year_dict = loc_year_dict
        self.avg_dict = avg_dict
        self.batch_size = batch_size
        self.is_training = is_training 
        self.mean_last_features = mean_last_features
        self.avg_yield_raw = avg_yield_raw
        
        self.sequences = []
        
        if is_training:
            # --- 訓練シーケンスの生成 ---
            loc_ids = df['place'].unique()
            all_years = sorted(df['year'].unique())
            
            for loc_id in loc_ids:
                # シーケンス長(TIME_STEPS)に合わせてループ範囲を調整
                # 3年シーケンスなら、len(all_years) - 2 回ループ
                for i in range(len(all_years) - (TIME_STEPS - 1)):
                    seq_years = all_years[i : i + TIME_STEPS]
                    if all((loc_id, year) in self.loc_year_dict for year in seq_years):
                        self.sequences.append({'loc_id': loc_id, 'years': seq_years})
            
            self.sequences = [s for s in self.sequences if 2018 not in s['years']]
            print(f"訓練ジェネレータが、{len(self.sequences)}個の有効な「地域-{TIME_STEPS}年」シーケンスを生成しました。")
            
        else:
            # --- 検証シーケンスの生成 ---
            loc_ids_2018 = df[df['year'] == 2018]['place'].unique()
            valid_loc_ids = []
            for loc_id in loc_ids_2018:
                if (loc_id, 2018) in self.loc_year_dict:
                     valid_loc_ids.append(loc_id)
            self.sequences = valid_loc_ids
            
            if self.mean_last_features is None or self.avg_yield_raw is None:
                raise ValueError("検証ジェネレータには 'mean_last_features' と 'avg_yield_raw' が必要です。")
            print(f"検証ジェネレータが、{len(self.sequences)}個の有効な「2018年地点データ」を生成しました。")
        
        self.indices = np.arange(len(self.sequences))
        self.on_epoch_end()

    def __len__(self):
        if len(self.sequences) == 0: return 0
        return int(np.floor(len(self.sequences) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        actual_batch_size = len(batch_indices)

        # TIME_STEPS (3) に合わせてshapeを変更
        X_dict = {
            'e_input': np.zeros((actual_batch_size, TIME_STEPS, 312)),
            'ybar_input': np.zeros((actual_batch_size, TIME_STEPS, 1))
        }
        Y_dict = {
            'Yhat1': np.zeros((actual_batch_size, 1)),
            'Yhat2': np.zeros((actual_batch_size, TIME_STEPS - 1, 1)) # 過去2年分
        }

        if self.is_training:
            batch_seq_info = [self.sequences[i] for i in batch_indices]
            for i, seq_info in enumerate(batch_seq_info):
                loc_id = seq_info['loc_id']
                years = seq_info['years']
                
                for j, year in enumerate(years):
                    sample = self.loc_year_dict[(loc_id, year)]
                    features = sample.iloc[3:].values
                    
                    X_dict['e_input'][i, j, :] = features[0:312]
                    X_dict['ybar_input'][i, j, 0] = self.avg_dict[str(year)]

                Y_dict['Yhat1'][i] = self.loc_year_dict[(loc_id, years[-1])]['yield']
                past_yields = [self.loc_year_dict[(loc_id, y)]['yield'] for y in years[:-1]]
                # reshape((TIME_STEPS - 1, 1)) -> (2, 1)
                Y_dict['Yhat2'][i] = np.array(past_yields).reshape(TIME_STEPS - 1, 1)
        
        else:
            # --- 検証バッチの生成 ---
            batch_loc_ids = [self.sequences[i] for i in batch_indices]
            
            # 検証時の「過去の年」を定義 (2018年がターゲットなので、その前の2年分)
            mean_last_years = [2016, 2017] 

            for i, loc_id in enumerate(batch_loc_ids):
                # タイムステップ 0〜(TIME_STEPS-2) : 過去の平均データ
                for j, year in enumerate(mean_last_years):
                    mean_data = self.mean_last_features[year]
                    X_dict['e_input'][i, j, :] = mean_data['features']
                    X_dict['ybar_input'][i, j, 0] = mean_data['ybar']
                
                # タイムステップ (TIME_STEPS-1) : 2018年の実測データ
                target_idx = TIME_STEPS - 1 
                year_2018 = 2018
                sample = self.loc_year_dict[(loc_id, year_2018)]
                features = sample.iloc[3:].values
                
                X_dict['e_input'][i, target_idx, :] = features[0:312]
                X_dict['ybar_input'][i, target_idx, 0] = self.avg_dict[str(year_2018)]

                Y_dict['Yhat1'][i] = sample['yield']
                past_yields_raw = [self.avg_yield_raw.get(y, 0) for y in mean_last_years]
                Y_dict['Yhat2'][i] = np.array(past_yields_raw).reshape(TIME_STEPS - 1, 1)

        return X_dict, Y_dict

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

# 2. モデル定義
def build_and_compile_model():
    """
    気象データ(E)と平均収量(Ybar)のみを使用するモデル。
    """
    # shapeのタイムステップを TIME_STEPS (3) に変更
    e_input = layers.Input(shape=(TIME_STEPS, 312), name="e_input")
    ybar_input = layers.Input(shape=(TIME_STEPS, 1), name="ybar_input")

    # --- 特徴量処理ブロック (CNN) ---
    # 気象データは6要素 x 52週 = 312次元
    e_cnn_input = layers.Input(shape=(52, 1), name="e_cnn_input")
    
    # TF1.xロジックに基づくCNN構成
    x = layers.Conv1D(8, 9, activation='relu', padding='valid')(e_cnn_input)
    x = layers.AveragePooling1D(2)(x)
    x = layers.Conv1D(12, 3, activation='relu', padding='valid')(x)
    x = layers.AveragePooling1D(2)(x)
    
    e_cnn_output = layers.Flatten()(x)
    shared_e_cnn = models.Model(inputs=e_cnn_input, outputs=e_cnn_output, name="Shared_E_CNN")
    e_proc_input = layers.Input(shape=(312,), name="e_proc_input")
    e_reshaped = layers.Reshape((6, 52, 1))(e_proc_input)
    e_sub_outputs = [shared_e_cnn(e_reshaped[:, i]) for i in range(6)]
    e_proc_output = layers.Concatenate()(e_sub_outputs)
    e_processor = models.Model(inputs=e_proc_input, outputs=e_proc_output, name="E_Processor")

    # --- TimeDistributed ---
    e_processed = layers.TimeDistributed(e_processor, name="TDD_E_Processor")(e_input)

    # --- LSTM ---
    merged = layers.Concatenate()([e_processed, ybar_input])
    x = layers.Dense(128, activation='relu')(merged)
    x = layers.LSTM(64, return_sequences=True, dropout=0.2)(x)
    output = layers.TimeDistributed(layers.Dense(1))(x)
    
    # --- 出力層 ---
    Yhat1 = output[:, -1, :]
    Yhat1 = layers.Identity(name='Yhat1')(Yhat1)
    Yhat2 = output[:, :-1, :]
    Yhat2 = layers.Identity(name='Yhat2')(Yhat2)

    model = models.Model(inputs=[e_input, ybar_input], outputs=[Yhat1, Yhat2])
    
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0003),
                  loss={'Yhat1': losses.Huber(), 'Yhat2': losses.Huber()},
                  loss_weights={'Yhat1': 1.0, 'Yhat2': 0.0},
                  metrics={'Yhat1': 'mae'})
    return model

# 3. 訓練と評価
def run_training_and_evaluation():
    print("\n データ読み込みと前処理...")
    # ★修正点★: 引数でファイルパスを指定 (必要に応じて変更可能)
    df = load_and_preprocess_data(path='./Data/Input/soybean_data_new.csv') 
    if df is None: return

    loc_year_dict, avg_dict, avg_yield_raw = create_year_loc_dict_and_avg(df)
    
    print("\n検証用の平均特徴量 (mean_last) を計算しています...")
    mean_last_features = {}
    feature_cols = df.columns[3:] 
    
    # 3年シーケンス(末尾2018)の検証に必要な過去年は [2016, 2017]
    years_for_mean = [2015, 2016, 2017]
    
    df_mean_features_by_year = df[df['year'].isin(years_for_mean)].groupby('year')[feature_cols].mean()
    
    for year in years_for_mean:
        if year in df_mean_features_by_year.index:
            features = df_mean_features_by_year.loc[year].values
            ybar = avg_dict.get(str(year), 0)
            mean_last_features[year] = {
                'features': features,
                'ybar': ybar
            }
        else:
            mean_last_features[year] = {
                'features': np.zeros(len(feature_cols)),
                'ybar': 0
            }
    
    print("\n データジェネレータの作成...")
    train_generator = SoybeanDataGenerator(df, loc_year_dict, avg_dict, batch_size=32, is_training=True)
    val_generator = SoybeanDataGenerator(df, loc_year_dict, avg_dict, batch_size=26, is_training=False, 
                                         mean_last_features=mean_last_features, 
                                         avg_yield_raw=avg_yield_raw)

    print("\n モデルの構築とコンパイル...")
    model = build_and_compile_model()
    model.summary(line_length=120)

    print("\n モデルの訓練を開始します...")
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    if len(train_generator) == 0:
        print("\n警告: 訓練データシーケンスが生成されませんでした。")
        return 
    
    if len(val_generator) > 0:
        model.fit(train_generator, validation_data=val_generator, epochs=200, callbacks=[early_stop], verbose=2)
    else:
        model.fit(train_generator, epochs=200, verbose=2)
            
    model.save("soybean_yield_model_weather_only_3yr.keras")
    print("\n モデル保存完了")

    print("\n モデルの評価を開始します...")
    if len(val_generator) > 0:
        val_generator.on_epoch_end = lambda: None
        loaded_model = models.load_model("soybean_yield_model_weather_only_3yr.keras")
        
        predictions = loaded_model.predict(val_generator)
        Y1_pred = predictions[0]
        Y1_test_true = np.concatenate([val_generator[i][1]['Yhat1'] for i in range(len(val_generator))])
        
        rmse = np.sqrt(mean_squared_error(Y1_test_true, Y1_pred))
        print(f"\n Test RMSE (final year): {rmse:.4f}")

        # === 相対誤差の計算・表示 ===
        mask = Y1_test_true != 0
        relative_errors = np.abs(Y1_pred[mask] - Y1_test_true[mask]) / np.abs(Y1_test_true[mask])
        mean_relative_error = np.mean(relative_errors)
        print(f"平均相対誤差 (Mean Relative Error): {mean_relative_error:.4f}")
        
        if len(Y1_test_true) >= 2:
            corr, _ = pearsonr(Y1_test_true.flatten(), Y1_pred.flatten())
            print(f" 相関係数: {corr:.4f}")

        np.savez("prediction_result_weather_only_3yr.npz", Y1_true=Y1_test_true, Y1_pred=Y1_pred)
    else:
        print("評価データがありません。")

if __name__ == "__main__":
    SEED = 42
    os.environ['PYTHONHASHSEED'] = str(SEED)
    tf.keras.utils.set_random_seed(SEED)
    print(" 大豆収量予測モデル (気象データのみ・3年シーケンス・Excel入力) - 実行")
    run_training_and_evaluation()