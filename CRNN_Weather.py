# -*- coding: utf-8 -*-
"""
先行研究のモデル(TF1.x)の検証ロジックをKeras 2.xで再現したもの。
土壌(S)・管理(P)データを除外し、環境(E)データと平均収量(Ybar)のみを使用する。
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, callbacks
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

# 1. データローダー関連
def load_and_preprocess_data(path='./Data/Input/soybean_data.csv'):
    """CSVファイルを読み込み、基本的な前処理（標準化など）を行う。"""
    if not os.path.exists(path):
        print(f"エラー: データファイル '{path}' が見つかりません。プログラムを終了します。")
        return None
    df = pd.read_csv(path)
    
    # 392カラムあると想定 (e: 312, s: 66, p: 14)
    feature_cols = df.columns[3:]
    train_df = df[df['year'] <= 2017]
    mean = train_df[feature_cols].mean()
    std = train_df[feature_cols].std()
    std[std == 0] = 1.0
    df[feature_cols] = (df[feature_cols] - mean) / std
    df = df.fillna(0)
    df = df[df['yield'] >= 5].reset_index(drop=True)
    return df

def create_year_loc_dict_and_avg(df):
    """
    データ辞書、標準化済み平均収量辞書、標準化前平均収量辞書を作成する。
    """
    loc_year_dict = { (row.loc_ID, int(row.year)): row for index, row in df.iterrows() }
    
    # 標準化前の年ごと平均収量 (Yhat2の正解ラベルとして使用)
    avg_yield_by_year_raw = df.groupby('year')['yield'].mean().to_dict()
    
    # 標準化済み年ごと平均収量 (ybar_inputとして使用)
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
    """
    Kerasモデルのためのカスタムデータジェネレータ (土壌・管理データなし版)
    検証時(is_training=False)は、TF1.x版のget_sample_teのロジックを再現する。
    """
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
            loc_ids = df['loc_ID'].unique()
            all_years = sorted(df['year'].unique())
            for loc_id in loc_ids:
                for i in range(len(all_years) - 4):
                    seq_years = all_years[i:i+5]
                    if all((loc_id, year) in self.loc_year_dict for year in seq_years):
                        self.sequences.append({'loc_id': loc_id, 'years': seq_years})
            self.sequences = [s for s in self.sequences if 2018 not in s['years']]
            print(f"訓練ジェネレータが、{len(self.sequences)}個の有効な「地域-5年」シーケンスを生成しました。")
        else:
            # --- 検証シーケンスの生成 ---
            loc_ids_2018 = df[df['year'] == 2018]['loc_ID'].unique()
            valid_loc_ids = []
            for loc_id in loc_ids_2018:
                if (loc_id, 2018) in self.loc_year_dict:
                     valid_loc_ids.append(loc_id)
            self.sequences = valid_loc_ids # 検証対象のloc_idリスト
            
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

        # 入力データを辞書形式で準備 (s_input と p_input を削除)
        X_dict = {
            'e_input': np.zeros((actual_batch_size, 5, 312)),
            # 's_input': (削除)
            # 'p_input': (削除)
            'ybar_input': np.zeros((actual_batch_size, 5, 1))
        }
        Y_dict = {
            'Yhat1': np.zeros((actual_batch_size, 1)),
            'Yhat2': np.zeros((actual_batch_size, 4, 1))
        }

        if self.is_training:
            # --- 訓練バッチの生成 ---
            batch_seq_info = [self.sequences[i] for i in batch_indices]
            for i, seq_info in enumerate(batch_seq_info):
                loc_id = seq_info['loc_id']
                years = seq_info['years']
                
                for j, year in enumerate(years):
                    sample = self.loc_year_dict[(loc_id, year)]
                    features = sample.iloc[3:].values # 全特徴量 (392)
                    
                    X_dict['e_input'][i, j, :] = features[0:312]  # 環境データ
                    # s_input (312-377) はスキップ
                    # p_input (378-391) はスキップ
                    X_dict['ybar_input'][i, j, 0] = self.avg_dict[str(year)]

                Y_dict['Yhat1'][i] = self.loc_year_dict[(loc_id, years[-1])]['yield']
                past_yields = [self.loc_year_dict[(loc_id, y)]['yield'] for y in years[:-1]]
                Y_dict['Yhat2'][i] = np.array(past_yields).reshape(4, 1)
        
        else:
            # --- 検証バッチの生成 (TF1.xロジック) ---
            batch_loc_ids = [self.sequences[i] for i in batch_indices]
            mean_last_years = [2014, 2015, 2016, 2017]

            for i, loc_id in enumerate(batch_loc_ids):
                # タイムステップ 0-3 (過去4年分) に「全地点の平均データ」を入力
                for j, year in enumerate(mean_last_years):
                    mean_data = self.mean_last_features[year]
                    features = mean_data['features'] # 平均特徴量 (392)
                    
                    X_dict['e_input'][i, j, :] = features[0:312] # 環境データ
                    # s_input (312-377) はスキップ
                    # p_input (378-391) はスキップ
                    X_dict['ybar_input'][i, j, 0] = mean_data['ybar']
                
                # タイムステップ 4 (5年目) に「2018年の実測データ」を入力
                year_2018 = 2018
                sample = self.loc_year_dict[(loc_id, year_2018)]
                features = sample.iloc[3:].values # 全特徴量 (392)
                X_dict['e_input'][i, 4, :] = features[0:312] # 環境データ
                # s_input (312-377) はスキップ
                # p_input (378-391) はスキップ
                X_dict['ybar_input'][i, 4, 0] = self.avg_dict[str(year_2018)]

                Y_dict['Yhat1'][i] = sample['yield']
                past_yields_raw = [self.avg_yield_raw[y] for y in mean_last_years]
                Y_dict['Yhat2'][i] = np.array(past_yields_raw).reshape(4, 1)

        return X_dict, Y_dict

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

# 2. モデル定義
def build_and_compile_model():
    """
    土壌(S)・管理(P)データ用CNNを除いたモデルを構築する。
    """
    # --- 入力層の定義 (s_input と p_input を削除) ---
    e_input = layers.Input(shape=(5, 312), name="e_input")
    # s_input = (削除)
    # p_input = (削除)
    ybar_input = layers.Input(shape=(5, 1), name="ybar_input")

    # --- 特徴量処理ブロックの定義 (サブモデルとして) ---
    # 環境(E)データ用 共有CNNモデル (変更なし)
    e_cnn_input = layers.Input(shape=(52, 1), name="e_cnn_input")
    x = layers.Conv1D(8, 9, activation='relu', padding='valid')(e_cnn_input)
    x = layers.AveragePooling1D(2)(x)
    x = layers.Conv1D(12, 3, activation='relu', padding='valid')(x)
    x = layers.AveragePooling1D(2)(x)
    e_cnn_output = layers.Flatten()(x)
    shared_e_cnn = models.Model(inputs=e_cnn_input, outputs=e_cnn_output, name="Shared_E_CNN")

    # 環境(E)データ ラッパーモデル (変更なし)
    e_proc_input = layers.Input(shape=(312,), name="e_proc_input")
    e_reshaped = layers.Reshape((6, 52, 1))(e_proc_input)
    e_sub_outputs = [shared_e_cnn(e_reshaped[:, i]) for i in range(6)]
    e_proc_output = layers.Concatenate()(e_sub_outputs)
    e_processor = models.Model(inputs=e_proc_input, outputs=e_proc_output, name="E_Processor")

    # 土壌(S)データ用CNNモデル (削除)
    # s_processor = (削除)

    # --- TimeDistributedで各タイムステップに特徴量処理を適用 ---
    e_processed = layers.TimeDistributed(e_processor, name="TDD_E_Processor")(e_input)
    # s_processed = (削除)
    # p_processed = (削除)

    # --- 全ての特徴量を結合し、LSTMに入力 (s_processed と p_processed を削除) ---
    merged = layers.Concatenate()([e_processed, ybar_input]) # <--- 修正
    x = layers.Dense(128, activation='relu')(merged)
    x = layers.LSTM(64, return_sequences=True, dropout=0.2)(x)
    output = layers.TimeDistributed(layers.Dense(1))(x)
    
    # Lambdaレイヤーを使わずにスライスで出力を定義
    Yhat1 = output[:, -1, :]
    Yhat1 = layers.Identity(name='Yhat1')(Yhat1)
    Yhat2 = output[:, :-1, :]
    Yhat2 = layers.Identity(name='Yhat2')(Yhat2)

    # モデルの定義 (入力から s_input と p_input を削除)
    model = models.Model(inputs=[e_input, ybar_input], outputs=[Yhat1, Yhat2]) # <--- 修正
    
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0003),
                  loss={'Yhat1': losses.Huber(), 'Yhat2': losses.Huber()},
                  loss_weights={'Yhat1': 1.0, 'Yhat2': 0.0},
                  metrics={'Yhat1': 'mae'})
    return model

# 3. 訓練と評価
def run_training_and_evaluation():
    print("\n データ読み込みと前処理...")
    df = load_and_preprocess_data()
    if df is None: return

    loc_year_dict, avg_dict, avg_yield_raw = create_year_loc_dict_and_avg(df)
    
    print("\n検証用の平均特徴量 (mean_last) を計算しています...")
    mean_last_features = {}
    feature_cols = df.columns[3:] 
    
    years_for_mean = [2014, 2015, 2016, 2017]
    
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
            print(f"警告: {year}年のデータが不足しており、平均特徴量を計算できません。0で埋めます。")
            mean_last_features[year] = {
                'features': np.zeros(len(feature_cols)), # 392カラム分の0
                'ybar': 0
            }
    # --- 計算完了 ---
    
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
        print("\nエラー: 訓練データがありません。データ生成プロセスを確認してください。")
        return

    if len(val_generator) > 0:
        model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=200,
            callbacks=[early_stop],
            verbose=2
        )
    else:
        print("\n警告: 検証データが見つからなかったため、検証なしで訓練します。")
        early_stop_train = callbacks.EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
        model.fit(train_generator, epochs=200, callbacks=[early_stop_train])
        
    # 保存ファイル名を変更
    model_filename = "soybean_yield_model_E_Ybar_only.keras"
    model.save(model_filename)
    print(f"\n モデル訓練完了・保存済み ({model_filename})")

    print("\n モデルの評価を開始します...")
    if len(val_generator) > 0:
        val_generator.on_epoch_end = lambda: None
        
        loaded_model = models.load_model(model_filename)
        print(" モデル読み込み完了")

        predictions = loaded_model.predict(val_generator)
        Y1_pred = predictions[0]
        
        Y1_test_true = np.concatenate([val_generator[i][1]['Yhat1'] for i in range(len(val_generator))])
        
        rmse = np.sqrt(mean_squared_error(Y1_test_true, Y1_pred))
        print(f"\n Test RMSE (final year): {rmse:.4f}")

        if len(Y1_test_true) >= 2:
            corr, _ = pearsonr(Y1_test_true.flatten(), Y1_pred.flatten())
            print(f" 相関係数 (final year): {corr:.4f}")

        # 結果ファイル名を変更
        result_filename = "prediction_result_E_Ybar_only.npz"
        np.savez(result_filename, Y1_true=Y1_test_true, Y1_pred=Y1_pred)
        print(f" 予測結果を '{result_filename}' に保存しました")
    else:
        print("評価データがありません。評価をスキップします。")

# 4. メイン実行ブロック
if __name__ == "__main__":
    print(" 大豆収量予測モデル (環境E + 平均収量Ybar のみ) - 総合実行スクリプト") # <--- タイトル変更
    run_training_and_evaluation()
    print("\n 全処理が完了しました！")

