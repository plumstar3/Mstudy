import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, callbacks
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from tensorflow.keras.layers import Bidirectional


# 1. データローダー関連
def load_and_preprocess_data(path='./Data/Input/soybean_data.csv'):
    """CSVファイルを読み込み、基本的な前処理（標準化など）を行う。"""
    if not os.path.exists(path):
        print(f"エラー: データファイル '{path}' が見つかりません。プログラムを終了します。")
        return None
    df = pd.read_csv(path)
    
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
    """年と地域(loc_ID)をキーにしたデータ辞書と、年ごとの平均収量辞書を作成する。"""
    loc_year_dict = { (row.loc_ID, int(row.year)): row for index, row in df.iterrows() }
    avg_yield_by_year = df.groupby('year')['yield'].mean()
    mean_yield = avg_yield_by_year.mean()
    std_yield = avg_yield_by_year.std()
    avg_dict = (avg_yield_by_year - mean_yield) / std_yield
    
    if 2018 not in avg_dict.index and 2017 in avg_dict.index:
        avg_dict[2018] = avg_dict.get(2017, 0)
        
    return loc_year_dict, {str(k): v for k, v in avg_dict.to_dict().items()}

class SoybeanDataGenerator(tf.keras.utils.Sequence):
    """Kerasモデルのためのカスタムデータジェネレータ（地域考慮・TimeDistributed対応版）"""
    def __init__(self, df, loc_year_dict, avg_dict, batch_size, is_training=True):
        self.loc_year_dict = loc_year_dict
        self.avg_dict = avg_dict
        self.batch_size = batch_size
        
        self.sequences = []
        loc_ids = df['loc_ID'].unique()
        all_years = sorted(df['year'].unique())

        for loc_id in loc_ids:
            for i in range(len(all_years) - 4):
                seq_years = all_years[i:i+5]
                if all((loc_id, year) in self.loc_year_dict for year in seq_years):
                    self.sequences.append({'loc_id': loc_id, 'years': seq_years})
        
        if is_training:
            self.sequences = [s for s in self.sequences if 2018 not in s['years']]
        else:
            self.sequences = [s for s in self.sequences if 2018 in s['years']]
        
        print(f"{'訓練' if is_training else '検証'}ジェネレータが、{len(self.sequences)}個の有効な「地域-5年」シーケンスを生成しました。")
        self.indices = np.arange(len(self.sequences))
        self.on_epoch_end()

    def __len__(self):
        if len(self.sequences) == 0: return 0
        return int(np.floor(len(self.sequences) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        batch_seq_info = [self.sequences[i] for i in batch_indices]
        actual_batch_size = len(batch_seq_info)

        #入力データを辞書形式で準備
        X_dict = {
            'e_input': np.zeros((actual_batch_size, 5, 312)),
            's_input': np.zeros((actual_batch_size, 5, 66)),
            'p_input': np.zeros((actual_batch_size, 5, 14)),
            'ybar_input': np.zeros((actual_batch_size, 5, 1))
        }
        Y_dict = {
            'Yhat1': np.zeros((actual_batch_size, 1)),
            'Yhat2': np.zeros((actual_batch_size, 4, 1))
        }

        for i, seq_info in enumerate(batch_seq_info):
            loc_id = seq_info['loc_id']
            years = seq_info['years']
            
            for j, year in enumerate(years):
                sample = self.loc_year_dict[(loc_id, year)]
                features = sample.iloc[3:].values # ID, year, yieldを除く
                
                # 特徴量を各入力に割り当て
                X_dict['e_input'][i, j, :] = features[0:312]
                X_dict['s_input'][i, j, :] = features[312:378]
                X_dict['p_input'][i, j, :] = features[378:392]
                X_dict['ybar_input'][i, j, 0] = self.avg_dict[str(year)]

            Y_dict['Yhat1'][i] = self.loc_year_dict[(loc_id, years[-1])]['yield']
            past_yields = [self.loc_year_dict[(loc_id, y)]['yield'] for y in years[:-1]]
            Y_dict['Yhat2'][i] = np.array(past_yields).reshape(4, 1)

        return X_dict, Y_dict

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

# 2. モデル定義
def create_cnn_block(input_layer, filters, kernel_sizes):
    """汎用的なCNNブロックを作成"""
    x = input_layer
    for f, k in zip(filters, kernel_sizes):
        x = layers.Conv1D(f, k, activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(pool_size=2, strides=1, padding='same')(x) # ストライドを調整
    return layers.Flatten()(x)

def build_and_compile_model():
    """
    論文の設計思想（重み共有とTimeDistributed）を忠実に再現したモデルを構築する。
    """
    # --- 入力層の定義 (年ごとの変化を捉えるため、5つのタイムステップを持つ) ---
    e_input = layers.Input(shape=(5, 312), name="e_input")
    s_input = layers.Input(shape=(5, 66), name="s_input")
    p_input = layers.Input(shape=(5, 14), name="p_input")
    ybar_input = layers.Input(shape=(5, 1), name="ybar_input")

    # --- 特徴量処理ブロックの定義 (サブモデルとして) ---
    # ✨【修正点1】: 6種類の気象データ全てに適用する「共有CNNブロック」を1つだけ定義
    e_cnn_input = layers.Input(shape=(52, 1), name="e_cnn_input")
    x = layers.Conv1D(8, 9, activation='relu', padding='valid')(e_cnn_input)
    x = layers.AveragePooling1D(2)(x)
    x = layers.Conv1D(12, 3, activation='relu', padding='valid')(x)
    x = layers.AveragePooling1D(2)(x)
    e_cnn_output = layers.Flatten()(x)
    shared_e_cnn = models.Model(inputs=e_cnn_input, outputs=e_cnn_output, name="Shared_E_CNN")

    # ✨【修正点2】: 6つの気象データを処理するためのラッパーモデルを定義
    e_proc_input = layers.Input(shape=(312,), name="e_proc_input")
    e_reshaped = layers.Reshape((6, 52, 1))(e_proc_input)
    # 6つの入力それぞれに、上で定義した「全く同じ共有CNN」を適用する
    e_sub_outputs = [shared_e_cnn(e_reshaped[:, i]) for i in range(6)]
    e_proc_output = layers.Concatenate()(e_sub_outputs)
    e_processor = models.Model(inputs=e_proc_input, outputs=e_proc_output, name="E_Processor")

    # 土壌(S)データ用CNNモデル
    s_proc_input = layers.Input(shape=(66,), name="s_proc_input")
    s_reshaped = layers.Reshape((6, 11))(s_proc_input)
    s_cnn_out = layers.Flatten()(layers.Conv1D(16, 3, activation='relu')(s_reshaped))
    s_processor = models.Model(inputs=s_proc_input, outputs=s_cnn_out, name="S_Processor")

    # --- TimeDistributedで各タイムステップに特徴量処理を適用 ---
    e_processed = layers.TimeDistributed(e_processor, name="TDD_E_Processor")(e_input)
    s_processed = layers.TimeDistributed(s_processor, name="TDD_S_Processor")(s_input)
    p_processed = layers.TimeDistributed(layers.Flatten(), name="TDD_P_Flatten")(p_input)

    # --- 全ての特徴量を結合し、LSTMに入力 ---
    merged = layers.Concatenate()([e_processed, s_processed, p_processed, ybar_input])
    x = layers.Dense(128, activation='relu')(merged)
    x = layers.LSTM(64, return_sequences=True, dropout=0.2)(x)
    output = layers.TimeDistributed(layers.Dense(1))(x)
    
    Yhat1 = layers.Identity(name='Yhat1')(output[:, -1, :])
    Yhat2 = layers.Identity(name='Yhat2')(output[:, :-1, :])

    model = models.Model(inputs=[e_input, s_input, p_input, ybar_input], outputs=[Yhat1, Yhat2])
    
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0003),
                  loss={'Yhat1': losses.Huber(), 'Yhat2': losses.Huber()},
                  loss_weights={'Yhat1': 1.0, 'Yhat2': 0.0},
                  metrics={'Yhat1': 'mae'})
    return model

# 3. 訓練と評価
def run_training_and_evaluation():
    print("\n モデルの訓練を開始します...")
    df = load_and_preprocess_data()
    if df is None: return

    loc_year_dict, avg_dict = create_year_loc_dict_and_avg(df)
    
    train_generator = SoybeanDataGenerator(df, loc_year_dict, avg_dict, batch_size=32, is_training=True)
    val_generator = SoybeanDataGenerator(df, loc_year_dict, avg_dict, batch_size=26, is_training=False)

    model = build_and_compile_model()
    model.summary(line_length=120)

    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    if len(val_generator) > 0:
        model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=200,
            callbacks=[early_stop],
            verbose=2
        )
    else:
        print("\n検証データが見つからなかったため、検証なしで訓練します。")
        model.fit(train_generator, epochs=200, callbacks=[callbacks.EarlyStopping(monitor='loss', patience=20)])
        
    model.save("soybean_yield_model.keras")
    print("\n モデル訓練完了・保存済み")

    print("\n モデルの評価を開始します...")
    if len(val_generator) > 0:
        val_generator.on_epoch_end = lambda: None
        
        loaded_model = models.load_model("soybean_yield_model.keras")
        print(" モデル読み込み完了")

        # 全ての検証データで一度に予測
        predictions = loaded_model.predict(val_generator)
        Y1_pred = predictions[0] # 最初の出力がYhat1
        
        # 全ての正解データをジェネレータから取得
        Y1_test_true = np.concatenate([val_generator[i][1]['Yhat1'] for i in range(len(val_generator))])
        
        rmse = np.sqrt(mean_squared_error(Y1_test_true, Y1_pred))
        print(f"\n Test RMSE (final year): {rmse:.4f}")

        if len(Y1_test_true) >= 2:
            corr, _ = pearsonr(Y1_test_true.flatten(), Y1_pred.flatten())
            print(f" 相関係数 (final year): {corr:.4f}")

        np.savez("prediction_result.npz", Y1_true=Y1_test_true, Y1_pred=Y1_pred)
        print(" 予測結果を 'prediction_result.npz' に保存しました")
    else:
        print("評価データがありません。評価をスキップします。")

# 4. メイン実行ブロック
if __name__ == "__main__":
    print(" 大豆収量予測モデル - 総合実行スクリプト")
    run_training_and_evaluation()
    print("\n 全処理が完了しました！")