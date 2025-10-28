# -*- coding: utf-8 -*-
"""先行研究のモデルから土壌データ用CNNを除いたバージョン"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, callbacks
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
# from tensorflow.keras.layers import Bidirectional # Bidirectionalは使われていなかったので削除


# 1. データローダー関連
def load_and_preprocess_data(path='./Data/Input/soybean_data.csv'):
    """CSVファイルを読み込み、基本的な前処理（標準化など）を行う。"""
    if not os.path.exists(path):
        print(f"エラー: データファイル '{path}' が見つかりません。プログラムを終了します。")
        return None
    df = pd.read_csv(path)
    
    # 最初の3列('loc_ID', 'year', 'yield')以外の列を特徴量とする
    feature_cols = df.columns[3:]
    train_df = df[df['year'] <= 2017]
    mean = train_df[feature_cols].mean()
    std = train_df[feature_cols].std()
    std[std == 0] = 1.0 # 標準偏差が0の場合は1で割る（変化なし）
    df[feature_cols] = (df[feature_cols] - mean) / std
    df = df.fillna(0) # 標準化後の欠損値を0で埋める
    df = df[df['yield'] >= 5].reset_index(drop=True) # 収量が5未満のデータを除外
    return df

def create_year_loc_dict_and_avg(df):
    """年と地域(loc_ID)をキーにしたデータ辞書と、年ごとの平均収量辞書を作成する。"""
    # 各行を (loc_ID, year) をキーとする辞書に格納
    loc_year_dict = { (row.loc_ID, int(row.year)): row for index, row in df.iterrows() }
    
    # 年ごとの平均収量を計算し、標準化
    avg_yield_by_year = df.groupby('year')['yield'].mean()
    mean_yield = avg_yield_by_year.mean()
    std_yield = avg_yield_by_year.std()
    avg_dict = (avg_yield_by_year - mean_yield) / std_yield
    
    # 2018年の平均収量データがない場合、2017年の値で代用（検証用）
    if 2018 not in avg_dict.index and 2017 in avg_dict.index:
        avg_dict[2018] = avg_dict.get(2017, 0) # 2017年もなければ0
        
    # キーを文字列に変換して返す（TensorFlowでの扱いやすさのため）
    return loc_year_dict, {str(k): v for k, v in avg_dict.to_dict().items()}

class SoybeanDataGenerator(tf.keras.utils.Sequence):
    """Kerasモデルのためのカスタムデータジェネレータ（土壌データCNNなし版）"""
    def __init__(self, df, loc_year_dict, avg_dict, batch_size, is_training=True):
        self.loc_year_dict = loc_year_dict
        self.avg_dict = avg_dict
        self.batch_size = batch_size
        
        self.sequences = []
        loc_ids = df['loc_ID'].unique()
        all_years = sorted(df['year'].unique())

        # 連続する5年分のデータが存在する地域-年の組み合わせを探す
        for loc_id in loc_ids:
            for i in range(len(all_years) - 4):
                seq_years = all_years[i:i+5]
                # 5年分のデータが辞書に全て存在するか確認
                if all((loc_id, year) in self.loc_year_dict for year in seq_years):
                    self.sequences.append({'loc_id': loc_id, 'years': seq_years})
        
        # 訓練用か検証用かでシーケンスをフィルタリング
        if is_training:
            # 訓練用：2018年を含まないシーケンスのみ
            self.sequences = [s for s in self.sequences if 2018 not in s['years']]
        else:
            # 検証用：最後の年が2018年であるシーケンスのみ
            self.sequences = [s for s in self.sequences if s['years'][-1] == 2018] # 2018年が最後に来るものだけ
        
        print(f"{'訓練' if is_training else '検証'}ジェネレータが、{len(self.sequences)}個の有効な「地域-5年」シーケンスを生成しました。")
        self.indices = np.arange(len(self.sequences))
        self.on_epoch_end() # 最初にインデックスをシャッフル

    def __len__(self):
        # 1エポックあたりのバッチ数を返す
        if len(self.sequences) == 0: return 0
        return int(np.floor(len(self.sequences) / self.batch_size))

    def __getitem__(self, index):
        # 指定されたインデックスのバッチデータを生成
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        batch_seq_info = [self.sequences[i] for i in batch_indices]
        actual_batch_size = len(batch_seq_info) # 最後のバッチはサイズが小さい可能性がある

        # 入力データを辞書形式で準備 (s_inputを削除)
        X_dict = {
            'e_input': np.zeros((actual_batch_size, 5, 312)),
            # 's_input': np.zeros((actual_batch_size, 5, 66)), # <--- 削除
            'p_input': np.zeros((actual_batch_size, 5, 14)),
            'ybar_input': np.zeros((actual_batch_size, 5, 1))
        }
        # 出力データを辞書形式で準備
        Y_dict = {
            'Yhat1': np.zeros((actual_batch_size, 1)),
            'Yhat2': np.zeros((actual_batch_size, 4, 1)) # 過去4年分の収量
        }

        for i, seq_info in enumerate(batch_seq_info):
            loc_id = seq_info['loc_id']
            years = seq_info['years']
            
            for j, year in enumerate(years):
                sample = self.loc_year_dict[(loc_id, year)]
                # loc_ID, year, yield を除いた特徴量部分を取得
                features = sample.iloc[3:].values 
                
                # 特徴量を各入力に割り当て (s_inputへの割り当てを削除)
                X_dict['e_input'][i, j, :] = features[0:312]    # 環境データ (0-311)
                # s_input部分 (312-377) をスキップ
                X_dict['p_input'][i, j, :] = features[378:392]  # 品種特性データ (378-391)
                X_dict['ybar_input'][i, j, 0] = self.avg_dict[str(year)] # 年ごとの標準化平均収量

            # 目的変数（収量）を設定
            # Yhat1: 最後の年の収量
            Y_dict['Yhat1'][i] = self.loc_year_dict[(loc_id, years[-1])]['yield']
            # Yhat2: 過去4年分の収量
            past_yields = [self.loc_year_dict[(loc_id, y)]['yield'] for y in years[:-1]]
            Y_dict['Yhat2'][i] = np.array(past_yields).reshape(4, 1)

        return X_dict, Y_dict

    def on_epoch_end(self):
        # 各エポック終了時にインデックスをシャッフル
        np.random.shuffle(self.indices)

# 2. モデル定義
def build_and_compile_model():
    """
    土壌データ用CNNを除いたモデルを構築する。
    """
    # --- 入力層の定義 (s_inputを削除) ---
    e_input = layers.Input(shape=(5, 312), name="e_input")
    # s_input = layers.Input(shape=(5, 66), name="s_input") # <--- 削除
    p_input = layers.Input(shape=(5, 14), name="p_input")
    ybar_input = layers.Input(shape=(5, 1), name="ybar_input")

    # --- 特徴量処理ブロックの定義 (サブモデルとして) ---
    # 環境(E)データ用 共有CNNモデル (変更なし)
    e_cnn_input = layers.Input(shape=(52, 1), name="e_cnn_input")
    x_e = layers.Conv1D(8, 9, activation='relu', padding='valid')(e_cnn_input)
    x_e = layers.AveragePooling1D(2)(x_e)
    x_e = layers.Conv1D(12, 3, activation='relu', padding='valid')(x_e)
    x_e = layers.AveragePooling1D(2)(x_e)
    e_cnn_output = layers.Flatten()(x_e)
    shared_e_cnn = models.Model(inputs=e_cnn_input, outputs=e_cnn_output, name="Shared_E_CNN")

    # 環境(E)データ ラッパーモデル (変更なし)
    e_proc_input = layers.Input(shape=(312,), name="e_proc_input")
    e_reshaped = layers.Reshape((6, 52, 1))(e_proc_input)
    e_sub_outputs = [shared_e_cnn(e_reshaped[:, i]) for i in range(6)]
    e_proc_output = layers.Concatenate()(e_sub_outputs)
    e_processor = models.Model(inputs=e_proc_input, outputs=e_proc_output, name="E_Processor")

    # 土壌(S)データ用CNNモデル (削除)
    # s_proc_input = layers.Input(shape=(66,), name="s_proc_input")
    # s_reshaped = layers.Reshape((6, 11))(s_proc_input)
    # s_cnn_out = layers.Flatten()(layers.Conv1D(16, 3, activation='relu')(s_reshaped))
    # s_processor = models.Model(inputs=s_proc_input, outputs=s_cnn_out, name="S_Processor") # <--- 削除

    # --- TimeDistributedで各タイムステップに特徴量処理を適用 ---
    e_processed = layers.TimeDistributed(e_processor, name="TDD_E_Processor")(e_input)
    # s_processed = layers.TimeDistributed(s_processor, name="TDD_S_Processor")(s_input) # <--- 削除
    # 品種特性(P)データは単純にFlattenする
    p_processed = layers.TimeDistributed(layers.Flatten(), name="TDD_P_Flatten")(p_input)

    # --- 全ての特徴量を結合し、LSTMに入力 (s_processedを削除) ---
    merged = layers.Concatenate()([e_processed, p_processed, ybar_input]) # <--- 修正
    
    # --- 後段のLSTMと出力層 (変更なし) ---
    x = layers.Dense(128, activation='relu')(merged)
    x = layers.LSTM(64, return_sequences=True, dropout=0.2)(x) # 過去の情報を考慮
    # TimeDistributed Denseで各タイムステップの収量を予測
    output = layers.TimeDistributed(layers.Dense(1))(x) 
    
    # 最後のタイムステップの予測 (Yhat1) と過去の予測 (Yhat2) を分離
    Yhat1 = layers.Identity(name='Yhat1')(output[:, -1, :])
    Yhat2 = layers.Identity(name='Yhat2')(output[:, :-1, :])

    # モデルの定義 (入力からs_inputを削除)
    model = models.Model(inputs=[e_input, p_input, ybar_input], outputs=[Yhat1, Yhat2]) # <--- 修正
    
    # モデルのコンパイル (変更なし)
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0003),
                  loss={'Yhat1': losses.Huber(), 'Yhat2': losses.Huber()},
                  loss_weights={'Yhat1': 1.0, 'Yhat2': 0.0}, # Yhat1(最終年予測)のみを損失計算に使用
                  metrics={'Yhat1': 'mae'}) # Yhat1のMAEを評価指標とする
    return model

# 3. 訓練と評価
def run_training_and_evaluation():
    print("\n データ読み込みと前処理...")
    df = load_and_preprocess_data()
    if df is None: return

    loc_year_dict, avg_dict = create_year_loc_dict_and_avg(df)
    
    print("\n データジェネレータの作成...")
    train_generator = SoybeanDataGenerator(df, loc_year_dict, avg_dict, batch_size=32, is_training=True)
    val_generator = SoybeanDataGenerator(df, loc_year_dict, avg_dict, batch_size=26, is_training=False) # 検証バッチサイズは検証データ数に合わせると良い場合がある

    print("\n モデルの構築とコンパイル...")
    model = build_and_compile_model()
    model.summary(line_length=120) # モデル構造を表示

    print("\n モデルの訓練を開始します...")
    # 早期終了設定：検証損失が20エポック改善しなければ停止し、最良の重みを復元
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    if len(train_generator) == 0:
        print("\nエラー: 訓練データがありません。データ生成プロセスを確認してください。")
        return

    if len(val_generator) > 0:
        # 訓練実行
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=200, # 最大エポック数
            callbacks=[early_stop],
            verbose=2 # 各エポックの結果を表示
        )
    else:
        print("\n警告: 検証データが見つからなかったため、検証なしで訓練します。")
        # 検証データがない場合の早期終了は訓練損失を監視
        early_stop_train = callbacks.EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
        history = model.fit(
            train_generator, 
            epochs=200, 
            callbacks=[early_stop_train],
            verbose=2
        )
        
    # モデルの保存
    model.save("soybean_yield_model_no_soil.keras")
    print("\n モデル訓練完了・保存済み (soybean_yield_model_no_soil.keras)")

    print("\n モデルの評価を開始します...")
    if len(val_generator) > 0:
        # 評価時にはジェネレータのシャッフルを無効化
        val_generator.on_epoch_end = lambda: None 
        
        # 保存した最良モデルを読み込み (EarlyStoppingで復元されているが念のため)
        loaded_model = models.load_model("soybean_yield_model_no_soil.keras")
        print(" モデル読み込み完了")

        # 全ての検証データで一度に予測
        # predict()は入力データ(X)のみを受け取り、予測結果(Yhat1, Yhat2)を返す
        predictions = loaded_model.predict(val_generator)
        # predictions はリスト形式 [Yhat1の予測値, Yhat2の予測値]
        Y1_pred = predictions[0] # 最初の出力がYhat1の予測
        
        # 全ての正解データをジェネレータから取得
        Y1_test_true = np.concatenate([val_generator[i][1]['Yhat1'] for i in range(len(val_generator))])
        
        # 評価指標の計算
        rmse = np.sqrt(mean_squared_error(Y1_test_true, Y1_pred))
        print(f"\n 検証 RMSE (最終年): {rmse:.4f}")

        # データが2つ以上あれば相関係数を計算
        if len(Y1_test_true) >= 2:
            corr, _ = pearsonr(Y1_test_true.flatten(), Y1_pred.flatten())
            print(f" 検証 相関係数 (最終年): {corr:.4f}")

        # 予測結果と正解値の保存
        np.savez("prediction_result_no_soil.npz", Y1_true=Y1_test_true, Y1_pred=Y1_pred)
        print(" 予測結果を 'prediction_result_no_soil.npz' に保存しました")
    else:
        print("評価データがありません。評価をスキップします。")

# 4. メイン実行ブロック
if __name__ == "__main__":
    print(" 大豆収量予測モデル (土壌CNNなし) - 総合実行スクリプト")
    run_training_and_evaluation()
    print("\n 全処理が完了しました！")
