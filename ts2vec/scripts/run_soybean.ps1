# run_soybean.ps1
# 大豆収量予測 2段階パイプライン実行スクリプト
# 使用例: .\scripts\run_soybean.ps1
#
# ts2vec ディレクトリから実行してください:
#   cd c:\Users\amilu\Projects\vsCodeFile\Mstudy\ts2vec
#   .\scripts\run_soybean.ps1

$DATASET_DIR = "..\data\processed\soybean_ts2vec"
$GPU         = "cpu"     # GPU使用時は "0" に変更（例: RTX搭載PCなら "0"）
$REPR_DIMS   = 320
$BATCH_SIZE  = 16

# ──────────────────────────────────────────────────────────────
# Stage 1: 事前学習（自己教師あり、ラベル不要）
#   - pretrain_X.npy (23,218サンプル) でTS2Vecを学習
# ──────────────────────────────────────────────────────────────
Write-Host "=== Stage 1: Pre-training ===" -ForegroundColor Cyan

python train.py "$DATASET_DIR" soybean_pretrain `
    --loader      soybean_pretrain `
    --gpu         $GPU `
    --batch-size  $BATCH_SIZE `
    --repr-dims   $REPR_DIMS `
    --epochs      50

if ($LASTEXITCODE -ne 0) {
    Write-Host "Stage 1 failed." -ForegroundColor Red
    exit 1
}

# 最新の事前学習済みモデルを自動検出
$PRETRAINED_MODEL = (Get-ChildItem "training\soybean_pretrain__*\model.pkl" |
                     Sort-Object LastWriteTime -Descending |
                     Select-Object -First 1).FullName

Write-Host ""
Write-Host "Pretrained model: $PRETRAINED_MODEL" -ForegroundColor Green

# ──────────────────────────────────────────────────────────────
# Stage 2: ファインチューニング＆収量予測評価
#   - 事前学習済みモデルをロード
#   - X_train (2015-2016) でさらに自己教師あり追加学習
#   - Ridge回帰で収量予測 → RMSE / MAE / R² を報告
# ──────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "=== Stage 2: Fine-tuning & Evaluation ===" -ForegroundColor Cyan

python train.py "$DATASET_DIR" soybean_finetune `
    --loader           soybean_finetune `
    --gpu              $GPU `
    --batch-size       $BATCH_SIZE `
    --repr-dims        $REPR_DIMS `
    --epochs           30 `
    --pretrained-model "$PRETRAINED_MODEL" `
    --eval

if ($LASTEXITCODE -ne 0) {
    Write-Host "Stage 2 failed." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=== Pipeline Complete ===" -ForegroundColor Green
