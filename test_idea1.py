"""
test_idea1.py (my_ideaモデル対応版)
学習済みのTwoStageTrajectoryPredictorモデルをロードし、
テストデータセットで最終的な性能（ADE/FDE）を評価します。
"""

import torch
import numpy as np
import logging
import os
import argparse

# -----------------------------------------------------------------------------
# 必要なモジュールをインポート
# -----------------------------------------------------------------------------
try:
    from model import TwoStageTrajectoryPredictor
    from utils import DataLoader
    # train_idea1からデータ処理関数をインポート
    from train_idea1 import process_batch, safe_eval_step
except ImportError as e:
    print(f"❌ 必要なモジュールが見つかりません: {e}")
    print("👉 model.py, utils.py, train_idea1.py が同じ階層にあるか確認してください。")
    exit()

# -----------------------------------------------------------------------------
# ログ設定
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# メイン処理
# -----------------------------------------------------------------------------
def main():
    """メイン関数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./model/TwoStagePredictor/best_model_social.pth',
                        help='評価する学習済みモデルのパス')
    # 訓練時と同じモデルパラメータを指定
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_pedestrians', type=int, default=15)
    parser.add_argument('--obs_len', type=int, default=8)
    parser.add_argument('--pred_len', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=64) # 評価時はバッチサイズを大きくできる
    parser.add_argument('--use_cuda', action="store_true", default=True)
    args = parser.parse_args()
    
    logger.info("🚀 学習済みモデルの評価を開始します")
    
    if not os.path.exists(args.model_path):
        logger.error(f"❌ モデルファイルが見つかりません: {args.model_path}")
        return

    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    logger.info(f"使用デバイス: {device}")
    
    # --- 1. データローダーの準備 ---
    seq_len = args.obs_len + args.pred_len
    dataloader = DataLoader('.', args.batch_size, seq_len, num_of_validation=0, forcePreProcess=True, is_train=False) # テストデータ用
    
    # --- 2. モデルの初期化と重みロード ---
    model = TwoStageTrajectoryPredictor(
        input_dim=2, output_dim=2, hidden_dim=args.hidden_dim, seq_len=args.obs_len,
        pred_len=args.pred_len, num_layers=args.num_layers, dropout=0.0, # 評価時はドロップアウトなし
        num_pedestrians=args.num_pedestrians
    ).to(device)
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    logger.info(f"✅ モデルの重みを '{args.model_path}' からロードしました。")
    
    # --- 3. 評価ループ ---
    logger.info("📊 評価を開始...")
    model.eval()
    final_ades, final_fdes = [], []
    
    dataloader.reset_batch_pointer()
    for _ in range(dataloader.num_batches):
        x, y, _, _, _, _ = dataloader.next_batch()
        if not x: continue
        
        input_traj, target_traj = process_batch(x, y, args.obs_len, args.pred_len, args.num_pedestrians)
        input_traj, target_traj = input_traj.to(device), target_traj.to(device)
        
        metrics = safe_eval_step(model, input_traj, target_traj)
        final_ades.append(metrics['ade'])
        final_fdes.append(metrics['fde'])

    # --- 4. 最終結果の表示 ---
    final_ade = np.mean([ade for ade in final_ades if ade is not None and not np.isinf(ade)])
    final_fde = np.mean([fde for fde in final_fdes if fde is not None and not np.isinf(fde)])
    
    logger.info("\n" + "="*60)
    logger.info("🏆 最終評価結果 🏆")
    logger.info(f"  モデルファイル: {os.path.basename(args.model_path)}")
    logger.info(f"  >> 最終ADE: {final_ade:.4f}")
    logger.info(f"  >> 最終FDE: {final_fde:.4f}")
    logger.info("="*60)


if __name__ == "__main__":
    main()

