"""
test_idea1.py (あなたのモデル対応・最終決定版)
学習済みのTwoStageTrajectoryPredictorモデルを、
Social-STGCNNベースのデータローダーを使ってテストデータで評価します。
"""
import torch
import torch.nn as nn
import logging
import os
import argparse
import numpy as np
from tqdm import tqdm

# -----------------------------------------------------------------------------
# 必要なモジュールをインポート
# -----------------------------------------------------------------------------
try:
    from model import TwoStageTrajectoryPredictor
    from utils import TrajectoryDataset
    from torch.utils.data import DataLoader, ConcatDataset
except ImportError as e:
    print(f"❌ 必要なモジュールが見つかりません: {e}")
    print("👉 model.py と utils.py が同じ階層にあるか確認してください。")
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
# データ変換と評価ステップ
# -----------------------------------------------------------------------------
def transform_batch_for_model(obs_traj, pred_traj_gt, num_peds_fixed):
    """
    TrajectoryDatasetからのバッチをTwoStageTrajectoryPredictorの入力形式に変換。
    """
    # 入力形状: (num_peds, 2, obs_len) -> (num_peds, obs_len, 2)
    obs_traj = obs_traj.permute(0, 2, 1)
    pred_traj_gt = pred_traj_gt.permute(0, 2, 1)

    num_peds, obs_len, features = obs_traj.shape
    pred_len = pred_traj_gt.shape[1]
    peds_to_use = min(num_peds, num_peds_fixed)

    # (1, obs_len, num_peds_fixed, 2) のテンソルを作成
    input_tensor = torch.zeros(1, obs_len, num_peds_fixed, features, device=obs_traj.device)
    input_tensor[0, :, :peds_to_use, :] = obs_traj[:peds_to_use, :, :].permute(1, 0, 2)
    
    # (1, pred_len, num_peds_fixed, 2) のテンソルを作成
    target_tensor = torch.zeros(1, pred_len, num_peds_fixed, features, device=pred_traj_gt.device)
    target_tensor[0, :, :peds_to_use, :] = pred_traj_gt[:peds_to_use, :, :].permute(1, 0, 2)
    
    return input_tensor, target_tensor


def safe_eval_step(model, input_traj, target_traj):
    """安全な検証ステップ"""
    model.eval()
    with torch.no_grad():
        try:
            final_pred, _, _ = model(input_traj)
            final_pred = final_pred.squeeze(0)
            target_gt = target_traj.squeeze(0)[:, 0, :]
            
            mask = (target_gt.abs().sum(dim=-1) > 0)
            if not mask.any(): return {'ade': 0.0, 'fde': 0.0}
            
            errors = torch.norm(final_pred - target_gt, dim=-1)
            errors[~mask] = 0
            
            epsilon = 1e-6
            ade = (errors.sum() / (mask.sum() + epsilon)).item()
            fde = (errors[-1].sum() / (mask[-1].sum() + epsilon)).item()
            
            return {'ade': ade, 'fde': fde}
        except Exception as e:
            logger.error(f"❌ 検証ステップでエラー: {e}", exc_info=True)
            return {'ade': float('inf'), 'fde': float('inf')}

# -----------------------------------------------------------------------------
# メイン処理
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./datasets', help='データセットが格納されているルートディレクトリ')
    parser.add_argument('--model_path', type=str, default='./model/MyTwoStagePredictor/trained_model.pth', help='評価するモデルのパス')
    parser.add_argument('--obs_len', type=int, default=8)
    parser.add_argument('--pred_len', type=int, default=12)
    parser.add_argument('--num_pedestrians', type=int, default=20)
    args = parser.parse_args()

    logger.info("🚀 学習済みモデルの最終評価を開始します")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用デバイス: {device}")

    if not os.path.exists(args.model_path):
        logger.error(f"❌ モデルファイルが見つかりません: {args.model_path}")
        return

    try:
        logger.info("🔄 テストデータセットを読み込んでいます...")
        all_dataset_folders = [d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))]
        test_dsets = []
        for dset_folder in all_dataset_folders:
            test_path = os.path.join(args.data_dir, dset_folder, 'test')
            if os.path.exists(test_path) and any(f.endswith('.txt') for f in os.listdir(test_path)):
                logger.info(f"  > テストデータを発見: {test_path}")
                dset = TrajectoryDataset(
                    data_dir=test_path, obs_len=args.obs_len, pred_len=args.pred_len, skip=1, delim='\t')
                if len(dset) > 0:
                    test_dsets.append(dset)

        if not test_dsets:
            raise FileNotFoundError(f"'{args.data_dir}' 内に有効なテストデータが見つかりませんでした。")
        
        full_test_dset = ConcatDataset(test_dsets)
        test_loader = DataLoader(full_test_dset, batch_size=1, shuffle=False, num_workers=0)
        logger.info(f"✅ データ読み込み完了. 全テストシーケンス数: {len(full_test_dset)}")
    
    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
        return

    model = TwoStageTrajectoryPredictor(
        input_dim=2, output_dim=2,
        seq_len=args.obs_len, pred_len=args.pred_len,
        num_pedestrians=args.num_pedestrians).to(device)
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    logger.info(f"✅ モデルを '{args.model_path}' からロードしました。")

    all_ades, all_fdes = [], []
    
    pbar = tqdm(test_loader, desc="評価中")
    for batch in pbar:
        obs_traj, pred_traj_gt = batch
        obs_traj, pred_traj_gt = obs_traj.to(device), pred_traj_gt.to(device)
        
        obs_traj = obs_traj.squeeze(0)
        pred_traj_gt = pred_traj_gt.squeeze(0)
        
        input_traj, target_traj = transform_batch_for_model(obs_traj, pred_traj_gt, args.num_pedestrians)
        
        metrics = safe_eval_step(model, input_traj, target_traj)
        all_ades.append(metrics['ade'])
        all_fdes.append(metrics['fde'])

    avg_ade = np.mean(all_ades)
    avg_fde = np.mean(all_fdes)

    logger.info("="*50)
    logger.info("🏆 最終評価結果 🏆")
    logger.info(f"  >> 平均ADE (Average Displacement Error): {avg_ade:.4f}")
    logger.info(f"  >> 平均FDE (Final Displacement Error):  {avg_fde:.4f}")
    logger.info("="*50)

if __name__ == '__main__':
    main()

