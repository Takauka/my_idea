"""
train_idea2.py (あなたのモデル対応・最終決定版)
Social-STGCNNの安定したデータローダーを使い、
あなたのTwoStageTrajectoryPredictorモデルを訓練します。
"""

import torch
import torch.optim as optim
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
# データ変換と訓練・評価ステップ
# -----------------------------------------------------------------------------
def transform_batch_for_model(obs_traj, pred_traj_gt, num_peds_fixed):
    """
    TrajectoryDatasetからのバッチをTwoStageTrajectoryPredictorの入力形式に変換。
    歩行者数を固定長にパディング/切り詰めします。
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


def safe_train_step(model, optimizer, input_traj, target_traj):
    """安全な訓練ステップ"""
    model.train()
    optimizer.zero_grad()
    
    try:
        final_pred, stage1_pred, _ = model(input_traj)
        
        # --- FIX: モデルの出力(batch=1)とGTの次元を合わせる ---
        final_pred = final_pred.squeeze(0)
        stage1_pred = stage1_pred.squeeze(0)
        target_gt = target_traj.squeeze(0)[:, 0, :]
        
        mask = (target_gt.abs().sum(dim=-1) > 0)
        if not mask.any(): return {'total_loss': 0.0, 'ade': 0.0, 'fde': 0.0}
        
        loss_final = nn.functional.mse_loss(final_pred[mask], target_gt[mask])
        
        short_pred_len = stage1_pred.shape[0]
        stage1_target = target_gt[:short_pred_len, :]
        stage1_mask = mask[:short_pred_len]

        if stage1_mask.any():
            stage1_loss = nn.functional.mse_loss(stage1_pred[stage1_mask], stage1_target[stage1_mask])
        else:
            stage1_loss = torch.tensor(0.0, device=input_traj.device)

        total_loss = loss_final + 0.3 * stage1_loss
        
        if torch.isnan(total_loss):
            logger.warning("損失がNaNになりました。スキップします。")
            return {'total_loss': 0.0, 'ade': 0.0, 'fde': 0.0}

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        with torch.no_grad():
            errors = torch.norm(final_pred - target_gt, dim=1)
            errors[~mask] = 0
            
            epsilon = 1e-6
            ade = (errors.sum() / (mask.sum() + epsilon)).item()
            fde = (errors[-1].sum() / (mask[-1].sum() + epsilon)).item()

        return {'total_loss': total_loss.item(), 'ade': ade, 'fde': fde}
    except Exception as e:
        logger.error(f"❌ 訓練ステップでエラー: {e}", exc_info=True)
        return {'total_loss': 0.0, 'ade': float('inf'), 'fde': float('inf')}

# -----------------------------------------------------------------------------
# メイン処理
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./datasets', help='データセットが格納されているルートディレクトリ')
    parser.add_argument('--obs_len', type=int, default=8)
    parser.add_argument('--pred_len', type=int, default=12)
    parser.add_argument('--num_epochs', type=int, default=200, help='エポック数')
    parser.add_argument('--lr', type=float, default=0.001, help='学習率')
    parser.add_argument('--num_pedestrians', type=int, default=20, help='シーン内で考慮する歩行者の最大数')
    args = parser.parse_args()

    logger.info("🚀 あなたの二段階モデルの訓練を開始します (Social-STGCNNデータローダー使用)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用デバイス: {device}")

    try:
        logger.info("🔄 データセットを読み込んでいます...")
        all_dataset_folders = [d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))]
        train_dsets = []
        for dset_folder in all_dataset_folders:
            train_path = os.path.join(args.data_dir, dset_folder, 'train')
            if os.path.exists(train_path) and any(f.endswith('.txt') for f in os.listdir(train_path)):
                logger.info(f"  > 訓練データを発見: {train_path}")
                dset = TrajectoryDataset(
                    data_dir=train_path, obs_len=args.obs_len, pred_len=args.pred_len, skip=1, delim='\t')
                if len(dset) > 0:
                    train_dsets.append(dset)

        if not train_dsets:
            raise FileNotFoundError(f"'{args.data_dir}' 内に有効な訓練データが見つかりませんでした。")

        full_train_dset = ConcatDataset(train_dsets)
        train_loader = DataLoader(full_train_dset, batch_size=1, shuffle=True, num_workers=0)
        logger.info(f"✅ データ読み込み完了. 全シーケンス数: {len(full_train_dset)}")

    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
        return

    model = TwoStageTrajectoryPredictor(
        input_dim=2, output_dim=2,
        seq_len=args.obs_len, pred_len=args.pred_len,
        num_pedestrians=args.num_pedestrians).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    logger.info("🎓 訓練開始")
    
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss, epoch_ade, epoch_fde = 0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch in pbar:
            obs_traj, pred_traj_gt = batch
            obs_traj, pred_traj_gt = obs_traj.to(device), pred_traj_gt.to(device)
            
            obs_traj = obs_traj.squeeze(0)
            pred_traj_gt = pred_traj_gt.squeeze(0)
            
            input_traj, target_traj = transform_batch_for_model(obs_traj, pred_traj_gt, args.num_pedestrians)
            
            # --- FIX: metrics辞書を正しく受け取る ---
            metrics = safe_train_step(model, optimizer, input_traj, target_traj)
            
            epoch_loss += metrics['total_loss']
            epoch_ade += metrics['ade']
            epoch_fde += metrics['fde']
            pbar.set_postfix(loss=f"{metrics['total_loss']:.4f}", ade=f"{metrics['ade']:.4f}", fde=f"{metrics['fde']:.4f}")
        
        scheduler.step()
        
        num_batches = len(train_loader)
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        avg_ade = epoch_ade / num_batches if num_batches > 0 else 0
        avg_fde = epoch_fde / num_batches if num_batches > 0 else 0
        logger.info(f"Epoch {epoch+1}完了, 平均Loss: {avg_loss:.4f}, ADE: {avg_ade:.4f}, FDE: {avg_fde:.4f}")

    save_directory = './model/MyTwoStagePredictor'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    final_model_path = os.path.join(save_directory, 'trained_model.pth')
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"🎉 訓練完了。最終モデルを '{final_model_path}' に保存しました。")


if __name__ == '__main__':
    main()


if __name__ == '__main__':
    main()
    
