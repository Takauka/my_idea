"""
train_idea2.py (my_ideaモデル対応・訓練特化版)
新しい二段階予測モデル(TwoStageTrajectoryPredictor)を訓練します。
このスクリプトは検証を行わず、訓練のみに専念します。
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
import os
import time
import pickle
import argparse
import math

# -----------------------------------------------------------------------------
# 必要なモジュールをインポート
# -----------------------------------------------------------------------------
try:
    # ユーザーが提供した新しいモデルをインポート
    from model import TwoStageTrajectoryPredictor
    from utils import DataLoader
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
# データ処理と訓練・評価ステップ (変更なし)
# -----------------------------------------------------------------------------
def process_batch(x_batch, y_batch, target_ids_batch, obs_len, pred_len, num_peds_fixed):
    """
    DataLoaderからのバッチをモデル入力形式に変換。
    ターゲット歩行者が必ずインデックス0に来るように並べ替える。
    """
    batch_size = len(x_batch)
    
    input_tensor = torch.zeros(batch_size, obs_len, num_peds_fixed, 2)
    target_tensor = torch.zeros(batch_size, pred_len, num_peds_fixed, 2)

    for i in range(batch_size):
        obs_traj_seq = x_batch[i]
        pred_traj_seq = y_batch[i]
        target_id = target_ids_batch[i]

        all_peds_in_seq = set()
        for frame in obs_traj_seq:
            if frame.size > 0:
                all_peds_in_seq.update(np.unique(frame[:, 0]))
        
        other_peds = sorted(list(all_peds_in_seq - {target_id}))
        ordered_peds = ([target_id] + other_peds)[:num_peds_fixed]
        
        ped_to_idx = {ped_id: idx for idx, ped_id in enumerate(ordered_peds)}
        
        for t in range(obs_len):
            frame_data_array = obs_traj_seq[t]
            for row in frame_data_array:
                ped_id, x, y = row[0], row[1], row[2]
                if ped_id in ped_to_idx:
                    input_tensor[i, t, ped_to_idx[ped_id], :] = torch.tensor([x, y])
                        
        for t in range(pred_len):
            frame_data_array = pred_traj_seq[t]
            for row in frame_data_array:
                ped_id, x, y = row[0], row[1], row[2]
                if ped_id in ped_to_idx:
                    target_tensor[i, t, ped_to_idx[ped_id], :] = torch.tensor([x, y])

    return input_tensor, target_tensor


def safe_train_step(model, optimizer, input_traj, target_traj):
    """安全な訓練ステップ"""
    model.train()
    optimizer.zero_grad()
    
    try:
        final_pred_target, stage1_pred_target, _ = model(input_traj)
        target_traj_target = target_traj[:, :, 0, :]
        
        mask = (target_traj_target.abs().sum(dim=-1) > 0)
        if not mask.any(): return {'total_loss': 0.0, 'ade': 0.0, 'fde': 0.0}
        
        main_loss = F.mse_loss(final_pred_target[mask], target_traj_target[mask])
        
        short_pred_len = stage1_pred_target.shape[1]
        stage1_target = target_traj_target[:, :short_pred_len, :]
        stage1_mask = mask[:, :short_pred_len]
        stage1_loss = F.mse_loss(stage1_pred_target[stage1_mask], stage1_target[stage1_mask])
        
        total_loss = main_loss + 0.3 * stage1_loss
        
        if torch.isnan(total_loss):
            logger.warning("訓練中に損失がNaNになりました。スキップします。")
            return {'total_loss': 0.0, 'ade': 0.0, 'fde': 0.0}

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        with torch.no_grad():
            errors = torch.norm(final_pred_target - target_traj_target, dim=-1)
            errors[~mask] = 0
            ade = (errors.sum() / mask.sum()).item()
            fde = (errors[:, -1].sum() / mask[:, -1].sum()).item()

        return {'total_loss': total_loss.item(), 'ade': ade, 'fde': fde}
    except Exception as e:
        logger.error(f"❌ 訓練ステップでエラー: {e}", exc_info=True)
        return {'total_loss': 0.0, 'ade': float('inf'), 'fde': float('inf')}

# -----------------------------------------------------------------------------
# メイン処理
# -----------------------------------------------------------------------------
def main():
    """メイン関数"""
    parser = argparse.ArgumentParser()
    # --- MODIFIED: 検証関連の引数を削除 ---
    parser.add_argument('--data_dir', type=str, default='./data/train', help='訓練データセットが格納されているディレクトリ')
    
    # モデルのハイパーパラメータ
    parser.add_argument('--hidden_dim', type=int, default=64, help='RNN/GNNの隠れ層の次元')
    parser.add_argument('--num_layers', type=int, default=2, help='LSTMの層数')
    parser.add_argument('--dropout', type=float, default=0.1, help='ドロップアウト率')
    parser.add_argument('--num_pedestrians', type=int, default=15, help='シーン内で考慮する歩行者の最大数')
    # データ関連
    parser.add_argument('--obs_len', type=int, default=8, help='観測長')
    parser.add_argument('--pred_len', type=int, default=12, help='予測長')
    # 訓練関連
    parser.add_argument('--batch_size', type=int, default=16, help='ミニバッチサイズ')
    parser.add_argument('--num_epochs', type=int, default=500, help='エポック数')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学習率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='重み減衰')
    # その他
    parser.add_argument('--use_cuda', action="store_true", default=True, help='GPUを使用するか')
    args = parser.parse_args()
    
    logger.info("🚀 新しい二段階モデルの訓練を開始します (訓練特化モード)")
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    logger.info(f"使用デバイス: {device}")
    
    # --- 1. データローダーの準備 ---
    try:
        seq_len = args.obs_len + args.pred_len
        # --- FIX: DataLoaderの呼び出しを修正。validation_size=0で全データを訓練に使用 ---
        dataloader = DataLoader(args.data_dir, args.batch_size, seq_len, 
                                validation_size=0,
                                forcePreProcess=True)
    except FileNotFoundError as e:
        logger.error(e)
        logger.info("訓練を中止します。")
        return

    save_directory = './model/TwoStagePredictor'
    if not os.path.exists(save_directory): os.makedirs(save_directory)

    # --- 2. モデル、最適化手法、スケジューラの定義 ---
    model = TwoStageTrajectoryPredictor(
        input_dim=2, output_dim=2, 
        hidden_dim=args.hidden_dim, seq_len=args.obs_len,
        pred_len=args.pred_len, num_layers=args.num_layers, dropout=args.dropout,
        num_pedestrians=args.num_pedestrians
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # --- 3. 訓練ループ ---
    logger.info("🎓 訓練開始")
    
    for epoch in range(args.num_epochs):
        logger.info(f"--- Epoch {epoch+1}/{args.num_epochs} ---")
        
        # --- 訓練 ---
        model.train()
        epoch_losses, epoch_ades, epoch_fdes = [], [], []
        dataloader.reset_batch_pointer()
        for _ in range(dataloader.num_batches):
            x, y, _, _, _, target_ids = dataloader.next_batch()
            if not x: continue
            input_traj, target_traj = process_batch(x, y, target_ids, args.obs_len, args.pred_len, args.num_pedestrians)
            input_traj, target_traj = input_traj.to(device), target_traj.to(device)
            
            metrics = safe_train_step(model, optimizer, input_traj, target_traj)
            epoch_losses.append(metrics['total_loss'])
            epoch_ades.append(metrics['ade'])
            epoch_fdes.append(metrics['fde'])
        
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0
        avg_ade = np.mean(epoch_ades) if epoch_ades else 0
        avg_fde = np.mean(epoch_fdes) if epoch_fdes else 0

        logger.info(f" [訓練] Loss: {avg_loss:.4f}, ADE: {avg_ade:.4f}, FDE: {avg_fde:.4f}")

        # --- MODIFIED: 検証ループを削除し、訓練ロスでスケジューラを更新 ---
        scheduler.step(avg_loss)

    # --- MODIFIED: 最終エポックのモデルを保存 ---
    final_model_path = os.path.join(save_directory, 'trained_model.pth')
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"🎉 訓練完了。最終モデルを '{final_model_path}' に保存しました。")

if __name__ == "__main__":
    main()

