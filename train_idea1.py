"""
train_idea1.py (my_ideaモデル対応・修正版)
新しい二段階予測モデル(TwoStageTrajectoryPredictor)を訓練します。
モデルの出力形式に合わせてデータ処理と損失計算を最適化。
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
# データ処理と訓練・評価ステップ
# -----------------------------------------------------------------------------
def process_batch(x_batch, y_batch, target_ids_batch, obs_len, pred_len, num_peds_fixed):
    """
    DataLoaderからのバッチをモデル入力形式に変換。
    ターゲット歩行者が必ずインデックス0に来るように並べ替える。
    """
    batch_size = len(x_batch)
    
    # モデル入力用のテンソルを初期化
    # (batch, seq, peds, feats)
    input_tensor = torch.zeros(batch_size, obs_len, num_peds_fixed, 2)
    target_tensor = torch.zeros(batch_size, pred_len, num_peds_fixed, 2)

    for i in range(batch_size):
        obs_traj_seq = x_batch[i]
        pred_traj_seq = y_batch[i]
        target_id = target_ids_batch[i]

        # シーケンス内の全歩行者IDを収集
        all_peds_in_seq = set()
        if obs_traj_seq.ndim >= 3 and obs_traj_seq.shape[2] > 1:
            all_peds_in_seq.update(np.unique(obs_traj_seq[:, :, 1]))
        
        # ターゲットを先頭にした歩行者リストを作成
        other_peds = sorted(list(all_peds_in_seq - {target_id}))
        ordered_peds = [target_id] + other_peds
        
        # 固定長に切り詰め
        ordered_peds = ordered_peds[:num_peds_fixed]
        
        ped_to_idx = {ped_id: idx for idx, ped_id in enumerate(ordered_peds)}
        
        # テンソルにデータを格納
        for t in range(obs_len):
            for frame_data in obs_traj_seq[t]:
                if len(frame_data) >= 4:
                    x, y, ped_id = frame_data[1:4] # frame, x, y, ped_id
                    if ped_id in ped_to_idx:
                        idx = ped_to_idx[ped_id]
                        input_tensor[i, t, idx, :] = torch.tensor([x, y])
                        
        for t in range(pred_len):
            for frame_data in pred_traj_seq[t]:
                if len(frame_data) >= 4:
                    x, y, ped_id = frame_data[1:4]
                    if ped_id in ped_to_idx:
                        idx = ped_to_idx[ped_id]
                        target_tensor[i, t, idx, :] = torch.tensor([x, y])

    return input_tensor, target_tensor


def safe_train_step(model, optimizer, input_traj, target_traj):
    """安全な訓練ステップ"""
    model.train()
    optimizer.zero_grad()
    
    try:
        # モデルはターゲット歩行者の予測のみを返す
        final_pred_target, stage1_pred_target, _ = model(input_traj)
        
        # 正解データもターゲット歩行者のもの（インデックス0）を抽出
        target_traj_target = target_traj[:, :, 0, :]
        
        # 損失計算 (ターゲットが存在する箇所のみ)
        # ターゲット歩行者がそのタイムステップに存在するかどうかのマスク
        mask = (target_traj_target.abs().sum(dim=-1) > 0)

        if not mask.any(): return {'total_loss': 0.0, 'ade': 0.0, 'fde': 0.0}
        
        # メイン損失
        main_loss = F.mse_loss(final_pred_target[mask], target_traj_target[mask])
        
        # Stage1の損失
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
            ade = errors.sum() / mask.sum()
            fde = errors[:, -1].sum() / mask[:, -1].sum()

        return {'total_loss': total_loss.item(), 'ade': ade.item(), 'fde': fde.item()}
    except Exception as e:
        logger.error(f"❌ 訓練ステップでエラー: {e}", exc_info=True)
        return {'total_loss': 0.0, 'ade': float('inf'), 'fde': float('inf')}


def safe_eval_step(model, input_traj, target_traj):
    """安全な検証ステップ"""
    model.eval()
    with torch.no_grad():
        try:
            final_pred_target, _, _ = model(input_traj)
            target_traj_target = target_traj[:, :, 0, :]
            mask = (target_traj_target.abs().sum(dim=-1) > 0)
            
            if not mask.any(): return {'ade': 0.0, 'fde': 0.0}
            
            errors = torch.norm(final_pred_target - target_traj_target, dim=-1)
            errors[~mask] = 0
            
            ade = errors.sum() / mask.sum()
            fde = errors[:, -1].sum() / mask[:, -1].sum()
            
            return {'ade': ade.item(), 'fde': fde.item()}
        except Exception as e:
            logger.error(f"❌ 検証ステップでエラー: {e}", exc_info=True)
            return {'ade': float('inf'), 'fde': float('inf')}


# -----------------------------------------------------------------------------
# メイン処理
# -----------------------------------------------------------------------------
def main():
    """メイン関数"""
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--num_epochs', type=int, default=100, help='エポック数')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学習率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='重み減衰')
    # その他
    parser.add_argument('--use_cuda', action="store_true", default=True, help='GPUを使用するか')
    args = parser.parse_args()
    
    logger.info("🚀 新しい二段階モデルの訓練を開始します")
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    logger.info(f"使用デバイス: {device}")
    
    # --- 1. データローダーの準備 ---
    save_directory = './model/TwoStagePredictor'
    if not os.path.exists(save_directory): os.makedirs(save_directory)

    seq_len = args.obs_len + args.pred_len
    dataloader = DataLoader('.', args.batch_size, seq_len, num_of_validation=1, forcePreProcess=True)
    val_dataloader = DataLoader('.', args.batch_size, seq_len, num_of_validation=1, forcePreProcess=True) # 検証用
    
    # --- 2. モデル、最適化手法、スケジューラの定義 ---
    model = TwoStageTrajectoryPredictor(
        # モデル定義は3次元(x, y, ?)だが、データは2次元(x, y)なので合わせる
        input_dim=2, output_dim=2, 
        hidden_dim=args.hidden_dim, seq_len=args.obs_len,
        pred_len=args.pred_len, num_layers=args.num_layers, dropout=args.dropout,
        num_pedestrians=args.num_pedestrians
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    # --- 3. 訓練ループ ---
    logger.info("🎓 訓練開始")
    best_val_ade = float('inf')
    
    for epoch in range(args.num_epochs):
        logger.info(f"--- Epoch {epoch+1}/{args.num_epochs} ---")
        
        # --- 訓練 ---
        model.train()
        epoch_losses, epoch_ades, epoch_fdes = [], [], []
        dataloader.reset_batch_pointer()
        for _ in range(dataloader.num_batches):
            x, y, _, _, _, target_ids = dataloader.next_batch()
            input_traj, target_traj = process_batch(x, y, target_ids, args.obs_len, args.pred_len, args.num_pedestrians)
            input_traj, target_traj = input_traj.to(device), target_traj.to(device)
            
            metrics = safe_train_step(model, optimizer, input_traj, target_traj)
            epoch_losses.append(metrics['total_loss'])
            epoch_ades.append(metrics['ade'])
            epoch_fdes.append(metrics['fde'])
        
        logger.info(f" [訓練] Loss: {np.mean(epoch_losses):.4f}, ADE: {np.mean(epoch_ades):.4f}, FDE: {np.mean(epoch_fdes):.4f}")

        # --- 検証 ---
        model.eval()
        val_ades, val_fdes = [], []
        val_dataloader.reset_batch_pointer(valid=True)
        for _ in range(val_dataloader.valid_num_batches):
            x, y, _, _, _, target_ids = val_dataloader.next_valid_batch()
            input_traj, target_traj = process_batch(x, y, target_ids, args.obs_len, args.pred_len, args.num_pedestrians)
            input_traj, target_traj = input_traj.to(device), target_traj.to(device)
            
            metrics = safe_eval_step(model, input_traj, target_traj)
            val_ades.append(metrics['ade'])
            val_fdes.append(metrics['fde'])

        avg_val_ade = np.mean(val_ades) if val_ades else float('inf')
        avg_val_fde = np.mean(val_fdes) if val_fdes else float('inf')
        logger.info(f" [検証] ADE: {avg_val_ade:.4f}, FDE: {avg_val_fde:.4f}")
        
        scheduler.step(avg_val_ade)
        
        if avg_val_ade < best_val_ade:
            best_val_ade = avg_val_ade
            torch.save(model.state_dict(), os.path.join(save_directory, 'best_model_social.pth'))
            logger.info(f"🎉 新しいベストモデルを保存しました！ (ADE: {best_val_ade:.4f})")

    logger.info("🎉 訓練完了")

if __name__ == "__main__":
    main()
