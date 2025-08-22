"""
train_idea1.py (ベストモデル保存機能付き)
各エポックで検証を行い、最も性能が良いモデルを保存し、最終的にその性能を報告する。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import os
import time
from typing import Dict, Any, Tuple, Optional

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_dummy_data(batch_size=32, seq_len=8, pred_len=12, num_pedestrians=5, feature_dim=3):
    """3次元のダミーデータを作成"""
    input_trajectories = torch.randn(batch_size, seq_len, num_pedestrians, feature_dim)
    target_trajectories = torch.randn(batch_size, pred_len, num_pedestrians, feature_dim)
    obstacle_maps = torch.randn(batch_size, 2)
    return input_trajectories, target_trajectories, obstacle_maps

def check_dataloader_availability(pred_len):
    """DataLoaderが使用可能かチェック"""
    try:
        if not os.path.exists('utils.py'): return False
        from utils import DataLoader
        data_dirs = ['data/train', 'data/test', 'data/validation']
        if not any(os.path.exists(d) and os.listdir(d) for d in data_dirs): return False
        loader = DataLoader(f_prefix='.', batch_size=2, seq_length=8, pred_len=pred_len)
        return True
    except Exception:
        return False

def process_dataloader_batch(dataloader, pred_len):
    """DataLoaderからバッチを安全に取得し、形状を検証"""
    try:
        x_batch, y_batch, _, _, _, _ = dataloader.next_batch()
        if not isinstance(x_batch, (list, np.ndarray)) or len(x_batch) == 0: return None

        input_trajectories = torch.from_numpy(np.array(x_batch)).float().permute(1, 0, 2, 3)
        target_trajectories = torch.from_numpy(np.array(y_batch)).float().permute(1, 0, 2, 3)
        
        if target_trajectories.shape[1] != pred_len:
            return None
            
        obstacle_maps = torch.randn(input_trajectories.shape[0], 2)
        
        return {
            'input_trajectories': input_trajectories,
            'target_trajectories': target_trajectories,
            'obstacle_map': obstacle_maps
        }
    except Exception:
        return None

def safe_train_step(model, optimizer, input_traj, target_traj, obstacle_map, 
                    grad_clip_value=1.0):
    """安全な訓練ステップ"""
    model.train()
    optimizer.zero_grad()
    
    try:
        final_pred, stage1_pred, _ = model(input_traj, obstacle_map)
        target_traj_for_loss = target_traj[:, :, 0, :]
        
        main_loss = F.mse_loss(final_pred, target_traj_for_loss)
        stage1_loss = F.mse_loss(stage1_pred, target_traj_for_loss[:, :stage1_pred.shape[1], :])
        
        total_loss = main_loss + 0.3 * stage1_loss
        
        if torch.isnan(total_loss):
            return create_zero_losses()

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
        optimizer.step()
        
        with torch.no_grad():
            displacement_errors = torch.norm(final_pred[..., :2] - target_traj_for_loss[..., :2], dim=-1)
            ade = displacement_errors.mean()
            fde = torch.norm(final_pred[:, -1, :2] - target_traj_for_loss[:, -1, :2], dim=-1).mean()
        
        return {'total_loss': total_loss.item(), 'ade': ade.item(), 'fde': fde.item()}
    except Exception as e:
        logger.error(f"❌ 訓練ステップでエラー: {e}", exc_info=True)
        return create_zero_losses()

def safe_eval_step(model, input_traj, target_traj, obstacle_map):
    """安全な検証ステップ（勾配計算なし）"""
    model.eval()
    with torch.no_grad():
        try:
            final_pred, stage1_pred, _ = model(input_traj, obstacle_map)
            target_traj_for_loss = target_traj[:, :, 0, :]
            
            main_loss = F.mse_loss(final_pred, target_traj_for_loss)
            stage1_loss = F.mse_loss(stage1_pred, target_traj_for_loss[:, :stage1_pred.shape[1], :])
            total_loss = main_loss + 0.3 * stage1_loss
            
            displacement_errors = torch.norm(final_pred[..., :2] - target_traj_for_loss[..., :2], dim=-1)
            ade = displacement_errors.mean()
            fde = torch.norm(final_pred[:, -1, :2] - target_traj_for_loss[:, -1, :2], dim=-1).mean()
            
            return {'total_loss': total_loss.item(), 'ade': ade.item(), 'fde': fde.item()}
        except Exception as e:
            logger.error(f"❌ 検証ステップでエラー: {e}", exc_info=True)
            return create_zero_losses()

def create_zero_losses():
    return {'total_loss': 0.0, 'ade': 0.0, 'fde': 0.0}

def main():
    """メイン関数"""
    logger.info("🚀 新モデル(ECAM+SocialSTGCNN)の訓練を開始します (ベストモデル保存機能付き)")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用デバイス: {device}")
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        from model import TwoStageTrajectoryPredictor
    except ImportError:
        logger.error("❌ model.py が見つかりません。")
        return
    
    config = {
        'batch_size': 32, 'seq_len': 8, 'pred_len': 12, 'num_pedestrians': 5,
        'hidden_dim': 64, 'num_epochs': 500, 'learning_rate': 0.0005,
        'weight_decay': 1e-5, 'feature_dim': 3
    }
    logger.info(f"設定: {config}")
    
    use_dataloader = check_dataloader_availability(config['pred_len'])
    train_dataloader = None # train_dataloaderを初期化
    if use_dataloader:
        try:
            from utils import DataLoader
            train_dataloader = DataLoader(
                f_prefix='.', batch_size=config['batch_size'], 
                seq_length=config['seq_len'], pred_len=config['pred_len']
            )
            logger.info("✅ DataLoader を使用します")
        except Exception as e:
            logger.error(f"❌ DataLoader 作成失敗: {e}")
            use_dataloader = False

    if not use_dataloader:
        logger.info("📊 ダミーデータを使用します")
    
    model = TwoStageTrajectoryPredictor(
        input_dim=config['feature_dim'], hidden_dim=config['hidden_dim'],
        output_dim=config['feature_dim'], seq_len=config['seq_len'],
        pred_len=config['pred_len'], num_pedestrians=config['num_pedestrians']
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    logger.info("🎓 訓練開始")
    
    best_val_ade = float('inf')
    best_epoch = 0
    
    for epoch in range(config['num_epochs']):
        logger.info(f"**************** Epoch {epoch+1}/{config['num_epochs']} ****************")
        
        # --- 訓練フェーズ ---
        epoch_losses, epoch_ades, epoch_fdes = [], [], []
        max_batches_per_epoch = 50
        model.train()
        for batch_idx in range(max_batches_per_epoch):
            if use_dataloader:
                batch_data = process_dataloader_batch(train_dataloader, config['pred_len'])
                if batch_data is None:
                    if train_dataloader.num_batches == 0: break
                    train_dataloader.reset_batch_pointer()
                    continue
                input_traj, target_traj, obstacle_map = [v.to(device) for v in batch_data.values()]
            else:
                input_traj, target_traj, obstacle_map = create_dummy_data(**config)
                input_traj, target_traj, obstacle_map = input_traj.to(device), target_traj.to(device), obstacle_map.to(device)

            losses = safe_train_step(model, optimizer, input_traj, target_traj, obstacle_map)
            epoch_losses.append(losses['total_loss'])
            epoch_ades.append(losses['ade'])
            epoch_fdes.append(losses['fde'])
        
        avg_loss, avg_ade, avg_fde = np.mean(epoch_losses), np.mean(epoch_ades), np.mean(epoch_fdes)
        logger.info(f"Epoch {epoch+1} [訓練] - Loss: {avg_loss:.4f}, ADE: {avg_ade:.4f}, FDE: {avg_fde:.4f}")

        # --- 検証フェーズ ---
        val_epoch_ades, val_epoch_fdes = [], []
        max_val_batches = 10
        model.eval()
        for _ in range(max_val_batches):
            input_traj, target_traj, obstacle_map = create_dummy_data(**config)
            input_traj, target_traj, obstacle_map = input_traj.to(device), target_traj.to(device), obstacle_map.to(device)
            
            metrics = safe_eval_step(model, input_traj, target_traj, obstacle_map)
            val_epoch_ades.append(metrics['ade'])
            val_epoch_fdes.append(metrics['fde'])
        
        avg_val_ade, avg_val_fde = np.mean(val_epoch_ades), np.mean(val_epoch_fdes)
        logger.info(f"Epoch {epoch+1} [検証] - ADE: {avg_val_ade:.4f}, FDE: {avg_val_fde:.4f}")

        # ### ★★★★★ 修正点 ★★★★★ ###
        # ベストモデルの保存
        if avg_val_ade < best_val_ade:
            best_val_ade = avg_val_ade
            best_epoch = epoch + 1
            torch.save(model.state_dict(), 'best_model_social.pth')
            logger.info(f"🎉 新しいベストモデルを保存しました！ (ADE: {best_val_ade:.4f} at Epoch {best_epoch})")

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        logger.info(f"現在の学習率: {current_lr:.6f}")
        scheduler.step()

    logger.info("🎉 訓練完了")
    torch.save(model.state_dict(), 'last_model_social.pth')
    logger.info(f"✅ 最終モデル保存完了: last_model_social.pth")
    
    # ### ★★★★★ 修正点 ★★★★★ ###
    # 最終的なベストモデルの性能を表示
    if os.path.exists('best_model_social.pth'):
        logger.info("\n" + "="*60)
        logger.info("🏆 ベストモデルの最終評価 🏆")
        logger.info(f"   (Epoch {best_epoch} で達成)")
        
        # ベストモデルの重みをロード
        model.load_state_dict(torch.load('best_model_social.pth'))
        
        # 再度、検証データで評価
        final_eval_ades, final_eval_fdes = [], []
        max_eval_batches = 50 # より多くのバッチで評価
        model.eval()
        for _ in range(max_eval_batches):
            input_traj, target_traj, obstacle_map = create_dummy_data(**config)
            input_traj, target_traj, obstacle_map = input_traj.to(device), target_traj.to(device), obstacle_map.to(device)
            metrics = safe_eval_step(model, input_traj, target_traj, obstacle_map)
            final_eval_ades.append(metrics['ade'])
            final_eval_fdes.append(metrics['fde'])
            
        final_ade = np.mean(final_eval_ades)
        final_fde = np.mean(final_eval_fdes)
        
        logger.info(f"   >> 最終ADE: {final_ade:.4f}")
        logger.info(f"   >> 最終FDE: {final_fde:.4f}")
        logger.info("="*60)
    else:
        logger.warning("ベストモデルファイル 'best_model_social.pth' が見つかりませんでした。")

if __name__ == "__main__":
    main()
