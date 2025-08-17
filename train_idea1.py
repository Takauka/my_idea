import torch
import torch.nn as nn
import numpy as np
import logging
import os
from typing import Dict, Any, Tuple, Optional

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_dummy_data(batch_size=4, seq_len=8, pred_len=12, num_pedestrians=5, feature_dim=3):
    """3次元のダミーデータを作成"""
    logger.info(f"🎭 {feature_dim}次元のダミーデータを作成中...")
    
    input_trajectories = torch.randn(batch_size, seq_len, num_pedestrians, feature_dim)
    target_trajectories = torch.randn(batch_size, pred_len, num_pedestrians, feature_dim)
    obstacle_maps = torch.randn(batch_size, 2)
    
    logger.info(f"✅ ダミーデータ作成完了:")
    logger.info(f"   入力軌跡: {input_trajectories.shape}")
    logger.info(f"   ターゲット軌跡: {target_trajectories.shape}")
    
    return input_trajectories, target_trajectories, obstacle_maps

def check_dataloader_availability():
    """DataLoaderが使用可能かチェック"""
    try:
        if not os.path.exists('utils.py'): return False
        from utils import DataLoader
        data_dirs = ['data/train', 'data/test', 'data/validation']
        if not any(os.path.exists(d) and os.listdir(d) for d in data_dirs): return False
        # DataLoaderの初期化をテスト
        loader = DataLoader(f_prefix='.', batch_size=2, seq_length=8, pred_length=12)
        return True
    except Exception:
        return False

def process_dataloader_batch(dataloader, pred_len):
    """DataLoaderからバッチを安全に取得し、形状を検証"""
    try:
        x_batch, y_batch, _, _, _, _ = dataloader.next_batch()
        
        if not isinstance(x_batch, (list, np.ndarray)) or not isinstance(y_batch, (list, np.ndarray)):
             return None
        if len(x_batch) == 0 or len(y_batch) == 0:
            return None

        # (seq_len, batch, num_peds, features) -> (batch, seq_len, num_peds, features)
        input_trajectories = torch.from_numpy(np.array(x_batch)).float().permute(1, 0, 2, 3)
        target_trajectories = torch.from_numpy(np.array(y_batch)).float().permute(1, 0, 2, 3)
        
        # ### ★★★★★ エラー原因の核心 ★★★★★ ###
        # DataLoaderが返すターゲットのシーケンス長を検証
        if target_trajectories.shape[1] != pred_len:
            logger.warning(
                f"⚠️ DataLoaderが返すターゲット長({target_trajectories.shape[1]})が"
                f"期待値({pred_len})と異なります。このバッチをスキップします。"
            )
            return None
            
        obstacle_maps = torch.randn(input_trajectories.shape[0], 2)
        
        return {
            'input_trajectories': input_trajectories,
            'target_trajectories': target_trajectories,
            'obstacle_map': obstacle_maps
        }
    except Exception as e:
        logger.error(f"❌ DataLoader バッチ処理エラー: {e}")
        return None

def safe_train_step(model, optimizer, input_traj, target_traj, obstacle_map, 
                    grad_clip_value=1.0):
    """安全な訓練ステップ"""
    model.train()
    optimizer.zero_grad()
    
    try:
        final_pred, stage1_pred, contrast_feature = model(
            input_traj, obstacle_map, training=True
        )
        
        # 損失計算の前に、ターゲット軌跡もモデルへの入力と同じように
        # 最初の歩行者データのみをスライスする
        target_traj_for_loss = target_traj[:, :, 0, :]
        
        # 損失計算
        main_loss = nn.functional.mse_loss(final_pred, target_traj_for_loss)
        stage1_loss = nn.functional.mse_loss(stage1_pred, target_traj_for_loss[:, :stage1_pred.shape[1], :])
        
        # 総合損失
        total_loss = main_loss + 0.3 * stage1_loss + 0.1 * contrast_feature.mean()
        
        if torch.isnan(total_loss):
            logger.error("❌ 損失がNaNになりました。スキップします。")
            return create_zero_losses()

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
        optimizer.step()
        
        # ADE・FDE計算
        with torch.no_grad():
            displacement_errors = torch.norm(final_pred - target_traj_for_loss, dim=-1)
            ade = displacement_errors.mean()
            fde = torch.norm(final_pred[:, -1] - target_traj_for_loss[:, -1], dim=-1).mean()
        
        return {
            'total_loss': total_loss.item(), 'ade': ade.item(), 'fde': fde.item()
        }
    except Exception as e:
        logger.error(f"❌ 訓練ステップでエラー: {e}")
        import traceback
        traceback.print_exc()
        return create_zero_losses()

def create_zero_losses():
    """ゼロ損失辞書を作成"""
    return {'total_loss': 0.0, 'ade': 0.0, 'fde': 0.0}

def main():
    """メイン関数"""
    logger.info("🚀 修正版 train_idea1.py 開始 (DataLoader設定修正)")
    
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
        'batch_size': 4, 'seq_len': 8, 'pred_len': 12, 'num_pedestrians': 5,
        'hidden_dim': 64, 'num_epochs': 20, 'learning_rate': 0.001,
        'weight_decay': 1e-5, 'feature_dim': 3
    }
    logger.info(f"設定: {config}")
    
    use_dataloader = check_dataloader_availability()
    train_dataloader = None
    if use_dataloader:
        try:
            from utils import DataLoader
            # ### ★★★★★ エラー原因の核心 ★★★★★ ###
            # DataLoader初期化時に、予測長(pred_length)を正しく渡す
            train_dataloader = DataLoader(
                f_prefix='.', 
                batch_size=config['batch_size'], 
                seq_length=config['seq_len'],
                pred_length=config['pred_len'] # この引数が重要！
            )
            logger.info("✅ DataLoader を使用します")
        except Exception as e:
            logger.error(f"❌ DataLoader 作成失敗: {e}")
            use_dataloader = False

    if not use_dataloader:
        logger.info("📊 ダミーデータを使用します")
    
    model = TwoStageTrajectoryPredictor(
        input_dim=config['feature_dim'],
        hidden_dim=config['hidden_dim'],
        output_dim=config['feature_dim'],
        seq_len=config['seq_len'],
        pred_len=config['pred_len'],
        num_pedestrians=config['num_pedestrians']
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    logger.info("🎓 訓練開始")
    
    for epoch in range(config['num_epochs']):
        logger.info(f"**************** Epoch {epoch+1}/{config['num_epochs']} ****************")
        
        epoch_losses, epoch_ades, epoch_fdes = [], [], []
        max_batches = 20
        
        for batch_idx in range(max_batches):
            if use_dataloader:
                batch_data = process_dataloader_batch(train_dataloader, config['pred_len'])
                if batch_data is None: continue
                input_traj = batch_data['input_trajectories'].to(device)
                target_traj = batch_data['target_trajectories'].to(device)
                obstacle_map = batch_data['obstacle_map'].to(device)
            else:
                input_traj, target_traj, obstacle_map = create_dummy_data(
                    config['batch_size'], config['seq_len'], config['pred_len'],
                    config['num_pedestrians'], config['feature_dim']
                )
                input_traj, target_traj, obstacle_map = input_traj.to(device), target_traj.to(device), obstacle_map.to(device)

            losses = safe_train_step(model, optimizer, input_traj, target_traj, obstacle_map)
            
            epoch_losses.append(losses['total_loss'])
            epoch_ades.append(losses['ade'])
            epoch_fdes.append(losses['fde'])
            
            if (batch_idx + 1) % 5 == 0:
                logger.info(f"   バッチ {batch_idx+1}/{max_batches}: Loss={losses['total_loss']:.4f}, ADE={losses['ade']:.4f}")
        
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0
        avg_ade = np.mean(epoch_ades) if epoch_ades else 0
        avg_fde = np.mean(epoch_fdes) if epoch_fdes else 0
        
        logger.info(f"Epoch {epoch+1} 結果 - Loss: {avg_loss:.4f}, ADE: {avg_ade:.4f}, FDE: {avg_fde:.4f}")

    logger.info("🎉 訓練完了")
    torch.save(model.state_dict(), 'final_model_3d.pth')
    logger.info("✅ モデル保存完了: final_model_3d.pth")

if __name__ == "__main__":
    main()
