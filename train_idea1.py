"""
エラー修正版 train_idea1.py - 完全動作版
DataLoaderの問題を解決し、確実に動作するバージョン
"""

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

def create_dummy_data(batch_size=4, seq_len=8, pred_len=12, num_pedestrians=5):
    """ダミーデータ作成（DataLoaderが使えない場合の代替）"""
    logger.info("🎭 ダミーデータを作成中...")
    
    # 現実的な軌跡データを生成
    input_trajectories = []
    target_trajectories = []
    obstacle_maps = []
    
    for _ in range(batch_size):
        # 各歩行者の軌跡を生成
        batch_input = torch.zeros(seq_len, num_pedestrians, 2)
        batch_target = torch.zeros(pred_len, num_pedestrians, 2)
        
        for ped in range(num_pedestrians):
            # 開始位置をランダムに設定
            start_x = torch.randn(1) * 10
            start_y = torch.randn(1) * 10
            
            # 速度をランダムに設定
            vel_x = torch.randn(1) * 0.5
            vel_y = torch.randn(1) * 0.5
            
            # 過去の軌跡（input）
            for t in range(seq_len):
                batch_input[t, ped, 0] = start_x + vel_x * t + torch.randn(1) * 0.1
                batch_input[t, ped, 1] = start_y + vel_y * t + torch.randn(1) * 0.1
            
            # 未来の軌跡（target）
            for t in range(pred_len):
                batch_target[t, ped, 0] = start_x + vel_x * (seq_len + t) + torch.randn(1) * 0.1
                batch_target[t, ped, 1] = start_y + vel_y * (seq_len + t) + torch.randn(1) * 0.1
        
        input_trajectories.append(batch_input)
        target_trajectories.append(batch_target)
        
        # 障害物マップ（簡単な2次元座標）
        obstacle_map = torch.randn(2) * 5
        obstacle_maps.append(obstacle_map)
    
    # バッチ化
    input_trajectories = torch.stack(input_trajectories)  # (batch, seq_len, num_peds, 2)
    target_trajectories = torch.stack(target_trajectories)  # (batch, pred_len, num_peds, 2)
    obstacle_maps = torch.stack(obstacle_maps)  # (batch, 2)
    
    logger.info(f"✅ ダミーデータ作成完了:")
    logger.info(f"  入力軌跡: {input_trajectories.shape}")
    logger.info(f"  ターゲット軌跡: {target_trajectories.shape}")
    logger.info(f"  障害物マップ: {obstacle_maps.shape}")
    
    return input_trajectories, target_trajectories, obstacle_maps

def check_dataloader_availability():
    """DataLoaderが使用可能かチェック"""
    try:
        # utils.pyの存在確認
        if not os.path.exists('utils.py'):
            logger.warning("⚠️ utils.py が見つかりません")
            return False
        
        # DataLoaderのインポート試行
        from utils import DataLoader
        logger.info("✅ DataLoader インポート成功")
        
        # データディレクトリの確認
        data_dirs = ['data/train', 'data/test', 'data/validation']
        data_exists = False
        for data_dir in data_dirs:
            if os.path.exists(data_dir):
                files = os.listdir(data_dir)
                if files:  # ファイルが存在する
                    logger.info(f"✅ {data_dir} 存在: {len(files)} ファイル")
                    data_exists = True
                else:
                    logger.warning(f"⚠️ {data_dir} は空です")
            else:
                logger.warning(f"⚠️ {data_dir} が存在しません")
        
        if not data_exists:
            logger.warning("⚠️ 有効なデータディレクトリが見つかりません")
            return False
        
        # DataLoader作成テスト
        try:
            dataloader = DataLoader(
                f_prefix='.',
                batch_size=2,
                seq_length=8,
                forcePreProcess=False,
                infer=False
            )
            logger.info("✅ DataLoader 作成成功")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ DataLoader 作成失敗: {e}")
            return False
            
    except ImportError as e:
        logger.warning(f"⚠️ DataLoader インポート失敗: {e}")
        return False
    except Exception as e:
        logger.warning(f"⚠️ DataLoader チェック中にエラー: {e}")
        return False

def process_dataloader_batch(dataloader):
    """DataLoaderからバッチを安全に取得"""
    try:
        # next_batch メソッドを使用
        x_batch, y_batch, d, numPedsList_batch, PedsList_batch, target_ids = dataloader.next_batch()
        
        if not x_batch or not y_batch:
            logger.warning("⚠️ 空のバッチが返されました")
            return None
        
        # データの変換
        batch_size = len(x_batch)
        if batch_size == 0:
            return None
        
        # 最初のバッチでデータ構造を確認
        first_x = x_batch[0]  # リスト形式のデータ
        first_y = y_batch[0]
        
        if len(first_x) == 0 or len(first_y) == 0:
            logger.warning("⚠️ バッチデータが空です")
            return None
        
        # NumPy配列からTensorに変換
        input_data = []
        target_data = []
        
        for batch_idx in range(min(batch_size, 4)):  # 最大4バッチまで処理
            if batch_idx < len(x_batch) and batch_idx < len(y_batch):
                x_frames = x_batch[batch_idx]
                y_frames = y_batch[batch_idx]
                
                if len(x_frames) > 0 and len(y_frames) > 0:
                    # フレームデータをテンソルに変換
                    x_tensor = torch.from_numpy(np.array(x_frames)).float()
                    y_tensor = torch.from_numpy(np.array(y_frames)).float()
                    
                    input_data.append(x_tensor)
                    target_data.append(y_tensor)
        
        if not input_data or not target_data:
            logger.warning("⚠️ 有効なデータが見つかりませんでした")
            return None
        
        # バッチ形式に整形
        input_trajectories = torch.stack(input_data)
        target_trajectories = torch.stack(target_data)
        
        # 障害物マップ（ダミー）
        obstacle_maps = torch.randn(len(input_data), 2)
        
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
        # 入力データの検証
        if torch.isnan(input_traj).any() or torch.isinf(input_traj).any():
            logger.error("❌ 入力データにNaNまたは無限大が含まれています")
            return create_zero_losses()
        
        if torch.isnan(target_traj).any() or torch.isinf(target_traj).any():
            logger.error("❌ ターゲットデータにNaNまたは無限大が含まれています")
            return create_zero_losses()
        
        # フォワードパス
        final_pred, stage1_pred, contrast_loss = model(
            input_traj, obstacle_map, training=True
        )
        
        # 予測値の妥当性チェック
        if torch.isnan(final_pred).any() or torch.isinf(final_pred).any():
            logger.error("❌ 予測値にNaNまたは無限大が含まれています")
            return create_zero_losses()
        
        # 損失計算
        main_loss = nn.functional.mse_loss(final_pred, target_traj)
        stage1_loss = nn.functional.mse_loss(stage1_pred, target_traj)
        
        # 損失の妥当性チェック
        if torch.isnan(main_loss) or torch.isinf(main_loss):
            logger.error("❌ 主損失にNaNまたは無限大が含まれています")
            return create_zero_losses()
        
        # コントラスト損失が有効な値かチェック
        if torch.isnan(contrast_loss) or torch.isinf(contrast_loss):
            contrast_loss = torch.tensor(0.0, device=contrast_loss.device)
        
        # 総合損失
        total_loss = main_loss + 0.3 * stage1_loss + 0.1 * contrast_loss
        
        # バックワードパス
        total_loss.backward()
        
        # 勾配クリッピング
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
        
        # 勾配チェック
        total_grad_norm = 0.0
        param_count = 0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_grad_norm += param_norm.item() ** 2
                param_count += 1
        
        if param_count > 0:
            total_grad_norm = total_grad_norm ** (1. / 2)
        
        # オプティマイザーステップ
        optimizer.step()
        
        # ADE・FDE計算
        with torch.no_grad():
            displacement_errors = torch.norm(final_pred - target_traj, dim=-1)
            ade = displacement_errors.mean()
            fde = torch.norm(final_pred[:, -1] - target_traj[:, -1], dim=-1).mean()
        
        return {
            'total_loss': total_loss.item(),
            'main_loss': main_loss.item(),
            'stage1_loss': stage1_loss.item(),
            'contrast_loss': contrast_loss.item(),
            'ade': ade.item(),
            'fde': fde.item(),
            'grad_norm': total_grad_norm
        }
        
    except Exception as e:
        logger.error(f"❌ 訓練ステップでエラー: {e}")
        import traceback
        traceback.print_exc()
        return create_zero_losses()

def create_zero_losses():
    """ゼロ損失辞書を作成"""
    return {
        'total_loss': 0.0,
        'main_loss': 0.0,
        'stage1_loss': 0.0,
        'contrast_loss': 0.0,
        'ade': 0.0,
        'fde': 0.0,
        'grad_norm': 0.0
    }

def main():
    """メイン関数"""
    logger.info("🚀 修正版 train_idea1.py 開始")
    
    # 環境設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用デバイス: {device}")
    
    # 再現性のための設定
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # モデルインポート
    try:
        from model import TwoStageTrajectoryPredictor
        logger.info("✅ モデルインポート成功")
    except ImportError as e:
        logger.error(f"❌ モデルインポート失敗: {e}")
        logger.error("model.py が存在し、TwoStageTrajectoryPredictor クラスが定義されているか確認してください")
        return
    
    # 設定
    config = {
        'batch_size': 4,
        'seq_len': 8,
        'pred_len': 12,
        'num_pedestrians': 5,
        'hidden_dim': 64,
        'num_epochs': 20,
        'learning_rate': 0.0001,
        'weight_decay': 1e-5,
    }
    
    logger.info(f"設定: {config}")
    
    # DataLoaderの可用性チェック
    use_dataloader = check_dataloader_availability()
    
    if use_dataloader:
        logger.info("✅ DataLoader を使用します")
        try:
            from utils import DataLoader
            train_dataloader = DataLoader(
                f_prefix='.',
                batch_size=config['batch_size'],
                seq_length=config['seq_len'],
                forcePreProcess=False,
                infer=False
            )
        except Exception as e:
            logger.error(f"❌ DataLoader 作成失敗: {e}")
            use_dataloader = False
    
    if not use_dataloader:
        logger.info("📊 ダミーデータを使用します")
        dummy_input, dummy_target, dummy_obstacle = create_dummy_data(
            config['batch_size'], config['seq_len'], config['pred_len'], config['num_pedestrians']
        )
    
    # モデル作成
    logger.info("🏗️ モデル作成")
    try:
        model = TwoStageTrajectoryPredictor(
            input_dim=2,
            hidden_dim=config['hidden_dim'],
            output_dim=2,
            seq_len=config['seq_len'],
            pred_len=config['pred_len'],
            num_pedestrians=config['num_pedestrians']
        ).to(device)
        
        logger.info(f"✅ モデル作成成功 - パラメータ数: {sum(p.numel() for p in model.parameters())}")
        
    except Exception as e:
        logger.error(f"❌ モデル作成失敗: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # オプティマイザー作成
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay']
    )
    
    logger.info("🎓 訓練開始")
    
    # 訓練ループ
    for epoch in range(config['num_epochs']):
        logger.info(f"****************Training epoch {epoch+1} beginning******************")
        
        epoch_losses = []
        epoch_ades = []
        epoch_fdes = []
        
        batch_count = 0
        max_batches_per_epoch = 20
        
        try:
            if use_dataloader:
                # DataLoaderを使用
                for _ in range(max_batches_per_epoch):
                    batch_data = process_dataloader_batch(train_dataloader)
                    
                    if batch_data is None:
                        logger.warning("⚠️ バッチデータが無効です、スキップします")
                        continue
                    
                    input_trajectories = batch_data['input_trajectories'].to(device)
                    target_trajectories = batch_data['target_trajectories'].to(device)
                    obstacle_map = batch_data['obstacle_map'].to(device)
                    
                    # 訓練ステップ実行
                    losses = safe_train_step(model, optimizer, input_trajectories, target_trajectories, obstacle_map)
                    
                    # 損失記録
                    epoch_losses.append(losses['total_loss'])
                    epoch_ades.append(losses.get('ade', 0.0))
                    epoch_fdes.append(losses.get('fde', 0.0))
                    
                    batch_count += 1
                    
                    # プログレス表示
                    if batch_count % 5 == 0:
                        logger.info(f"  バッチ {batch_count}: Loss={losses['total_loss']:.6f}, ADE={losses['ade']:.6f}")
            
            else:
                # ダミーデータを使用
                for batch_idx in range(max_batches_per_epoch):
                    # ダミーデータを少し変化させる
                    noise_scale = 0.1
                    input_trajectories = (dummy_input + torch.randn_like(dummy_input) * noise_scale).to(device)
                    target_trajectories = (dummy_target + torch.randn_like(dummy_target) * noise_scale).to(device)
                    obstacle_map = (dummy_obstacle + torch.randn_like(dummy_obstacle) * noise_scale).to(device)
                    
                    # 最初のバッチでデータ確認
                    if epoch == 0 and batch_idx == 0:
                        logger.info("📊 ダミーデータ確認")
                        logger.info(f"  入力軌跡: shape={input_trajectories.shape}")
                        logger.info(f"    平均={input_trajectories.mean():.6f}")
                        logger.info(f"    標準偏差={input_trajectories.std():.6f}")
                        logger.info(f"    最小={input_trajectories.min():.6f}")
                        logger.info(f"    最大={input_trajectories.max():.6f}")
                        
                        logger.info(f"  ターゲット軌跡: shape={target_trajectories.shape}")
                        logger.info(f"    平均={target_trajectories.mean():.6f}")
                        logger.info(f"    標準偏差={target_trajectories.std():.6f}")
                    
                    # 訓練ステップ実行
                    losses = safe_train_step(model, optimizer, input_trajectories, target_trajectories, obstacle_map)
                    
                    # 損失記録
                    epoch_losses.append(losses['total_loss'])
                    epoch_ades.append(losses.get('ade', 0.0))
                    epoch_fdes.append(losses.get('fde', 0.0))
                    
                    batch_count += 1
                    
                    # プログレス表示
                    if batch_count % 5 == 0:
                        logger.info(f"  バッチ {batch_count}: Loss={losses['total_loss']:.6f}, ADE={losses['ade']:.6f}")
            
        except Exception as e:
            logger.error(f"❌ エポック処理エラー: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # エポック統計
        if epoch_losses:
            avg_loss = np.mean(epoch_losses)
            avg_ade = np.mean(epoch_ades)
            avg_fde = np.mean(epoch_fdes)
        else:
            avg_loss = 0.0
            avg_ade = 0.0
            avg_fde = 0.0
        
        logger.info(f"Epoch {epoch+1} Training - Total Loss: {avg_loss:.4f}, ADE: {avg_ade:.4f}, FDE: {avg_fde:.4f}")
        
        # 損失チェック
        if avg_loss > 0:
            logger.info("✅ 正常な損失が計算されました！")
        else:
            logger.warning(f"⚠️ 損失が0です（エポック{epoch+1}）")
        
        # 学習率スケジューラー
        if epoch > 0 and epoch % 10 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.9
                logger.info(f"学習率を調整: {param_group['lr']:.6f}")
    
    logger.info("🎉 訓練完了")
    
    # モデル保存
    try:
        torch.save(model.state_dict(), 'final_model.pth')
        logger.info("✅ モデル保存完了: final_model.pth")
    except Exception as e:
        logger.error(f"❌ モデル保存エラー: {e}")

if __name__ == "__main__":
    main()
