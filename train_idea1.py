"""
最終修正版 train_idea1.py - イテレーター対応
DataLoaderのイテレーター機能に対応し、0出力問題を完全解決
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

def debug_data_loader():
    """データローダーの問題を診断"""
    print("🔍 データローダー診断開始")
    
    # utils.pyの存在確認
    if not os.path.exists('utils.py'):
        print("❌ utils.pyが見つかりません")
        return False
    
    try:
        from utils import DataLoader
        print("✅ DataLoaderインポート成功")
    except Exception as e:
        print(f"❌ DataLoaderインポートエラー: {e}")
        return False
    
    # データディレクトリの確認
    data_dirs = ['data/train', 'data/test', 'data/validation']
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            files = os.listdir(data_dir)
            print(f"✅ {data_dir} 存在: {len(files)} ファイル")
        else:
            print(f"❌ {data_dir} が存在しません")
    
    # DataLoader作成テスト
    try:
        print("\nDataLoader作成テスト...")
        dataloader = DataLoader(
            f_prefix='.',  # 現在のディレクトリ
            batch_size=2,
            seq_length=8,
            forcePreProcess=False,
            infer=False
        )
        print("✅ DataLoader作成成功")
        
        # バッチ取得テスト
        print("バッチ取得テスト...")
        x_batch, y_batch, d, numPedsList_batch, PedsList_batch, target_ids = dataloader.next_batch()
        
        print(f"バッチ情報:")
        print(f"  x_batch長さ: {len(x_batch)}")
        print(f"  y_batch長さ: {len(y_batch)}")
        
        if len(x_batch) > 0:
            print(f"  最初のx_batch形状: {len(x_batch[0])}")
            print("✅ バッチ取得成功")
            
            # データ内容確認
            first_batch = x_batch[0]
            if len(first_batch) > 0:
                first_frame = first_batch[0]
                print(f"  最初のフレーム形状: {first_frame.shape}")
                print(f"  最初のフレーム内容: {first_frame}")
                
                if first_frame.size == 0:
                    print("❌ フレームデータが空です")
                    return False
            else:
                print("❌ バッチが空です")
                return False
        else:
            print("❌ バッチが空です")
            return False
            
    except Exception as e:
        print(f"❌ DataLoaderテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def safe_train_step(model, optimizer, input_traj, target_traj, obstacle_map, 
                   grad_clip_value=1.0):
    """安全な訓練ステップ"""
    model.train()
    optimizer.zero_grad()
    
    try:
        # フォワードパス
        final_pred, stage1_pred, contrast_loss = model(
            input_traj, obstacle_map, training=True
        )
        
        # 予測値の妥当性チェック
        if torch.isnan(final_pred).any() or torch.isinf(final_pred).any():
            logger.error("❌ 予測値にNaNまたは無限大が含まれています")
            return {
                'total_loss': 0.0,
                'main_loss': 0.0,
                'stage1_loss': 0.0,
                'contrast_loss': 0.0,
                'ade': 0.0,
                'fde': 0.0,
            }
        
        # 損失計算
        main_loss = nn.functional.mse_loss(final_pred, target_traj)
        stage1_loss = nn.functional.mse_loss(stage1_pred, target_traj)
        
        # 損失の妥当性チェック
        if torch.isnan(main_loss) or torch.isinf(main_loss):
            logger.error("❌ 主損失にNaNまたは無限大が含まれています")
            return {
                'total_loss': 0.0,
                'main_loss': 0.0,
                'stage1_loss': 0.0,
                'contrast_loss': 0.0,
                'ade': 0.0,
                'fde': 0.0,
            }
        
        # 総合損失
        total_loss = main_loss + 0.3 * stage1_loss + 0.1 * contrast_loss
        
        # バックワードパス
        total_loss.backward()
        
        # 勾配クリッピング
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
        
        # 勾配チェック
        total_grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_grad_norm += param_norm.item() ** 2
        total_grad_norm = total_grad_norm ** (1. / 2)
        
        if total_grad_norm == 0.0:
            logger.warning("⚠️ 勾配がゼロです")
        
        # オプティマイザーステップ
        optimizer.step()
        
        # ADE・FDE計算
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
    logger.info("🚀 最終修正版 train_idea1.py 開始")
    
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
        from model import TwoStageTrajectoryPredictor, TrajectoryPredictionTrainer
        logger.info("✅ モデルインポート成功")
    except ImportError as e:
        logger.error(f"❌ モデルインポート失敗: {e}")
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
    
    # データローダー診断
    logger.info("📊 データローダー診断実行")
    original_loader_works = debug_data_loader()
    
    if not original_loader_works:
        logger.error("❌ DataLoaderが正常に動作しません")
        return
    
    # DataLoader作成
    logger.info("✅ 修正版DataLoaderを使用")
    try:
        from utils import DataLoader
        train_dataloader = DataLoader(
            f_prefix='.',
            batch_size=config['batch_size'],
            seq_length=config['seq_len'],
            forcePreProcess=False,
            infer=False
        )
        logger.info("✅ DataLoader作成成功")
    except Exception as e:
        logger.error(f"❌ DataLoader作成失敗: {e}")
        return
    
    # モデル作成
    logger.info("🏗️ モデル作成")
    model = TwoStageTrajectoryPredictor(
        input_dim=2,
        hidden_dim=config['hidden_dim'],
        output_dim=2,
        seq_len=config['seq_len'],
        pred_len=config['pred_len'],
        num_pedestrians=config['num_pedestrians']
    ).to(device)
    
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
        
        try:
            # イテレーターを使用してバッチを取得
            for batch_data in train_dataloader:
                try:
                    # バッチデータ取得
                    input_trajectories = batch_data['input_trajectories'].to(device)
                    target_trajectories = batch_data['target_trajectories'].to(device)
                    obstacle_map = batch_data['obstacle_map'].to(device)
                    
                    # 最初のバッチでデータ確認
                    if epoch == 0 and batch_count == 0:
                        logger.info("📊 最初のバッチデータ確認")
                        logger.info(f"  入力軌跡: shape={input_trajectories.shape}")
                        logger.info(f"    平均={input_trajectories.mean():.6f}")
                        logger.info(f"    標準偏差={input_trajectories.std():.6f}")
                        logger.info(f"    最小={input_trajectories.min():.6f}")
                        logger.info(f"    最大={input_trajectories.max():.6f}")
                        logger.info(f"    ゼロ割合={(input_trajectories == 0).float().mean():.4f}")
                        
                        logger.info(f"  ターゲット軌跡: shape={target_trajectories.shape}")
                        logger.info(f"    平均={target_trajectories.mean():.6f}")
                        logger.info(f"    標準偏差={target_trajectories.std():.6f}")
                        
                        # データが全て0かチェック
                        if (input_trajectories == 0).all():
                            logger.error("❌ 入力データが全て0です！")
                            return
                        
                        if (target_trajectories == 0).all():
                            logger.error("❌ ターゲットデータが全て0です！")
                            return
                    
                    # 訓練ステップ実行
                    losses = safe_train_step(model, optimizer, input_trajectories, target_trajectories, obstacle_map)
                    
                    # 損失記録
                    epoch_losses.append(losses['total_loss'])
                    epoch_ades.append(losses.get('ade', 0.0))
                    epoch_fdes.append(losses.get('fde', 0.0))
                    
                    # 最初のバッチで詳細確認
                    if epoch == 0 and batch_count == 0:
                        logger.info("🔍 最初のバッチ損失詳細:")
                        for key, value in losses.items():
                            logger.info(f"  {key}: {value:.8f}")
                        
                        if losses['total_loss'] == 0.0:
                            logger.error("❌ 損失が0です！これは異常です")
                            
                            # デバッグ用: モデル出力確認
                            model.eval()
                            with torch.no_grad():
                                pred, _, _ = model(input_trajectories, obstacle_map, training=False)
                                logger.info(f"  モデル出力統計:")
                                logger.info(f"    平均={pred.mean():.8f}")
                                logger.info(f"    標準偏差={pred.std():.8f}")
                                logger.info(f"    最小={pred.min():.8f}")
                                logger.info(f"    最大={pred.max():.8f}")
                                
                                # 予測とターゲットの差
                                diff = (pred - target_trajectories).abs().mean()
                                logger.info(f"    予測-ターゲット差平均={diff:.8f}")
                            model.train()
                            
                            if (pred == 0).all():
                                logger.error("❌ モデル出力が全て0です！")
                                return
                        else:
                            logger.info("✅ 正常な損失が計算されました！")
                    
                    batch_count += 1
                    
                    # プログレス表示
                    if batch_count % 5 == 0:
                        logger.info(f"  バッチ {batch_count}: Loss={losses['total_loss']:.6f}, ADE={losses['ade']:.6f}")
                    
                    # 適度な数のバッチ処理後に次のエポックへ
                    if batch_count >= 20:  # 1エポックあたり最大20バッチ
                        break
                        
                except Exception as e:
                    logger.error(f"❌ バッチ処理エラー: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
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
        
        # 損失が0でない場合は成功
        if avg_loss > 0:
            logger.info("✅ 正常な損失が計算されました！")
        else:
            logger.warning(f"⚠️ 損失が0です（エポック{epoch+1}）")
        
        # 学習率スケジューラー（オプション）
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
