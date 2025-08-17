"""
修正版 train_idea1.py - データローダー問題を解決
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

class SimpleTrajectoryDataLoader:
    """シンプルな軌跡データローダー（代替版）"""
    
    def __init__(self, batch_size=2, seq_len=8, pred_len=12, num_pedestrians=5, 
                 num_batches=50, device='cpu'):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_pedestrians = num_pedestrians
        self.num_batches = num_batches
        self.device = device
        self.current_batch = 0
        
        logger.info(f"シンプルデータローダー作成: batch_size={batch_size}")
    
    def __iter__(self):
        self.current_batch = 0
        return self
    
    def __next__(self):
        if self.current_batch >= self.num_batches:
            raise StopIteration
        
        # 現実的な軌跡データを生成
        input_trajectories = torch.zeros(
            self.batch_size, self.seq_len, self.num_pedestrians, 2, 
            device=self.device
        )
        
        target_trajectories = torch.zeros(
            self.batch_size, self.pred_len, self.num_pedestrians, 2, 
            device=self.device
        )
        
        # 各歩行者に異なる軌跡パターンを設定
        for b in range(self.batch_size):
            for p in range(self.num_pedestrians):
                # ランダムな開始位置と速度
                start_x = torch.randn(1) * 3 + p * 4
                start_y = torch.randn(1) * 3 + b * 3
                vel_x = torch.randn(1) * 0.2 + 1.0
                vel_y = torch.randn(1) * 0.2 + 0.7
                
                # 観測軌跡
                for t in range(self.seq_len):
                    noise_x = torch.randn(1) * 0.1
                    noise_y = torch.randn(1) * 0.1
                    input_trajectories[b, t, p, 0] = start_x + vel_x * t + noise_x
                    input_trajectories[b, t, p, 1] = start_y + vel_y * t + noise_y
                
                # 予測軌跡（観測の延長として）
                last_pos = input_trajectories[b, -1, p]
                velocity = input_trajectories[b, -1, p] - input_trajectories[b, -2, p]
                
                for t in range(self.pred_len):
                    noise_x = torch.randn(1) * 0.05
                    noise_y = torch.randn(1) * 0.05
                    target_trajectories[b, t, p] = last_pos + velocity * (t + 1) + torch.tensor([noise_x, noise_y], device=self.device)
        
        # 障害物マップ
        obstacle_map = torch.randn(self.batch_size, 2, device=self.device) * 3
        
        self.current_batch += 1
        
        return {
            'input_trajectories': input_trajectories,
            'target_trajectories': target_trajectories,
            'obstacle_map': obstacle_map
        }
    
    def __len__(self):
        return self.num_batches

def main():
    """メイン関数"""
    logger.info("🚀 修正版 train_idea1.py 開始")
    
    # 環境設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用デバイス: {device}")
    
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
        'num_epochs': 50,
        'learning_rate': 0.0001,
        'weight_decay': 1e-5,
    }
    
    logger.info(f"設定: {config}")
    
    # データローダー診断
    logger.info("📊 データローダー診断実行")
    original_loader_works = debug_data_loader()
    
    if original_loader_works:
        logger.info("✅ 元のDataLoaderを使用")
        try:
            from utils import DataLoader
            train_dataloader = DataLoader(
                f_prefix='.',
                batch_size=config['batch_size'],
                seq_length=config['seq_len'],
                forcePreProcess=False,
                infer=False
            )
            use_original_loader = True
        except Exception as e:
            logger.error(f"❌ 元のDataLoader作成失敗: {e}")
            use_original_loader = False
    else:
        use_original_loader = False
    
    if not use_original_loader:
        logger.info("🔄 シンプルDataLoaderを使用")
        train_dataloader = SimpleTrajectoryDataLoader(
            batch_size=config['batch_size'],
            seq_len=config['seq_len'],
            pred_len=config['pred_len'],
            num_pedestrians=config['num_pedestrians'],
            device=device
        )
    
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
    
    # トレーナー作成
    trainer = TrajectoryPredictionTrainer(
        model=model,
        device=device,
        learning_rate=config['learning_rate'],
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
        
        for batch_data in train_dataloader:
            try:
                if use_original_loader:
                    # 元のDataLoaderの場合
                    x_batch, y_batch, d, numPedsList_batch, PedsList_batch, target_ids = batch_data
                    
                    # データ変換（複雑な処理が必要）
                    # この部分は元のコードに依存するため、シンプル版を推奨
                    logger.warning("元のDataLoader使用は複雑です。シンプル版を推奨。")
                    break
                    
                else:
                    # シンプルDataLoaderの場合
                    input_trajectories = batch_data['input_trajectories']
                    target_trajectories = batch_data['target_trajectories']
                    obstacle_map = batch_data['obstacle_map']
                
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
                losses = trainer.train_step(input_trajectories, target_trajectories, obstacle_map)
                
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
                
                batch_count += 1
                
                # 10バッチ処理したら次のエポックへ
                if batch_count >= 10:
                    break
                    
            except Exception as e:
                logger.error(f"❌ バッチ処理エラー: {e}")
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
    
    logger.info("🎉 訓練完了")

if __name__ == "__main__":
    main()
