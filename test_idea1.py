"""
test_idea1.py (実データ評価版)
学習済みモデル（.pthファイル）をロードし、
実際のETH/UCYテストデータセットを使って性能（ADE/FDE）を評価します。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import os
import argparse
from typing import Dict, Any, Tuple, Optional

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_dataloader_batch(dataloader, pred_len):
    """DataLoaderからバッチを安全に取得し、形状を検証"""
    try:
        # DataLoaderから次のバッチを取得
        x_batch, y_batch, _, _, _, _ = dataloader.next_batch()
        if not isinstance(x_batch, (list, np.ndarray)) or len(x_batch) == 0: return None

        # NumPy配列をTensorに変換し、形状を整える
        # (seq_len, batch, num_peds, features) -> (batch, seq_len, num_peds, features)
        input_trajectories = torch.from_numpy(np.array(x_batch)).float().permute(1, 0, 2, 3)
        target_trajectories = torch.from_numpy(np.array(y_batch)).float().permute(1, 0, 2, 3)
        
        # ターゲットのシーケンス長が期待値と一致するか検証
        if target_trajectories.shape[1] != pred_len:
            logger.warning(f"DataLoaderが返すターゲット長({target_trajectories.shape[1]})が期待値({pred_len})と異なります。スキップします。")
            return None
            
        # 障害物マップはダミーで生成（ECAMの入力として必要）
        obstacle_maps = torch.randn(input_trajectories.shape[0], 2)
        
        return {
            'input_trajectories': input_trajectories,
            'target_trajectories': target_trajectories,
            'obstacle_map': obstacle_maps
        }
    except Exception:
        return None

def safe_eval_step(model, input_traj, target_traj, obstacle_map):
    """安全な検証ステップ（勾配計算なし）"""
    model.eval()
    with torch.no_grad():
        try:
            # モデルは全歩行者の予測を返す (batch, peds, pred, feat)
            final_pred_all, _, _ = model(input_traj, obstacle_map)
            
            # ターゲットの形状を合わせる: (batch, pred, peds, feat) -> (batch, peds, pred, feat)
            target_traj_all = target_traj.permute(0, 2, 1, 3)
            
            # 座標(x,y)のみを使って誤差を計算
            displacement_errors = torch.norm(final_pred_all[..., :2] - target_traj_all[..., :2], dim=-1)
            ade = displacement_errors.mean().item()
            fde = torch.norm(final_pred_all[:, :, -1, :2] - target_traj_all[:, :, -1, :2], dim=-1).mean().item()
            
            return {'ade': ade, 'fde': fde}
        except Exception as e:
            logger.error(f"❌ 評価ステップでエラー: {e}", exc_info=True)
            return {'ade': float('inf'), 'fde': float('inf')}

def main(args):
    """メイン関数"""
    logger.info("🚀 モデルの評価を開始します (実データ使用)")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用デバイス: {device}")
    
    try:
        from model import TwoStageTrajectoryPredictor
        from utils import DataLoader
    except ImportError as e:
        logger.error(f"❌ 必要なモジュールが見つかりません: {e}")
        logger.error("model.py と utils.py が同じ階層にあるか確認してください。")
        return

    if not os.path.exists(args.model_path):
        logger.error(f"❌ モデルファイルが見つかりません: {args.model_path}")
        return

    # --- モデルとデータの構成を訓練時と合わせる ---
    config = {
        'batch_size': 64, # 評価時は大きめのバッチサイズでもOK
        'seq_len': 8, 
        'pred_len': 12, 
        'num_pedestrians': 5,
        'hidden_dim': 128, # 訓練時と同じ値
        'feature_dim': 3
    }
    logger.info(f"テスト設定: {config}")
    
    # ### ★★★★★ 修正点 ★★★★★ ###
    # 実際のテストデータセットをロード
    logger.info("📖 テストデータセットを読み込んでいます...")
    try:
        test_dataloader = DataLoader(
            f_prefix='.', 
            batch_size=config['batch_size'], 
            seq_length=config['seq_len'],
            pred_len=config['pred_len'],
            forcePreProcess=False, 
            # infer=True を設定してテストモードでデータをロード
            infer=True 
        )
        logger.info(f"✅ テストデータ読み込み完了: {test_dataloader.num_batches} バッチ")
    except Exception as e:
        logger.error(f"❌ DataLoaderの初期化に失敗しました: {e}")
        return

    # モデルのアーキテクチャを定義
    model = TwoStageTrajectoryPredictor(
        input_dim=config['feature_dim'], hidden_dim=config['hidden_dim'],
        output_dim=config['feature_dim'], seq_len=config['seq_len'],
        pred_len=config['pred_len'], num_pedestrians=config['num_pedestrians']
    ).to(device)
    
    # 保存された重みをロード
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    logger.info(f"✅ モデルの重みを '{args.model_path}' からロードしました。")
    
    # --- 評価ループ ---
    logger.info("📊 評価を開始...")
    final_eval_ades, final_eval_fdes = [], []
    
    # DataLoaderの全バッチに対して評価を実行
    for i in range(test_dataloader.num_batches):
        batch_data = process_dataloader_batch(test_dataloader, config['pred_len'])
        if batch_data is None: continue
            
        input_traj, target_traj, obstacle_map = [v.to(device) for v in batch_data.values()]
        
        metrics = safe_eval_step(model, input_traj, target_traj, obstacle_map)
        final_eval_ades.append(metrics['ade'])
        final_eval_fdes.append(metrics['fde'])

        if (i + 1) % 10 == 0:
            logger.info(f"  バッチ {i+1}/{test_dataloader.num_batches} 完了...")

    final_ade = np.mean(final_eval_ades)
    final_fde = np.mean(final_eval_fdes)
    
    logger.info("\n" + "="*60)
    logger.info("🏆 最終評価結果 🏆")
    logger.info(f"   モデルファイル: {args.model_path}")
    logger.info(f"   >> 最終ADE: {final_ade:.4f}")
    logger.info(f"   >> 最終FDE: {final_fde:.4f}")
    logger.info("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='best_model_social.pth',
                        help='評価する学習済みモデルのパス (best_model_social.pth を推奨)')
    args = parser.parse_args()
    
    main(args)
