"""
test_idea1.py (モデル評価用スクリプト)
学習済みモデル（.pthファイル）をロードし、その性能（ADE/FDE）を評価します。
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

def create_dummy_data(batch_size=32, seq_len=8, pred_len=12, num_pedestrians=5, feature_dim=3):
    """評価用の3次元ダミーデータを作成"""
    input_trajectories = torch.randn(batch_size, seq_len, num_pedestrians, feature_dim)
    target_trajectories = torch.randn(batch_size, pred_len, num_pedestrians, feature_dim)
    obstacle_maps = torch.randn(batch_size, 2)
    return input_trajectories, target_trajectories, obstacle_maps

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
    logger.info("🚀 モデルの評価を開始します")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用デバイス: {device}")
    
    # model.pyが存在するか確認
    try:
        from model import TwoStageTrajectoryPredictor
    except ImportError:
        logger.error("❌ model.py が見つかりません。train_idea1.pyと同じ階層に配置してください。")
        return

    # モデルの重みファイルが存在するか確認
    if not os.path.exists(args.model_path):
        logger.error(f"❌ モデルファイルが見つかりません: {args.model_path}")
        logger.error("train_idea1.pyを実行して、先にモデルを学習・保存してください。")
        return

    # --- モデルの構成を訓練時と合わせる ---
    config = {
        'batch_size': 32, 
        'seq_len': 8, 
        'pred_len': 12, 
        'num_pedestrians': 5,
        'hidden_dim': 128, # 訓練時と同じ値
        'feature_dim': 3
    }
    logger.info(f"テスト設定: {config}")
    
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
    max_eval_batches = 100 # 評価に使用するバッチ数
    
    for i in range(max_eval_batches):
        input_traj, target_traj, obstacle_map = create_dummy_data(**config)
        input_traj, target_traj, obstacle_map = input_traj.to(device), target_traj.to(device), obstacle_map.to(device)
        
        metrics = safe_eval_step(model, input_traj, target_traj, obstacle_map)
        final_eval_ades.append(metrics['ade'])
        final_eval_fdes.append(metrics['fde'])

        if (i + 1) % 10 == 0:
            logger.info(f"  バッチ {i+1}/{max_eval_batches} 完了...")

    final_ade = np.mean(final_eval_ades)
    final_fde = np.mean(final_eval_fdes)
    
    logger.info("\n" + "="*60)
    logger.info("🏆 最終評価結果 🏆")
    logger.info(f"   モデルファイル: {args.model_path}")
    logger.info(f"   >> 最終ADE: {final_ade:.4f}")
    logger.info(f"   >> 最終FDE: {final_fde:.4f}")
    logger.info("="*60)

if __name__ == "__main__":
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='last_model_social.pth',
                        help='評価する学習済みモデルのパス')
    args = parser.parse_args()
    
    main(args)
