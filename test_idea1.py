"""
test_idea1.py - 新しい二段階モデル(Social-STGCNN対応)用のテストファイル
モデルの各コンポーネントと、訓練プロセス全体が正しく動作するかを検証する。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import logging

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 現在のディレクトリをPythonパスに追加
sys.path.append('.')

def create_dummy_batch(batch_size=4, seq_len=8, pred_len=12, num_pedestrians=5, feature_dim=3):
    """
    テスト用の正しい形状のダミーデータバッチを生成するシンプルな関数
    """
    input_traj = torch.randn(batch_size, seq_len, num_pedestrians, feature_dim)
    target_traj = torch.randn(batch_size, pred_len, num_pedestrians, feature_dim)
    obstacle_map = torch.randn(batch_size, 2)
    return input_traj, target_traj, obstacle_map

def test_model_components():
    """モデルの個別コンポーネントが正しく動作するかをテスト"""
    logger.info("--- 個別コンポーネントテスト開始 ---")
    success = True
    try:
        from model import EnvironmentalAttentionModule, SingularTrajectoryPredictor, SocialTemporalGNN

        # 1. ECAM テスト
        ecam = EnvironmentalAttentionModule(embedding_dim=64)
        traj_features = torch.randn(2 * 5, 8, 64) # (batch*peds, seq, hidden)
        attended, _ = ecam(traj_features)
        assert attended.shape == traj_features.shape
        logger.info("✅ 1. EnvironmentalAttentionModule: 動作確認OK")

        # 2. 第1段階予測器テスト
        predictor = SingularTrajectoryPredictor(input_dim=3, hidden_dim=64, output_dim=3, seq_len=8, pred_len=6)
        input_traj_flat = torch.randn(2 * 5, 8, 3) # (batch*peds, seq, feat)
        pred, _ = predictor(input_traj_flat)
        assert pred.shape == (2 * 5, 6, 3)
        logger.info("✅ 2. SingularTrajectoryPredictor (第1段階): 動作確認OK")

        # 3. SocialTemporalGNN (第2段階) テスト
        st_gnn = SocialTemporalGNN(input_dim=3, hidden_dim=64, output_dim=3, pred_len=12)
        social_input = torch.randn(2, 5, 8 + 6, 3) # (batch, peds, seq+pred, feat)
        final_pred = st_gnn(social_input)
        assert final_pred.shape == (2, 5, 12, 3)
        logger.info("✅ 3. SocialTemporalGNN (第2段階): 動作確認OK")

    except Exception as e:
        logger.error(f"❌ コンポーネントテストでエラー: {e}", exc_info=True)
        success = False
    
    return success

def test_full_model_forward_pass():
    """モデル全体の順伝播と出力形状をテスト"""
    logger.info("--- モデル全体の順伝播テスト開始 ---")
    success = True
    try:
        from model import TwoStageTrajectoryPredictor
        
        # モデルの初期化
        model = TwoStageTrajectoryPredictor(
            input_dim=3, hidden_dim=64, output_dim=3, seq_len=8, 
            pred_len=12, num_pedestrians=5
        )
        model.eval()

        # ダミーデータの生成
        input_traj, _, obstacle_map = create_dummy_batch(
            batch_size=4, seq_len=8, pred_len=12, num_pedestrians=5, feature_dim=3
        )
        
        with torch.no_grad():
            final_pred, stage1_pred, _ = model(input_traj, obstacle_map)

        # 出力形状の検証
        assert final_pred.shape == (4, 12, 3)
        assert stage1_pred.shape == (4, 6, 3)
        logger.info(f"✅ 順伝播成功: final_pred shape={final_pred.shape}, stage1_pred shape={stage1_pred.shape}")

    except Exception as e:
        logger.error(f"❌ 順伝播テストでエラー: {e}", exc_info=True)
        success = False

    return success

def test_training_step():
    """1回の訓練ステップ（順伝播、損失計算、逆伝播）をテスト"""
    logger.info("--- 訓練ステップテスト開始 ---")
    success = True
    try:
        from model import TwoStageTrajectoryPredictor
        
        # モデルとオプティマイザの初期化
        model = TwoStageTrajectoryPredictor(
            input_dim=3, hidden_dim=64, output_dim=3, seq_len=8, 
            pred_len=12, num_pedestrians=5
        )
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

        # ダミーデータの生成
        input_traj, target_traj, obstacle_map = create_dummy_batch(
            batch_size=4, seq_len=8, pred_len=12, num_pedestrians=5, feature_dim=3
        )

        # 順伝播
        final_pred, stage1_pred, _ = model(input_traj, obstacle_map)
        
        # ターゲット歩行者(index 0)の正解データをスライス
        target_traj_for_loss = target_traj[:, :, 0, :]
        
        # 損失計算
        main_loss = F.mse_loss(final_pred, target_traj_for_loss)
        stage1_loss = F.mse_loss(stage1_pred, target_traj_for_loss[:, :stage1_pred.shape[1], :])
        total_loss = main_loss + 0.3 * stage1_loss
        
        # 逆伝播とパラメータ更新
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        logger.info(f"✅ 訓練ステップ成功: Loss={total_loss.item():.4f}")
        # パラメータが更新されたか簡易的にチェック
        assert model.stage2_corrector.output_projection.weight.grad is not None

    except Exception as e:
        logger.error(f"❌ 訓練ステップテストでエラー: {e}", exc_info=True)
        success = False

    return success

def main():
    """メインテスト関数"""
    logger.info("="*60)
    logger.info("Social-STGCNN対応モデルのテストを開始します")
    logger.info("="*60)
    
    # デバイス確認
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用デバイス: {device}")
    logger.info(f"PyTorchバージョン: {torch.__version__}")
    
    # テストの実行
    tests = {
        "個別コンポーネント": test_model_components,
        "モデル全体の順伝播": test_full_model_forward_pass,
        "訓練ステップ": test_training_step,
    }
    
    all_success = True
    for test_name, test_func in tests.items():
        logger.info(f"\n--- [{test_name}] を実行中... ---")
        success = test_func()
        if success:
            logger.info(f"✅ [{test_name}] 成功")
        else:
            logger.error(f"❌ [{test_name}] 失敗")
            all_success = False
            
    # 最終結果
    logger.info("\n" + "="*60)
    if all_success:
        logger.info("🎉 全てのテストに成功しました！")
        logger.info("train_idea1.py を実行して、本格的な訓練を開始できます。")
    else:
        logger.error("❌ いくつかのテストが失敗しました。上記のエラーログを確認してください。")
    logger.info("="*60)

if __name__ == "__main__":
    main()
