"""
train_idea1.py (改善版)
学習プラトーを解消するため、学習率スケジューラ(ReduceLROnPlateau)を導入。
訓練ループを構造化し、ロギングを改善しています。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import logging
import os
import time
import pickle
import argparse
import subprocess

# -----------------------------------------------------------------------------
# 必要なヘルパー関数とユーティリティをインポート
# -----------------------------------------------------------------------------
try:
    from model import SocialModel
    from utils import DataLoader
    from grid import getSequenceGridMask
    from helper import *
except ImportError as e:
    print(f"❌ 必要なモジュールが見つかりません: {e}")
    print("👉 model.py, utils.py, helper.py, grid.py が同じ階層にあるか確認してください。")
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
# 安全な訓練・評価ステップ
# -----------------------------------------------------------------------------

def safe_train_step(net, optimizer, x_seq, grid_seq, peds_list_seq, num_peds_list_seq, dataloader, lookup_seq):
    """安全な訓練ステップ"""
    net.train()
    optimizer.zero_grad()
    
    try:
        # Forward prop
        outputs, _, _ = net(x_seq, grid_seq, peds_list_seq, num_peds_list_seq, dataloader, lookup_seq)
        
        # Compute loss
        loss = Gaussian2DLikelihood(outputs, x_seq, peds_list_seq, lookup_seq)
        
        if torch.isnan(loss):
            logger.warning("訓練中に損失がNaNになりました。このバッチをスキップします。")
            return {'total_loss': 0.0}

        # Compute gradients
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)

        # Update parameters
        optimizer.step()
        
        return {'total_loss': loss.item()}
    
    except Exception as e:
        logger.error(f"❌ 訓練ステップでエラー: {e}", exc_info=True)
        return {'total_loss': 0.0}

def safe_eval_step(net, x_seq, grid_seq, peds_list_seq, num_peds_list_seq, dataloader, lookup_seq, args):
    """安全な検証ステップ（勾配計算なし）"""
    net.eval()
    with torch.no_grad():
        try:
            # Forward prop
            outputs, _, _ = net(x_seq[:-1], grid_seq[:-1], peds_list_seq[:-1], num_peds_list_seq, dataloader, lookup_seq)
            
            # Compute loss
            loss = Gaussian2DLikelihood(outputs, x_seq[1:], peds_list_seq[1:], lookup_seq)

            # Sample from the bivariate Gaussian
            mux, muy, sx, sy, corr = getCoef(outputs)
            next_x, next_y = sample_gaussian_2d(mux.data, muy.data, sx.data, sy.data, corr.data, peds_list_seq[-1], lookup_seq)

            next_vals = torch.zeros_like(x_seq[-1])
            next_vals[:, 0] = next_x
            next_vals[:, 1] = next_y

            # ADEとFDEを計算
            ade = get_mean_error(next_vals.unsqueeze(0), x_seq[-1].data.unsqueeze(0), [peds_list_seq[-1]], [peds_list_seq[-1]], args.use_cuda, lookup_seq)
            fde = get_final_error(next_vals.unsqueeze(0), x_seq[-1].data.unsqueeze(0), [peds_list_seq[-1]], [peds_list_seq[-1]], lookup_seq)

            return {'total_loss': loss.item(), 'ade': ade, 'fde': fde}
        
        except Exception as e:
            logger.error(f"❌ 検証ステップでエラー: {e}", exc_info=True)
            return {'total_loss': float('inf'), 'ade': float('inf'), 'fde': float('inf')}

# -----------------------------------------------------------------------------
# メイン処理
# -----------------------------------------------------------------------------
def main():
    """メイン関数"""
    parser = argparse.ArgumentParser()
    # 基本的な設定
    parser.add_argument('--rnn_size', type=int, default=128, help='size of RNN hidden state')
    parser.add_argument('--embedding_size', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--batch_size', type=int, default=8, help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=20, help='RNN sequence length')
    parser.add_argument('--pred_length', type=int, default=12, help='prediction length')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.003, help='learning rate')
    parser.add_argument('--grad_clip', type=float, default=10., help='clip gradients at this value')
    parser.add_argument('--lambda_param', type=float, default=0.0005, help='L2 regularization parameter')
    # モデルとデータに関する設定
    parser.add_argument('--gru', action="store_true", default=False, help='True : GRU cell, False: LSTM cell')
    parser.add_argument('--neighborhood_size', type=int, default=32, help='Neighborhood size')
    parser.add_argument('--grid_size', type=int, default=4, help='Grid size')
    # 実行環境に関する設定
    parser.add_argument('--use_cuda', action="store_true", default=True, help='Use GPU or not')
    parser.add_argument('--drive', action="store_true", default=False, help='Use Google drive or not')

    args = parser.parse_args()
    
    logger.info("🚀 Social-LSTM/GRU モデルの訓練を開始します (改善版)")
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    logger.info(f"使用デバイス: {device}")

    # --- 1. ディレクトリとデータローダーの準備 ---
    f_prefix = '.'
    if args.drive:
        f_prefix = 'drive/semester_project/social_lstm_final' # 必要に応じて変更
    
    model_name = "GRU" if args.gru else "LSTM"
    method_name = "SOCIALLSTM"
    
    # モデル保存用ディレクトリを作成
    save_directory = os.path.join(f_prefix, 'model', method_name, model_name)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # 設定ファイルを保存
    with open(os.path.join(save_directory,'config.pkl'), 'wb') as f:
        pickle.dump(args, f)

    # データローダーの初期化
    try:
        dataloader = DataLoader(f_prefix, args.batch_size, args.seq_length, num_of_validation=2, forcePreProcess=True)
        logger.info("✅ DataLoader を使用します")
    except Exception as e:
        logger.error(f"❌ DataLoader 作成失敗: {e}", exc_info=True)
        return

    # --- 2. モデル、最適化手法、スケジューラの定義 ---
    net = SocialModel(args).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.lambda_param)
    
    # 検証スコアが停滞したら学習率を下げるスケジューラ
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # --- 3. 訓練ループ ---
    logger.info("🎓 訓練開始")
    best_val_metric = float('inf')
    best_epoch = 0

    for epoch in range(args.num_epochs):
        logger.info(f"**************** Epoch {epoch+1}/{args.num_epochs} ****************")
        
        # --- 訓練フェーズ ---
        dataloader.reset_batch_pointer(valid=False)
        epoch_losses = []
        
        for batch in range(dataloader.num_batches):
            x, _, _, num_peds_list, peds_list, _ = dataloader.next_batch()
            
            for sequence in range(dataloader.batch_size):
                x_seq, peds_list_seq = x[sequence], peds_list[sequence]
                x_seq, lookup_seq = dataloader.convert_proper_array(x_seq, num_peds_list[sequence], peds_list_seq)
                
                grid_seq = getSequenceGridMask(x_seq, dataloader.get_dataset_dimension(), peds_list_seq, args.neighborhood_size, args.grid_size, args.use_cuda)
                x_seq, _ = vectorize_seq(x_seq, peds_list_seq, lookup_seq)
                x_seq = x_seq.to(device)
                
                metrics = safe_train_step(net, optimizer, x_seq, grid_seq, peds_list, num_peds_list, dataloader, lookup_seq)
                epoch_losses.append(metrics['total_loss'])

        avg_loss = np.mean(epoch_losses) if epoch_losses else 0
        logger.info(f"Epoch {epoch+1} [訓練] - 平均Loss: {avg_loss:.4f}")
        
        # --- 検証フェーズ ---
        if dataloader.valid_num_batches > 0:
            dataloader.reset_batch_pointer(valid=True)
            val_epoch_losses, val_epoch_ades, val_epoch_fdes = [], [], []
            
            for _ in range(dataloader.valid_num_batches):
                x, _, _, num_peds_list, peds_list, _ = dataloader.next_valid_batch()
                
                for sequence in range(dataloader.batch_size):
                    x_seq, peds_list_seq = x[sequence], peds_list[sequence]
                    x_seq, lookup_seq = dataloader.convert_proper_array(x_seq, num_peds_list[sequence], peds_list_seq)
                    
                    grid_seq = getSequenceGridMask(x_seq, dataloader.get_dataset_dimension(), peds_list_seq, args.neighborhood_size, args.grid_size, args.use_cuda)
                    x_seq, _ = vectorize_seq(x_seq, peds_list_seq, lookup_seq)
                    x_seq = x_seq.to(device)
                    
                    metrics = safe_eval_step(net, x_seq, grid_seq, peds_list_seq, num_peds_list, dataloader, lookup_seq, args)
                    val_epoch_losses.append(metrics['total_loss'])
                    val_epoch_ades.append(metrics['ade'])
                    val_epoch_fdes.append(metrics['fde'])

            avg_val_loss = np.mean(val_epoch_losses)
            avg_val_ade = np.mean(val_epoch_ades)
            avg_val_fde = np.mean(val_epoch_fdes)
            current_val_metric = (avg_val_ade + avg_val_fde) / 2 # ADEとFDEの平均を評価指標とする
            
            logger.info(f"Epoch {epoch+1} [検証] - Loss: {avg_val_loss:.4f}, ADE: {avg_val_ade:.4f}, FDE: {avg_val_fde:.4f}")

            if current_val_metric < best_val_metric:
                best_val_metric = current_val_metric
                best_epoch = epoch + 1
                
                # ベストモデルを保存
                torch.save(net.state_dict(), os.path.join(save_directory, 'best_model_social.pth'))
                logger.info(f"🎉 新しいベストモデルを保存しました！ (Metric: {best_val_metric:.4f} at Epoch {best_epoch})")

            # 検証メトリックを元にスケジューラを更新
            scheduler.step(current_val_metric)

    logger.info("🎉 訓練完了")
    logger.info(f"🏆 最良モデル: Epoch {best_epoch} (Metric: {best_val_metric:.4f})")
    logger.info(f"✅ 最良モデルは '{os.path.join(save_directory, 'best_model_social.pth')}' に保存されました。")

if __name__ == "__main__":
    main()
