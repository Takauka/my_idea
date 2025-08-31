""
test_idea1.py (実データ評価版)
学習済みのSocial-LSTM/GRUモデルをロードし、
ETH/UCYテストデータセットを使って性能（ADE/FDE）を評価します。
"""
import os
import pickle
import argparse
import time
import subprocess
import torch
from torch.autograd import Variable
import numpy as np
import logging

# -----------------------------------------------------------------------------
# 必要なヘルパー関数とユーティリティをインポート
# (これらのファイルが同じディレクトリにあることを確認してください)
# -----------------------------------------------------------------------------
try:
    from utils import DataLoader
    from helper import get_method_name, get_model, getCoef, sample_gaussian_2d
    from grid import getSequenceGridMask, getGridMask
    from helper import vectorize_seq, revert_seq
except ImportError as e:
    print(f"❌ 必要なモジュールが見つかりません: {e}")
    print("👉 utils.py, helper.py, grid.py が同じ階層にあるか確認してください。")
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
# 評価指標計算関数
# -----------------------------------------------------------------------------
def calculate_metrics(pred_traj, true_traj, obs_len):
    """
    予測軌道と真値軌道からADEとFDEを計算する。
    Args:
        pred_traj (torch.Tensor): 予測された軌道 (seq, peds, 2)
        true_traj (torch.Tensor): 真値の軌道 (seq, peds, 2)
        obs_len (int): 観測長
    Returns:
        Tuple[float, float]: (ade, fde)
    """
    # 予測期間のみを抽出
    pred_traj_gt = pred_traj[obs_len:, :, :]
    true_traj_gt = true_traj[obs_len:, :, :]

    # 座標の差を計算 (seq, peds, 2)
    errors = pred_traj_gt - true_traj_gt
    
    # ユークリッド距離を計算 (seq, peds)
    # 0で割るのを防ぐために微小値を追加
    displacement_errors = torch.sqrt(torch.sum(errors ** 2, dim=-1) + 1e-12)
    
    # ADE: 平均変位誤差
    ade = torch.mean(displacement_errors).item()
    
    # FDE: 最終変位誤差 (最後のタイムステップの誤差の平均)
    fde = torch.mean(displacement_errors[-1, :]).item()
    
    return ade, fde

# -----------------------------------------------------------------------------
# メイン処理
# -----------------------------------------------------------------------------
def main(args):
    """メイン関数"""
    logger.info("🚀 モデルの評価を開始します (実データ使用)")
    
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    logger.info(f"使用デバイス: {device}")

    # --- 1. モデルと設定のロード ---
    method_name = get_method_name(args.method)
    model_name = "GRU" if args.gru else "LSTM"
    logger.info(f"評価対象: Method={method_name}, Model={model_name}, Epoch={args.epoch}")

    # モデルと設定ファイルのパスを構築
    f_prefix = '.'
    save_directory = os.path.join(f_prefix, 'model', method_name, model_name)
    config_path = os.path.join(save_directory, 'config.pkl')
    
    save_tar_name = f"{method_name}_{'gru' if args.gru else 'lstm'}_model_"
    checkpoint_path = os.path.join(save_directory, f"{save_tar_name}{args.epoch}.tar")

    if not os.path.exists(checkpoint_path):
        logger.error(f"❌ モデルファイルが見つかりません: {checkpoint_path}")
        return
    if not os.path.exists(config_path):
        logger.error(f"❌ 設定ファイルが見つかりません: {config_path}")
        return

    # 学習時の設定をロード
    try:
        with open(config_path, 'rb') as f:
            saved_args = pickle.load(f)
    except Exception as e:
        logger.error(f"❌ 設定ファイル({config_path})の読み込みに失敗しました: {e}")
        return
        
    # --- 2. データローダーの準備 ---
    obs_len = saved_args.obs_length
    pred_len = saved_args.pred_length
    seq_length = obs_len + pred_len

    logger.info(f"📖 テストデータセットを読み込んでいます... (観測長: {obs_len}, 予測長: {pred_len})")
    try:
        dataloader = DataLoader(
            f_prefix, 
            batch_size=1,  # 評価時はバッチサイズ1で処理
            seq_length=seq_length, 
            forcePreProcess=False, 
            infer=True # テストモードでロード
        )
        logger.info(f"✅ テストデータ読み込み完了: {dataloader.num_batches} バッチ")
    except Exception as e:
        logger.error(f"❌ DataLoaderの初期化に失敗しました: {e}", exc_info=True)
        return

    # --- 3. モデルの初期化と重みロード ---
    logger.info("🔧 モデルを構築し、学習済み重みをロードしています...")
    net = get_model(args.method, saved_args, is_test=True).to(device)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        net.load_state_dict(checkpoint['state_dict'])
        logger.info(f"✅ モデルの重みを '{checkpoint_path}' (Epoch: {checkpoint['epoch']}) からロードしました。")
    except Exception as e:
        logger.error(f"❌ モデルの重みロードに失敗しました: {e}", exc_info=True)
        return

    net.eval() # 評価モードに設定

    # --- 4. 評価ループ ---
    logger.info("📊 評価を開始...")
    total_ade, total_fde = 0, 0
    batch_count = 0

    for batch in range(dataloader.num_batches):
        start_time = time.time()
        
        # データを取得
        x, _, _, _, _, _ = dataloader.next_batch()
        if not x: continue
        
        x_seq, peds_list_seq, lookup_seq, target_id = dataloader.get_test_sequence_data()
        
        # オリジナルの座標を保持
        orig_x_seq = x_seq.clone().to(device)

        # グリッドマスクを計算
        grid_seq = None
        if args.method in [1, 2]: # social-lstm or obstacle-lstm
             grid_seq = getSequenceGridMask(x_seq, dataloader.get_dataset_dimension(), peds_list_seq, saved_args.neighborhood_size, saved_args.grid_size, args.use_cuda)

        # データを相対座標に変換
        x_seq, first_values_dict = vectorize_seq(x_seq.clone(), peds_list_seq, lookup_seq)
        x_seq = x_seq.to(device)

        # 軌道を予測
        with torch.no_grad():
            obs_traj = x_seq[:obs_len]
            obs_peds_list = peds_list_seq[:obs_len]
            obs_grid = grid_seq[:obs_len] if grid_seq is not None else None

            # sample関数で予測を実行
            pred_x_seq_relative = sample(
                obs_traj, obs_peds_list, saved_args, net, x_seq, peds_list_seq, 
                saved_args, dataloader, lookup_seq, device, args.gru, obs_grid
            )

        # 予測結果を絶対座標に戻す
        pred_x_seq_abs = revert_seq(pred_x_seq_relative.clone(), peds_list_seq, lookup_seq, first_values_dict).to(device)

        # このバッチのADEとFDEを計算 (ターゲット歩行者のみ)
        target_idx = lookup_seq[target_id]
        ade, fde = calculate_metrics(
            pred_x_seq_abs[:, target_idx:target_idx+1, :],
            orig_x_seq[:, target_idx:target_idx+1, :],
            obs_len
        )
        
        total_ade += ade
        total_fde += fde
        batch_count += 1
        
        end_time = time.time()
        
        if (batch + 1) % 20 == 0:
            logger.info(f"  バッチ {batch+1}/{dataloader.num_batches} 完了 (ADE: {ade:.4f}, FDE: {fde:.4f}, Time: {end_time - start_time:.2f}s)")
            
    # --- 5. 最終結果の表示 ---
    if batch_count > 0:
        final_ade = total_ade / batch_count
        final_fde = total_fde / batch_count
        
        logger.info("\n" + "="*60)
        logger.info("🏆 最終評価結果 🏆")
        logger.info(f"  モデルファイル: {os.path.basename(checkpoint_path)}")
        logger.info(f"  評価データ数: {batch_count}")
        logger.info(f"  >> 平均ADE: {final_ade:.4f}")
        logger.info(f"  >> 平均FDE: {final_fde:.4f}")
        logger.info("="*60)
    else:
        logger.warning("評価できるデータがありませんでした。")


def sample(x_seq, ped_list, args, net, true_x_seq, true_ped_list, saved_args, dataloader, look_up, device, is_gru, grid=None):
    """
    軌道をサンプリング（予測）する関数
    """
    num_peds = len(look_up)
    
    with torch.no_grad():
        hidden_states = torch.zeros(num_peds, net.args.rnn_size, device=device)
        cell_states = torch.zeros(num_peds, net.args.rnn_size, device=device) if not is_gru else None

        ret_x_seq = torch.zeros(args.obs_length + args.pred_length, num_peds, 2, device=device)
        ret_x_seq[:args.obs_length, :, :] = x_seq[:args.obs_length].clone()

        # --- 観測期間のフォワードパス (隠れ状態を更新) ---
        for t in range(args.obs_length - 1):
            if grid is None: # vanilla lstm
                _, hidden_states, cell_states = net(
                    ret_x_seq[t].unsqueeze(0), hidden_states, cell_states, 
                    [ped_list[t]], dataloader, look_up
                )
            else: # social/obstacle lstm
                _, hidden_states, cell_states = net(
                    ret_x_seq[t].unsqueeze(0), [grid[t]], hidden_states, cell_states,
                    [ped_list[t]], dataloader, look_up
                )

        # --- 予測期間のフォワードパス ---
        for t in range(args.obs_length - 1, args.pred_length + args.obs_length - 1):
            if grid is None:
                outputs, hidden_states, cell_states = net(
                    ret_x_seq[t].unsqueeze(0), hidden_states, cell_states,
                    [true_ped_list[t]], dataloader, look_up
                )
            else:
                # 予測した座標から新しいグリッドを計算
                current_peds_indices = [look_up.get(p) for p in true_ped_list[t] if p in look_up]
                current_x_coords = ret_x_seq[t, current_peds_indices, :]
                
                if args.method == 2: # obstacle-lstm
                    new_grid = getGridMask(current_x_coords.cpu(), dataloader.get_dataset_dimension(), len(true_ped_list[t]), saved_args.neighborhood_size, saved_args.grid_size, is_obstacle=True)
                else: # social-lstm
                    new_grid = getGridMask(current_x_coords.cpu(), dataloader.get_dataset_dimension(), len(true_ped_list[t]), saved_args.neighborhood_size, saved_args.grid_size)
                
                new_grid = torch.from_numpy(new_grid).float().to(device)

                outputs, hidden_states, cell_states = net(
                    ret_x_seq[t].unsqueeze(0), [new_grid], hidden_states, cell_states,
                    [true_ped_list[t]], dataloader, look_up
                )
            
            mux, muy, sx, sy, corr = getCoef(outputs)
            next_x, next_y = sample_gaussian_2d(mux, muy, sx, sy, corr, true_ped_list[t+1], look_up)
            
            ret_x_seq[t + 1, :, 0] = next_x
            ret_x_seq[t + 1, :, 1] = next_y
            
        return ret_x_seq

# -----------------------------------------------------------------------------
# スクリプト実行
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # --- 必須の引数 ---
    parser.add_argument('--method', type=int, required=True,
                        help='使用するモデルの手法 (1: social-lstm, 2: obstacle-lstm, 3: vanilla-lstm)')
    parser.add_argument('--epoch', type=int, required=True,
                        help='評価するモデルのエポック番号')
                        
    # --- オプションの引数 ---
    parser.add_argument('--gru', action="store_true", default=False,
                        help='GRUモデルを使用する場合は指定')
    parser.add_argument('--use_cuda', action="store_true", default=True,
                        help='GPUを使用しない場合は --use_cuda False を指定')

    args = parser.parse_args()
    
    main(args)
