"""
train_idea2.py (Social-STGCNN対応・最終版)
Social-STGCNNのデータローダーとモデルを使い、訓練を実行します。
"""

import torch
import torch.optim as optim
import torch.nn as nn
import logging
import os
import argparse
from tqdm import tqdm

# -----------------------------------------------------------------------------
# 必要なモジュールをインポート
# -----------------------------------------------------------------------------
try:
    # --- FIX: Social-STGCNNのモデルとデータセットクラスを正しくインポート ---
    from model import SocialSTGCNN
    from utils import TrajectoryDataset, seq_to_graph
    from torch.utils.data import DataLoader
except ImportError as e:
    print(f"❌ 必要なモジュールが見つかりません: {e}")
    print("👉 model.py と utils.py が同じ階層にあるか確認してください。")
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
# ヘルパー関数
# -----------------------------------------------------------------------------
def bivariate_loss(V_pred, V_tr):
    """二変量正規分布の負の対数尤度損失を計算"""
    #mux, muy, sx, sy, corr
    #assert V_pred.shape == V_tr.shape
    normx = V_tr[:,:,0]- V_pred[:,:,0]
    normy = V_tr[:,:,1]- V_pred[:,:,1]

    sx = torch.exp(V_pred[:,:,2]) #sx
    sy = torch.exp(V_pred[:,:,3]) #sy
    corr = torch.tanh(V_pred[:,:,4]) #corr
    
    sxsy = sx * sy
    z = (normx/sx)**2 + (normy/sy)**2 - 2*((corr*normx*normy)/sxsy)
    neg_rho = 1 - corr**2
    
    #missing max case
    result = torch.exp(-z/(2*neg_rho)) / (2*torch.pi*sxsy*torch.sqrt(neg_rho))
    epsilon = 1e-20
    
    loss = -torch.log(torch.clamp(result, min=epsilon))
    return torch.mean(loss)

# -----------------------------------------------------------------------------
# メイン処理
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./datasets', help='データセットが格納されているルートディレクトリ')
    parser.add_argument('--obs_len', type=int, default=8)
    parser.add_argument('--pred_len', type=int, default=12)
    parser.add_argument('--num_epochs', type=int, default=200, help='エポック数')
    parser.add_argument('--lr', type=float, default=0.01, help='学習率')
    args = parser.parse_args()

    logger.info("🚀 Social-STGCNNモデルの訓練を開始します")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用デバイス: {device}")

    # --- 1. データローダーの準備 ---
    try:
        logger.info("🔄 データセットを読み込んでいます...")
        train_path = os.path.join(args.data_dir, 'train')
        train_dset = TrajectoryDataset(
            data_dir=train_path, # --- FIX: 正しい引数名を使用 ---
            obs_len=args.obs_len,
            pred_len=args.pred_len,
            skip=1,
            delim='\t')
        
        train_loader = DataLoader(
            train_dset,
            batch_size=1, # バッチ処理はシーケンスごとに行う
            shuffle=True,
            num_workers=0)
        logger.info("✅ データ読み込み完了")
    except (FileNotFoundError, NotADirectoryError):
        logger.error(f"❌ 訓練データが見つかりません。'{train_path}' ディレクトリ、またはそのサブディレクトリに.txtファイルがあるか確認してください。")
        return

    # --- 2. モデル、最適化手法の定義 ---
    model = SocialSTGCNN(n_stgcnn=1, n_txpcnn=5,
                         output_feat=5, seq_len=args.obs_len,
                         pred_seq_len=args.pred_len).to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # --- 3. 訓練ループ ---
    logger.info("🎓 訓練開始")
    
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch in pbar:
            # バッチデータをGPUに転送
            batch = [tensor.to(device) for tensor in batch]
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, _, _, \
            v_obs, A_obs, v_pred, A_pred = batch
            
            optimizer.zero_grad()
            
            # 観測データを(N, C, T, V)の形式に変換
            v_obs = v_obs.permute(0, 3, 1, 2)
            
            # モデルで予測
            V_pred, _ = model(v_obs, A_obs.squeeze(0))
            
            V_pred = V_pred.permute(0,2,1)
            V_pred = V_pred.contiguous()
            
            num_nodes = v_pred.shape[2] # --- FIX: 正しい次元からノード数を取得 ---
            V_pred = V_pred.view(-1,num_nodes,5)
            V_tr = v_pred.squeeze(0).permute(1,0,2) # --- FIX: 形状を合わせる ---
            V_tr = V_tr.contiguous().view(-1, V_tr.shape[2])


            # 損失計算
            loss = bivariate_loss(V_pred, V_tr)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
            
            pbar.set_postfix(loss=loss.item())
        
        scheduler.step()
        
        avg_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else 0
        logger.info(f"Epoch {epoch+1}完了, 平均損失: {avg_loss:.4f}")

    # --- モデルの保存 ---
    save_directory = './model/SocialSTGCNN'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    final_model_path = os.path.join(save_directory, 'trained_model.pth')
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"🎉 訓練完了。最終モデルを '{final_model_path}' に保存しました。")


if __name__ == '__main__':
    main()
    
