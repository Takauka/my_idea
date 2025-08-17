import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, Dict, Any

# 修正版: EnvironmentalAttentionModule

class EnvironmentalAttentionModule(nn.Module):
    """環境認識型注意機構 (Environmental Context Attention Module: ECAM)"""
    
    def __init__(self, embedding_dim: int = 64, env_dim: int = 32, dropout: float = 0.1):
        super(EnvironmentalAttentionModule, self).__init__()
        self.embedding_dim = embedding_dim
        self.env_dim = env_dim
        
        # 環境情報エンコーダ（改善版）
        self.env_encoder = nn.Sequential(
            nn.Linear(2, env_dim),
            nn.LayerNorm(env_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(env_dim, env_dim),
            nn.LayerNorm(env_dim),
            nn.ReLU()
        )
        
        # 軌跡エンコーダ（修正版）- 重要な変更点！
        self.traj_encoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),  # 入力がembedding_dimの場合
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # マルチヘッド注意機構
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 環境-軌跡融合層
        self.fusion_layer = nn.Sequential(
            nn.Linear(embedding_dim + env_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # コントラスト学習用プロジェクタ
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # 重み初期化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """重み初期化メソッド"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, trajectory: torch.Tensor, 
                obstacle_map: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            trajectory: (batch_size, seq_len, embedding_dim) - エンコード済み軌跡データ
            obstacle_map: (batch_size, 2) - 環境情報
        Returns:
            attended_trajectory: 注意機構適用後の軌跡
            contrast_feature: コントラスト学習用特徴量
        """
        batch_size, seq_len, input_dim = trajectory.shape
        
        # 入力次元チェックとデバッグ情報
        print(f"ECAM入力形状: {trajectory.shape}")
        print(f"期待される入力次元: {self.embedding_dim}")
        
        # 軌跡エンコーディング
        if input_dim == self.embedding_dim:
            # 既にエンコード済みの場合
            traj_encoded = self.traj_encoder(trajectory)
        else:
            # 次元不整合の場合の対処
            print(f"⚠️ 次元不整合検出: 入力{input_dim}, 期待{self.embedding_dim}")
            if input_dim == 2:
                # 座標データの場合、embedding_dimに変換
                coord_encoder = nn.Linear(2, self.embedding_dim).to(trajectory.device)
                trajectory = coord_encoder(trajectory)
                traj_encoded = self.traj_encoder(trajectory)
            else:
                raise ValueError(f"予期しない入力次元: {input_dim}")
        
        if obstacle_map is not None:
            # 環境エンコーディング
            env_encoded = self.env_encoder(obstacle_map)  # (batch_size, env_dim)
            env_expanded = env_encoded.unsqueeze(1).expand(-1, seq_len, -1)
            
            # 軌跡と環境情報を融合
            fused_features = torch.cat([traj_encoded, env_expanded], dim=-1)
            attended_traj = self.fusion_layer(fused_features)
            
            # セルフアテンション適用
            attended_traj, _ = self.multihead_attention(
                attended_traj, attended_traj, attended_traj
            )
        else:
            attended_traj = traj_encoded
        
        # コントラスト学習用特徴量（時系列平均）
        contrast_feature = self.projector(attended_traj.mean(dim=1))
        
        return attended_traj, contrast_feature

# より根本的な解決策: SingularTrajectoryPredictorの修正

class SingularTrajectoryPredictor(nn.Module):
    """単体軌跡予測器（修正版）"""
    
    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, output_dim: int = 2,
                 seq_len: int = 8, pred_len: int = 12, num_layers: int = 2, dropout: float = 0.1):
        super(SingularTrajectoryPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_layers = num_layers
        self.input_dim = input_dim  # 重要：input_dimを保存
        
        # 改良されたLSTMエンコーダ（双方向 + 複数層）
        self.encoder_lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=True
        )
        
        # エンコーダ出力次元調整
        self.encoder_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # デコーダLSTM
        self.decoder_lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,  # input_dimを使用
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # 出力層（残差接続付き）
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # 環境認識型注意機構（修正版）
        self.ecam = EnvironmentalAttentionModule(hidden_dim, dropout=dropout)
        
        # 重み初期化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """重み初期化メソッド"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:  # LSTM input-hidden weights
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:  # LSTM hidden-hidden weights
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:  # LSTM biases
                nn.init.zeros_(param.data)
                # forget gate biasを1に設定（重要！）
                if 'bias_ih' in name:
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1.)
        
        # 線形層の初期化
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, input_traj: torch.Tensor, 
                obstacle_map: Optional[torch.Tensor] = None,
                training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_traj: (batch_size, seq_len, input_dim) - 通常は2次元座標
            obstacle_map: (batch_size, 2)
            training: 訓練モードフラグ
        Returns:
            predicted_traj: (batch_size, pred_len, output_dim)
            contrast_feature: コントラスト学習用特徴量
        """
        batch_size = input_traj.shape[0]
        
        # デバッグ情報
        print(f"SingularTrajectoryPredictor入力形状: {input_traj.shape}")
        print(f"期待される入力次元: {self.input_dim}")
        
        # 入力次元チェック
        if input_traj.shape[-1] != self.input_dim:
            print(f"⚠️ 入力次元が期待値と異なります: {input_traj.shape[-1]} != {self.input_dim}")
            # 必要に応じて次元調整
            if input_traj.shape[-1] > self.input_dim:
                input_traj = input_traj[..., :self.input_dim]  # 切り詰め
            elif input_traj.shape[-1] < self.input_dim:
                # パディング
                padding_size = self.input_dim - input_traj.shape[-1]
                padding = torch.zeros(*input_traj.shape[:-1], padding_size, device=input_traj.device)
                input_traj = torch.cat([input_traj, padding], dim=-1)
        
        # エンコーダで軌跡を符号化
        encoded_seq, (h_n, c_n) = self.encoder_lstm(input_traj)
        
        # 双方向LSTM出力を射影
        encoded_seq = self.encoder_projection(encoded_seq)
        
        print(f"エンコード後の形状: {encoded_seq.shape}")
        
        # ECAM適用（encoded_seqは(batch_size, seq_len, hidden_dim)）
        attended_seq, contrast_feature = self.ecam(encoded_seq, obstacle_map)
        
        # デコーダの初期状態（最後の隠れ状態を使用）
        h_n = h_n[-self.num_layers:].contiguous()  # 前方向のみ使用
        c_n = c_n[-self.num_layers:].contiguous()
        
        # 予測軌跡生成
        predicted_traj = []
        decoder_input = input_traj[:, -1:, :]  # 最後の観測点（元の次元）
        decoder_hidden = (h_n, c_n)
        
        for t in range(self.pred_len):
            # デコーダステップ
            decoder_output, decoder_hidden = self.decoder_lstm(decoder_input, decoder_hidden)
            
            # 出力予測
            pred_step = self.output_layer(decoder_output)
            predicted_traj.append(pred_step)
            
            # 次のステップの入力
            decoder_input = pred_step
        
        predicted_traj = torch.cat(predicted_traj, dim=1)
        
        return predicted_traj, contrast_feature

# 簡単な修正版（一時的対処）
def quick_fix_ecam():
    """一時的な修正：ECAMの軌跡エンコーダーを座標入力用に変更"""
    
    # model.pyの該当行を以下に変更：
    
    # 修正前（84行目付近）:
    # self.traj_encoder = nn.Sequential(
    #     nn.Linear(embedding_dim, embedding_dim),  # これが問題
    #     ...
    # )
    
    # 修正後:
    # self.traj_encoder = nn.Sequential(
    #     nn.Linear(2, embedding_dim),  # 座標入力(2次元)を想定
    #     nn.LayerNorm(embedding_dim),
    #     nn.ReLU(),
    #     nn.Dropout(dropout),
    #     nn.Linear(embedding_dim, embedding_dim)
    # )
    
    pass

# デバッグ用関数
def debug_tensor_flow():
    """テンソルの流れをデバッグ"""
    print("=== テンソル流れのデバッグ ===")
    
    # サンプルデータ
    batch_size = 16
    seq_len = 8
    input_dim = 2  # 座標データ
    
    input_traj = torch.randn(batch_size, seq_len, input_dim)
    print(f"1. 初期入力: {input_traj.shape}")
    
    # LSTMエンコーダー
    hidden_dim = 64
    encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
    encoded_seq, (h_n, c_n) = encoder_lstm(input_traj)
    print(f"2. LSTM出力: {encoded_seq.shape}")  # (16, 8, 128)
    
    # 射影
    encoder_projection = nn.Linear(hidden_dim * 2, hidden_dim)
    encoded_seq = encoder_projection(encoded_seq)
    print(f"3. 射影後: {encoded_seq.shape}")  # (16, 8, 64)
    
    # ECAM入力
    print(f"4. ECAM入力期待: (batch_size, seq_len, embedding_dim)")
    print(f"   実際の形状: {encoded_seq.shape}")
    
    return encoded_seq

if __name__ == "__main__":
    debug_tensor_flow()
