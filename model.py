"""
Social-STGCNNベースの社会的時空間補正モジュールを組み込んだ、
二段階軌跡予測モデルの完全版。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, Dict, Any

# --- 第1段階で使用するモジュール (変更なし) ---
class EnvironmentalAttentionModule(nn.Module):
    """環境認識型注意機構 (ECAM)"""
    def __init__(self, embedding_dim: int = 64, env_dim: int = 32, dropout: float = 0.1):
        super(EnvironmentalAttentionModule, self).__init__()
        self.embedding_dim = embedding_dim
        self.env_encoder = nn.Sequential(
            nn.Linear(2, env_dim), nn.LayerNorm(env_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(env_dim, env_dim), nn.LayerNorm(env_dim), nn.ReLU()
        )
        self.traj_encoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim), nn.LayerNorm(embedding_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(embedding_dim, embedding_dim)
        )
        self.multihead_attention = nn.MultiheadAttention(embedding_dim, 8, dropout=dropout, batch_first=True)
        self.fusion_layer = nn.Sequential(
            nn.Linear(embedding_dim + env_dim, embedding_dim), nn.LayerNorm(embedding_dim),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(embedding_dim, embedding_dim)
        )
        self.projector = nn.Sequential(nn.Linear(embedding_dim, 64), nn.ReLU(), nn.Linear(64, 32))
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, trajectory: torch.Tensor, obstacle_map: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len, input_dim = trajectory.shape[1], trajectory.shape[2]
        if input_dim != self.embedding_dim:
            raise ValueError(f"ECAM入力次元エラー: 期待値{self.embedding_dim}, 実際値{input_dim}")
        
        traj_encoded = self.traj_encoder(trajectory)
        if obstacle_map is not None:
            env_encoded = self.env_encoder(obstacle_map).unsqueeze(1).expand(-1, seq_len, -1)
            fused = torch.cat([traj_encoded, env_encoded], dim=-1)
            attended_traj = self.fusion_layer(fused)
            attended_traj, _ = self.multihead_attention(attended_traj, attended_traj, attended_traj)
        else:
            attended_traj = traj_encoded
        
        contrast_feature = self.projector(attended_traj.mean(dim=1))
        return attended_traj, contrast_feature

class SingularTrajectoryPredictor(nn.Module):
    """第1段階：個々の歩行者のための環境回避型粗予測モジュール"""
    def __init__(self, input_dim: int = 3, hidden_dim: int = 64, output_dim: int = 3,
                 seq_len: int = 8, pred_len: int = 12, num_layers: int = 2, dropout: float = 0.1):
        super(SingularTrajectoryPredictor, self).__init__()
        self.pred_len, self.num_layers = pred_len, num_layers
        self.input_dim, self.output_dim = input_dim, output_dim
        
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.encoder_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        self.decoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, output_dim)
        )
        self.ecam = EnvironmentalAttentionModule(hidden_dim, dropout=dropout)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name: nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name: nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.zeros_(param.data)
                        n = param.size(0)
                        param.data[n//4:n//2].fill_(1.)

    def forward(self, input_traj: torch.Tensor, obstacle_map: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if input_traj.shape[-1] != self.input_dim:
            raise ValueError(f"SingularPredictor入力次元エラー: 期待値{self.input_dim}, 実際値{input_traj.shape[-1]}")

        encoded_seq, (h_n, c_n) = self.encoder_lstm(input_traj)
        encoded_seq = self.encoder_projection(encoded_seq)
        
        attended_seq, contrast_feature = self.ecam(encoded_seq, obstacle_map)
        
        decoder_h = attended_seq[:, -1, :].unsqueeze(0).repeat(self.num_layers, 1, 1)
        c_n_bi = self.encoder_projection(torch.cat((c_n[-2,:,:], c_n[-1,:,:]), dim=1))
        decoder_c = c_n_bi.unsqueeze(0).repeat(self.num_layers, 1, 1)
        decoder_hidden = (decoder_h, decoder_c)
        
        decoder_input = input_traj[:, -1:, :]
        predicted_traj = []
        for _ in range(self.pred_len):
            output, decoder_hidden = self.decoder_lstm(decoder_input, decoder_hidden)
            pred_step = self.output_layer(output)
            predicted_traj.append(pred_step)
            decoder_input = pred_step
            
        return torch.cat(predicted_traj, dim=1), contrast_feature

# --- ★★★ 新しい第2段階モジュール ★★★ ---
class SocialTemporalGNN(nn.Module):
    """
    第2段階：社会的時空間補正モジュール (Social-STGCNNベース)
    グラフアテンションで歩行者間の相互作用を、畳み込みで時間的連続性を捉える。
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, pred_len: int, dropout=0.1):
        super(SocialTemporalGNN, self).__init__()
        self.pred_len = pred_len
        self.output_dim = output_dim

        # 1. 時間的特徴を抽出する畳み込み層
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # 2. 歩行者間の空間的相互作用をモデル化するグラフアテンション層
        self.graph_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)

        # 3. 相互作用後の特徴を再度時間的に処理する層
        self.temporal_decoder = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # 4. 最終的な軌跡を出力する全結合層
        self.output_projection = nn.Linear(hidden_dim, pred_len * output_dim)

    def forward(self, trajectories: torch.Tensor) -> torch.Tensor:
        # 入力形状: (batch_size, num_pedestrians, seq_len, feature_dim)
        b, p, s, f = trajectories.shape

        # 時間的エンコーダのため形状変更: (batch * peds, feat_dim, seq_len)
        x = trajectories.reshape(b * p, s, f).permute(0, 2, 1)
        x = self.temporal_encoder(x)  # -> (b*p, hidden, s)
        
        # グラフアテンションのため形状変更: (batch * seq, peds, hidden)
        x = x.permute(0, 2, 1).reshape(b, p, s, -1).permute(0, 2, 1, 3)
        x = x.reshape(b * s, p, -1)
        
        # グラフアテンション適用 (残差接続付き)
        res = x
        x, _ = self.graph_attention(x, x, x)
        x = self.norm1(x + res)  # -> (b*s, p, hidden)
        
        # 時間的デコーダのため形状変更: (batch * peds, hidden, seq_len)
        x = x.reshape(b, s, p, -1).permute(0, 2, 3, 1).reshape(b * p, -1, s)
        x = self.temporal_decoder(x)
        
        # 最後のタイムステップの特徴量を使って未来を予測
        final_features = x[:, :, -1]  # -> (b*p, hidden)
        
        # 最終的な軌跡を予測
        predictions = self.output_projection(final_features) # -> (b*p, pred_len * out_dim)
        
        # 出力形状を整形: (batch, peds, pred_len, out_dim)
        return predictions.view(b, p, self.pred_len, self.output_dim)

# --- ★★★ 全体を統括するメインモデル (修正版) ★★★ ---
class TwoStageTrajectoryPredictor(nn.Module):
    """構想を実現した二段階軌跡予測器"""
    def __init__(self, input_dim: int = 3, hidden_dim: int = 64, output_dim: int = 3,
                 seq_len: int = 8, pred_len: int = 12, num_layers: int = 2, dropout: float = 0.1,
                 num_pedestrians: int = 5):
        super(TwoStageTrajectoryPredictor, self).__init__()
        
        self.short_pred_len = pred_len // 2
        # 第2段階は最終的な予測長全体を出力する
        self.final_pred_len = pred_len 

        # --- 第1段階：環境回避型粗予測モジュール ---
        self.stage1_predictor = SingularTrajectoryPredictor(
            input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
            seq_len=seq_len, pred_len=self.short_pred_len, num_layers=num_layers,
            dropout=dropout
        )
        
        # --- 第2段階：社会的時空間補正モジュール ---
        # 入力は「過去軌跡(seq_len)＋粗予測(short_pred_len)」
        stage2_input_len = seq_len + self.short_pred_len
        self.stage2_corrector = SocialTemporalGNN(
            input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
            pred_len=self.final_pred_len, dropout=dropout
        )

    def forward(self, input_traj: torch.Tensor, 
                obstacle_map: Optional[torch.Tensor] = None,
                training: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 入力形状: (batch, seq_len, num_peds, feat_dim)
        b, s, p, f = input_traj.shape

        # --- STAGE 1: 環境を考慮した粗予測 ---
        # 各歩行者を独立したバッチとして処理するため形状変更
        stage1_input = input_traj.permute(0, 2, 1, 3).reshape(b * p, s, f)
        
        # 障害物マップを各歩行者に合わせて拡張
        if obstacle_map is not None:
            obstacle_map_flat = obstacle_map.unsqueeze(1).repeat(1, p, 1).view(b * p, -1)
        else:
            obstacle_map_flat = None
            
        rough_preds_flat, contrast_features_flat = self.stage1_predictor(stage1_input, obstacle_map_flat)
        
        # --- STAGE 2: 社会的相互作用を考慮した補正 ---
        # 粗予測の形状を元に戻す: (batch, peds, short_pred_len, feat)
        rough_preds = rough_preds_flat.view(b, p, self.short_pred_len, f)
        
        # 第2段階の入力を作成: [過去軌跡 + 粗予測軌跡]
        history = input_traj.permute(0, 2, 1, 3) # -> (b, p, s, f)
        stage2_input = torch.cat([history, rough_preds], dim=2) # -> (b, p, s + short_pred, f)
        
        # 全歩行者の最終予測を取得
        final_preds_all_peds = self.stage2_corrector(stage2_input) # -> (b, p, final_pred_len, f)
        
        # --- 出力 ---
        # 訓練スクリプトに合わせて、ターゲット歩行者(index 0)の結果のみを返す
        final_pred_target = final_preds_all_peds[:, 0, :, :] # -> (b, final_pred_len, f)
        
        # 第1段階の予測もターゲット歩行者のものを返す
        stage1_pred_target = rough_preds[:, 0, :, :] # -> (b, short_pred_len, f)
        
        # コントラスト特徴量もターゲット歩行者のものを返す
        contrast_features = contrast_features_flat.view(b, p, -1)
        contrast_feature_target = contrast_features[:, 0, :] # -> (b, feature_dim)
        
        return final_pred_target, stage1_pred_target, contrast_feature_target
