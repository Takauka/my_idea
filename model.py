"""
エラー修正版 model.py - 3次元データ対応・アテンション機構修正済み
train_idea1.pyの最終版と連携して動作するバージョン
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, Dict, Any

class EnvironmentalAttentionModule(nn.Module):
    """環境認識型注意機構 (Environmental Context Attention Module: ECAM)"""
    
    def __init__(self, embedding_dim: int = 64, env_dim: int = 32, dropout: float = 0.1):
        super(EnvironmentalAttentionModule, self).__init__()
        self.embedding_dim = embedding_dim
        self.env_dim = env_dim
        
        # 環境情報エンコーダ
        self.env_encoder = nn.Sequential(
            nn.Linear(2, env_dim),
            nn.LayerNorm(env_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(env_dim, env_dim),
            nn.LayerNorm(env_dim),
            nn.ReLU()
        )
        
        # 軌跡エンコーダ
        self.traj_encoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
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
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, trajectory: torch.Tensor, 
                obstacle_map: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, input_dim = trajectory.shape
        
        if input_dim != self.embedding_dim:
             raise ValueError(f"ECAMへの入力次元が不正です。期待値: {self.embedding_dim}, 実際の値: {input_dim}")

        traj_encoded = self.traj_encoder(trajectory)
        
        if obstacle_map is not None:
            env_encoded = self.env_encoder(obstacle_map)
            env_expanded = env_encoded.unsqueeze(1).expand(-1, seq_len, -1)
            fused_features = torch.cat([traj_encoded, env_expanded], dim=-1)
            attended_traj = self.fusion_layer(fused_features)
            attended_traj, _ = self.multihead_attention(
                attended_traj, attended_traj, attended_traj
            )
        else:
            attended_traj = traj_encoded
        
        contrast_feature = self.projector(attended_traj.mean(dim=1))
        return attended_traj, contrast_feature

class SingularTrajectoryPredictor(nn.Module):
    """単体軌跡予測器（修正版）"""
    
    def __init__(self, input_dim: int = 3, hidden_dim: int = 64, output_dim: int = 3,
                 seq_len: int = 8, pred_len: int = 12, num_layers: int = 2, dropout: float = 0.1,
                 num_pedestrians: int = 5):
        super(SingularTrajectoryPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.pred_len = pred_len
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.encoder_lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=True
        )
        
        self.encoder_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        
        self.decoder_lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self.ecam = EnvironmentalAttentionModule(hidden_dim, dropout=dropout)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """モデルの重みを初期化する"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.zeros_(param.data)
                        # 忘却ゲートのバイアスを1に初期化
                        n = param.size(0)
                        param.data[n//4:n//2].fill_(1.)
    
    def forward(self, input_traj: torch.Tensor, 
                obstacle_map: Optional[torch.Tensor] = None,
                training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if input_traj.shape[-1] != self.input_dim:
            raise ValueError(
                f"SingularTrajectoryPredictorへの入力次元が不正です。"
                f"期待値: {self.input_dim}, 実際の値: {input_traj.shape[-1]}"
            )

        encoded_seq, (h_n, c_n) = self.encoder_lstm(input_traj)
        encoded_seq = self.encoder_projection(encoded_seq)
        
        # ECAM（アテンション）の計算結果を、後続のデコーダで正しく使用する
        attended_seq, contrast_feature = self.ecam(encoded_seq, obstacle_map)
        
        # デコーダの初期状態を、アテンション適用後の特徴量から生成する
        decoder_h = attended_seq[:, -1, :].unsqueeze(0).repeat(self.num_layers, 1, 1)

        # セル状態は、元のエンコーダの最終状態を再利用する
        c_n_forward = c_n[-2,:,:]
        c_n_backward = c_n[-1,:,:]
        decoder_c = torch.cat([c_n_forward, c_n_backward], dim=1)
        decoder_c = self.encoder_projection(decoder_c).unsqueeze(0).repeat(self.num_layers, 1, 1)

        decoder_hidden = (decoder_h, decoder_c)
        
        predicted_traj = []
        decoder_input = input_traj[:, -1:, :]
        
        for _ in range(self.pred_len):
            decoder_output, decoder_hidden = self.decoder_lstm(decoder_input, decoder_hidden)
            pred_step = self.output_layer(decoder_output)
            predicted_traj.append(pred_step)
            decoder_input = pred_step
            
        predicted_traj = torch.cat(predicted_traj, dim=1)
        return predicted_traj, contrast_feature

class TwoStageTrajectoryPredictor(nn.Module):
    """二段階軌跡予測器 - 短期予測と長期予測を組み合わせたモデル"""
    
    def __init__(self, input_dim: int = 3, hidden_dim: int = 64, output_dim: int = 3,
                 seq_len: int = 8, pred_len: int = 12, num_layers: int = 2, dropout: float = 0.1,
                 num_pedestrians: int = 5):
        super(TwoStageTrajectoryPredictor, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.pred_len = pred_len
        self.short_pred_len = pred_len // 2
        self.long_pred_len = pred_len - self.short_pred_len
        
        # パラメータをインスタンス変数として保存し、一貫性を保つ
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_pedestrians = num_pedestrians
        
        self.short_term_predictor = SingularTrajectoryPredictor(
            input_dim=input_dim, 
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            seq_len=seq_len,
            pred_len=self.short_pred_len,
            num_layers=self.num_layers,
            dropout=self.dropout,
            num_pedestrians=self.num_pedestrians
        )
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32)
        )
        
        self.trajectory_refinement = nn.Sequential(
            nn.Linear(output_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, output_dim)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_traj: torch.Tensor, 
                obstacle_map: Optional[torch.Tensor] = None,
                training: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # 4次元以上の場合は、最初の歩行者のデータのみ使用する
        if input_traj.dim() > 3:
            input_traj = input_traj[:, :, 0, :]
        
        # Stage 1: 短期予測
        short_pred, short_contrast = self.short_term_predictor(
            input_traj, obstacle_map, training
        )
        
        extended_input = torch.cat([input_traj, short_pred], dim=1)
        
        # Stage 2: 長期予測
        # ハードコードされていた値をインスタンス変数から取得するように変更
        long_term_model = SingularTrajectoryPredictor(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            seq_len=extended_input.shape[1],
            pred_len=self.long_pred_len,
            num_layers=self.num_layers,
            dropout=self.dropout,
            num_pedestrians=self.num_pedestrians
        ).to(extended_input.device)
        
        long_pred, long_contrast = long_term_model(
            extended_input, obstacle_map, training
        )
        
        full_prediction = torch.cat([short_pred, long_pred], dim=1)
        
        # 予測軌跡の微調整
        refined_prediction = full_prediction + self.trajectory_refinement(full_prediction)
        
        # コントラスト特徴量の融合
        combined_contrast = torch.cat([short_contrast, long_contrast], dim=-1)
        final_contrast = self.feature_fusion(combined_contrast)
        
        return refined_prediction, short_pred, final_contrast

if __name__ == '__main__':
    # テスト用の簡単な関数
    def test_models():
        print("=== モデルテスト開始 ===")
        
        batch_size = 4
        seq_len = 8
        pred_len = 12
        input_dim = 3
        num_pedestrians = 5
        
        # 4次元テンソルのテスト
        input_traj_4d = torch.randn(batch_size, seq_len, num_pedestrians, input_dim)
        print(f"テスト入力形状: {input_traj_4d.shape}")
        
        obstacle_map = torch.randn(batch_size, 2)
        
        try:
            model = TwoStageTrajectoryPredictor(
                input_dim=input_dim,
                output_dim=input_dim,
                seq_len=seq_len,
                pred_len=pred_len,
                num_pedestrians=num_pedestrians
            )
            
            final_pred, stage1_pred, contrast = model(input_traj_4d, obstacle_map)
            
            print("✅ テスト成功")
            print(f"   最終予測形状: {final_pred.shape} (期待値: {batch_size, pred_len, input_dim})")
            print(f"   Stage1予測形状: {stage1_pred.shape} (期待値: {batch_size, pred_len//2, input_dim})")
            print(f"   コントラスト特徴量形状: {contrast.shape}")

            assert final_pred.shape == (batch_size, pred_len, input_dim)
            assert stage1_pred.shape == (batch_size, pred_len//2, input_dim)

        except Exception as e:
            print(f"❌ テスト失敗: {e}")
            import traceback
            traceback.print_exc()
        
        print("=== モデルテスト終了 ===")

    test_models()
