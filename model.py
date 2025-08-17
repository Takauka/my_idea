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
                 seq_len: int = 8, pred_len: int = 12, num_layers: int = 2, dropout: float = 0.1,
                 num_pedestrians: int = 5):
        super(SingularTrajectoryPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_layers = num_layers
        self.input_dim = input_dim  # 重要：input_dimを保存
        self.num_pedestrians = num_pedestrians  # 歩行者数パラメータを追加
        
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
            input_traj: 入力軌跡データ（形状は動的に処理）
            obstacle_map: (batch_size, 2)
            training: 訓練モードフラグ
        Returns:
            predicted_traj: (batch_size, pred_len, output_dim)
            contrast_feature: コントラスト学習用特徴量
        """
        
        # デバッグ情報
        print(f"SingularTrajectoryPredictor入力形状: {input_traj.shape}")
        print(f"期待される入力次元: {self.input_dim}")
        
        # 入力テンソルの次元を正規化
        original_shape = input_traj.shape
        
        # 4次元以上の場合は3次元に変換
        if len(original_shape) > 3:
            print(f"⚠️ 4次元以上のテンソルを3次元に変換: {original_shape}")
            # 最初の3次元以外をflatenして結合
            if len(original_shape) == 4:
                batch_size, seq_len, num_peds, feature_dim = original_shape
                # 複数歩行者の場合は平均化または最初の歩行者のみ使用
                input_traj = input_traj[:, :, 0, :]  # 最初の歩行者のみ使用
                print(f"   変換後形状: {input_traj.shape}")
            else:
                # さらに高次元の場合は reshape
                input_traj = input_traj.view(original_shape[0], original_shape[1], -1)
                print(f"   reshape後形状: {input_traj.shape}")
        
        batch_size = input_traj.shape[0]
        
        # 入力次元チェックと調整
        current_input_dim = input_traj.shape[-1]
        if current_input_dim != self.input_dim:
            print(f"⚠️ 入力次元が期待値と異なります: {current_input_dim} != {self.input_dim}")
            
            if current_input_dim > self.input_dim:
                # 次元が多い場合は切り詰め（最初のinput_dim次元のみ使用）
                input_traj = input_traj[..., :self.input_dim]
                print(f"   切り詰め後形状: {input_traj.shape}")
            elif current_input_dim < self.input_dim:
                # 次元が少ない場合はパディング
                padding_size = self.input_dim - current_input_dim
                padding = torch.zeros(*input_traj.shape[:-1], padding_size, device=input_traj.device)
                input_traj = torch.cat([input_traj, padding], dim=-1)
                print(f"   パディング後形状: {input_traj.shape}")
        
        # 最終的な形状チェック
        if len(input_traj.shape) != 3:
            raise ValueError(f"LSTM入力は3次元である必要があります。現在の形状: {input_traj.shape}")
        
        print(f"最終入力形状: {input_traj.shape}")
        
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


# 新しく追加：TwoStageTrajectoryPredictor クラス

class TwoStageTrajectoryPredictor(nn.Module):
    """二段階軌跡予測器 - 短期予測と長期予測を組み合わせたモデル"""
    
    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, output_dim: int = 2,
                 seq_len: int = 8, pred_len: int = 12, num_layers: int = 2, dropout: float = 0.1,
                 num_pedestrians: int = 5):
        super(TwoStageTrajectoryPredictor, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_pedestrians = num_pedestrians  # 歩行者数パラメータを追加
        self.short_pred_len = pred_len // 2  # 短期予測長
        self.long_pred_len = pred_len - self.short_pred_len  # 長期予測長
        
        # Stage 1: 短期予測器
        self.short_term_predictor = SingularTrajectoryPredictor(
            input_dim=input_dim, 
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            seq_len=seq_len,
            pred_len=self.short_pred_len,
            num_layers=num_layers,
            dropout=dropout,
            num_pedestrians=num_pedestrians  # パラメータを渡す
        )
        
        # Stage 2: 長期予測器（短期予測結果も入力として使用）
        self.long_term_predictor = SingularTrajectoryPredictor(
            input_dim=input_dim, 
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            seq_len=seq_len + self.short_pred_len,  # 元の軌跡 + 短期予測
            pred_len=self.long_pred_len,
            num_layers=num_layers,
            dropout=dropout,
            num_pedestrians=num_pedestrians  # パラメータを渡す
        )
        
        # 特徴量融合層
        self.feature_fusion = nn.Sequential(
            nn.Linear(64, hidden_dim),  # コントラスト特徴量は32次元×2=64次元
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32)  # 最終的なコントラスト特徴量
        )
        
        # 予測軌跡の調整層（オプション）
        self.trajectory_refinement = nn.Sequential(
            nn.Linear(output_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, output_dim)
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
    
    def forward(self, input_traj: torch.Tensor, 
                obstacle_map: Optional[torch.Tensor] = None,
                training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        二段階予測を実行
        
        Args:
            input_traj: 入力軌跡データ（形状は動的に処理）
            obstacle_map: (batch_size, 2) - 障害物マップ
            training: 訓練モードフラグ
            
        Returns:
            predicted_traj: (batch_size, pred_len, output_dim) - 予測軌跡
            contrast_feature: コントラスト学習用特徴量
        """
        
        print(f"TwoStageTrajectoryPredictor入力形状: {input_traj.shape}")
        
        # 入力の形状を正規化
        original_shape = input_traj.shape
        
        # 4次元以上の場合は3次元に変換
        if len(original_shape) > 3:
            print(f"⚠️ 4次元以上のテンソルを3次元に変換: {original_shape}")
            if len(original_shape) == 4:
                batch_size, seq_len, num_peds, feature_dim = original_shape
                # 複数歩行者がある場合は最初の歩行者のみ使用
                input_traj = input_traj[:, :, 0, :]
                print(f"   変換後形状: {input_traj.shape}")
        
        batch_size = input_traj.shape[0]
        print(f"短期予測長: {self.short_pred_len}, 長期予測長: {self.long_pred_len}")
        
        # Stage 1: 短期予測
        short_pred, short_contrast = self.short_term_predictor(
            input_traj, obstacle_map, training
        )
        
        print(f"短期予測結果形状: {short_pred.shape}")
        
        # Stage 1の予測結果を元の軌跡に結合して拡張入力を作成
        extended_input = torch.cat([input_traj, short_pred], dim=1)
        print(f"拡張入力形状: {extended_input.shape}")
        
        # Stage 2: 長期予測（拡張入力を使用）
        long_pred, long_contrast = self.long_term_predictor(
            extended_input, obstacle_map, training
        )
        
        print(f"長期予測結果形状: {long_pred.shape}")
        
        # 短期と長期の予測を結合
        full_prediction = torch.cat([short_pred, long_pred], dim=1)
        print(f"最終予測形状: {full_prediction.shape}")
        
        # 予測軌跡の微調整（オプション）
        refined_prediction = []
        for t in range(full_prediction.shape[1]):
            step_pred = full_prediction[:, t:t+1, :]
            refined_step = step_pred + self.trajectory_refinement(step_pred)
            refined_prediction.append(refined_step)
        
        refined_prediction = torch.cat(refined_prediction, dim=1)
        
        # コントラスト特徴量の融合
        combined_contrast = torch.cat([short_contrast, long_contrast], dim=-1)
        final_contrast = self.feature_fusion(combined_contrast)
        
        print(f"最終コントラスト特徴量形状: {final_contrast.shape}")
        
        # train_idea1.pyとの互換性のため、3つの値を返す
        # (final_pred, stage1_pred, contrast_loss)の形式で返す
        return refined_prediction, short_pred, final_contrast


# デバッグ用関数（以前のコードから）
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


# デバッグ用の入力テンソル処理関数
def process_input_tensor(input_tensor: torch.Tensor, expected_dim: int = 2) -> torch.Tensor:
    """
    入力テンソルを正規化して3次元(batch_size, seq_len, feature_dim)に変換
    
    Args:
        input_tensor: 入力テンソル
        expected_dim: 期待される特徴量次元
    
    Returns:
        正規化されたテンソル
    """
    print(f"🔧 入力テンソル処理開始")
    print(f"   元の形状: {input_tensor.shape}")
    
    original_shape = input_tensor.shape
    
    # 2次元の場合 (batch_size, feature_dim) -> (batch_size, 1, feature_dim)
    if len(original_shape) == 2:
        input_tensor = input_tensor.unsqueeze(1)
        print(f"   2次元→3次元: {input_tensor.shape}")
    
    # 4次元の場合 (batch_size, seq_len, num_agents, feature_dim)
    elif len(original_shape) == 4:
        batch_size, seq_len, num_agents, feature_dim = original_shape
        # 最初のエージェントのみ使用
        input_tensor = input_tensor[:, :, 0, :]
        print(f"   4次元→3次元（最初のエージェント）: {input_tensor.shape}")
    
    # 5次元以上の場合
    elif len(original_shape) > 4:
        # 最初の2次元をbatch_size, seq_lenとして扱い、残りをflattenする
        batch_size, seq_len = original_shape[:2]
        feature_dim = np.prod(original_shape[2:])
        input_tensor = input_tensor.view(batch_size, seq_len, feature_dim)
        print(f"   高次元→3次元（flatten）: {input_tensor.shape}")
    
    # 特徴量次元の調整
    current_feature_dim = input_tensor.shape[-1]
    if current_feature_dim != expected_dim:
        print(f"   特徴量次元調整: {current_feature_dim} → {expected_dim}")
        
        if current_feature_dim > expected_dim:
            # 切り詰め
            input_tensor = input_tensor[..., :expected_dim]
        else:
            # パディング
            padding_size = expected_dim - current_feature_dim
            padding = torch.zeros(*input_tensor.shape[:-1], padding_size, 
                                device=input_tensor.device, dtype=input_tensor.dtype)
            input_tensor = torch.cat([input_tensor, padding], dim=-1)
    
    print(f"   最終形状: {input_tensor.shape}")
    return input_tensor


# テスト用の簡単な関数
def test_models():
    """モデルのテスト"""
    print("=== モデルテスト開始 ===")
    
    # テストデータ（異なる形状でテスト）
    batch_size = 4
    seq_len = 8
    input_dim = 2
    pred_len = 12
    
    # 3次元テンソルのテスト
    input_traj_3d = torch.randn(batch_size, seq_len, input_dim)
    print(f"3次元テスト入力形状: {input_traj_3d.shape}")
    
    # 4次元テンソルのテスト（複数歩行者）
    num_pedestrians = 3
    input_traj_4d = torch.randn(batch_size, seq_len, num_pedestrians, input_dim)
    print(f"4次元テスト入力形状: {input_traj_4d.shape}")
    
    obstacle_map = torch.randn(batch_size, 2)
    
    # TwoStageTrajectoryPredictorのテスト
    try:
        model = TwoStageTrajectoryPredictor(
            input_dim=input_dim,
            seq_len=seq_len,
            pred_len=pred_len,
            num_pedestrians=num_pedestrians
        )
        
        # 3次元入力テスト
        try:
            result = model(input_traj_3d, obstacle_map)
            print(f"✅ 3次元入力テスト成功")
            print(f"   戻り値数: {len(result)}")
            for i, tensor in enumerate(result):
                print(f"   戻り値{i}形状: {tensor.shape}")
        except Exception as e:
            print(f"❌ 3次元入力テスト失敗: {e}")
        
        # 4次元入力テスト
        try:
            result = model(input_traj_4d, obstacle_map)
            print(f"✅ 4次元入力テスト成功")
            print(f"   戻り値数: {len(result)}")
            for i, tensor in enumerate(result):
                print(f"   戻り値{i}形状: {tensor.shape}")
        except Exception as e:
            print(f"❌ 4次元入力テスト失敗: {e}")
        
    except Exception as e:
        print(f"❌ TwoStageTrajectoryPredictor テスト失敗: {e}")
    
    print("=== モデルテスト終了 ===")


if __name__ == "__main__":
    # デバッグとテストの実行
    debug_tensor_flow()
    print("\n")
    test_models()
