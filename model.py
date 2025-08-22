from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from .anchor import AdaptiveAnchor
from .space import SingularSpace
from . import homography as hm

from baseline.transformerdiffusion.nce.map_nce import MapNceLoss, MapQueryEmbedder, MapKeyEmbedder
import baseline.transformerdiffusion.model_utils as model_utils


# ### ★★★★★ 新しく追加するECAMモジュール ★★★★★ ###
class EnvironmentalAttentionModule(nn.Module):
    """
    ECAM (Environmental Context Attention Module)
    観測された軌跡と障害物マップから、環境を考慮したコンテキストベクトルを生成する。
    """
    def __init__(self, traj_input_dim=2, map_input_channels=1, hidden_dim=64, context_dim=32, dropout=0.1):
        super().__init__()
        
        # 軌跡を処理するためのエンコーダ (LSTM)
        self.traj_encoder = nn.LSTM(traj_input_dim, hidden_dim, batch_first=True)
        
        # 障害物マップのパッチを処理するためのエンコーダ (CNN)
        # map_patches (B, 1, H, W) を想定
        self.map_encoder = nn.Sequential(
            nn.Conv2d(map_input_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), # -> (B, 32, 1, 1)
            nn.Flatten() # -> (B, 32)
        )
        
        # 軌跡特徴量とマップ特徴量を統合し、最終的なコンテキストベクトルを生成する
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim + 32, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, context_dim)
        )

    def forward(self, obs_traj: torch.Tensor, map_patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs_traj (torch.Tensor): 観測軌跡 (B, T, 2)
            map_patches (torch.Tensor): 各歩行者周辺の障害物マップパッチ (B, 1, H, W)
        
        Returns:
            torch.Tensor: 環境コンテキストベクトル (B, context_dim)
        """
        # 軌跡特徴量の抽出
        _, (traj_hidden, _) = self.traj_encoder(obs_traj)
        traj_features = traj_hidden.squeeze(0) # -> (B, hidden_dim)
        
        # マップ特徴量の抽出
        map_features = self.map_encoder(map_patches) # -> (B, 32)
        
        # 特徴量の結合
        combined_features = torch.cat([traj_features, map_features], dim=1)
        
        # 最終的なコンテキストベクトルの生成
        ecam_context = self.fusion_layer(combined_features)
        
        return ecam_context


class SingularTrajectory(nn.Module):
    r"""The SingularTrajectory model (ECAM統合版)"""

    def __init__(self, baseline_model, hook_func, hyper_params, device):
        super().__init__()

        self.baseline_model = baseline_model
        self.hook_func = hook_func
        self.hyper_params = hyper_params
        self.t_obs, self.t_pred = hyper_params.obs_len, hyper_params.pred_len
        self.obs_svd, self.pred_svd = hyper_params.obs_svd, hyper_params.pred_svd
        self.k = hyper_params.k
        self.s = hyper_params.num_samples
        self.dim = hyper_params.traj_dim
        self.static_dist = hyper_params.static_dist
        self.device = device

        self.Singular_space_m = SingularSpace(hyper_params=hyper_params, norm_sca=True)
        self.Singular_space_s = SingularSpace(hyper_params=hyper_params, norm_sca=False)
        self.adaptive_anchor_m = AdaptiveAnchor(hyper_params=hyper_params)
        self.adaptive_anchor_s = AdaptiveAnchor(hyper_params=hyper_params)

        # ### ★★★★★ ECAMモジュールをインスタンス化 ★★★★★ ###
        # baseline_modelのコンテキストサイズに合わせてECAMの出力次元を設定するのが理想
        # ここでは仮に32次元とする
        self.ecam_module = EnvironmentalAttentionModule(context_dim=32)

        #####################
        ## MapNCE (変更なし)
        query_proj = MapQueryEmbedder(256, 16)
        event_encoder = MapKeyEmbedder(2, 16)
        self.map_nce = MapNceLoss(obs_len=self.t_obs, pred_len=self.t_pred,
                                  num_contour_points=10, query_embedder=query_proj,
                                  key_embedder=event_encoder, temperature=0.3)

    # (calculate_parameters, calculate_adaptive_anchor, calculate_mask は変更なし)
    def calculate_parameters(self, obs_traj_BT2, pred_traj_BT2):
        mask_B = self.calculate_mask(obs_traj_BT2)
        obs_m_traj_BT2, pred_m_traj_BT2 = obs_traj_BT2[mask_B], pred_traj_BT2[mask_B]
        obs_s_traj_BT2, pred_s_traj_BT2 = obs_traj_BT2[~mask_B], pred_traj_BT2[~mask_B]
        data_m = self.Singular_space_m.parameter_initialization(obs_m_traj_BT2, pred_m_traj_BT2)
        data_s = self.Singular_space_s.parameter_initialization(obs_s_traj_BT2, pred_s_traj_BT2)
        self.adaptive_anchor_m.anchor_initialization(*data_m)
        self.adaptive_anchor_s.anchor_initialization(*data_s)

    def calculate_adaptive_anchor(self, dataset):
        obs_traj_BT2, pred_traj_BT2 = dataset.obs_traj_BT2, dataset.pred_traj_BT2
        scene_id_B = dataset.scene_id
        vector_field = dataset.vector_field
        homography = dataset.homography
        obs_traj_BT2 = obs_traj_BT2.to(self.device)
        pred_traj_BT2 = pred_traj_BT2.to(self.device)
        mask_B = self.calculate_mask(obs_traj_BT2)
        mask_cpu_B = mask_B.cpu().numpy()
        obs_m_traj_BT2, scene_id_m_B = obs_traj_BT2[mask_B], scene_id_B[mask_cpu_B]
        obs_s_traj_BT2, scene_id_s_B = obs_traj_BT2[~mask_B], scene_id_B[~mask_cpu_B]
        n_ped = pred_traj_BT2.size(0)
        anchor_BKN = torch.zeros((n_ped, self.k, self.s), dtype=torch.float)
        anchor_BKN[mask_B] = self.adaptive_anchor_m.adaptive_anchor_calculation(obs_m_traj_BT2, scene_id_m_B, vector_field, homography, self.Singular_space_m)
        anchor_BKN[~mask_B] = self.adaptive_anchor_s.adaptive_anchor_calculation(obs_s_traj_BT2, scene_id_s_B, vector_field, homography, self.Singular_space_s)
        return anchor_BKN

    def calculate_mask(self, obs_traj_BT2):
        if obs_traj_BT2.size(1) <= 2:
            mask_B = (obs_traj_BT2[:, -1] - obs_traj_BT2[:, -2]).div(1).norm(p=2, dim=-1) > self.static_dist
        else:
            mask_B = (obs_traj_BT2[:, -1] - obs_traj_BT2[:, -3]).div(2).norm(p=2, dim=-1) > self.static_dist
        return mask_B

    def forward(self, obs_traj_BT2, adaptive_anchor_BKN, pred_traj_BT2=None, addl_info=None):
        n_ped = obs_traj_BT2.size(0)

        # (中略: 既存の静的/動的歩行者の分離処理、Projection処理は変更なし)
        mask_B = self.calculate_mask(obs_traj_BT2)
        obs_m_traj_BT2, obs_s_traj_BT2 = obs_traj_BT2[mask_B], obs_traj_BT2[~mask_B]
        pred_m_traj_gt_BT2 = pred_traj_BT2[mask_B] if pred_traj_BT2 is not None else None
        pred_s_traj_gt_BT2 = pred_traj_BT2[~mask_B] if pred_traj_BT2 is not None else None
        C_m_obs_KB, C_m_pred_gt_KB = self.Singular_space_m.projection(obs_m_traj_BT2, pred_m_traj_gt_BT2)
        C_s_obs_KB, C_s_pred_gt_KB = self.Singular_space_s.projection(obs_s_traj_BT2, pred_s_traj_gt_BT2)
        C_obs_KB = torch.zeros((self.k, n_ped), dtype=torch.float, device=obs_traj_BT2.device)
        C_obs_KB[:, mask_B], C_obs_KB[:, ~mask_B] = C_m_obs_KB, C_s_obs_KB
        obs_m_ori_2B = self.Singular_space_m.traj_normalizer.traj_ori_B12.squeeze(dim=1).T
        obs_s_ori_2B = self.Singular_space_s.traj_normalizer.traj_ori_B12.squeeze(dim=1).T
        obs_ori_2B = torch.zeros((2, n_ped), dtype=torch.float, device=obs_traj_BT2.device)
        obs_ori_2B[:, mask_B], obs_ori_2B[:, ~mask_B] = obs_m_ori_2B, obs_s_ori_2B
        obs_ori_2B -= obs_ori_2B.mean(dim=1, keepdim=True)
        C_anchor_KBN = adaptive_anchor_BKN.permute(1, 0, 2)
        addl_info["anchor"] = C_anchor_KBN.clone()
        addl_info["original_obs_traj"] = obs_traj_BT2

        # Trajectory prediction
        input_data = self.hook_func.model_forward_pre_hook(C_obs_KB, obs_ori_2B, addl_info)
        
        # ### ★★★★★ ECAMの実行とコンテキストの注入 ★★★★★ ###
        # 1. baseline_modelを一度実行して、必要なmap_patchesを取得
        #    注: 本来はmap_patchesを事前に計算するのが望ましいが、既存のフック構造を尊重
        _, _, map_patches_B1HW = self.hook_func.model_forward(input_data, self.baseline_model, get_map_only=True)
        
        # 2. ECAMモジュールで環境コンテキストを計算
        ecam_context = self.ecam_module(obs_traj_BT2, map_patches_B1HW)
        
        # 3. 計算したコンテキストをbaseline_modelへの入力に追加
        input_data['ecam_context'] = ecam_context
        # ### 修正ここまで ###
        
        # baseline_modelの本格的な順伝播
        output_data_BNK1, context_BK, _ = self.hook_func.model_forward(input_data, self.baseline_model)
        
        C_pred_refine_KBN = self.hook_func.model_forward_post_hook(output_data_BNK1, addl_info) * 0.1
        C_m_pred_KBN = self.adaptive_anchor_m(C_pred_refine_KBN[:, mask_B], C_anchor_KBN[:, mask_B])
        C_s_pred_KBN = self.adaptive_anchor_s(C_pred_refine_KBN[:, ~mask_B], C_anchor_KBN[:, ~mask_B])
        
        # (中略: Reconstructionと損失計算は変更なし)
        pred_m_traj_recon_NBT2 = self.Singular_space_m.reconstruction(C_m_pred_KBN)
        pred_s_traj_recon_NBT2 = self.Singular_space_s.reconstruction(C_s_pred_KBN)
        pred_traj_recon_NBT2 = torch.zeros((self.s, n_ped, self.t_pred, self.dim), dtype=torch.float, device=obs_traj_BT2.device)
        pred_traj_recon_NBT2[:, mask_B], pred_traj_recon_NBT2[:, ~mask_B] = pred_m_traj_recon_NBT2, pred_s_traj_recon_NBT2
        output = {"recon_traj": pred_traj_recon_NBT2}
        
        if pred_traj_BT2 is not None:
            # ... (既存の損失計算ロジック) ...
            C_pred_KBN = torch.zeros((self.k, n_ped, self.s), dtype=torch.float, device=obs_traj_BT2.device)
            C_pred_KBN[:, mask_B], C_pred_KBN[:, ~mask_B] = C_m_pred_KBN, C_s_pred_KBN
            C_pred_gt_KB = torch.zeros((self.k, n_ped), dtype=torch.float, device=obs_traj_BT2.device)
            C_pred_gt_KB[:, mask_B], C_pred_gt_KB[:, ~mask_B] = C_m_pred_gt_KB, C_s_pred_gt_KB
            C_pred_gt_KB = C_pred_gt_KB.detach()
            error_coefficient = (C_pred_KBN - C_pred_gt_KB.unsqueeze(dim=-1)).norm(p=2, dim=0)
            error_displacement_NBT = (pred_traj_recon_NBT2 - pred_traj_BT2.unsqueeze(dim=0)).norm(p=2, dim=-1)
            output["loss_eigentraj"] = error_coefficient.min(dim=-1)[0].mean()
            output["loss_euclidean_ade"] = error_displacement_NBT.mean(dim=-1).min(dim=0)[0].mean()
            output["loss_euclidean_fde"] = error_displacement_NBT[:, :, -1].min(dim=0)[0].mean()
            output["loss_diversity"] = torch.tensor(0.0, device=obs_traj_BT2.device)
            traj_BT2 = torch.cat([obs_traj_BT2, pred_traj_BT2], dim=1)
            
            if self.hyper_params.baseline_use_map:
                map_nce_loss = self.map_nce(traj_BT2, context_BK, map_patches_B1HW)
                output["loss_map_nce"] = map_nce_loss
                env_collision_loss_total = 0
                scene_ids_B = addl_info["scene_ids"]
                maps_dict = addl_info["maps"]
                homography_dict = addl_info["homography"]
                vector_field_dict = addl_info["vector_field"]
                for dataset_name in maps_dict.keys():
                    map_mask_1HW = (maps_dict[dataset_name]).to(device=obs_traj_BT2.device) * 255
                    hom_meters2mask = torch.from_numpy(homography_dict[dataset_name]["meters2mask"]).to(obs_traj_BT2.device)
                    hom_meters2image = torch.from_numpy(homography_dict[dataset_name]["meters2image"]).to(obs_traj_BT2.device)
                    hom_image2meters = torch.from_numpy(homography_dict[dataset_name]["image2meters"]).to(obs_traj_BT2.device)
                    vector_field = torch.from_numpy(vector_field_dict[dataset_name]).to(obs_traj_BT2.device)
                    img_size = torch.tensor(vector_field.shape[1::-1], device=obs_traj_BT2.device) // 2
                    scene_pred_BT2 = pred_traj_BT2[scene_ids_B == dataset_name]
                    scene_pred_hat_NBT2 = pred_traj_recon_NBT2[:, scene_ids_B == dataset_name]
                    scene_pred_hat_AT2 = scene_pred_hat_NBT2.view(-1, self.t_pred, 2)
                    if scene_pred_BT2.size(0) == 0:
                        continue
                    mode = self.hyper_params.env_col_loss_mode
                    env_collision_loss = self.compute_env_col_loss(scene_pred_BT2, scene_pred_hat_AT2, map_mask_1HW, hom_meters2mask, hom_meters2image, hom_image2meters, img_size, vector_field, mode=mode)
                    env_collision_loss_total += env_collision_loss
                output["loss_env_collision"] = env_collision_loss_total
            else:
                output["loss_map_nce"] = torch.tensor(0.0, device=obs_traj_BT2.device)
                output["loss_env_collision"] = torch.tensor(0.0, device=obs_traj_BT2.device)
        return output

    # (generate_artificial_gt, compute_env_col_loss, etc. は変更なし)
    @torch.no_grad()
    def generate_artificial_gt(self, scene_pred_hat_AT2, vector_field, map_mask_1HW, hom_meters2mask, hom_meters2image, hom_image2meters, img_size, min_margin, max_margin):
        traj_image_AT2 = hm.project(scene_pred_hat_AT2, hom_meters2image).int()
        traj_image_AT2 = torch.clamp(traj_image_AT2, min=-img_size//2, max=img_size + img_size//2 - 1)
        idx_h = traj_image_AT2[:, :, 1] + img_size[1] // 2
        idx_w = traj_image_AT2[:, :, 0] + img_size[0] // 2
        closest_valid_pos_img_AT2 = vector_field[idx_h, idx_w]
        closest_valid_pos_img_AT2 = closest_valid_pos_img_AT2.flip(2) - img_size // 2
        closest_valid_pos_AT2 = hm.project(closest_valid_pos_img_AT2, hom_image2meters)
        displ_AT2 = (closest_valid_pos_AT2 - scene_pred_hat_AT2).float()
        norm_AT1 = displ_AT2.norm(p=2, dim=-1, keepdim=True)
        almost_zero = torch.isclose(norm_AT1, torch.zeros_like(norm_AT1), atol=1e-1)
        norm_AT1[almost_zero] = 1
        dir_AT2 = displ_AT2 / norm_AT1
        dir_AT2[almost_zero.expand_as(dir_AT2)] = 0
        artificial_gt_AT2 = (scene_pred_hat_AT2 + displ_AT2).float()
        margin_range = max_margin - min_margin
        margin_A = torch.rand(scene_pred_hat_AT2.size(0), device=scene_pred_hat_AT2.device) * margin_range + min_margin
        margin_AT2 = dir_AT2 * margin_A[:, None, None]
        artificial_gt_AT2 = artificial_gt_AT2 + margin_AT2
        artificial_gt_aug_AT2 = model_utils.augment_traj_resolution(artificial_gt_AT2, parts=1)
        env_gt_collisions_A = model_utils.check_env_collisions(artificial_gt_aug_AT2, map_mask_1HW, torch.eye(3).to(artificial_gt_AT2.device), hom_meters2mask)
        valid_env_gt_collisions_A = ~env_gt_collisions_A
        return artificial_gt_AT2, valid_env_gt_collisions_A

    def compute_env_col_loss(self, scene_pred_BT2, scene_pred_hat_AT2, map_mask_1HW, hom_meters2mask, hom_meters2image, hom_image2meters, img_size, vector_field, mode: Literal["true-gt", "synth-gt"]):
        env_collisions_AP = model_utils.check_env_collisions_precise(scene_pred_hat_AT2, map_mask_1HW, torch.eye(3).to(scene_pred_hat_AT2.device), hom_meters2mask)
        if mode == "synth-gt":
            env_collisions_A = env_collisions_AP.any(dim=-1)
            up_to_first_col_included_AP = env_collisions_AP.cumsum(dim=-1) <= 1
            min_margin = self.hyper_params.env_col_loss_synth_gt_min_margin
            max_margin = self.hyper_params.env_col_loss_synth_gt_max_margin
            artificial_gt_AT2, valid_env_gt_collisions_A = self.generate_artificial_gt(scene_pred_hat_AT2, vector_field, map_mask_1HW, hom_meters2mask, hom_meters2image, hom_image2meters, img_size, min_margin=min_margin, max_margin=max_margin)
            fake_env_collisions_AP = env_collisions_A.unsqueeze(dim=-1).expand_as(env_collisions_AP)
            loss_mask_AT = fake_env_collisions_AP & up_to_first_col_included_AP
            gt_C2 = artificial_gt_AT2[loss_mask_AT]
            env_collision_loss = torch.tensor(0.0, device=scene_pred_hat_AT2.device)
            if scene_pred_hat_AT2[loss_mask_AT].size(0) > 0:
                env_collision_loss = F.mse_loss(scene_pred_hat_AT2[loss_mask_AT], gt_C2, reduction='mean')
                if torch.isnan(env_collision_loss):
                    env_collision_loss = torch.tensor(0.0, device=env_collision_loss.device)
        else:
            env_collisions_A = env_collisions_AP.any(dim=-1)
            env_collisions_NB = env_collisions_A.view(self.s, -1)
            loss_mask_A = env_collisions_A
            _, gt_index_A = torch.where(env_collisions_NB)
            gt_AT2 = scene_pred_BT2[gt_index_A]
