import math
import torch
import torch.nn as nn
from torch.nn import Module, Linear
import numpy as np
from .layers import PositionalEncoding, ConcatSquashLinear

from baseline.transformerdiffusion.mask_autoenc.mask_autoencoder import PatchEncoder
import baseline.transformerdiffusion.model_utils as model_utils


class st_encoder(nn.Module):
    """Transformer Denoising Model
    codebase borrowed from https://github.com/MediaBrain-SJTU/LED"""
    def __init__(self):
        super().__init__()
        channel_in, channel_out, dim_kernel = 2, 32, 3
        self.dim_embedding_key = 256
        self.spatial_conv = nn.Conv1d(channel_in, channel_out, dim_kernel, stride=1, padding=1)
        self.temporal_encoder = nn.GRU(channel_out, self.dim_embedding_key, 1, batch_first=True)
        self.relu = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.spatial_conv.weight)
        nn.init.kaiming_normal_(self.temporal_encoder.weight_ih_l0)
        nn.init.kaiming_normal_(self.temporal_encoder.weight_hh_l0)
        nn.init.zeros_(self.spatial_conv.bias)
        nn.init.zeros_(self.temporal_encoder.bias_ih_l0)
        nn.init.zeros_(self.temporal_encoder.bias_hh_l0)

    def forward(self, X):
        X_t = torch.transpose(X, 1, 2)
        X_after_spatial = self.relu(self.spatial_conv(X_t))
        X_embed = torch.transpose(X_after_spatial, 1, 2)
        _, state_x = self.temporal_encoder(X_embed)
        return state_x.squeeze(0)


class social_transformer(nn.Module):
    """Transformer Denoising Model
    codebase borrowed from https://github.com/MediaBrain-SJTU/LED"""
    # ### ★★★★★ 修正点 ★★★★★ ###
    # Transformerの層数とヘッド数を設定可能にする
    def __init__(self, cfg, additional_dim, num_layers=2, num_heads=2):
        super(social_transformer, self).__init__()
        
        self.encode_past = nn.Linear(cfg.k*cfg.s+cfg.k+2 + additional_dim, 256, bias=False)
        
        # 設定ファイルから層数とヘッド数を指定できるように変更
        self.layer = nn.TransformerEncoderLayer(d_model=256, nhead=num_heads, dim_feedforward=256, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=num_layers)

    def forward(self, h_B_Ktotal, mask_BB):
        h_feat_B1K = self.encode_past(h_B_Ktotal).unsqueeze(1)
        h_feat__B1K = self.transformer_encoder(h_feat_B1K, mask_BB)
        h_feat_B1K = h_feat_B1K + h_feat__B1K
        return h_feat_B1K


class TransformerDenoisingModel(Module):
    """Transformer Denoising Model
    codebase borrowed from https://github.com/MediaBrain-SJTU/LED"""
    def __init__(self, context_dim=256, cfg=None):
        super().__init__()
        assert cfg is not None, "cfg must be provided"

        ECAM_CONTEXT_DIM = 32 
        MAP_EMB_DIM = 32 if cfg.baseline_use_map else 0
        
        additional_dim = 0
        if cfg.baseline_use_map:
            additional_dim += MAP_EMB_DIM
        additional_dim += ECAM_CONTEXT_DIM

        self.context_dim = context_dim
        self.spatial_dim = 1
        self.temporal_dim = cfg.k
        self.n_samples = cfg.s
        
        # ### ★★★★★ 修正点 ★★★★★ ###
        # 設定ファイルからTransformerのサイズを渡す
        num_layers = cfg.get('num_transformer_layers', 2) # デフォルトは2層
        num_heads = cfg.get('num_transformer_heads', 2)   # デフォルトは2ヘッド
        self.encoder_context = social_transformer(cfg, additional_dim, num_layers, num_heads)

        if cfg.baseline_use_map:
            self.map_encoder = PatchEncoder(64)
            checkpoint = torch.load("checkpoints/patch_enc_ps.ckpt", map_location="cpu", weights_only=True)
            encoder_weights = {k.replace("autoencoder.encoder.", ""): v for k, v in checkpoint["state_dict"].items() if k.startswith("autoencoder.encoder.")}
            self.map_encoder.load_state_dict(encoder_weights)
            self.map_encoder.requires_grad_(False)
            self.map_mlp = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, MAP_EMB_DIM))
            
        self.pos_emb = PositionalEncoding(d_model=2*context_dim, dropout=0.1, max_len=24)
        self.concat1 = ConcatSquashLinear(self.n_samples*self.spatial_dim*self.temporal_dim, 2*context_dim, context_dim+3)
        self.concat3 = ConcatSquashLinear(2*context_dim,context_dim,context_dim+3)
        self.concat4 = ConcatSquashLinear(context_dim,context_dim//2,context_dim+3)
        self.linear = ConcatSquashLinear(context_dim//2, self.n_samples*self.spatial_dim*self.temporal_dim, context_dim+3)

    def encode_context(self, context_B_Ktotal, mask_BB):
        mask_BB = mask_BB.float().masked_fill(mask_BB == 0, float('-inf')).masked_fill(mask_BB == 1, float(0.0))
        context_B1K = self.encoder_context(context_B_Ktotal, mask_BB)
        return context_B1K

    def generate_accelerate(self, x_BNK1, beta_B111, context_B1K):
        beta_B1 = beta_B111.view(beta_B111.size(0), 1)
        time_emb_B3 = torch.cat([beta_B1, torch.sin(beta_B1), torch.cos(beta_B1)], dim=-1)
        ctx_emb_BK = torch.cat([time_emb_B3, context_B1K.view(-1, self.context_dim*self.spatial_dim)], dim=-1)
        trans_BC = self.concat1.batch_generate(ctx_emb_BK, x_BNK1.view(-1, self.n_samples*self.temporal_dim*self.spatial_dim))
        trans_BC = self.concat3.batch_generate(ctx_emb_BK, trans_BC)
        trans_BC = self.concat4.batch_generate(ctx_emb_BK, trans_BC)
        return self.linear.batch_generate(ctx_emb_BK, trans_BC).view(-1, self.n_samples, self.temporal_dim, self.spatial_dim)

    def encode_maps(self, map_B1HW):
        map_emb_BK = self.map_encoder(map_B1HW)
        map_emb_BK = self.map_mlp(map_emb_BK)
        return map_emb_BK


class DiffusionModel(Module):
    """Transformer Denoising Model
    codebase borrowed from https://github.com/MediaBrain-SJTU/LED"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = TransformerDenoisingModel(context_dim=256, cfg=cfg)
        self.register_buffer("betas_D", self.make_beta_schedule(schedule=self.cfg.beta_schedule, n_timesteps=self.cfg.steps, start=self.cfg.beta_start, end=self.cfg.beta_end))
        self.register_buffer("alphas_D", 1 - self.betas_D)
        self.register_buffer("alphas_prod", torch.cumprod(self.alphas_D, dim=0))
        self.register_buffer("alphas_bar_sqrt", torch.sqrt(self.alphas_prod))
        self.register_buffer("one_minus_alphas_bar_sqrt", torch.sqrt(1 - self.alphas_prod))

    def make_beta_schedule(self, schedule: str = 'linear', n_timesteps: int = 1000, start: float = 1e-5, end: float = 1e-2) -> torch.Tensor:
        if schedule == 'linear': return torch.linspace(start, end, n_timesteps)
        elif schedule == "quad": return torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
        elif schedule == "sigmoid":
            betas = torch.linspace(-6, 6, n_timesteps)
            return torch.sigmoid(betas) * (end - start) + start

    def extract(self, input_D, t_B, x_BNK1):
        shape = x_BNK1.shape
        out_D = torch.gather(input_D, 0, t_B.to(input_D.device))
        reshape = [t_B.shape[0]] + [1] * (len(shape) - 1)
        return out_D.reshape(*reshape)

    def forward(self, past_traj_BK1, traj_mask_BB, loc_BNK1, maps_dict, homography_dict,
                scene_ids, orig_obs_traj_BT2, addl_info=None, get_map_only=False):
        
        map_masks = [maps_dict[scene_ids[i]] * 255 for i in range(len(scene_ids))]
        max_h = max(mask.shape[1] for mask in map_masks)
        max_w = max(mask.shape[2] for mask in map_masks)
        padded_masks = [F.pad(mask, (0, max_w - mask.shape[2], 0, max_h - mask.shape[1]), value=0) for mask in map_masks]
        map_masks_B1HW = torch.stack(padded_masks, dim=0)
        scene_transform_matrix_B33 = torch.eye(3, device=orig_obs_traj_BT2.device).unsqueeze(0).expand(orig_obs_traj_BT2.shape[0], -1, -1)
        hom_meters2mask_B33 = torch.stack([torch.from_numpy(homography_dict[scene_id]["meters2mask"]).to(orig_obs_traj_BT2.device) for scene_id in scene_ids], dim=0)
        patch_B1HW, _, _ = model_utils.extract_patches_batched(orig_obs_traj_BT2, map_masks_B1HW, scene_transform_matrix_B33, hom_meters2mask_B33, patch_size_px=100, back_dist_px=10)
        
        if get_map_only:
            return None, None, patch_B1HW

        map_emb_BK = self.model.encode_maps(patch_B1HW) if self.cfg.baseline_use_map else torch.zeros(loc_BNK1.shape[0], 0, device=loc_BNK1.device)
        
        ecam_context = addl_info.get('ecam_context', None)
        
        pred_traj_BNK1, context_BK = self.p_sample_forward(past_traj_BK1, traj_mask_BB, loc_BNK1, map_emb_BK, ecam_context)
        return pred_traj_BNK1, context_BK, patch_B1HW

    def p_sample(self, cur_y_BNK1, t, context_B1K):
        t_tensor = torch.tensor([t], device=cur_y_BNK1.device)
        beta_B111 = self.extract(self.betas_D, t_tensor.repeat(context_B1K.shape[0]), cur_y_BNK1)
        eps_theta = self.model.generate_accelerate(cur_y_BNK1, beta_B111, context_B1K)
        eps_factor = ((1 - self.extract(self.alphas_D, t_tensor, cur_y_BNK1)) / self.extract(self.one_minus_alphas_bar_sqrt, t_tensor, cur_y_BNK1))
        mean = (1 / self.extract(self.alphas_D, t_tensor, cur_y_BNK1).sqrt()) * (cur_y_BNK1 - (eps_factor * eps_theta))
        rng = torch.Generator(device=cur_y_BNK1.device).manual_seed(0)
        z_BNK1 = torch.normal(mean=0.0, std=1.0, size=cur_y_BNK1.shape, generator=rng, device=cur_y_BNK1.device)
        sigma_t = self.extract(self.betas_D, t_tensor, cur_y_BNK1).sqrt()
        return mean + sigma_t * z_BNK1 * 0.00001

    def p_sample_forward(self, x_BK1, mask_BB, loc_BNK1, map_emb_BK, ecam_context=None):
        rng = torch.Generator(device=x_BK1.device).manual_seed(0)
        cur_y_BNK1 = torch.normal(mean=0.0, std=1.0, size=loc_BNK1.shape, generator=rng, device=x_BK1.device)
        
        batch_size = x_BK1.shape[1]
        social_context_flat = x_BK1.permute(1, 0, 2).reshape(batch_size, -1)
        
        combined_context_list = [social_context_flat]
        if self.cfg.baseline_use_map:
            combined_context_list.append(map_emb_BK)
        if ecam_context is not None:
            combined_context_list.append(ecam_context)
            
        combined_context = torch.cat(combined_context_list, dim=1)
        context_B1K = self.model.encode_context(combined_context, mask_BB)
        
        for i in reversed(range(self.cfg.steps)):
            cur_y_BNK1 = self.p_sample(cur_y_BNK1, i, context_B1K)
            
        return cur_y_BNK1, context_B1K.squeeze(1)
