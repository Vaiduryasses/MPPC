import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class NoiseScheduler:
    """噪声调度器，用于扩散过程"""
    
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps
        
        # 线性噪声调度
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 计算扩散过程中需要的系数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # 设备标记，用于确保tensor在正确设备上
        self._device = None
        
    def to(self, device):
        """将调度器参数移动到指定设备"""
        self._device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        return self
        
    def _ensure_device(self, timesteps):
        """确保调度器参数在正确的设备上"""
        if self._device != timesteps.device:
            self.to(timesteps.device)
        
    def add_noise(self, x_start, noise, timesteps):
        """向原始数据添加噪声"""
        self._ensure_device(timesteps)
        
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1)
        
        return sqrt_alphas_cumprod * x_start + sqrt_one_minus_alphas_cumprod * noise
    
    def sample_timesteps(self, batch_size, device):
        """随机采样时间步"""
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device)


class TimestepEmbedding(nn.Module):
    """时间步嵌入"""
    
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, timesteps):
        half = self.dim // 2
        freqs = torch.exp(
            -np.log(self.max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if self.dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        
        return self.mlp(embedding)


class MultiScaleTokenExtractor(nn.Module):
    """多尺度Token提取器"""
    
    def __init__(self, embed_dim, scales=[1, 2, 4]):
        super().__init__()
        self.scales = scales
        self.embed_dim = embed_dim
        
        # 为每个尺度创建特征提取器
        self.scale_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embed_dim, embed_dim, kernel_size=scale, stride=scale, padding=0),
                nn.BatchNorm1d(embed_dim),
                nn.GELU(),
                nn.Conv1d(embed_dim, embed_dim, 1)
            ) for scale in scales
        ])
        
        # 尺度融合模块
        self.scale_fusion = nn.Sequential(
            nn.Linear(embed_dim * len(scales), embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, N, C] 输入特征
        Returns:
            multi_scale_features: [B, N, C] 多尺度融合特征
        """
        B, N, C = x.shape
        x_transposed = x.transpose(1, 2)  # [B, C, N]
        
        scale_features = []
        for i, extractor in enumerate(self.scale_extractors):
            scale_feat = extractor(x_transposed)  # [B, C, N//scale]
            
            # 上采样到原始尺寸
            if scale_feat.size(2) != N:
                scale_feat = F.interpolate(scale_feat, size=N, mode='linear', align_corners=False)
            
            scale_features.append(scale_feat.transpose(1, 2))  # [B, N, C]
        
        # 拼接所有尺度特征
        multi_scale_feat = torch.cat(scale_features, dim=-1)  # [B, N, C*scales]
        
        # 融合特征
        fused_features = self.scale_fusion(multi_scale_feat)  # [B, N, C]
        
        return fused_features + x  # 残差连接


class LightweightDiTBlock(nn.Module):
    """轻量级扩散Transformer块"""
    
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # 时间步条件化
        self.time_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim * 2)  # 用于scale和shift
        )
        
        # 自注意力
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        # MLP
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
        # 多尺度token提取
        self.multi_scale_extractor = MultiScaleTokenExtractor(dim)
        
    def forward(self, x, time_emb):
        """
        Args:
            x: [B, N, C] 输入特征
            time_emb: [B, C] 时间步嵌入
        """
        B, N, C = x.shape
        
        # 时间步条件化
        time_scale_shift = self.time_mlp(time_emb).view(B, 1, 2 * C)
        time_scale, time_shift = time_scale_shift.chunk(2, dim=-1)
        
        # 多尺度特征提取
        x_multi_scale = self.multi_scale_extractor(x)
        
        # 应用时间步条件
        x_conditioned = x_multi_scale * (1 + time_scale) + time_shift
        
        # 自注意力
        x_norm1 = self.norm1(x_conditioned)
        attn_out, _ = self.attn(x_norm1, x_norm1, x_norm1)
        x = x + attn_out
        
        # MLP
        x_norm2 = self.norm2(x)
        mlp_out = self.mlp(x_norm2)
        x = x + mlp_out
        
        return x


class TwoStageDiffusionModule(nn.Module):
    """两阶段扩散模块，集成到现有decoder中"""
    
    def __init__(self, embed_dim, num_layers=4, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_diffusion = True
        
        # 时间步嵌入
        self.time_embedding = TimestepEmbedding(embed_dim)
        
        # 粗生成阶段
        self.coarse_layers = nn.ModuleList([
            LightweightDiTBlock(embed_dim, num_heads) 
            for _ in range(num_layers // 2)
        ])
        
        # 细化阶段
        self.refine_layers = nn.ModuleList([
            LightweightDiTBlock(embed_dim, num_heads) 
            for _ in range(num_layers // 2)
        ])
        
        # 阶段间特征融合
        self.stage_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # 噪声调度器
        self.noise_scheduler = NoiseScheduler()
        
    def forward(self, x, timesteps=None, training=True):
        """
        Args:
            x: [B, N, C] 输入特征
            timesteps: [B] 时间步 (可选)
            training: 是否为训练模式
        """
        B, N, C = x.shape
        
        # 如果没有提供时间步，采样随机时间步（训练时）或使用0（推理时）
        if timesteps is None:
            if training:
                timesteps = self.noise_scheduler.sample_timesteps(B, x.device)
            else:
                timesteps = torch.zeros(B, dtype=torch.long, device=x.device)
        
        # 时间步嵌入
        time_emb = self.time_embedding(timesteps)
        
        # 粗生成阶段
        coarse_x = x
        for layer in self.coarse_layers:
            coarse_x = layer(coarse_x, time_emb)
        
        # 细化阶段
        refine_x = coarse_x
        for layer in self.refine_layers:
            refine_x = layer(refine_x, time_emb)
        
        # 阶段融合
        fused_features = torch.cat([coarse_x, refine_x], dim=-1)
        output = self.stage_fusion(fused_features)
        
        return output