"""
Q-Former bridge module for fusing multi-modal features
(image, LiDAR, radar) into the LLM token space.

Inspired by BLIP-2's Q-Former: uses a set of learnable query tokens
that cross-attend to the multi-modal features, producing a fixed
number of output tokens that are projected into the LLM embedding space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention: queries attend to key-value pairs."""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, query, key_value, key_padding_mask=None):
        """
        Args:
            query: [B, Nq, D]
            key_value: [B, Nkv, D]
            key_padding_mask: [B, Nkv] bool, True = ignore
        Returns:
            [B, Nq, D]
        """
        B, Nq, D = query.shape
        Nkv = key_value.shape[1]
        H = self.num_heads
        
        q = self.q_proj(query).view(B, Nq, H, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).view(B, Nkv, H, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).view(B, Nkv, H, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, Nq, Nkv]
        
        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),  # [B, 1, 1, Nkv]
                float('-inf')
            )
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, Nq, D)
        return self.out_proj(out)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention."""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.mhca = MultiHeadCrossAttention(dim, num_heads, dropout)
    
    def forward(self, x):
        return self.mhca(x, x)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)


class QFormerBlock(nn.Module):
    """
    Single Q-Former block:
      1. Self-attention on query tokens
      2. Cross-attention from query tokens to multi-modal features
      3. Feed-forward network
    """
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.cross_attn = MultiHeadCrossAttention(dim, num_heads, dropout)
        self.ffn = FeedForward(dim, dropout=dropout)
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
    
    def forward(self, queries, features, key_padding_mask=None):
        """
        Args:
            queries: [B, Nq, D] learnable query tokens
            features: [B, Nf, D] multi-modal features (concat of image+lidar+radar)
            key_padding_mask: [B, Nf] bool
        Returns:
            updated queries: [B, Nq, D]
        """
        # Self-attention
        queries = queries + self.self_attn(self.norm1(queries))
        # Cross-attention to multi-modal features
        queries = queries + self.cross_attn(
            self.norm2(queries), 
            self.norm_kv(features),
            key_padding_mask
        )
        # FFN
        queries = queries + self.ffn(self.norm3(queries))
        return queries


class QFormerBridge(nn.Module):
    """
    Q-Former bridge module that fuses multi-modal features and projects
    them into the LLM token embedding space.
    
    Architecture:
      1. Project each modality feature to a shared dimension
      2. Add modality-specific type embeddings
      3. Concatenate all modality features
      4. Use learnable query tokens to cross-attend to the concatenated features
      5. Project query outputs to LLM embedding dimension
    
    Input:
      - Image features: [B, N_img, D_img] from DINOv3 (after spatial pooling)
      - LiDAR features: [B, D_lidar] from VoxelNeXt (global pooled)
      - Radar features: [B, D_radar] from RadarNeXt (global pooled)
    
    Output:
      - [B, num_query_tokens, D_llm] tokens to be prepended/merged into LLM input
    """
    def __init__(self,
                 image_dim: int = 768,       # DINOv3 base hidden_size
                 lidar_dim: int = 256,       # VoxelNeXt output dim
                 radar_dim: int = 256,       # RadarNeXt output dim
                 qformer_dim: int = 512,     # Q-Former internal dimension
                 llm_dim: int = 896,         # Qwen2.5-0.5B hidden_size
                 num_query_tokens: int = 64, # Number of output tokens
                 num_layers: int = 4,        # Number of Q-Former blocks
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        self.num_query_tokens = num_query_tokens
        self.qformer_dim = qformer_dim
        
        # Modality projection layers
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, qformer_dim),
            nn.LayerNorm(qformer_dim),
        )
        self.lidar_proj = nn.Sequential(
            nn.Linear(lidar_dim, qformer_dim),
            nn.LayerNorm(qformer_dim),
        )
        self.radar_proj = nn.Sequential(
            nn.Linear(radar_dim, qformer_dim),
            nn.LayerNorm(qformer_dim),
        )
        
        # Modality type embeddings
        self.image_type_embed = nn.Parameter(torch.randn(1, 1, qformer_dim) * 0.02)
        self.lidar_type_embed = nn.Parameter(torch.randn(1, 1, qformer_dim) * 0.02)
        self.radar_type_embed = nn.Parameter(torch.randn(1, 1, qformer_dim) * 0.02)
        
        # Learnable query tokens
        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, qformer_dim) * 0.02)
        
        # Q-Former blocks
        self.layers = nn.ModuleList([
            QFormerBlock(qformer_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection to LLM space
        self.output_proj = nn.Sequential(
            nn.Linear(qformer_dim, llm_dim),
            nn.LayerNorm(llm_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, 
                image_features: torch.Tensor,
                lidar_features: torch.Tensor,
                radar_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_features: [B, N_img, D_img] - DINOv3 patch features
            lidar_features: [B, D_lidar]      - VoxelNeXt global features
            radar_features: [B, D_radar]      - RadarNeXt global features
        
        Returns:
            output_tokens: [B, num_query_tokens, D_llm]
        """
        B = image_features.shape[0]
        
        # Project each modality to shared dimension
        img_tokens = self.image_proj(image_features)   # [B, N_img, qformer_dim]
        lid_tokens = self.lidar_proj(lidar_features.unsqueeze(1))  # [B, 1, qformer_dim]
        rad_tokens = self.radar_proj(radar_features.unsqueeze(1))  # [B, 1, qformer_dim]
        
        # Add modality type embeddings
        img_tokens = img_tokens + self.image_type_embed
        lid_tokens = lid_tokens + self.lidar_type_embed
        rad_tokens = rad_tokens + self.radar_type_embed
        
        # Concatenate all modality features
        # Image: ~256 tokens, LiDAR: 1 token, Radar: 1 token
        all_features = torch.cat([img_tokens, lid_tokens, rad_tokens], dim=1)  # [B, N_img+2, D]
        
        # Expand learnable queries for batch
        queries = self.query_tokens.expand(B, -1, -1)  # [B, num_query, D]
        
        # Apply Q-Former blocks
        for layer in self.layers:
            queries = layer(queries, all_features)
        
        # Project to LLM dimension
        output_tokens = self.output_proj(queries)  # [B, num_query, D_llm]
        
        return output_tokens


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Testing QFormerBridge...")
    bridge = QFormerBridge(
        image_dim=768,
        lidar_dim=256,
        radar_dim=256,
        qformer_dim=512,
        llm_dim=896,
        num_query_tokens=64,
        num_layers=4,
    ).to(device)
    
    # Simulate inputs
    img_feat = torch.randn(2, 256, 768).to(device)    # DINOv3 patches (256 = 16x16 - 1 + pad)
    lidar_feat = torch.randn(2, 256).to(device)        # VoxelNeXt global
    radar_feat = torch.randn(2, 256).to(device)        # RadarNeXt global
    
    out = bridge(img_feat, lidar_feat, radar_feat)
    print(f"  Output shape: {out.shape}")  # [2, 64, 896]
    print(f"  QFormer params: {sum(p.numel() for p in bridge.parameters()):,}")
