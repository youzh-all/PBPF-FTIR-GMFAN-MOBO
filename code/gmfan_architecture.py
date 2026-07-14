from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1DBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        pool_size: int = 2,
        dropout: float = 0.2,
        use_residual: bool = True,
        use_channel_attention: bool = True,
        use_spatial_attention: bool = True,
        attention_reduction: int = 16,
    ) -> None:
        super().__init__()
        self.use_residual = use_residual
        self.use_channel_attention = use_channel_attention
        self.use_spatial_attention = use_spatial_attention

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(pool_size) if pool_size > 1 else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.residual_proj = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1)

        if use_channel_attention:
            self.channel_attn = ChannelAttention(out_channels, reduction=attention_reduction)
        if use_spatial_attention:
            self.spatial_attn = SpatialAttention(kernel_size=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.use_residual:
            x = x + self.residual_proj(residual)
        x = F.gelu(x)
        if self.use_channel_attention:
            x = self.channel_attn(x)
        if self.use_spatial_attention:
            x = self.spatial_attn(x)
        x = self.pool(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = torch.mean(x, dim=2)
        excite = F.relu(self.fc1(squeeze))
        excite = torch.sigmoid(self.fc2(excite)).unsqueeze(2)
        return x * excite


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([max_pool, avg_pool], dim=1)))
        return x * attn


class AttentionPooling(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.attention = nn.Linear(in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xt = x.transpose(1, 2)
        w = torch.softmax(self.attention(xt), dim=1)
        return torch.sum(xt * w, dim=1)


class EnhancedFTIREncoder(nn.Module):
    def __init__(self, latent_dim: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        channels = [64, 128, 256, 256]
        kernels = [15, 7, 7, 7]
        pools = [2, 2, 2, 2]

        self.input_proj = nn.Conv1d(1, channels[0], 1)
        self.blocks = nn.ModuleList()
        in_ch = channels[0]
        for out_ch, k, p in zip(channels[1:], kernels[1:], pools[1:]):
            self.blocks.append(
                Conv1DBlock(
                    in_ch,
                    out_ch,
                    kernel_size=k,
                    pool_size=p,
                    dropout=dropout,
                    use_residual=True,
                    use_channel_attention=True,
                    use_spatial_attention=True,
                    attention_reduction=16,
                )
            )
            in_ch = out_ch

        self.pooling = AttentionPooling(in_ch)
        self.fc = nn.Linear(in_ch, latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.pooling(x)
        x = self.fc(x)
        x = self.dropout(x)
        return x


class MetaEncoder(nn.Module):
    def __init__(self, input_dim: int = 11, hidden_dim: int = 128, latent_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class GatedFusion(nn.Module):
    def __init__(self, ftir_dim: int = 256, meta_dim: int = 64, output_dim: int = 256) -> None:
        super().__init__()
        self.ftir_proj = nn.Linear(ftir_dim, output_dim)
        self.meta_proj = nn.Linear(meta_dim, output_dim)
        self.gate_net = nn.Sequential(
            nn.Linear(ftir_dim + meta_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.Sigmoid(),
        )
        self.fusion_layer = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim),
        )
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, ftir_features: torch.Tensor, meta_features: torch.Tensor) -> torch.Tensor:
        ftir_proj = self.ftir_proj(ftir_features)
        meta_proj = self.meta_proj(meta_features)
        gate = self.gate_net(torch.cat([ftir_features, meta_features], dim=1))
        gated = gate * ftir_proj + (1.0 - gate) * meta_proj
        combined = torch.cat([gated, ftir_proj + meta_proj], dim=1)
        fused = self.fusion_layer(combined)
        return self.norm(fused + gated)


class AttentionBackbone(nn.Module):
    def __init__(self, meta_input_dim: int = 11) -> None:
        super().__init__()
        self.ftir_encoder = EnhancedFTIREncoder(latent_dim=256, dropout=0.2)
        self.meta_encoder = MetaEncoder(input_dim=meta_input_dim, hidden_dim=128, latent_dim=64, dropout=0.1)
        self.fusion = GatedFusion(ftir_dim=256, meta_dim=64, output_dim=256)

    def forward(self, ftir: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        z = self.ftir_encoder(ftir)
        m = self.meta_encoder(meta)
        return self.fusion(z, m)


class DSCAttentionModel(nn.Module):
    def __init__(self, dsc_coeff_dim: int, dsc_scalar_dim: int) -> None:
        super().__init__()
        self.backbone = AttentionBackbone()
        self.curve_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, dsc_coeff_dim),
        )
        self.scalar_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, dsc_scalar_dim),
        )

    def forward(self, ftir: torch.Tensor, meta: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.backbone(ftir, meta)
        return {"latent": latent, "curve_coeff": self.curve_head(latent), "scalar": self.scalar_head(latent)}


class DirectUTMAttentionModel(nn.Module):
    def __init__(self, utm_coeff_dim: int, utm_scalar_dim: int) -> None:
        super().__init__()
        self.backbone = AttentionBackbone()
        self.curve_head = nn.Sequential(
            nn.Linear(256, 160),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(160, utm_coeff_dim),
        )
        self.scalar_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, utm_scalar_dim),
        )

    def forward(self, ftir: torch.Tensor, meta: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.backbone(ftir, meta)
        return {"latent": latent, "curve_coeff": self.curve_head(latent), "scalar": self.scalar_head(latent)}


class HierarchicalUTMAttentionModel(nn.Module):
    def __init__(self, dsc_cond_dim: int, utm_coeff_dim: int, utm_scalar_dim: int) -> None:
        super().__init__()
        self.backbone = AttentionBackbone()
        self.cond_proj = nn.Sequential(
            nn.Linear(dsc_cond_dim, 64),
            nn.GELU(),
            nn.LayerNorm(64),
        )
        self.fuse = nn.Sequential(
            nn.Linear(256 + 64, 256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )
        self.curve_head = nn.Sequential(
            nn.Linear(256, 160),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(160, utm_coeff_dim),
        )
        self.scalar_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, utm_scalar_dim),
        )

    def forward(self, ftir: torch.Tensor, meta: torch.Tensor, dsc_cond: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.backbone(ftir, meta)
        cond = self.cond_proj(dsc_cond)
        fused = self.fuse(torch.cat([latent, cond], dim=1))
        return {"latent": fused, "curve_coeff": self.curve_head(fused), "scalar": self.scalar_head(fused)}
