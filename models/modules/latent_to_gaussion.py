import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """简单的残差块，保持维度不变"""

    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(4, channels)  # 16通道用4组很合适
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.act = nn.SiLU()
        self.norm2 = nn.GroupNorm(4, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return x + h


class LatentVAE(nn.Module):
    """
    轻量级适配器：
    Input:  TransformerUNet 的 Latent (B, 16, H, W) -> 均值不为0，方差不为1
    Output: 标准高斯 Latent (B, 16, H, W) -> 均值0，方差1
    """

    def __init__(self, in_channels=16, latent_channels=16):
        super().__init__()

        # --- Encoder: 把特征变成分布 ---
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1),  # 稍微升维增加非线性能力
            nn.SiLU(),
            ResBlock(32),
            ResBlock(32),
            nn.Conv2d(32, latent_channels * 2, 3, 1, 1)  # 输出 mean 和 logvar
        )

        # --- Decoder: 把采样点还原回特征 ---
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 32, 3, 1, 1),
            nn.SiLU(),
            ResBlock(32),
            ResBlock(32),
            nn.Conv2d(32, in_channels, 3, 1, 1)  # 还原回原始 Transformer Latent 维度
        )

    def reparameterize(self, mean, logvar):
        """重参数化采样"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std
        else:
            return mean  # 推理时使用均值

    def forward(self, x):
        # 1. Encode
        moments = self.encoder(x)
        mean, logvar = torch.chunk(moments, 2, dim=1)

        # 2. Sample (这是真正的 Gaussian Latent，用于扩散模型)
        z = self.reparameterize(mean, logvar)

        # 3. Decode (还原尝试)
        recon_x = self.decoder(z)

        return recon_x, mean, logvar, z
    # 重建的内容