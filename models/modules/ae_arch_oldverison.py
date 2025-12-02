import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numbers
from einops import rearrange
import clip


# ============================================================================
# Part 1: ControlFusionNet 核心组件 (CBAM, CrossAttention, TransformerBlock等)
# ============================================================================

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


# --- 1. Cross Attention 模块 (新增) ---
class Cross_attention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=16):
        super().__init__()
        self.n_head = n_head
        self.norm_A = nn.GroupNorm(norm_groups, in_channel)
        self.norm_B = nn.GroupNorm(norm_groups, in_channel)
        self.qkv_A = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out_A = nn.Conv2d(in_channel, in_channel, 1)

        self.qkv_B = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out_B = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, x_A, x_B):
        batch, channel, height, width = x_A.shape

        n_head = self.n_head
        head_dim = channel // n_head

        x_A = self.norm_A(x_A)
        qkv_A = self.qkv_A(x_A).view(batch, n_head, head_dim * 3, height, width)
        query_A, key_A, value_A = qkv_A.chunk(3, dim=2)

        x_B = self.norm_B(x_B)
        qkv_B = self.qkv_B(x_B).view(batch, n_head, head_dim * 3, height, width)
        query_B, key_B, value_B = qkv_B.chunk(3, dim=2)

        # Attention A: Query from B, Key from A (B looks at A)
        attn_A = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query_B, key_A
        ).contiguous() / math.sqrt(channel)
        attn_A = attn_A.view(batch, n_head, height, width, -1)
        attn_A = torch.softmax(attn_A, -1)
        attn_A = attn_A.view(batch, n_head, height, width, height, width)

        out_A = torch.einsum("bnhwyx, bncyx -> bnchw", attn_A, value_A).contiguous()
        out_A = self.out_A(out_A.view(batch, channel, height, width))
        out_A = out_A + x_A

        # Attention B: Query from A, Key from B (A looks at B)
        attn_B = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query_A, key_B
        ).contiguous() / math.sqrt(channel)
        attn_B = attn_B.view(batch, n_head, height, width, -1)
        attn_B = torch.softmax(attn_B, -1)
        attn_B = attn_B.view(batch, n_head, height, width, height, width)

        out_B = torch.einsum("bnhwyx, bncyx -> bnchw", attn_B, value_B).contiguous()
        out_B = self.out_B(out_B.view(batch, channel, height, width))
        out_B = out_B + x_B

        return out_A, out_B


# --- 2. CBAM 及其子模块 (新增) ---
class ChannelAttention(nn.Module):
    def __init__(self, n_channels_in, reduction_ratio):
        super(ChannelAttention, self).__init__()
        self.n_channels_in = n_channels_in
        self.reduction_ratio = reduction_ratio
        self.middle_layer_size = int(self.n_channels_in / float(self.reduction_ratio))

        self.bottleneck = nn.Sequential(
            nn.Linear(self.n_channels_in, self.middle_layer_size),
            nn.ReLU(),
            nn.Linear(self.middle_layer_size, self.n_channels_in)
        )

    def forward(self, x):
        kernel = (x.size()[2], x.size()[3])
        avg_pool = F.avg_pool2d(x, kernel)
        max_pool = F.max_pool2d(x, kernel)

        avg_pool = avg_pool.view(avg_pool.size()[0], -1)
        max_pool = max_pool.view(max_pool.size()[0], -1)

        avg_pool_bck = self.bottleneck(avg_pool)
        max_pool_bck = self.bottleneck(max_pool)

        pool_sum = avg_pool_bck + max_pool_bck

        sig_pool = torch.sigmoid(pool_sum)
        sig_pool = sig_pool.unsqueeze(2).unsqueeze(3)

        out = sig_pool.repeat(1, 1, kernel[0], kernel[1])
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size

        assert kernel_size % 2 == 1, "Odd kernel size required"
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size,
                              padding=int((kernel_size - 1) / 2))

    def forward(self, x):
        max_pool = self.agg_channel(x, "max")
        avg_pool = self.agg_channel(x, "avg")
        pool = torch.cat([max_pool, avg_pool], dim=1)
        conv = self.conv(pool)
        conv = conv.repeat(1, x.size()[1], 1, 1)
        att = torch.sigmoid(conv)
        return att

    def agg_channel(self, x, pool="max"):
        b, c, h, w = x.size()
        x = x.view(b, c, h * w)
        x = x.permute(0, 2, 1)
        if pool == "max":
            x = F.max_pool1d(x, c)
        elif pool == "avg":
            x = F.avg_pool1d(x, c)
        x = x.permute(0, 2, 1)
        x = x.view(b, 1, h, w)
        return x


class CBAM(nn.Module):
    def __init__(self, n_channels_in, reduction_ratio, kernel_size):
        super(CBAM, self).__init__()
        self.n_channels_in = n_channels_in
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size

        self.channel_attention = ChannelAttention(n_channels_in, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, f):
        chan_att = self.channel_attention(f)
        fp = chan_att * f
        spat_att = self.spatial_attention(fp)
        fpp = spat_att * fp
        return fpp


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x = F.gelu(x)
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):
        """
        Args:
            dim_in: 输入通道数
            dim_out: 输出通道数
            num_heads: Attention 的头数 (注意：这是针对 dim_out 的头数)
            ...
        """
        super(TransformerBlock, self).__init__()

        # 1. 维度投影 (Projection)
        # 如果 dim_in != dim_out，我们需要先用 1x1 卷积将维度统一到 dim_out
        # 否则 Transformer 内部的残差连接 x = x + attn(x) 无法相加
        if dim_in != dim_out:
            self.proj = nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=bias)
        else:
            self.proj = nn.Identity()

        # 2. 核心组件 (全部基于 dim_out 构建)
        # 因为经过 proj 后，特征流已经是 dim_out 了
        self.norm1 = LayerNorm(dim_out, LayerNorm_type)
        self.attn = Attention(dim_out, num_heads, bias)
        self.norm2 = LayerNorm(dim_out, LayerNorm_type)
        self.ffn = FeedForward(dim_out, ffn_expansion_factor, bias)

    def forward(self, x):
        # x shape: [B, dim_in, H, W]

        # 步骤 1: 维度对齐
        x = self.proj(x)
        # 现在 x shape: [B, dim_out, H, W]

        # 步骤 2: Transformer 计算 (Pre-Norm 结构)
        # 这里的残差连接是在 dim_out 维度上进行的
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class Downsample(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Downsample, self).__init__()
        # PixelUnshuffle(2) 会在空间下采样 2 倍的同时，将通道数扩大 4 倍。
        # 为了让最终输出通道数等于 dim_out，我们需要让卷积输出为 dim_out // 4。
        # 例如：dim_in=64, dim_out=128 -> Conv输出 32 -> Unshuffle输出 128 (符合预期)

        self.body = nn.Sequential(
            nn.Conv2d(dim_in, dim_out // 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )
        # PixelUnshuffle(2) 会将空间尺寸减半，通道数扩大 4 倍。

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Upsample, self).__init__()
        # PixelShuffle(2) 会在空间上采样 2 倍，同时将通道数减少为原来的 1/4。
        # 为了让最终输出通道数为 dim_out，卷积层需要输出 dim_out * 4 的通道数。
        self.body = nn.Sequential(
            nn.Conv2d(dim_in, dim_out * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=True):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        # 这里的 in_channels 是 CLIP context 的维度 (例如 768)
        # out_channels 是当前图像特征图的通道数
        self.MLP = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.LeakyReLU(),
            nn.Linear(in_channels * 2, out_channels * (1 + self.use_affine_level))
        )

    def forward(self, x, text_embed):
        # text_embed: [Batch, C_text] or [Batch, 1, C_text]
        if text_embed.dim() == 3:
            text_embed = text_embed.squeeze(1)

        batch = x.shape[0]
        if self.use_affine_level:
            # 预测 gamma 和 beta
            gamma, beta = self.MLP(text_embed).view(batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            beta = self.MLP(text_embed).view(batch, -1, 1, 1)
            x = x + beta
        return x


#  就是简单将图像的通道维度映射到embed_dim的维度；
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=64, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x

    # -------------------------------------------------------------------------


# 1. Residual (残差连接包装器)
# -------------------------------------------------------------------------
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


# -------------------------------------------------------------------------
# 2. PreNorm (前置归一化包装器)
# -------------------------------------------------------------------------
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        # 针对图像数据 (Batch, Channel, Height, Width)
        # 使用 GroupNorm(1, dim) 等价于 LayerNorm，但无需 permute 维度
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x, **kwargs):
        # 先进行归一化，再送入函数
        x = self.norm(x)
        return self.fn(x, **kwargs)


# -------------------------------------------------------------------------
# 3. LinearAttention (线性注意力机制)
# -------------------------------------------------------------------------
class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        # 生成 Q, K, V
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        # 输出投影
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            nn.GroupNorm(1, dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        # 1. 生成 Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim=1)

        # 2. 调整维度以适应多头注意力
        # b (h c) x y -> b h c (x y)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        # 3. 线性注意力核心技巧 (Linear Attention Trick)
        # 通过在特征维度(c)而非序列维度(n)上做 Softmax，避免生成 N*N 的注意力图
        # 公式: Q * (K^T * V)
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        # 4. 计算 Context (K^T * V) -> 形状 (b, h, c, c)
        # 这一步复杂度是 O(N * C^2)，通常 N(HW) >> C，所以比标准注意力的 O(N^2 * C) 快得多
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        # 5. 计算 Output (Q * Context) -> 形状 (b, h, c, n)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)

        # 6. 恢复维度
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)

        return self.to_out(out)


class CrossAttention(nn.Module):
    """
    标准的交叉注意力模块
    Query: 来自模态 A
    Key, Value: 来自模态 B
    """

    def __init__(self, dim, num_heads=4, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        # 确保维度能被头数整除，这是 Transformer 的硬性要求
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 线性投影层，使用 1x1 卷积实现，保留空间结构直到 reshape
        self.to_q = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.to_k = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.to_v = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_query, x_context):
        """
        x_query:   [B, C, H, W] -> 主动去查询的特征
        x_context: [B, C, H, W] -> 提供信息的特征 (Key/Value)
        """
        B, C, H, W = x_query.shape
        N = H * W

        # 1. 生成 Q, K, V
        # q: [B, Heads, C/Heads, N]
        q = self.to_q(x_query).view(B, self.num_heads, C // self.num_heads, N)
        k = self.to_k(x_context).view(B, self.num_heads, C // self.num_heads, N)
        v = self.to_v(x_context).view(B, self.num_heads, C // self.num_heads, N)

        # 2. 计算 Attention Score (Scaled Dot-Product)
        # q: [B, H, d, N] @ k.T: [B, H, N, d] -> attn: [B, H, N, N]
        # 这一步计算复杂度是 O(N^2)，如果图像很大 (如 256x256)，显存会爆。
        # 建议在下采样后的特征图 (如 32x32 或 64x64) 上使用。
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 3. 聚合 Value
        # attn: [B, H, N, N] @ v.T: [B, H, N, d] -> x: [B, H, N, d]
        x = (attn @ v.transpose(-2, -1)).transpose(-2, -1)

        # 4. 还原形状 [B, C, H, W]
        x = x.reshape(B, C, H, W)
        x = self.proj(x)
        x = self.proj_drop(x)

        # 这里返回的是"从 Context 中提取出的对 Query 有用的信息"
        return x


class _FeedForward(nn.Module):
    """
    Transformer 中的 FFN (Feed-Forward Network)，用于特征变换和非线性增强
    """

    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, dim, kernel_size=1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class BiModalFusionBlock(nn.Module):
    """
    双向交叉注意力融合模块
    """

    def __init__(self, dim, num_heads=4, mlp_ratio=4., drop=0.):
        super().__init__()

        # 1. 归一化层 (Pre-Norm)
        self.norm_ir = nn.GroupNorm(1, dim)  # 使用 GroupNorm 对 CNN 特征更友好
        self.norm_vis = nn.GroupNorm(1, dim)

        # 2. 双向交叉注意力
        # IR 查 VIS
        self.cross_attn_ir = CrossAttention(dim, num_heads=num_heads, attn_drop=drop, proj_drop=drop)
        # VIS 查 IR
        self.cross_attn_vis = CrossAttention(dim, num_heads=num_heads, attn_drop=drop, proj_drop=drop)

        # 3. FFN 部分 (可选，为了增强特征表达能力)
        self.norm_ir_ffn = nn.GroupNorm(1, dim)
        self.norm_vis_ffn = nn.GroupNorm(1, dim)
        self.ffn_ir = _FeedForward(dim, int(dim * mlp_ratio), dropout=drop)
        self.ffn_vis = _FeedForward(dim, int(dim * mlp_ratio), dropout=drop)

        # 4. 最终融合层
        # 将两路特征拼接后压缩回原始维度
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x_ir, x_vis):
        """
        输入:
            x_ir:  [B, C, H, W]
            x_vis: [B, C, H, W]
        输出:
            x_fused: [B, C, H, W]
        """

        # --- 分支 1: 增强 IR 特征 ---
        # IR 此时作为 Query，它想知道 VIS 里的哪些纹理对自己有用
        # 采用残差连接: 原 IR + (用 IR 去 VIS 里查到的信息)
        res_ir = x_ir
        ir_norm = self.norm_ir(x_ir)
        vis_norm = self.norm_vis(x_vis)

        feat_ir_enhanced = self.cross_attn_ir(x_query=ir_norm, x_context=vis_norm)
        x_ir = res_ir + feat_ir_enhanced

        # FFN
        x_ir = x_ir + self.ffn_ir(self.norm_ir_ffn(x_ir))

        # --- 分支 2: 增强 VIS 特征 ---
        # VIS 此时作为 Query，它想知道 IR 里的哪些热源对自己有用
        res_vis = x_vis
        # 注意：这里再次使用 norm 后的变量，或者重新 norm 取决于架构设计，这里复用上面的 norm 即可

        feat_vis_enhanced = self.cross_attn_vis(x_query=vis_norm, x_context=ir_norm)
        x_vis = res_vis + feat_vis_enhanced

        # FFN
        x_vis = x_vis + self.ffn_vis(self.norm_vis_ffn(x_vis))

        # --- 融合阶段 ---
        # 拼接两个互补增强后的特征
        concat_feat = torch.cat([x_ir, x_vis], dim=1)

        # 卷积融合
        x_fused = self.fusion_conv(concat_feat)

        return x_fused


def default_conv(dim_in, dim_out, kernel_size=3, bias=False):
    return nn.Conv2d(dim_in, dim_out, kernel_size, padding=(kernel_size // 2), bias=bias)


class Attention_spatial(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=16):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head
        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)
        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x


# 假设您已经有这些组件的实现：
# - TransformerBlock
# - CBAM
# - Attention_spatial (ControlFusionNet 中的 Attention_spatial)
# - OverlapPatchEmbed
# - Downsample, Upsample
# - Identity

class TransformerUNet(nn.Module):
    """
    优化后的 TransformerUNet

    改进点：
    1. 每层 Encoder 添加 CBAM 特征增强
    2. Bottleneck 使用 Attention_spatial 建模全局关系
    3. 保持单模态独立编码
    4. 支持交叉重建训练（decode 可接收不同来源的 latent 和 skips）
    """

    def __init__(self, in_ch=3, out_ch=3, ch=48, ch_mult=[1, 2, 4, 4], embed_dim=16,
                 ffn_expansion_factor=2.0, bias=False, LayerNorm_type='WithBias'):
        """
        Args:
            in_ch: 输入图像通道
            out_ch: 输出图像通道
            ch: 基础通道数
            ch_mult: 通道倍率列表
            embed_dim: Latent 的维度 (对应原 UNet 的 embed_dim)
            ffn_expansion_factor: FeedForward 的扩展因子
            bias: 是否使用 bias
            LayerNorm_type: LayerNorm 类型
        """
        super().__init__()

        self.depth = len(ch_mult)

        # 初始卷积
        self.init_conv = OverlapPatchEmbed(in_c=in_ch, embed_dim=ch, bias=bias)

        # ====================================================================
        # Encoder: 使用 CBAM 增强特征
        # ====================================================================
        self.encoder = nn.ModuleList([])
        self.encoder_cbam = nn.ModuleList([])  # 每层的 CBAM

        ch_mult = [1] + ch_mult  # [1, 1, 2, 2, 4]

        for i in range(self.depth):
            dim_in = ch * ch_mult[i]
            dim_out = ch * ch_mult[i + 1]

            # CBAM 注意力模块
            self.encoder_cbam.append(
                CBAM(n_channels_in=dim_in, reduction_ratio=16, kernel_size=7)
            )

            # Encoder 块
            self.encoder.append(nn.ModuleList([
                # TransformerBlock 1
                TransformerBlock(
                    dim_in=dim_in,
                    dim_out=dim_in,
                    num_heads=max(1, dim_in // 48),
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type
                ),
                # TransformerBlock 2
                TransformerBlock(
                    dim_in=dim_in,
                    dim_out=dim_in,
                    num_heads=max(1, dim_in // 48),
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type
                ),
                # Downsample
                Downsample(dim_in, dim_out) if i != (self.depth - 1)
                else nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=bias)
            ]))

        # ====================================================================
        # Bottleneck: Spatial Attention + Latent 提取
        # ====================================================================
        mid_dim = ch * ch_mult[-1]

        # Spatial Attention (替代原来的 LinearAttention)
        self.spatial_attention = Attention_spatial(
            in_channel=mid_dim,
            n_head=1,
            norm_groups=min(16, mid_dim)
        )

        # Latent 空间
        self.latent_conv = nn.Conv2d(mid_dim, embed_dim, kernel_size=1, bias=bias)
        self.post_latent_conv = nn.Conv2d(embed_dim, mid_dim, kernel_size=1, bias=bias)

        # ====================================================================
        # Decoder: 支持交叉重建
        # ====================================================================
        self.decoder = nn.ModuleList([])

        for i in range(self.depth):
            dim_out = ch * ch_mult[self.depth - i]
            dim_in = ch * ch_mult[self.depth - i - 1]

            self.decoder.append(nn.ModuleList([
                # TransformerBlock 1 (接收 concat 后的特征)
                TransformerBlock(
                    dim_in=dim_out + dim_in,  # concat 维度
                    dim_out=dim_out,
                    num_heads=max(1, dim_out // 48),
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type
                ),
                # TransformerBlock 2
                TransformerBlock(
                    dim_in=dim_out + dim_in,
                    dim_out=dim_out,
                    num_heads=max(1, dim_out // 48),
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type
                ),
                # Upsample
                Upsample(dim_out, dim_in) if i != self.depth - 1
                else nn.Conv2d(dim_out, dim_in, kernel_size=1, bias=bias)
            ]))

        # 最终输出
        self.final_conv = nn.Conv2d(ch, out_ch, 3, 1, 1)

    # ========================================================================
    # 工具方法
    # ========================================================================

    def check_image_size(self, x, h, w):
        """确保图像尺寸可以被下采样"""
        s = int(math.pow(2, self.depth))
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    # ========================================================================
    # 核心方法：Encode 和 Decode
    # ========================================================================

    def encode(self, x):
        """
        编码器：提取 Latent 和多尺度特征

        Args:
            x: 输入图像 [B, 3, H, W]

        Returns:
            latent: [B, embed_dim, H/16, W/16] (退化信息)
            h: List of skip connections (内容信息)
        """
        h, w = x.shape[2:]
        self.H, self.W = h, w

        # Padding
        x = self.check_image_size(x, h, w)

        # 初始卷积
        x = self.init_conv(x)

        # 保存 skip connections
        skips = [x]

        # Encoder 逐层处理
        for i, (b1, b2, downsample) in enumerate(self.encoder):
            # *** 关键改进 1: CBAM 特征增强 ***
            x = self.encoder_cbam[i](x)

            # TransformerBlock 1
            x = b1(x)
            skips.append(x)

            # TransformerBlock 2
            x = b2(x)
            skips.append(x)

            # Downsample
            x = downsample(x)

        # *** 关键改进 2: Spatial Attention 建模全局关系 ***
        x = self.spatial_attention(x)

        # 提取 Latent
        x = self.latent_conv(x)

        return x, skips

    def insight_data(self, h, hs):
        print(f'h.shape: {h.shape}')
        print(f'h.range: {h.min()} to {h.max()}')
        for i, layer in enumerate(hs):
            print(f'hs[{i}] shape: {layer.shape}')
            print(f'hs[{i}] range: {layer.min()} to {layer.max()}')
        print('=' * 70)
        return

    def insight(self, x, name):
        print(f'name.shape: {x.shape}')
        print(f'name.range: {x.min()} to {x.max()}')
        print('=' * 70)
        return

    def decode(self, x, h):
        """
        解码器：从 Latent + Skips 重建图像

        Args:
            x: Latent 特征 [B, embed_dim, H/16, W/16]
            h: Skip connections (可以来自不同模态，用于交叉重建)

        Returns:
            output: 重建图像 [B, out_ch, H, W]
        """
        self.insight_data(x, h)

        self.insight(x, 'latent feature')

        x = self.post_latent_conv(x)

        self.insight(x, 'post latent feature')

        # Decoder 逐层处理
        for i, (b1, b2, upsample) in enumerate(self.decoder):
            # Concat skip connection 1
            x = torch.cat([x, h[-(i * 2 + 1)]], dim=1)
            x = b1(x)

            # Concat skip connection 2
            x = torch.cat([x, h[-(i * 2 + 2)]], dim=1)
            x = b2(x)

            # Upsample
            x = upsample(x)

        # 最终输出
        x = self.final_conv(x + h[0])

        # 裁剪到原始尺寸
        return x[..., :self.H, :self.W]

    # ========================================================================
    # 前向传播
    # ========================================================================

    def forward(self, x):
        """
        前向传播：完整的编码-解码流程

        Args:
            x: 输入图像 [B, 3, H, W]

        Returns:
            output: 重建图像 [B, 3, H, W]
        """
        latent, skips = self.encode(x)
        # print(f'latent shape: {latent.shape}')
        # for i,layer in enumerate(skips):
        #     print(f'skips[{i}] shape: {layer.shape}')

        output = self.decode(latent, skips)
        return output


# ============================================================================
# 使用示例
# ============================================================================

def load_model_checkpoint(model, checkpoint_path):
    """加载模型权重"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    print(f"✓ 模型权重已加载: {checkpoint_path}")
    return model


def load_image_to_x(
        image_path=r'E:\Recurrence\OmniFuse\DDL-12\IR_Low_contrast\IR_Low_contrast_slight\train\Infrared\00019N.png',
        device='cuda:0'):
    from PIL import Image
    from torchvision import transforms
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)  # 添加 batch 维度
    return img_tensor


if __name__ == "__main__":
    device = "cpu"

    # ====================================================================
    # 1. 创建模型
    # ====================================================================
    print("=" * 70)
    print("创建优化后的 TransformerUNet")
    print("=" * 70)

    model = TransformerUNet(
        in_ch=3,
        out_ch=3,
        ch=48,
        ch_mult=[1, 2, 4, 4],
        embed_dim=16,  # 对应原始 UNet 的 embed_dim
        ffn_expansion_factor=2,
        bias=False,
        LayerNorm_type='WithBias'
    ).to(device)

    model=load_model_checkpoint(model,r'E:\Recurrence\OmniFuse\pretrained\56025_G.pth')
    # model = model.to(device)

    print(f"✓ 模型创建成功")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # ====================================================================
    # 2. 测试单模态编码-解码
    # ====================================================================
    print("\n" + "=" * 70)
    print("测试 1: 单模态编码-解码")
    print("=" * 70)

    # vis_lq = torch.rand(1, 3, 512, 512).to(device)
    vis_lq = load_image_to_x(
        image_path=r'E:\Recurrence\OmniFuse\DDL-12\VI_Noise_Low\test\Infrared\010225.png',
        device='cpu')
    # 前向传播
    # output = model(vis_lq)
    h,hs= model.encode(vis_lq)
    from utils.yaml_utils import plot_tensor_hist
    # plot_tensor_hist()

    for i in range(h.shape[1]):
        ret=plot_tensor_hist(h[0,i,:,:].unsqueeze(0),(-8,8),f'latent channel {i} histogram')
        print(f"layer{i}, mean: {ret['mean']}, std: {ret['std']}, max: {ret['max']}, min: {ret['min']}")


    print(f"输入: {vis_lq.shape}")
    # print(f"输出: {output.shape}")
    print(f"✓ 单模态测试通过")







    # # ====================================================================
    # # 3. 测试交叉重建（您的训练流程）
    # # ====================================================================
    # print("\n" + "=" * 70)
    # print("测试 2: 交叉重建训练流程")
    # print("=" * 70)
    #
    # # 模拟数据
    # X_LQ = torch.randn(2, 3, 128, 128).to(device)  # Vis LQ
    # X_GT = torch.randn(2, 3, 128, 128).to(device)  # Vis GT
    # Y_LQ = torch.randn(2, 3, 128, 128).to(device)  # IR LQ
    # Y_GT = torch.randn(2, 3, 128, 128).to(device)  # IR GT
    #
    # # 编码
    # X_L_lq, X_H_lq = model.encode(X_LQ)
    # X_L_gt, X_H_gt = model.encode(X_GT)
    # Y_L_lq, Y_H_lq = model.encode(Y_LQ)
    # Y_L_gt, Y_H_gt = model.encode(Y_GT)
    #
    # print("编码结果:")
    # print(f"  Vis LQ Latent: {X_L_lq.shape}, Skips: {len(X_H_lq)}")
    # print(f"  Vis GT Latent: {X_L_gt.shape}, Skips: {len(X_H_gt)}")
    # print(f"  IR LQ Latent:  {Y_L_lq.shape}, Skips: {len(Y_H_lq)}")
    # print(f"  IR GT Latent:  {Y_L_gt.shape}, Skips: {len(Y_H_gt)}")
    #
    # # 同模态重建
    # X_rec_llq_hlq = model.decode(X_L_lq, X_H_lq)  # Vis: LQ → LQ
    # X_rec_lgt_hgt = model.decode(X_L_gt, X_H_gt)  # Vis: GT → GT
    #
    # # 交叉重建
    # X_rec_llq_hgt = model.decode(X_L_lq, X_H_gt)  # LQ 退化 + GT 内容
    # X_rec_lgt_hlq = model.decode(X_L_gt, X_H_lq)  # GT 退化 + LQ 内容
    #
    # # 跨模态重建
    # rec_X_llq_Y_hgt = model.decode(X_L_lq, Y_H_gt)  # Vis 退化 + IR 内容
    # rec_Y_llq_X_hgt = model.decode(Y_L_lq, X_H_gt)  # IR 退化 + Vis 内容
    #
    # print("\n重建结果:")
    # print(f"  ✓ 同模态重建: {X_rec_llq_hlq.shape}")
    # print(f"  ✓ 交叉重建: {X_rec_llq_hgt.shape}")
    # print(f"  ✓ 跨模态重建: {rec_X_llq_Y_hgt.shape}")
    #
    # # ====================================================================
    # # 4. 模拟完整训练步骤
    # # ====================================================================
    # print("\n" + "=" * 70)
    # print("测试 3: 完整训练步骤（损失计算）")
    # print("=" * 70)
    #
    # # 损失函数
    # loss_fn = nn.L1Loss()
    #
    # # X 模态重建损失
    # X_loss_rec = (
    #         loss_fn(X_rec_llq_hlq, X_LQ) +
    #         loss_fn(X_rec_lgt_hgt, X_GT) +
    #         loss_fn(X_rec_lgt_hlq, X_GT) +
    #         loss_fn(X_rec_llq_hgt, X_LQ)
    # )
    #
    # # Y 模态重建损失
    # Y_rec_llq_hlq = model.decode(Y_L_lq, Y_H_lq)
    # Y_rec_lgt_hgt = model.decode(Y_L_gt, Y_H_gt)
    # Y_rec_lgt_hlq = model.decode(Y_L_gt, Y_H_lq)
    # Y_rec_llq_hgt = model.decode(Y_L_lq, Y_H_gt)
    #
    # Y_loss_rec = (
    #         loss_fn(Y_rec_llq_hlq, Y_LQ) +
    #         loss_fn(Y_rec_lgt_hgt, Y_GT) +
    #         loss_fn(Y_rec_lgt_hlq, Y_GT) +
    #         loss_fn(Y_rec_llq_hgt, Y_LQ)
    # )
    #
    # # 跨模态重建损失
    # rec_X_lgt_Y_hlq = model.decode(X_L_gt, Y_H_lq)
    # rec_X_lgt_Y_hgt = model.decode(X_L_gt, Y_H_gt)
    # rec_X_llq_Y_hlq = model.decode(X_L_lq, Y_H_lq)
    #
    # X_Y_loss_rec = (
    #         loss_fn(rec_X_llq_Y_hlq, X_LQ) +
    #         loss_fn(rec_X_lgt_Y_hlq, X_GT) +
    #         loss_fn(rec_X_lgt_Y_hgt, X_GT) +
    #         loss_fn(rec_X_llq_Y_hgt, X_LQ)
    # )
    #
    # rec_Y_lgt_X_hlq = model.decode(Y_L_gt, X_H_lq)
    # rec_Y_lgt_X_hgt = model.decode(Y_L_gt, X_H_gt)
    # rec_Y_llq_X_hlq = model.decode(Y_L_lq, X_H_lq)
    #
    # Y_X_loss_rec = (
    #         loss_fn(rec_Y_llq_X_hlq, Y_LQ) +
    #         loss_fn(rec_Y_lgt_X_hlq, Y_GT) +
    #         loss_fn(rec_Y_lgt_X_hgt, Y_GT) +
    #         loss_fn(rec_Y_llq_X_hgt, Y_LQ)
    # )
    #
    # total_loss = X_loss_rec + Y_loss_rec + X_Y_loss_rec + Y_X_loss_rec
    #
    # print("损失统计:")
    # print(f"  X_loss_rec:   {X_loss_rec.item():.6f}")
    # print(f"  Y_loss_rec:   {Y_loss_rec.item():.6f}")
    # print(f"  X_Y_loss_rec: {X_Y_loss_rec.item():.6f}")
    # print(f"  Y_X_loss_rec: {Y_X_loss_rec.item():.6f}")
    # print(f"  Total Loss:   {total_loss.item():.6f}")
    #
    # # ====================================================================
    # # 5. 总结
    # # ====================================================================
    # print("\n" + "=" * 70)
    # print("✓ 所有测试通过！")
    # print("=" * 70)
    # print("\n关键改进总结:")
    # print("  1. ✅ 每层 Encoder 添加 CBAM 特征增强")
    # print("  2. ✅ Bottleneck 使用 Spatial Attention 替代 LinearAttention")
    # print("  3. ✅ 保持单模态独立编码")
    # print("  4. ✅ 完全兼容您的 optimize_parameters 训练流程")
    # print("  5. ✅ 不改变原有接口签名 encode(x) → (latent, skips)")
