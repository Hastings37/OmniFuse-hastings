import torch
import torch.nn as nn
import torch.nn.functional as F
import math



from models.modules.ae_arch_oldverison import *


# 假设这些组件已经在你的环境中定义，保持引用不变
# from ae_arch import OverlapPatchEmbed, CBAM, TransformerBlock, Downsample, Attention_spatial, Upsample, ...
# 如果是在同一个文件里，不需要改动导入部分
class TransformerUNet(nn.Module):
    """
    【最终优化版】TransformerUNet

    针对扩散模型训练的优化：
    1. [Fix] Encoder 末尾添加 GroupNorm，解决 Transformer 导致的 Latent 方差爆炸问题。
    2. [Feat] 保留 CBAM 和 Spatial Attention 增强特征提取能力。
    3. [Mode] 确定性 Latent 输出 (非 VAE 概率采样)，适合作为 Conditional 输入。
    """

    def __init__(self, in_ch=3, out_ch=3, ch=48, ch_mult=[1, 2, 4, 4], embed_dim=8,
                 ffn_expansion_factor=2.0, bias=False, LayerNorm_type='WithBias'):
        super().__init__()

        self.depth = len(ch_mult)

        # 1. 初始卷积
        self.init_conv = OverlapPatchEmbed(in_c=in_ch, embed_dim=ch, bias=bias)

        # 2. Encoder: 使用 CBAM 增强特征
        self.encoder = nn.ModuleList([])
        self.encoder_cbam = nn.ModuleList([])

        ch_mult = [1] + ch_mult  # e.g., [1, 1, 2, 4, 4]

        for i in range(self.depth):
            dim_in = ch * ch_mult[i]
            dim_out = ch * ch_mult[i + 1]

            # CBAM 注意力模块
            self.encoder_cbam.append(
                CBAM(n_channels_in=dim_in, reduction_ratio=16, kernel_size=7)
            )

            # Encoder 块
            self.encoder.append(nn.ModuleList([
                TransformerBlock(
                    dim_in=dim_in, dim_out=dim_in,
                    num_heads=max(1, dim_in // 48),
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias, LayerNorm_type=LayerNorm_type
                ),
                TransformerBlock(
                    dim_in=dim_in, dim_out=dim_in,
                    num_heads=max(1, dim_in // 48),
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias, LayerNorm_type=LayerNorm_type
                ),
                # Downsample
                Downsample(dim_in, dim_out) if i != (self.depth - 1)
                else nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=bias)
            ]))

        # 3. Bottleneck
        mid_dim = ch * ch_mult[-1]

        # Spatial Attention
        self.spatial_attention = Attention_spatial(
            in_channel=mid_dim,
            n_head=1,
            norm_groups=min(16, mid_dim)
        )
        # 新增的归一化层内容
        self.norm_before_latent = nn.GroupNorm(1, mid_dim)


        # Latent 映射 (确定性输出)
        self.latent_conv = nn.Conv2d(mid_dim, embed_dim*2, kernel_size=1, bias=bias)
        self.post_latent_conv = nn.Conv2d(embed_dim, mid_dim, kernel_size=1, bias=bias)

        # 4. Decoder
        self.decoder = nn.ModuleList([])

        for i in range(self.depth):
            dim_out = ch * ch_mult[self.depth - i]
            dim_in = ch * ch_mult[self.depth - i - 1]

            self.decoder.append(nn.ModuleList([
                TransformerBlock(
                    dim_in=dim_out + dim_in,  # Concat 维度
                    dim_out=dim_out,
                    num_heads=max(1, dim_out // 48),
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias, LayerNorm_type=LayerNorm_type
                ),
                TransformerBlock(
                    dim_in=dim_out + dim_in,
                    dim_out=dim_out,
                    num_heads=max(1, dim_out // 48),
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias, LayerNorm_type=LayerNorm_type
                ),
                Upsample(dim_out, dim_in) if i != self.depth - 1
                else nn.Conv2d(dim_out, dim_in, kernel_size=1, bias=bias)
            ]))

        # 最终输出
        self.final_conv = nn.Conv2d(ch, out_ch, 3, 1, 1)

    def check_image_size(self, x, h, w):
        """确保图像尺寸符合 Patch/Window 要求"""
        s = int(math.pow(2, self.depth))
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def encode(self, x):
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            latent: [B, embed_dim, H/16, W/16] -> 数值范围将变得正常
            skips: List of tensors
        """
        h, w = x.shape[2:]
        self.H, self.W = h, w

        x = self.check_image_size(x, h, w)
        x = self.init_conv(x)
        skips = [x]

        for i, (b1, b2, downsample) in enumerate(self.encoder):
            x = self.encoder_cbam[i](x)  # CBAM
            x = b1(x)
            skips.append(x)
            x = b2(x)
            skips.append(x)
            x = downsample(x)

        # Bottleneck 处理
        x = self.spatial_attention(x)

        # ====================================================================
        # [关键修改] 应用归一化
        # ====================================================================
        # 在进入 Latent 卷积之前清洗数据
        x = self.norm_before_latent(x)


        # 开始预测分布的参数内容；
        moments=self.latent_conv(x)
        mean,logvar=torch.chunk(moments,2,dim=1)
        # 将其分割为两个块；

        if self.training:
            std=torch.exp(0.5*logvar)
            eps=torch.randn_like(std)
            z=mean+eps*std
        else:
            z=mean

        return z,(mean,logvar),skips

    def decode(self, x, h):
        """
        Args:
            x: Latent 特征
            h: Skip Connections
        """
        x = self.post_latent_conv(x)

        for i, (b1, b2, upsample) in enumerate(self.decoder):
            # 拼接 skip connection (倒序取)
            x = torch.cat([x, h[-(i * 2 + 1)]], dim=1)
            x = b1(x)

            x = torch.cat([x, h[-(i * 2 + 2)]], dim=1)
            x = b2(x)

            x = upsample(x)

        x = self.final_conv(x + h[0])
        return x[..., :self.H, :self.W]

    def forward(self, x):
        z,(mean,logvar),skips=self.encoder(x)
        recon=self.decode(z, skips)
        return recon




if __name__ == "__main__":
    device =  "cpu"

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
        embed_dim=8,  # 对应原始 UNet 的 embed_dim
        ffn_expansion_factor=2,
        bias=False,
        LayerNorm_type='WithBias'
    ).to(device)

    model=load_model_checkpoint(model,r'E:\Recurrence\OmniFuse\experiments\OmniFuse\latent_AutoEncoder\models\5000_G.pth')

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
    vis_lq = load_image_to_x(image_path=r'E:\Recurrence\OmniFuse\DDL-12\IR_Low_contrast\IR_Low_contrast_slight\train\Infrared\00019N.png',
                             device='cpu')
    # 前向传播
    z,(mean,var),skips=model.encode(vis_lq)
    from utils.yaml_utils import plot_tensor_hist
    # plot_tensor_hist(h[:, 0, ...],(-3.0,3.0),'visual latent after encode')
    for i in range(z.shape[1]): # 这里的z就是类似的潜在特征了；；
        ret=plot_tensor_hist(z[:, i, ...],(-1.0,1.0),f'visual latent channel {i} after encode')
        print(f'channel {i} : mean={ret["mean"]}, std={ret["std"]} min:{ret["min"]} max:{ret["max"]}')



    # output = model(vis_lq)

    print(f"输入: {vis_lq.shape}")
    # print(f"输出: {output.shape}")
    print(f"✓ 单模态测试通过")