import os
import yaml
from typing import Any, Dict, Optional

from models.modules.ae_arch import TransformerUNet #最新的版本的 ；

import torch

import torch
import torch.nn.functional as F
from math import exp

import numpy as np
import matplotlib.pyplot as plt

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

def plot_tensor_hist(tensor, range_lim, title="Data Distribution", bins=100):
    """
    绘制 BCHW Tensor 的像素分布直方图，并返回真实统计指标。

    参数:
        tensor: 输入张量 (Torch Tensor 或 Numpy Array)
        range_lim: 绘图显示的范围 (min, max)，仅影响可视化，不影响统计数值。
        title: 标题
        bins: 直方图柱子数量

    返回:
        dict: 包含 mean, var, std, min, max, overflow_ratio 的字典
    """
    # -------------------------------------------------
    # 1. 数据预处理
    # -------------------------------------------------
    if isinstance(tensor, torch.Tensor):
        data = tensor.detach().cpu().numpy().flatten()
    else:
        data = np.array(tensor).flatten()

    if len(data) == 0:
        print(f"Error: [{title}] 输入 Tensor 为空。")
        return None

    # -------------------------------------------------
    # 2. 计算“真实”统计指标 (基于原始数据)
    # -------------------------------------------------
    real_mean = float(np.mean(data))
    real_var = float(np.var(data))  # 方差
    real_std = float(np.std(data))  # 标准差
    real_min = float(np.min(data))
    real_max = float(np.max(data))

    # -------------------------------------------------
    # 3. 计算“溢出”比例 (用于监控 Hard Clip 风险)
    # -------------------------------------------------
    clip_min, clip_max = range_lim
    out_of_bounds_mask = (data < clip_min) | (data > clip_max)
    out_ratio_percent = float(np.sum(out_of_bounds_mask) / len(data) * 100)

    # -------------------------------------------------
    # 4. 构建返回字典
    # -------------------------------------------------
    stats_dict = {
        "title": title,
        "mean": real_mean,
        "var": real_var,
        "std": real_std,
        "min": real_min,
        "max": real_max,
        "overflow_percent": out_ratio_percent
    }

    # -------------------------------------------------
    # 5. 绘图 (可视化部分)
    # -------------------------------------------------
    plt.figure(figsize=(10, 5))

    # 直方图：range=range_lim 仅限制绘图区间的显示
    plt.hist(data, bins=bins, range=range_lim,
             color='#4F81BD', edgecolor='black', alpha=0.7, linewidth=0.5, density=True)

    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.axvline(0, color='r', linestyle='--', alpha=0.3, linewidth=1)

    # 标题包含方差信息
    stats_msg = (f"Mean={real_mean:.4f} | Var={real_var:.4f} | Std={real_std:.4f}\n"
                 f"Range: [{real_min:.4f}, {real_max:.4f}]")
    plt.title(f"{title}\n{stats_msg}", fontsize=11)
    plt.xlabel("Value")
    plt.ylabel("Density")

    # 警告标注
    if out_ratio_percent > 0.1:
        warn_msg = f"⚠️ Hidden: {out_ratio_percent:.2f}%"
        color = 'red'
    else:
        warn_msg = f"✔ Coverage: {100 - out_ratio_percent:.2f}%"
        color = 'green'

    plt.text(0.98, 0.95, warn_msg, transform=plt.gca().transAxes,
             ha='right', va='top', color=color, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.9, edgecolor=color, boxstyle='round'))

    plt.tight_layout()
    plt.show()

    return stats_dict



def calculate_psnr(img1, img2):
    """
    计算 PSNR
    :param img1: [0, 1] 范围的 Tensor
    :param img2: [0, 1] 范围的 Tensor
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def calculate_ssim(img1, img2, window_size=11, size_average=True):
    """
    计算 SSIM (标准高斯窗口)
    :param img1: [0, 1] 范围的 Tensor
    :param img2: [0, 1] 范围的 Tensor
    """
    channel = img1.size(1)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


@torch.no_grad()
def latent_to_common(prediction_norm, type_name, mean_var_dict, task='vis', device='cuda:0'):
    """
    common_to_latent 的逆操作。
    功能：将扩散模型输出的归一化特征 ([-1, 1]) 映射回原始潜在空间 ([-35, 35])。
    注意：始终使用 GT (Ground Truth) 的统计量进行恢复，因为预测目标是干净图像。
    """

    # 1. 确定字典键名
    # 既然是恢复成干净图像，我们只需要读取 GT 的统计数据
    # task='vis' -> 读取 'vis_gt'
    # task='ir'  -> 读取 'ir_gt'
    key_img = 'vis' if task == 'vis' else 'ir'
    key_gt = f'{key_img}_gt'

    # 2. 向量化准备统计参数
    batch_size = len(type_name)

    # 提取 GT 的均值和标准差
    means_gt = [mean_var_dict[t][key_gt]['mean'] for t in type_name]
    stds_gt = [mean_var_dict[t][key_gt]['std'] for t in type_name]

    # 3. 转换为 Tensor 并调整维度 (Broadcasting)
    # Shape: [B, 1, 1, 1]
    mean_gt_tensor = torch.tensor(means_gt, device=device).view(batch_size, 1, 1, 1)
    std_gt_tensor = torch.tensor(stds_gt, device=device).view(batch_size, 1, 1, 1)

    # 4. 反向归一化 (Inverse Normalization)
    # 公式：x = x_norm * std + mean
    # prediction_norm 是扩散模型的输出
    restored_latent = prediction_norm * std_gt_tensor + mean_gt_tensor

    return restored_latent


@torch.no_grad()
def common_to_latent(ae, img, img_gt, type_name, mean_var_dict, task='vis', device='cuda:0'):
    """
    优化版：
    1. 移除了 for 循环中的 ae.encode，改为一次性 Batch 处理。
    2. 使用向量化操作构建 mean/std 张量，避免在循环中反复 cat。
    3. 统一处理 vis 和 ir 逻辑。
    """

    # 1. 确定字典键名 (根据 task 自动切换)
    # 如果 task 是 'vis'，读取 'vis' 和 'vis_gt'
    # 如果 task 是 'ir'， 读取 'ir'  和 'ir_gt'
    key_img = 'vis' if task == 'vis' else 'ir'
    key_gt = f'{key_img}_gt'

    # 2. 批量编码 (Batch Encode) - 这是性能提升的关键
    # 假设 ae.encode 输出 shape: [B, C, H, W]
    cond, cond_hs = ae.encode(img.to(device))
    cond_gt, cond_gt_hs = ae.encode(img_gt.to(device))

    # 3. 向量化准备统计参数 (Vectorize Statistics)
    # 避免在循环中反复 .to(device) 和 torch.cat
    # 此时 type_name 是一个长度为 B 的 list
    batch_size = len(type_name)

    # 使用列表推导式快速提取当前 batch 所有样本对应的 mean/std
    # 结果是一个 list of floats
    means = [mean_var_dict[t][key_img]['mean'] for t in type_name]
    stds = [mean_var_dict[t][key_img]['std'] for t in type_name]

    means_gt = [mean_var_dict[t][key_gt]['mean'] for t in type_name]
    stds_gt = [mean_var_dict[t][key_gt]['std'] for t in type_name]

    # 4. 转换为 Tensor 并调整维度以支持广播 (Broadcasting)
    # 目标 Shape: [B, 1, 1, 1]，这样可以自动广播到 [B, C, H, W]
    mean_tensor = torch.tensor(means, device=device).view(batch_size, 1, 1, 1)
    std_tensor = torch.tensor(stds, device=device).view(batch_size, 1, 1, 1)

    mean_gt_tensor = torch.tensor(means_gt, device=device).view(batch_size, 1, 1, 1)
    std_gt_tensor = torch.tensor(stds_gt, device=device).view(batch_size, 1, 1, 1)

    # 5. 批量归一化 (Batch Normalization)
    cond_norm = (cond - mean_tensor) / std_tensor
    cond_gt_norm = (cond_gt - mean_gt_tensor) / std_gt_tensor

    return cond_norm, cond_hs, cond_gt_norm, cond_gt_hs



def load_ae(device='cuda:0'):
    model = TransformerUNet()
    pretrained_path = './pretrained/56025_G.pth'
    model.load_state_dict(torch.load(pretrained_path),strict=True)
    for param in model.parameters():
        param.requires_grad = False
    model.eval().to(device)
    return model
# from models.modules.UNet_arch import UNet
# def load_ae(device='cuda:0'):
#     model = UNet(in_ch=3,out_ch=3,ch=8,ch_mult=[4,8,8,16],embed_dim=8)
#     pretrained_path = './pretrained/AutoEncoder.pth'
#     model.load_state_dict(torch.load(pretrained_path),strict=True)
#     for param in model.parameters():
#         param.requires_grad = False
#     model.eval().to(device)
#     return model



def dict_to_yaml(config: Dict[str, Any], yaml_path: str) -> bool:
    """
    将 Python 字典保存到 YAML 文件。

    Args:
        config (Dict[str, Any]): 待保存的字典内容
        yaml_path (str): 输出 YAML 文件路径

    Returns:
        bool: 保存是否成功
    """
    try:
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)

        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(
                config,
                f,
                allow_unicode=True,  # 中文支持
                default_flow_style=False
            )
        return True

    except Exception as e:
        print(f"[Error] 保存 YAML 失败: {yaml_path}\n原因: {e}")
        return False


def yaml_to_config(yaml_path: str) -> Optional[Dict[str, Any]]:
    """
    将 YAML 文件解析为 Python 字典。

    Args:
        yaml_path (str): YAML 文件路径

    Returns:
        dict: 解析成功的字典
        None: 失败时返回 None
    """
    if not os.path.isfile(yaml_path):
        print(f"[Error] 文件不存在: {yaml_path}")
        return None

    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        if not isinstance(config, dict):
            print(f"[Error] YAML 内容必须是 dict: {yaml_path}")
            return None

        return config

    except yaml.YAMLError as e:
        print(f"[Error] YAML 格式错误: {yaml_path}\n原因: {e}")
        return None
    except Exception as e:
        print(f"[Error] 读取 YAML 失败: {yaml_path}\n原因: {e}")
        return None