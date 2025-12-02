# 将代码内容上传到git的操作
# 本地的编辑器中的内容和远程的git仓库中的东西合并起来调试的操作；
import os
import sys
import random
import clip
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

import os.path as osp
import torch
import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as F

'''
根据指定的路径位置获取到train test的文件的内容；
但是内部的只是和图像内容关联起来
'''


def read_val_data(val_root):
    '''
    Args:
        val_root: 路径下面的 Visible 和 Infrared 两个保存图片的文件夹内容；
    Returns:

    '''

    assert os.path.exists(val_root), f'val root: {val_root} does not exist.'
    val_images_visible_path = []
    val_images_infrared_path = []
    supported = [".jpg", ".JPG", ".png", ".PNG", ".bmp", 'tif', 'TIF']  # 支持的文件后缀类型
    val_images_visible_root = os.path.join(val_root, "Visible")
    val_images_infrared_root = os.path.join(val_root, "Infrared")
    val_visible_path = [os.path.join(val_images_visible_root, i) for i in os.listdir(val_images_visible_root) if
                        os.path.splitext(i)[-1] in supported]
    val_infrared_path = [os.path.join(val_images_infrared_root, i) for i in os.listdir(val_images_infrared_root) if
                         os.path.splitext(i)[-1] in supported]
    # 设定为按照数字进行的排序的操作；
    val_visible_path.sort(key=lambda x: osp.splitext(osp.basename(x))[0])
    val_infrared_path.sort(key=lambda x: osp.splitext(osp.basename(x))[0])
    assert len(val_visible_path) == len(
        val_infrared_path), ' The length of val dataset does not match. low:{}, high:{}'.format(len(val_visible_path),
                                                                                                len(val_infrared_path))
    # print("Visible and Infrared images check finish")
    return {"val_visible_path": val_visible_path, "val_infrared_path": val_infrared_path}


def read_train_data(train_root):
    assert osp.exists(train_root), f'train root: {train_root} does not exist.'
    train_visible_path = []
    train_infrared_path = []
    train_visible_gt_path = []
    train_infrared_gt_path = []
    supported = [".jpg", ".JPG", ".png", ".PNG", ".bmp", 'tif', 'TIF']  # 支持的文件后缀类型
    train_visible_root = os.path.join(train_root, "Visible")
    train_infrared_root = os.path.join(train_root, "Infrared")
    train_visible_gt_root = os.path.join(train_root, "Visible_gt")
    train_infrared_gt_root = os.path.join(train_root, "Infrared_gt")
    train_visible_path = [os.path.join(train_visible_root, i) for i in os.listdir(train_visible_root) if
                          os.path.splitext(i)[-1] in supported]
    train_infrared_path = [os.path.join(train_infrared_root, i) for i in os.listdir(train_infrared_root) if
                           os.path.splitext(i)[-1] in supported]
    train_visible_gt_path = [os.path.join(train_visible_gt_root, i) for i in os.listdir(train_visible_gt_root) if
                             os.path.splitext(i)[-1] in supported]
    train_infrared_gt_path = [os.path.join(train_infrared_gt_root, i) for i in os.listdir(train_infrared_gt_root) if
                              os.path.splitext(i)[-1] in supported]
    train_visible_path.sort(key=lambda x: osp.splitext(osp.basename(x))[0])
    train_infrared_path.sort(key=lambda x: osp.splitext(osp.basename(x))[0])
    train_visible_gt_path.sort(key=lambda x: osp.splitext(osp.basename(x))[0])
    train_infrared_gt_path.sort(key=lambda x: osp.splitext(osp.basename(x))[0])
    assert len(train_visible_path) == len(
        train_infrared_path), ' The length of train dataset does not match. low:{}, high:{}'.format(
        len(train_visible_path), len(train_infrared_path))
    return {"train_visible_path": train_visible_path, "train_infrared_path": train_infrared_path,
            "train_visible_gt_path": train_visible_gt_path, "train_infrared_gt_path": train_infrared_gt_path}


def imgToTensor(img):
    '''
    Args:
        img: 单个的RGB形式的Numpy Unit8 H W C / H W 类型的转换为
    Returns:
        tensor 类型并且是 0-1 C H W 的图片
    '''
    img=np.array(img) # 开始读取到的形式为一个 PIL Image 对象；
    if not isinstance(img,np.ndarray):
        raise TypeError("imgToTensor function only support np.ndarray")
    if img.ndim ==2:
        img=np.stack([img]*3,axis=2)
    elif img.ndim==3 and img.shape[2]==1:
        img=np.concatenate([img]*3,axis=2)
        # stack 和 concatenate 区别在于不会创建新的维度；
    elif img.ndim==3 and img.shape[2] not in (1,3):
        raise ValueError("imgToTensor function only support gray or RGB image")

    img=img.astype(np.float32)/255.0

    tensor=torch.from_numpy(np.transpose(img, (2, 0, 1))) # 这里同时使用contiguous 就是多余的状态；
    return tensor


# ================= IR Low Contrast =================
ir_low_contrast_slight_prompt_path = "DDL-12/IR_Low_contrast/IR_Low_contrast_slight/train/text.txt"
assert os.path.exists(ir_low_contrast_slight_prompt_path), "text prompt root: {} does not exist.".format(ir_low_contrast_slight_prompt_path)
with open(ir_low_contrast_slight_prompt_path, 'r', encoding='utf-8') as file:
    ir_low_contrast_slight_lines = file.readlines()

ir_low_contrast_moderate_prompt_path = "DDL-12/IR_Low_contrast/IR_Low_contrast_moderate/train/text.txt"
assert os.path.exists(ir_low_contrast_moderate_prompt_path), "text prompt root: {} does not exist.".format(ir_low_contrast_moderate_prompt_path)
with open(ir_low_contrast_moderate_prompt_path, 'r', encoding='utf-8') as file:
    ir_low_contrast_moderate_lines = file.readlines()

ir_low_contrast_average_prompt_path = "DDL-12/IR_Low_contrast/IR_Low_contrast_average/train/text.txt"
assert os.path.exists(ir_low_contrast_average_prompt_path), "text prompt root: {} does not exist.".format(ir_low_contrast_average_prompt_path)
with open(ir_low_contrast_average_prompt_path, 'r', encoding='utf-8') as file:
    ir_low_contrast_average_lines = file.readlines()

ir_low_contrast_extreme_prompt_path = "DDL-12/IR_Low_contrast/IR_Low_contrast_extreme/train/text.txt"
assert os.path.exists(ir_low_contrast_extreme_prompt_path), "text prompt root: {} does not exist.".format(ir_low_contrast_extreme_prompt_path)
with open(ir_low_contrast_extreme_prompt_path, 'r', encoding='utf-8') as file:
    ir_low_contrast_extreme_lines = file.readlines()


# ================= IR Noise =================
ir_noise_slight_prompt_path = "DDL-12/IR_Noise/IR_Noise_slight/train/text.txt"
assert os.path.exists(ir_noise_slight_prompt_path), "text prompt root: {} does not exist.".format(ir_noise_slight_prompt_path)
with open(ir_noise_slight_prompt_path, 'r', encoding='utf-8') as file:
    ir_noise_slight_lines = file.readlines()

ir_noise_moderate_prompt_path = "DDL-12/IR_Noise/IR_Noise_moderate/train/text.txt"
assert os.path.exists(ir_noise_moderate_prompt_path), "text prompt root: {} does not exist.".format(ir_noise_moderate_prompt_path)
with open(ir_noise_moderate_prompt_path, 'r', encoding='utf-8') as file:
    ir_noise_moderate_lines = file.readlines()

ir_noise_average_prompt_path = "DDL-12/IR_Noise/IR_Noise_average/train/text.txt"
assert os.path.exists(ir_noise_average_prompt_path), "text prompt root: {} does not exist.".format(ir_noise_average_prompt_path)
with open(ir_noise_average_prompt_path, 'r', encoding='utf-8') as file:
    ir_noise_average_lines = file.readlines()

ir_noise_extreme_prompt_path = "DDL-12/IR_Noise/IR_Noise_extreme/train/text.txt"
assert os.path.exists(ir_noise_extreme_prompt_path), "text prompt root: {} does not exist.".format(ir_noise_extreme_prompt_path)
with open(ir_noise_extreme_prompt_path, 'r', encoding='utf-8') as file:
    ir_noise_extreme_lines = file.readlines()


# ================= IR Stripe Noise =================
ir_stripe_noise_slight_prompt_path = "DDL-12/IR_Stripe_noise/IR_Stripe_noise_slight/train/text.txt"
assert os.path.exists(ir_stripe_noise_slight_prompt_path), "text prompt root: {} does not exist.".format(ir_stripe_noise_slight_prompt_path)
with open(ir_stripe_noise_slight_prompt_path, 'r', encoding='utf-8') as file:
    ir_stripe_noise_slight_lines = file.readlines()

ir_stripe_noise_moderate_prompt_path = "DDL-12/IR_Stripe_noise/IR_Stripe_noise_moderate/train/text.txt"
assert os.path.exists(ir_stripe_noise_moderate_prompt_path), "text prompt root: {} does not exist.".format(ir_stripe_noise_moderate_prompt_path)
with open(ir_stripe_noise_moderate_prompt_path, 'r', encoding='utf-8') as file:
    ir_stripe_noise_moderate_lines = file.readlines()

ir_stripe_noise_average_prompt_path = "DDL-12/IR_Stripe_noise/IR_Stripe_noise_average/train/text.txt"
assert os.path.exists(ir_stripe_noise_average_prompt_path), "text prompt root: {} does not exist.".format(ir_stripe_noise_average_prompt_path)
with open(ir_stripe_noise_average_prompt_path, 'r', encoding='utf-8') as file:
    ir_stripe_noise_average_lines = file.readlines()

ir_stripe_noise_extreme_prompt_path = "DDL-12/IR_Stripe_noise/IR_Stripe_noise_extreme/train/text.txt"
assert os.path.exists(ir_stripe_noise_extreme_prompt_path), "text prompt root: {} does not exist.".format(ir_stripe_noise_extreme_prompt_path)
with open(ir_stripe_noise_extreme_prompt_path, 'r', encoding='utf-8') as file:
    ir_stripe_noise_extreme_lines = file.readlines()


# ================= VI Blur =================
vi_blur_slight_prompt_path = "DDL-12/VI_Blur/VI_Blur_slight/train/text.txt"
assert os.path.exists(vi_blur_slight_prompt_path), "text prompt root: {} does not exist.".format(vi_blur_slight_prompt_path)
with open(vi_blur_slight_prompt_path, 'r', encoding='utf-8') as file:
    vi_blur_slight_lines = file.readlines()

vi_blur_moderate_prompt_path = "DDL-12/VI_Blur/VI_Blur_moderate/train/text.txt"
assert os.path.exists(vi_blur_moderate_prompt_path), "text prompt root: {} does not exist.".format(vi_blur_moderate_prompt_path)
with open(vi_blur_moderate_prompt_path, 'r', encoding='utf-8') as file:
    vi_blur_moderate_lines = file.readlines()

vi_blur_average_prompt_path = "DDL-12/VI_Blur/VI_Blur_average/train/text.txt"
assert os.path.exists(vi_blur_average_prompt_path), "text prompt root: {} does not exist.".format(vi_blur_average_prompt_path)
with open(vi_blur_average_prompt_path, 'r', encoding='utf-8') as file:
    vi_blur_average_lines = file.readlines()

vi_blur_extreme_prompt_path = "DDL-12/VI_Blur/VI_Blur_extreme/train/text.txt"
assert os.path.exists(vi_blur_extreme_prompt_path), "text prompt root: {} does not exist.".format(vi_blur_extreme_prompt_path)
with open(vi_blur_extreme_prompt_path, 'r', encoding='utf-8') as file:
    vi_blur_extreme_lines = file.readlines()


# ================= VI Haze =================
vi_haze_slight_prompt_path = "DDL-12/VI_Haze/VI_Haze_slight/train/text.txt"
assert os.path.exists(vi_haze_slight_prompt_path), "text prompt root: {} does not exist.".format(vi_haze_slight_prompt_path)
with open(vi_haze_slight_prompt_path, 'r', encoding='utf-8') as file:
    vi_haze_slight_lines = file.readlines()

vi_haze_moderate_prompt_path = "DDL-12/VI_Haze/VI_Haze_moderate/train/text.txt"
assert os.path.exists(vi_haze_moderate_prompt_path), "text prompt root: {} does not exist.".format(vi_haze_moderate_prompt_path)
with open(vi_haze_moderate_prompt_path, 'r', encoding='utf-8') as file:
    vi_haze_moderate_lines = file.readlines()

vi_haze_average_prompt_path = "DDL-12/VI_Haze/VI_Haze_average/train/text.txt"
assert os.path.exists(vi_haze_average_prompt_path), "text prompt root: {} does not exist.".format(vi_haze_average_prompt_path)
with open(vi_haze_average_prompt_path, 'r', encoding='utf-8') as file:
    vi_haze_average_lines = file.readlines()

vi_haze_extreme_prompt_path = "DDL-12/VI_Haze/VI_Haze_extreme/train/text.txt"
assert os.path.exists(vi_haze_extreme_prompt_path), "text prompt root: {} does not exist.".format(vi_haze_extreme_prompt_path)
with open(vi_haze_extreme_prompt_path, 'r', encoding='utf-8') as file:
    vi_haze_extreme_lines = file.readlines()


# ================= VI Haze Low（单级） =================
vi_haze_low_prompt_path = "DDL-12/VI_Haze_Low/train/text.txt"
assert os.path.exists(vi_haze_low_prompt_path), "text prompt root: {} does not exist.".format(vi_haze_low_prompt_path)
with open(vi_haze_low_prompt_path, 'r', encoding='utf-8') as file:
    vi_haze_low_lines = file.readlines()


# ================= VI Low Light =================
vi_low_light_slight_prompt_path = "DDL-12/VI_Low_light/VI_Low_light_slight/train/text.txt"
assert os.path.exists(vi_low_light_slight_prompt_path), "text prompt root: {} does not exist.".format(vi_low_light_slight_prompt_path)
with open(vi_low_light_slight_prompt_path, 'r', encoding='utf-8') as file:
    vi_low_light_slight_lines = file.readlines()

vi_low_light_moderate_prompt_path = "DDL-12/VI_Low_light/VI_Low_light_moderate/train/text.txt"
assert os.path.exists(vi_low_light_moderate_prompt_path), "text prompt root: {} does not exist.".format(vi_low_light_moderate_prompt_path)
with open(vi_low_light_moderate_prompt_path, 'r', encoding='utf-8') as file:
    vi_low_light_moderate_lines = file.readlines()

vi_low_light_average_prompt_path = "DDL-12/VI_Low_light/VI_Low_light_average/train/text.txt"
assert os.path.exists(vi_low_light_average_prompt_path), "text prompt root: {} does not exist.".format(vi_low_light_average_prompt_path)
with open(vi_low_light_average_prompt_path, 'r', encoding='utf-8') as file:
    vi_low_light_average_lines = file.readlines()

vi_low_light_extreme_prompt_path = "DDL-12/VI_Low_light/VI_Low_light_extreme/train/text.txt"
assert os.path.exists(vi_low_light_extreme_prompt_path), "text prompt root: {} does not exist.".format(vi_low_light_extreme_prompt_path)
with open(vi_low_light_extreme_prompt_path, 'r', encoding='utf-8') as file:
    vi_low_light_extreme_lines = file.readlines()


# ================= VI Noise =================
vi_noise_slight_prompt_path = "DDL-12/VI_Noise/VI_Noise_slight/train/text.txt"
assert os.path.exists(vi_noise_slight_prompt_path), "text prompt root: {} does not exist.".format(vi_noise_slight_prompt_path)
with open(vi_noise_slight_prompt_path, 'r', encoding='utf-8') as file:
    vi_noise_slight_lines = file.readlines()

vi_noise_moderate_prompt_path = "DDL-12/VI_Noise/VI_Noise_moderate/train/text.txt"
assert os.path.exists(vi_noise_moderate_prompt_path), "text prompt root: {} does not exist.".format(vi_noise_moderate_prompt_path)
with open(vi_noise_moderate_prompt_path, 'r', encoding='utf-8') as file:
    vi_noise_moderate_lines = file.readlines()

vi_noise_average_prompt_path = "DDL-12/VI_Noise/VI_Noise_average/train/text.txt"
assert os.path.exists(vi_noise_average_prompt_path), "text prompt root: {} does not exist.".format(vi_noise_average_prompt_path)
with open(vi_noise_average_prompt_path, 'r', encoding='utf-8') as file:
    vi_noise_average_lines = file.readlines()

vi_noise_extreme_prompt_path = "DDL-12/VI_Noise/VI_Noise_extreme/train/text.txt"
assert os.path.exists(vi_noise_extreme_prompt_path), "text prompt root: {} does not exist.".format(vi_noise_extreme_prompt_path)
with open(vi_noise_extreme_prompt_path, 'r', encoding='utf-8') as file:
    vi_noise_extreme_lines = file.readlines()


# ================= VI Noise Low（单级） =================
vi_noise_low_prompt_path = "DDL-12/VI_Noise_Low/train/text.txt"
assert os.path.exists(vi_noise_low_prompt_path), "text prompt root: {} does not exist.".format(vi_noise_low_prompt_path)
with open(vi_noise_low_prompt_path, 'r', encoding='utf-8') as file:
    vi_noise_low_lines = file.readlines()


# ================= VI Over Exposure =================
over_exposure_slight_prompt_path = "DDL-12/VI_Over_exposure/VI_Over_exposure_slight/train/text.txt"
assert os.path.exists(over_exposure_slight_prompt_path), "text prompt root: {} does not exist.".format(over_exposure_slight_prompt_path)
with open(over_exposure_slight_prompt_path, 'r', encoding='utf-8') as file:
    over_exposure_slight_lines = file.readlines()

over_exposure_moderate_prompt_path = "DDL-12/VI_Over_exposure/VI_Over_exposure_moderate/train/text.txt"
assert os.path.exists(over_exposure_moderate_prompt_path), "text prompt root: {} does not exist.".format(over_exposure_moderate_prompt_path)
with open(over_exposure_moderate_prompt_path, 'r', encoding='utf-8') as file:
    over_exposure_moderate_lines = file.readlines()

over_exposure_average_prompt_path = "DDL-12/VI_Over_exposure/VI_Over_exposure_average/train/text.txt"
assert os.path.exists(over_exposure_average_prompt_path), "text prompt root: {} does not exist.".format(over_exposure_average_prompt_path)
with open(over_exposure_average_prompt_path, 'r', encoding='utf-8') as file:
    over_exposure_average_lines = file.readlines()

over_exposure_extreme_prompt_path = "DDL-12/VI_Over_exposure/VI_Over_exposure_extreme/train/text.txt"
assert os.path.exists(over_exposure_extreme_prompt_path), "text prompt root: {} does not exist.".format(over_exposure_extreme_prompt_path)
with open(over_exposure_extreme_prompt_path, 'r', encoding='utf-8') as file:
    over_exposure_extreme_lines = file.readlines()


# ================= VI Rain =================
vi_rain_slight_prompt_path = "DDL-12/VI_Rain/VI_Rain_slight/train/text.txt"
assert os.path.exists(vi_rain_slight_prompt_path), "text prompt root: {} does not exist.".format(vi_rain_slight_prompt_path)
with open(vi_rain_slight_prompt_path, 'r', encoding='utf-8') as file:
    vi_rain_slight_lines = file.readlines()

vi_rain_moderate_prompt_path = "DDL-12/VI_Rain/VI_Rain_moderate/train/text.txt"
assert os.path.exists(vi_rain_moderate_prompt_path), "text prompt root: {} does not exist.".format(vi_rain_moderate_prompt_path)
with open(vi_rain_moderate_prompt_path, 'r', encoding='utf-8') as file:
    vi_rain_moderate_lines = file.readlines()

vi_rain_average_prompt_path = "DDL-12/VI_Rain/VI_Rain_average/train/text.txt"
assert os.path.exists(vi_rain_average_prompt_path), "text prompt root: {} does not exist.".format(vi_rain_average_prompt_path)
with open(vi_rain_average_prompt_path, 'r', encoding='utf-8') as file:
    vi_rain_average_lines = file.readlines()

vi_rain_extreme_prompt_path = "DDL-12/VI_Rain/VI_Rain_extreme/train/text.txt"
assert os.path.exists(vi_rain_extreme_prompt_path), "text prompt root: {} does not exist.".format(vi_rain_extreme_prompt_path)
with open(vi_rain_extreme_prompt_path, 'r', encoding='utf-8') as file:
    vi_rain_extreme_lines = file.readlines()


# ================= VI Rain Haze（单级） =================
vi_rain_haze_prompt_path = "DDL-12/VI_Rain_Haze/train/text.txt"
assert os.path.exists(vi_rain_haze_prompt_path), "text prompt root: {} does not exist.".format(vi_rain_haze_prompt_path)
with open(vi_rain_haze_prompt_path, 'r', encoding='utf-8') as file:
    vi_rain_haze_lines = file.readlines()


# ================= Double 退化 =================
llsn_prompt_path = "DDL-12/llsn/train/text.txt"
assert os.path.exists(llsn_prompt_path), "text prompt root: {} does not exist.".format(llsn_prompt_path)
with open(llsn_prompt_path, 'r', encoding='utf-8') as file:
    llsn_lines = file.readlines()

oelc_prompt_path = "DDL-12/oelc/train/text.txt"
assert os.path.exists(oelc_prompt_path), "text prompt root: {} does not exist.".format(oelc_prompt_path)
with open(oelc_prompt_path, 'r', encoding='utf-8') as file:
    oelc_lines = file.readlines()

rhrn_prompt_path = "DDL-12/rhrn/train/text.txt"
assert os.path.exists(rhrn_prompt_path), "text prompt root: {} does not exist.".format(rhrn_prompt_path)
with open(rhrn_prompt_path, 'r', encoding='utf-8') as file:
    rhrn_lines = file.readlines()


# =================  获取不同的文本提示内容  =================
def get_ir_low_contrast_slight_prompt():
    random_line = random.choice(ir_low_contrast_slight_lines)
    return random_line.strip()

def get_ir_low_contrast_moderate_prompt():
    random_line = random.choice(ir_low_contrast_moderate_lines)
    return random_line.strip()

def get_ir_low_contrast_average_prompt():
    random_line = random.choice(ir_low_contrast_average_lines)
    return random_line.strip()

def get_ir_low_contrast_extreme_prompt():
    random_line = random.choice(ir_low_contrast_extreme_lines)
    return random_line.strip()


def get_ir_noise_slight_prompt():
    random_line = random.choice(ir_noise_slight_lines)
    return random_line.strip()

def get_ir_noise_moderate_prompt():
    random_line = random.choice(ir_noise_moderate_lines)
    return random_line.strip()

def get_ir_noise_average_prompt():
    random_line = random.choice(ir_noise_average_lines)
    return random_line.strip()

def get_ir_noise_extreme_prompt():
    random_line = random.choice(ir_noise_extreme_lines)
    return random_line.strip()


def get_ir_stripe_noise_slight_prompt():
    random_line = random.choice(ir_stripe_noise_slight_lines)
    return random_line.strip()

def get_ir_stripe_noise_moderate_prompt():
    random_line = random.choice(ir_stripe_noise_moderate_lines)
    return random_line.strip()

def get_ir_stripe_noise_average_prompt():
    random_line = random.choice(ir_stripe_noise_average_lines)
    return random_line.strip()

def get_ir_stripe_noise_extreme_prompt():
    random_line = random.choice(ir_stripe_noise_extreme_lines)
    return random_line.strip()


def get_vi_blur_slight_prompt():
    random_line = random.choice(vi_blur_slight_lines)
    return random_line.strip()

def get_vi_blur_moderate_prompt():
    random_line = random.choice(vi_blur_moderate_lines)
    return random_line.strip()

def get_vi_blur_average_prompt():
    random_line = random.choice(vi_blur_average_lines)
    return random_line.strip()

def get_vi_blur_extreme_prompt():
    random_line = random.choice(vi_blur_extreme_lines)
    return random_line.strip()


def get_vi_haze_slight_prompt():
    random_line = random.choice(vi_haze_slight_lines)
    return random_line.strip()

def get_vi_haze_moderate_prompt():
    random_line = random.choice(vi_haze_moderate_lines)
    return random_line.strip()

def get_vi_haze_average_prompt():
    random_line = random.choice(vi_haze_average_lines)
    return random_line.strip()

def get_vi_haze_extreme_prompt():
    random_line = random.choice(vi_haze_extreme_lines)
    return random_line.strip()


def get_vi_haze_low_prompt():
    random_line = random.choice(vi_haze_low_lines)
    return random_line.strip()


def get_vi_low_light_slight_prompt():
    random_line = random.choice(vi_low_light_slight_lines)
    return random_line.strip()

def get_vi_low_light_moderate_prompt():
    random_line = random.choice(vi_low_light_moderate_lines)
    return random_line.strip()

def get_vi_low_light_average_prompt():
    random_line = random.choice(vi_low_light_average_lines)
    return random_line.strip()

def get_vi_low_light_extreme_prompt():
    random_line = random.choice(vi_low_light_extreme_lines)
    return random_line.strip()


def get_vi_noise_slight_prompt():
    random_line = random.choice(vi_noise_slight_lines)
    return random_line.strip()

def get_vi_noise_moderate_prompt():
    random_line = random.choice(vi_noise_moderate_lines)
    return random_line.strip()

def get_vi_noise_average_prompt():
    random_line = random.choice(vi_noise_average_lines)
    return random_line.strip()

def get_vi_noise_extreme_prompt():
    random_line = random.choice(vi_noise_extreme_lines)
    return random_line.strip()


def get_vi_noise_low_prompt():
    random_line = random.choice(vi_noise_low_lines)
    return random_line.strip()


def get_over_exposure_slight_prompt():
    random_line = random.choice(over_exposure_slight_lines)
    return random_line.strip()

def get_over_exposure_moderate_prompt():
    random_line = random.choice(over_exposure_moderate_lines)
    return random_line.strip()

def get_over_exposure_average_prompt():
    random_line = random.choice(over_exposure_average_lines)
    return random_line.strip()

def get_over_exposure_extreme_prompt():
    random_line = random.choice(over_exposure_extreme_lines)
    return random_line.strip()


def get_vi_rain_slight_prompt():
    random_line = random.choice(vi_rain_slight_lines)
    return random_line.strip()

def get_vi_rain_moderate_prompt():
    random_line = random.choice(vi_rain_moderate_lines)
    return random_line.strip()

def get_vi_rain_average_prompt():
    random_line = random.choice(vi_rain_average_lines)
    return random_line.strip()

def get_vi_rain_extreme_prompt():
    random_line = random.choice(vi_rain_extreme_lines)
    return random_line.strip()


def get_vi_rain_haze_prompt():
    random_line = random.choice(vi_rain_haze_lines)
    return random_line.strip()


def get_llsn_prompt():
    random_line = random.choice(llsn_lines)
    return random_line.strip()

def get_oelc_prompt():
    random_line = random.choice(oelc_lines)
    return random_line.strip()

def get_rhrn_prompt():
    random_line = random.choice(rhrn_lines)
    return random_line.strip()