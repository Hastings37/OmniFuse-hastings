import random

from PIL import Image
import torch
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode


def imgToTensor(path):
    '''
    Args:
        img: 单个的RGB形式的Numpy Unit8 H W C / H W 类型的转换为
    Returns:
        tensor 类型并且是 0-1 C H W 的图片
    '''
    img=np.array(Image.open(path),dtype=np.uint8)  # H W C
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


def pad_if_smaller(img, target_size, fill=0.0):
    # img: tensor C H W  (0-1) 数据范围；
    height,width = img.shape[1:3] # H W
    pad_height = max(target_size[0] - height, 0)
    pad_width = max(target_size[1] - width, 0)
    if pad_height > 0 or pad_width > 0:
        # left right top button  when input is tensor
        # left right top buttom when input is PIL Image
        img = F.pad(img, (0, 0, pad_width, pad_height), fill=fill)
    return img


def resize(img,target_size):
    # target_size if only one number then the miner side to target_size while keep the aspect ratio
    # img: tensor C H W  data range: 0-1
    img=F.resize(img,target_size,interpolation=InterpolationMode.NEAREST)
    return img

# question: i don't know why the function is here
def resize_16(img):
    height,width=img.shape[1:3]
    new_height=(height//16)*16
    new_width=(width//16)*16
    # is not make it can be diveded by 16 but make it smaller than orginal size ???
    img=F.resize(img,(new_height,new_width),interpolation=InterpolationMode.NEAREST)
    # i get it , it is used to make the new pixel values ;
    return img

def random_horizontal_filp(img,flip_prob):
    if torch.rand(1).item()<flip_prob:
        img=F.hflip(img)
    return img

def random_vertical_flip(img,flip_prob):
    if torch.rand(1).item()<flip_prob:
        # produce a random number between 0 and 1 (unincluded)
        img=F.vflip(img)
    return img

def random_rotation(img,degree=(-30,30),fill=0):
    angle=random.uniform(*degree)
    # produce a random number between degree[0] and degree[1]
    img=F.rotate(img,angle,interpolation=InterpolationMode.BILINEAR,fill=fill)
    return img

def center_crop(img,crop_size):
    img=pad_if_smaller(img,crop_size)
    # the function will make the imgsize higher than crop size
    # crop size should be on a single number
    img=F.center_crop(img,crop_size)
    return img

def random_crop(img,crop_size):
    img=pad_if_smaller(img,crop_size)
    corp_transformer=T.RandomCrop(crop_size)
    return corp_transformer(img)

# the corp_size should be a couple of number (2)

def argument_hastings(trans_opt, ir, vis, ir_gt, vis_gt, full):
    # 1. 将所有图片打包成列表，方便统一循环处理
    # img_list 中的元素顺序对应：[ir, vis, ir_gt, vis_gt, full]
    img_list = [ir, vis, ir_gt, vis_gt, full]

    # 辅助函数：安全获取配置
    def safe_num(x):
        return 0 if x is None else x

    def safe_bool(x):
        return False if x is None else x

    # ----------------------------------------------------------
    # 1. Resize 16 (确定性变换，不需要随机同步，直接对每个图应用)
    # ----------------------------------------------------------
    if safe_bool(trans_opt.get('resize_16')):
        img_list = [resize_16(img) for img in img_list]

    # ----------------------------------------------------------
    # 2. Random Horizontal Flip (水平翻转 - 同步)
    # ----------------------------------------------------------
    h_prob = safe_num(trans_opt.get('use_hflip'))
    if h_prob > 0:
        # 关键：只生成一次随机数！
        if random.random() < h_prob:
            img_list = [F.hflip(img) for img in img_list]

    # ----------------------------------------------------------
    # 3. Random Vertical Flip (垂直翻转 - 同步)
    # ----------------------------------------------------------
    v_prob = safe_num(trans_opt.get('use_vflip'))
    if v_prob > 0:
        # 关键：只生成一次随机数！
        if random.random() < v_prob:
            img_list = [F.vflip(img) for img in img_list]

    # ----------------------------------------------------------
    # 4. Random Rotation (随机旋转 - 同步)
    # ----------------------------------------------------------
    if safe_num(trans_opt.get('use_rot')) > 0:
        # 关键：只生成一次角度！
        angle = random.uniform(-30, 30)
        # 对所有图应用同一个角度
        img_list = [F.rotate(img, angle, interpolation=InterpolationMode.BILINEAR, fill=0) for img in img_list]

    # ----------------------------------------------------------
    # 5. Random Crop (随机裁剪 - 同步)
    # ----------------------------------------------------------
    rc = trans_opt.get('random_crop')
    if isinstance(rc, dict) and rc.get('height') and rc.get('width'):
        target_h, target_w = rc['height'], rc['width']

        # A. 先把所有图 Pad 一下，防止尺寸比裁剪框还小
        img_list = [pad_if_smaller(img, (target_h, target_w)) for img in img_list]

        # B. 获取当前尺寸 (取第一张图即可，假设大家尺寸一样)
        _, h, w = img_list[0].shape

        # C. 关键：计算一次裁剪坐标 (Top, Left)
        if h > target_h or w > target_w:
            top = random.randint(0, h - target_h)
            left = random.randint(0, w - target_w)
        else:
            top, left = 0, 0  # 理论上 pad 之后不会进这里，防个万一

        # D. 对所有图应用同一个坐标裁剪
        img_list = [F.crop(img, top, left, target_h, target_w) for img in img_list]

    # ----------------------------------------------------------
    # 6. Center Crop (中心裁剪 - 确定性)
    # ----------------------------------------------------------
    cc = trans_opt.get('center_crop')
    if isinstance(cc, dict) and cc.get('height') and cc.get('width'):
        target_h, target_w = cc['height'], cc['width']
        # 同样先 Pad 防止尺寸不够
        img_list = [pad_if_smaller(img, (target_h, target_w)) for img in img_list]
        # 执行中心裁剪
        img_list = [F.center_crop(img, (target_h, target_w)) for img in img_list]

    # 解包返回 (对应传入的顺序)
    return img_list[0], img_list[1], img_list[2], img_list[3], img_list[4]


def argument_hastings_val(trans_opt, ir, vis, full):
    # 1. 将所有图片打包成列表，方便统一循环处理
    # img_list 中的元素顺序对应：[ir, vis, ir_gt, vis_gt, full]
    img_list = [ir, vis,  full]

    # 辅助函数：安全获取配置
    def safe_num(x):
        return 0 if x is None else x

    def safe_bool(x):
        return False if x is None else x

    # ----------------------------------------------------------
    # 1. Resize 16 (确定性变换，不需要随机同步，直接对每个图应用)
    # ----------------------------------------------------------
    if safe_bool(trans_opt.get('resize_16')):
        img_list = [resize_16(img) for img in img_list]

    # ----------------------------------------------------------
    # 2. Random Horizontal Flip (水平翻转 - 同步)
    # ----------------------------------------------------------
    h_prob = safe_num(trans_opt.get('use_hflip'))
    if h_prob > 0:
        # 关键：只生成一次随机数！
        if random.random() < h_prob:
            img_list = [F.hflip(img) for img in img_list]

    # ----------------------------------------------------------
    # 3. Random Vertical Flip (垂直翻转 - 同步)
    # ----------------------------------------------------------
    v_prob = safe_num(trans_opt.get('use_vflip'))
    if v_prob > 0:
        # 关键：只生成一次随机数！
        if random.random() < v_prob:
            img_list = [F.vflip(img) for img in img_list]

    # ----------------------------------------------------------
    # 4. Random Rotation (随机旋转 - 同步)
    # ----------------------------------------------------------
    if safe_num(trans_opt.get('use_rot')) > 0:
        # 关键：只生成一次角度！
        angle = random.uniform(-30, 30)
        # 对所有图应用同一个角度
        img_list = [F.rotate(img, angle, interpolation=InterpolationMode.BILINEAR, fill=0) for img in img_list]

    # ----------------------------------------------------------
    # 5. Random Crop (随机裁剪 - 同步)
    # ----------------------------------------------------------
    rc = trans_opt.get('random_crop')
    if isinstance(rc, dict) and rc.get('height') and rc.get('width'):
        target_h, target_w = rc['height'], rc['width']

        # A. 先把所有图 Pad 一下，防止尺寸比裁剪框还小
        img_list = [pad_if_smaller(img, (target_h, target_w)) for img in img_list]

        # B. 获取当前尺寸 (取第一张图即可，假设大家尺寸一样)
        _, h, w = img_list[0].shape

        # C. 关键：计算一次裁剪坐标 (Top, Left)
        if h > target_h or w > target_w:
            top = random.randint(0, h - target_h)
            left = random.randint(0, w - target_w)
        else:
            top, left = 0, 0  # 理论上 pad 之后不会进这里，防个万一

        # D. 对所有图应用同一个坐标裁剪
        img_list = [F.crop(img, top, left, target_h, target_w) for img in img_list]

    # ----------------------------------------------------------
    # 6. Center Crop (中心裁剪 - 确定性)
    # ----------------------------------------------------------
    cc = trans_opt.get('center_crop')
    if isinstance(cc, dict) and cc.get('height') and cc.get('width'):
        target_h, target_w = cc['height'], cc['width']
        # 同样先 Pad 防止尺寸不够
        img_list = [pad_if_smaller(img, (target_h, target_w)) for img in img_list]
        # 执行中心裁剪
        img_list = [F.center_crop(img, (target_h, target_w)) for img in img_list]

    # 解包返回 (对应传入的顺序)
    return img_list[0], img_list[1], img_list[2]


