from copy import deepcopy
import numpy as np
from PIL import Image
import cv2

'''
读取图像的内容都是 H W C 形式的numpy 内容弄给
同样的红外的就是H W 的形式 
范围都是 0-255 之间的uint8 内容 
不同的是Image：RGB CV2：BGR
需要反转一下然后 .copy() 保证其是连续的内存；
'''
from torch.utils.data import Dataset
import os
import random
import os.path as osp

from torch.utils import data as data
from torchvision import transforms as T

from complex_degradation.data_control_fusion_util import (
    get_ir_low_contrast_slight_prompt,
    get_ir_low_contrast_moderate_prompt,
    get_ir_low_contrast_average_prompt,
    get_ir_low_contrast_extreme_prompt,
    get_ir_noise_slight_prompt,
    get_ir_noise_moderate_prompt,
    get_ir_noise_average_prompt,
    get_ir_noise_extreme_prompt,
    get_ir_stripe_noise_slight_prompt,
    get_ir_stripe_noise_moderate_prompt,
    get_ir_stripe_noise_average_prompt,
    get_ir_stripe_noise_extreme_prompt,
    get_vi_blur_slight_prompt,
    get_vi_blur_moderate_prompt,
    get_vi_blur_average_prompt,
    get_vi_blur_extreme_prompt,
    get_vi_haze_slight_prompt,
    get_vi_haze_moderate_prompt,
    get_vi_haze_average_prompt,
    get_vi_haze_extreme_prompt,
    get_vi_haze_low_prompt,
    get_vi_low_light_slight_prompt,
    get_vi_low_light_moderate_prompt,
    get_vi_low_light_average_prompt,
    get_vi_low_light_extreme_prompt,
    get_vi_noise_slight_prompt,
    get_vi_noise_moderate_prompt,
    get_vi_noise_average_prompt,
    get_vi_noise_extreme_prompt,
    get_vi_noise_low_prompt,
    get_over_exposure_slight_prompt,
    get_over_exposure_moderate_prompt,
    get_over_exposure_average_prompt,
    get_over_exposure_extreme_prompt,
    get_vi_rain_slight_prompt,
    get_vi_rain_moderate_prompt,
    get_vi_rain_average_prompt,
    get_vi_rain_extreme_prompt,
    get_vi_rain_haze_prompt,
    get_llsn_prompt,
    get_oelc_prompt,
    get_rhrn_prompt,
)


def imfrombytes(content, flag='color', float32=False):
    """Read an image from bytes.

    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale` and `unchanged`.
        float32 (bool): Whether to change to float32., If True, will also norm
            to [0, 1]. Default: False.

    Returns:
        ndarray: Loaded image array.
    """
    img_np = np.frombuffer(content, np.uint8)
    imread_flags = {'color': cv2.IMREAD_COLOR, 'grayscale': cv2.IMREAD_GRAYSCALE, 'unchanged': cv2.IMREAD_UNCHANGED}
    img = cv2.imdecode(img_np, imread_flags[flag])
    # 读取的是 H W C 并且是BGR的格式的；
    if float32:
        img = img.astype(np.float32) / 255.
    return img


import torch
####################   我们自己定义的引用的内容 ################################################
from complex_degradation.data_control_fusion_util import read_train_data, read_val_data, imgToTensor
from complex_degradation.data_argument_hastings import argument_hastings, argument_hastings_val


class ControlFusionDataset(Dataset):
    '''

    '''

    def __init__(self, dataset_opt):

        '''
        Args:
            dataset_opt: 直接一次性构建全部的dataset内容出来；
        '''
        super(ControlFusionDataset, self).__init__()
        self.opt = deepcopy(dataset_opt)
        self.is_train = self.opt.pop('phase') == 'train'  # 判断现在是不是在构建训练数据集的阶段中的操作；
        # 这里的内容是可以通过遍历数据集的数据计算出来的；
        self.get_paths_indices()

    def get_paths_indices(self):
        if self.is_train:
            self.paths = {}
            for name, path in self.opt.items():
                if not name.endswith('_path'):
                    continue
                name = name[:-len('_path')]  # 这里构建的时候已经出现过一次问题了看样子；
                self.paths[name] = read_train_data(path)
                # dict 内嵌套 dict 的形式；
        elif not self.is_train:
            self.paths = {}
            for name, path in self.opt.items():
                if not name.endswith('_path'):
                    continue
                name = name[:-len('_path')]
                self.paths[name] = read_train_data(path)  # read_val_data(path)

        self.class_indices = {}
        for class_key, class_paths in self.paths.items():
            '''
            {
              ir_low_contrast_slight:{
                    'train_infrared_path':[],
                    'train_visible_path':[],
                    'train_infrared_gt_path':[],
                    'train_visible_gt_path':[]
              }
                ... 
            }'''
            self.class_indices[class_key] = list(
                range(
                    len(
                        class_paths[
                            list(class_paths.keys())[0]
                        ]
                    )
                )
            )
            # 对应的一个大类别其中的四个内容具备的长度信息；
            # 将其keys 转换为list 然后取其第一个的长度 真是奇妙啊这个内容；

    def type_to_text(self, t):
        self.text_line = []
        # ================= IR =================
        if t == 'ir_low_contrast_slight':
            self.text_line.append(get_ir_low_contrast_slight_prompt())
        elif t == 'ir_low_contrast_average':
            self.text_line.append(get_ir_low_contrast_average_prompt())
        elif t == 'ir_low_contrast_moderate':
            self.text_line.append(get_ir_low_contrast_moderate_prompt())
        elif t == 'ir_low_contrast_extreme':
            self.text_line.append(get_ir_low_contrast_extreme_prompt())

        elif t == 'ir_noise_slight':
            self.text_line.append(get_ir_noise_slight_prompt())
        elif t == 'ir_noise_average':
            self.text_line.append(get_ir_noise_average_prompt())
        elif t == 'ir_noise_moderate':
            self.text_line.append(get_ir_noise_moderate_prompt())
        elif t == 'ir_noise_extreme':
            self.text_line.append(get_ir_noise_extreme_prompt())

        elif t == 'ir_stripe_noise_slight':
            self.text_line.append(get_ir_stripe_noise_slight_prompt())
        elif t == 'ir_stripe_noise_average':
            self.text_line.append(get_ir_stripe_noise_average_prompt())
        elif t == 'ir_stripe_noise_moderate':
            self.text_line.append(get_ir_stripe_noise_moderate_prompt())
        elif t == 'ir_stripe_noise_extreme':
            self.text_line.append(get_ir_stripe_noise_extreme_prompt())

        # ================= VI Blur =================
        elif t == 'vi_blur_slight':
            self.text_line.append(get_vi_blur_slight_prompt())
        elif t == 'vi_blur_average':
            self.text_line.append(get_vi_blur_average_prompt())
        elif t == 'vi_blur_moderate':
            self.text_line.append(get_vi_blur_moderate_prompt())
        elif t == 'vi_blur_extreme':
            self.text_line.append(get_vi_blur_extreme_prompt())

        # ================= VI Haze =================
        elif t == 'vi_haze_slight':
            self.text_line.append(get_vi_haze_slight_prompt())
        elif t == 'vi_haze_average':
            self.text_line.append(get_vi_haze_average_prompt())
        elif t == 'vi_haze_moderate':
            self.text_line.append(get_vi_haze_moderate_prompt())
        elif t == 'vi_haze_extreme':
            self.text_line.append(get_vi_haze_extreme_prompt())

        # 单级 VI_Haze_Low
        elif t == 'vi_haze_low':
            self.text_line.append(get_vi_haze_low_prompt())

        # ================= VI Low Light =================
        elif t == 'vi_low_light_slight':
            self.text_line.append(get_vi_low_light_slight_prompt())
        elif t == 'vi_low_light_average':
            self.text_line.append(get_vi_low_light_average_prompt())
        elif t == 'vi_low_light_moderate':
            self.text_line.append(get_vi_low_light_moderate_prompt())
        elif t == 'vi_low_light_extreme':
            self.text_line.append(get_vi_low_light_extreme_prompt())

        # ================= VI Noise =================
        elif t == 'vi_noise_slight':
            self.text_line.append(get_vi_noise_slight_prompt())
        elif t == 'vi_noise_average':
            self.text_line.append(get_vi_noise_average_prompt())
        elif t == 'vi_noise_moderate':
            self.text_line.append(get_vi_noise_moderate_prompt())
        elif t == 'vi_noise_extreme':
            self.text_line.append(get_vi_noise_extreme_prompt())

        # 单级 VI_Noise_Low
        elif t == 'vi_noise_low':
            self.text_line.append(get_vi_noise_low_prompt())

        # ================= VI Over Exposure =================
        elif t == 'over_exposure_slight':
            self.text_line.append(get_over_exposure_slight_prompt())
        elif t == 'over_exposure_average':
            self.text_line.append(get_over_exposure_average_prompt())
        elif t == 'over_exposure_moderate':
            self.text_line.append(get_over_exposure_moderate_prompt())
        elif t == 'over_exposure_extreme':
            self.text_line.append(get_over_exposure_extreme_prompt())

        # ================= VI Rain =================
        elif t == 'vi_rain_slight':
            self.text_line.append(get_vi_rain_slight_prompt())
        elif t == 'vi_rain_average':
            self.text_line.append(get_vi_rain_average_prompt())
        elif t == 'vi_rain_moderate':
            self.text_line.append(get_vi_rain_moderate_prompt())
        elif t == 'vi_rain_extreme':
            self.text_line.append(get_vi_rain_extreme_prompt())

        # 单级 VI_Rain_Haze
        elif t == 'vi_rain_haze':
            self.text_line.append(get_vi_rain_haze_prompt())

        # ================= Double 退化 =================
        elif t == 'llsn':
            self.text_line.append(get_llsn_prompt())
        elif t == 'oelc':
            self.text_line.append(get_oelc_prompt())
        elif t == 'rhrn':
            self.text_line.append(get_rhrn_prompt())
        else:
            raise NotImplementedError(f'未知的退化类型：{t}，请检查代码或数据集。')

        return self.text_line[0]

    def __getitem__(self, index):
        # 一个随机的退化类型的名字； 其中包含了类型和对应的程度情况；
        #  这里随机的为类型和一个随机的程度情况；
        class_key = random.choice(list(self.paths.keys()))
        class_indices = self.class_indices[class_key]  # 一个从 0-N-1 的列表内容；
        # 闭合区间的平均采样结果
        item_index = random.randint(0, len(class_indices) - 1)
        # 但是其形式不就是；
        # if self.is_train:
        ir_path = self.paths[class_key]['train_infrared_path'][item_index]
        vis_path = self.paths[class_key]['train_visible_path'][item_index]
        ir_gt_path = self.paths[class_key]['train_infrared_gt_path'][item_index]
        vis_gt_path = self.paths[class_key]['train_visible_gt_path'][item_index]

        type = class_key

        name = osp.splitext(osp.basename(ir_path))[0]
        ir = imgToTensor(Image.open(ir_path))
        vis = imgToTensor(Image.open(vis_path))
        ir_gt = imgToTensor(Image.open(ir_gt_path))
        vis_gt = imgToTensor(Image.open(vis_gt_path))
        # 这里的形式都是 C H W 的tensor 内容；
        full = vis.clone().detach()
        if self.opt['transforms'] is not None:
            ir, vis, ir_gt, vis_gt, full = argument_hastings(self.opt['transforms'], ir, vis, ir_gt, vis_gt, full)

        text = self.type_to_text(type)

        return vis, ir, vis_gt, ir_gt, full, text, name

    def __len__(self):
        total_len = 0
        for class_key, class_paths in self.paths.items():
            total_len += len(class_paths[list(class_paths.keys())[0]])
        return total_len

    @staticmethod
    def collate_fn(batch):
        vis, ir, vis_gt, ir_gt, full, text, name = zip(*batch)
        ir = torch.stack(ir, dim=0)
        vis = torch.stack(vis, dim=0)
        ir_gt = torch.stack(ir_gt, dim=0)
        vis_gt = torch.stack(vis_gt, dim=0)
        full = torch.stack(full, dim=0)
        return vis, ir, vis_gt, ir_gt, full, text, name


def joint_train_transform(images, MAX_SIZE, CROP_SIZE, flip_prob=0.5):
    """
    输入: images list [vis, vis_gt, ir, ir_gt] (PIL Images)
    输出: list of Tensors [C, H, W], float32, 0-1, RGB
    """

    # --- 1. 统一 Resize 策略 ---
    def resize_min_side(im):
        w, h = im.size
        min_side = min(w, h)

        # 如果图像小于裁剪尺寸，先放大
        if min_side < CROP_SIZE:
            scale = CROP_SIZE / min_side
            new_size = tuple(round(x * scale) for x in (w, h))
            im = im.resize(new_size, resample=Image.BICUBIC)

        # 更新尺寸
        w, h = im.size
        min_side = min(w, h)

        # 如果图像过大，限制最大边长（防止显存爆炸）
        if min_side >= MAX_SIZE:
            while min(w, h) >= 2 * MAX_SIZE:
                w, h = w // 2, h // 2
                im = im.resize((w, h), resample=Image.BOX)

            scale = MAX_SIZE / min(w, h)
            new_size = tuple(round(x * scale) for x in (w, h))
            im = im.resize(new_size, resample=Image.BICUBIC)
        return im

    # 对所有图片应用相同的 Resize
    for i in range(len(images)):
        images[i] = resize_min_side(images[i])

    # --- 2. 生成随机参数 ---
    do_h_flip = random.random() < flip_prob
    do_v_flip = random.random() < flip_prob

    w, h = images[0].size
    left, top = 0, 0
    if w >= CROP_SIZE and h >= CROP_SIZE:
        left = random.randint(0, w - CROP_SIZE)
        top = random.randint(0, h - CROP_SIZE)

    # --- 3. 应用变换并转 Tensor ---
    results = []
    for img in images:
        # 3.1 强制转 RGB (防止单通道灰度图导致维度不对)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # 3.2 翻转
        if do_h_flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if do_v_flip:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

        # 3.3 裁剪
        if w >= CROP_SIZE and h >= CROP_SIZE:
            box = (left, top, left + CROP_SIZE, top + CROP_SIZE)
            img = img.crop(box)

        # 3.4 归一化与维度变换
        # PIL (H, W, 3) -> Numpy (H, W, 3) float32 [0, 1]
        arr = np.array(img).astype(np.float32) / 255.0

        # (H, W, C) -> (C, H, W)
        arr = np.transpose(arr, (2, 0, 1))

        # 3.5 确保尺寸是 16 的倍数 (尽管 Crop 已经是固定值，但这步是保险)
        c, hh, ww = arr.shape
        arr = arr[:, :hh - hh % 16, :ww - ww % 16]

        # Numpy -> Tensor
        results.append(torch.from_numpy(arr).float())  # 显式 .float() 确保是 float32

    return results


def test_transform(img, MAX_SIZE):
    """
    修复后的测试集变换
    输入: PIL Image
    输出: Tensor [C, H, W], float32, 0-1, RGB
    """
    # 1. Resize 逻辑 (保持原样)
    if min(*img.size) >= MAX_SIZE:
        while min(*img.size) >= 2 * MAX_SIZE:
            img = img.resize(
                tuple(x // 2 for x in img.size), resample=Image.BOX
            )
        scale = MAX_SIZE / min(*img.size)
        img = img.resize(
            tuple(round(x * scale) for x in img.size), resample=Image.BICUBIC
        )

    # 2. 强制转 RGB
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # 3. 归一化与转 Tensor (修复了变量名混淆的 Bug)
    arr = np.array(img).astype(np.float32) / 255.0  # HWC, 0-1
    arr = np.transpose(arr, (2, 0, 1))  # CHW

    # 4. Padding/Cropping 确保 16 倍数
    c, h, w = arr.shape
    arr = arr[:, :h - h % 16, :w - w % 16]

    return torch.from_numpy(arr).float()


def type_to_text(t):
    text_line = None

    # ================= IR (红外) =================
    if t == 'ir_low_contrast_slight':
        text_line = get_ir_low_contrast_slight_prompt()
    elif t == 'ir_low_contrast_average':
        text_line = get_ir_low_contrast_average_prompt()
    elif t == 'ir_low_contrast_moderate':
        text_line = get_ir_low_contrast_moderate_prompt()
    elif t == 'ir_low_contrast_extreme':
        text_line = get_ir_low_contrast_extreme_prompt()

    elif t == 'ir_noise_slight':
        text_line = get_ir_noise_slight_prompt()
    elif t == 'ir_noise_average':
        text_line = get_ir_noise_average_prompt()
    elif t == 'ir_noise_moderate':
        text_line = get_ir_noise_moderate_prompt()
    elif t == 'ir_noise_extreme':
        text_line = get_ir_noise_extreme_prompt()

    elif t == 'ir_stripe_noise_slight':
        text_line = get_ir_stripe_noise_slight_prompt()
    elif t == 'ir_stripe_noise_average':
        text_line = get_ir_stripe_noise_average_prompt()
    elif t == 'ir_stripe_noise_moderate':
        text_line = get_ir_stripe_noise_moderate_prompt()
    elif t == 'ir_stripe_noise_extreme':
        text_line = get_ir_stripe_noise_extreme_prompt()

    # ================= VI Blur (可见光模糊) =================
    elif t == 'vi_blur_slight':
        text_line = get_vi_blur_slight_prompt()
    elif t == 'vi_blur_average':
        text_line = get_vi_blur_average_prompt()
    elif t == 'vi_blur_moderate':
        text_line = get_vi_blur_moderate_prompt()
    elif t == 'vi_blur_extreme':
        text_line = get_vi_blur_extreme_prompt()

    # ================= VI Haze (可见光雾霾) =================
    elif t == 'vi_haze_slight':
        text_line = get_vi_haze_slight_prompt()
    elif t == 'vi_haze_average':
        text_line = get_vi_haze_average_prompt()
    elif t == 'vi_haze_moderate':
        text_line = get_vi_haze_moderate_prompt()
    elif t == 'vi_haze_extreme':
        text_line = get_vi_haze_extreme_prompt()

    # 单级 VI_Haze_Low
    elif t == 'vi_haze_low':
        text_line = get_vi_haze_low_prompt()

    # ================= VI Low Light (可见光弱光) =================
    elif t == 'vi_low_light_slight':
        text_line = get_vi_low_light_slight_prompt()
    elif t == 'vi_low_light_average':
        text_line = get_vi_low_light_average_prompt()
    elif t == 'vi_low_light_moderate':
        text_line = get_vi_low_light_moderate_prompt()
    elif t == 'vi_low_light_extreme':
        text_line = get_vi_low_light_extreme_prompt()

    # ================= VI Noise (可见光噪点) =================
    elif t == 'vi_noise_slight':
        text_line = get_vi_noise_slight_prompt()
    elif t == 'vi_noise_average':
        text_line = get_vi_noise_average_prompt()
    elif t == 'vi_noise_moderate':
        text_line = get_vi_noise_moderate_prompt()
    elif t == 'vi_noise_extreme':
        text_line = get_vi_noise_extreme_prompt()

    # 单级 VI_Noise_Low
    elif t == 'vi_noise_low':
        text_line = get_vi_noise_low_prompt()

    # ================= VI Over Exposure (可见光过曝) =================
    elif t == 'vi_over_exposure_slight':
        text_line = get_over_exposure_slight_prompt()
    elif t == 'vi_over_exposure_average':
        text_line = get_over_exposure_average_prompt()
    elif t == 'vi_over_exposure_moderate':
        text_line = get_over_exposure_moderate_prompt()
    elif t == 'vi_over_exposure_extreme':
        text_line = get_over_exposure_extreme_prompt()

    # ================= VI Rain (可见光雨) =================
    elif t == 'vi_rain_slight':
        text_line = get_vi_rain_slight_prompt()
    elif t == 'vi_rain_average':
        text_line = get_vi_rain_average_prompt()
    elif t == 'vi_rain_moderate':
        text_line = get_vi_rain_moderate_prompt()
    elif t == 'vi_rain_extreme':
        text_line = get_vi_rain_extreme_prompt()

    # 单级 VI_Rain_Haze
    elif t == 'vi_rain_haze':
        text_line = get_vi_rain_haze_prompt()

    # ================= Double 退化 (双重退化) =================
    elif t == 'llsn':
        text_line = get_llsn_prompt()
    elif t == 'oelc':
        text_line = get_oelc_prompt()
    elif t == 'rhrn':
        text_line = get_rhrn_prompt()

    # ================= 异常处理 =================
    else:
        raise NotImplementedError(f'未知的退化类型：{t}，请检查代码或数据集。')

    return text_line


class DiffusionFusionDataset_train(Dataset):
    def __init__(self, MAX_SIZE, CROP_SIZE, dataset_opt):
        super(DiffusionFusionDataset_train, self).__init__()
        self.opt = deepcopy(dataset_opt)
        self.is_train = self.opt.pop('phase') == 'train'
        self.MAX_SIZE = MAX_SIZE
        self.CROP_SIZE = CROP_SIZE

        # --- 修正点 1: 必须先初始化容器，再调用构建函数 ---
        self.paths = {}
        self.flat_file_list = []
        self.get_paths_indices()

    def get_paths_indices(self):
        # 读取配置中的路径
        for name, path in self.opt.items():
            if not name.endswith('_path'):
                continue
            deg_type = name[:-len('_path')]
            self.paths[deg_type] = read_train_data(path)

        for deg_type, data_dict in self.paths.items():
            # 假设所有模态文件数量一致，取第一个key的长度
            first_key = list(data_dict.keys())[0]
            num_samples = len(data_dict[first_key])
            for i in range(num_samples):
                self.flat_file_list.append((deg_type, i))

    def loader(self, path):
        return Image.open(path).convert('RGB')

    def __len__(self):
        return len(self.flat_file_list)

    def __getitem__(self, index):
        type_name, item_index = self.flat_file_list[index]

        # 获取路径
        ir_path = self.paths[type_name]['train_infrared_path'][item_index]
        vis_path = self.paths[type_name]['train_visible_path'][item_index]
        ir_gt_path = self.paths[type_name]['train_infrared_gt_path'][item_index]
        vis_gt_path = self.paths[type_name]['train_visible_gt_path'][item_index]

        vis = self.loader(vis_path)
        ir = self.loader(ir_path)
        vis_gt = self.loader(vis_gt_path)
        ir_gt = self.loader(ir_gt_path)

        # 联合变换
        tensors = joint_train_transform([vis, vis_gt, ir, ir_gt],
                                        self.MAX_SIZE,
                                        self.CROP_SIZE)

        t_vis, t_vis_gt, t_ir, t_ir_gt = tensors

        text = type_to_text(type_name)
        vis_name = osp.splitext(osp.basename(vis_path))[0]
        ir_name = osp.splitext(osp.basename(ir_path))[0]

        return t_vis, t_vis_gt, t_ir, t_ir_gt, type_name, text, vis_name, ir_name

    @staticmethod
    def collate_fn(batch):
        t_vis, t_vis_gt, t_ir, t_ir_gt, type_name, text, vis_name, ir_name = zip(*batch)

        t_vis = torch.stack(t_vis, dim=0)
        t_vis_gt = torch.stack(t_vis_gt, dim=0)
        t_ir = torch.stack(t_ir, dim=0)
        t_ir_gt = torch.stack(t_ir_gt, dim=0)

        return t_vis, t_vis_gt, t_ir, t_ir_gt, type_name, text, vis_name, ir_name


class DiffusionFusionDataset_test(Dataset):
    def __init__(self, MAX_SIZE, dataset_opt):
        super(DiffusionFusionDataset_test, self).__init__()
        self.opt = deepcopy(dataset_opt)
        self.is_train = False
        self.MAX_SIZE = MAX_SIZE

        self.paths = {}
        self.flat_file_list = []
        self._get_paths_indices()

    def _get_paths_indices(self):
        # 逻辑与 Train 相同
        for name, path in self.opt.items():
            if not name.endswith('_path'):
                continue
            deg_type = name[:-len('_path')]
            self.paths[deg_type] = read_train_data(path)

        for deg_type, data_dict in self.paths.items():
            first_key = list(data_dict.keys())[0]
            num_samples = len(data_dict[first_key])
            for i in range(num_samples):
                self.flat_file_list.append((deg_type, i))

    def loader(self, path):
        # --- 修正点 2: 恢复正确的图片读取逻辑 ---
        return Image.open(path).convert('RGB')

    def __len__(self):
        # --- 修正点 3: 直接返回列表长度，去除冗余循环 ---
        return len(self.flat_file_list)

    def __getitem__(self, index):
        type_name, item_index = self.flat_file_list[index]

        ir_path = self.paths[type_name]['train_infrared_path'][item_index]
        vis_path = self.paths[type_name]['train_visible_path'][item_index]
        ir_gt_path = self.paths[type_name]['train_infrared_gt_path'][item_index]
        vis_gt_path = self.paths[type_name]['train_visible_gt_path'][item_index]

        vis = self.loader(vis_path)
        ir = self.loader(ir_path)
        vis_gt = self.loader(vis_gt_path)
        ir_gt = self.loader(ir_gt_path)

        # 测试变换
        t_vis = test_transform(vis, self.MAX_SIZE)
        t_vis_gt = test_transform(vis_gt, self.MAX_SIZE)
        t_ir = test_transform(ir, self.MAX_SIZE)
        t_ir_gt = test_transform(ir_gt, self.MAX_SIZE)

        text = type_to_text(type_name)

        vis_name = osp.splitext(osp.basename(vis_path))[0]
        ir_name = osp.splitext(osp.basename(ir_path))[0]

        return t_vis, t_vis_gt, t_ir, t_ir_gt, type_name, text, vis_name, ir_name

    @staticmethod
    def collate_fn(batch):
        t_vis, t_vis_gt, t_ir, t_ir_gt, type_name, text, vis_name, ir_name = zip(*batch)
        t_vis = torch.stack(t_vis, dim=0)
        t_vis_gt = torch.stack(t_vis_gt, dim=0)
        t_ir = torch.stack(t_ir, dim=0)
        t_ir_gt = torch.stack(t_ir_gt, dim=0)
        return t_vis, t_vis_gt, t_ir, t_ir_gt, type_name, text, vis_name, ir_name


if __name__ == "__main__":
    print("Complex degradation dataset module test code.")
