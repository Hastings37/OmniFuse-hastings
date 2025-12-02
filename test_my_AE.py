import argparse
import logging
import os.path
import sys
import time
import shutil  # [Added] 用于跨平台文件操作
from collections import OrderedDict
import torchvision.utils as tvutils

import numpy as np
import torch
# from IPython import embed # [Optional] 如果没有安装 IPython 可以注释掉
import lpips

import options as option
from models import create_model

import utils as util
from data import create_dataloader, create_dataset
from data.util import bgr2ycbcr

#### options
parser = argparse.ArgumentParser()
parser.add_argument("-opt", type=str, default='options/test/test_my_AE.yaml', help="Path to options YMAL file.")
# 解析参数
args = parser.parse_args()
opt = option.parse(args.opt, is_train=False)

opt = option.dict_to_nonedict(opt)

#### mkdir and logger
util.mkdirs(
    (
        path
        for key, path in opt["path"].items()
        if not key == "experiments_root"
           and "pretrain_model" not in key
           and "resume" not in key
    )
)

# ---------------- [修改开始] 跨平台兼容处理 ----------------
result_link_path = "./result_my"

# 1. 安全删除旧的 result 链接或文件夹
if os.path.exists(result_link_path):
    # 如果是软链接 (Symlink)
    if os.path.islink(result_link_path):
        os.remove(result_link_path)
    # 如果是目录 (Directory)
    elif os.path.isdir(result_link_path):
        shutil.rmtree(result_link_path)
    # 如果是文件 (File)
    else:
        os.remove(result_link_path)

# 2. 尝试创建新的软链接
# 注意：在 Windows 上，os.symlink 通常需要管理员权限或开启开发者模式
target_path = os.path.abspath(os.path.join(opt["path"]["results_root"], ".."))
try:
    os.symlink(target_path, result_link_path)
except OSError:
    # 如果创建失败（通常是 Windows 权限问题），打印警告但不报错退出
    print(f"[Warning] Unable to create symbolic link from '{target_path}' to '{result_link_path}'.")
    print("If you are on Windows, you might need Administrator privileges or Developer Mode enabled.")
    # 如果真的非常需要这个路径，可以在这里选择使用 shutil.copytree 复制，但对于大数据集不推荐
# ---------------- [修改结束] ----------------

util.setup_logger(
    "base",
    opt["path"]["log"],
    "test_" + opt["name"],
    level=logging.INFO,
    screen=True,
    tofile=True,
)
logger = logging.getLogger("base")
logger.info(option.dict2str(opt))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt["datasets"].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info(
        "Number of test images in [{:s}]: {:d}".format(
            dataset_opt["name"], len(test_set)
        )
    )
    test_loaders.append(test_loader)

# load pretrained model by default
model = create_model(opt)
device = model.device
lpips_fn = lpips.LPIPS(net='alex').to(device)

for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt["name"]  # path opt['']
    logger.info("\nTesting [{:s}]...".format(test_set_name))
    test_start_time = time.time()
    dataset_dir = os.path.join(opt["path"]["results_root"], test_set_name)
    util.mkdir(dataset_dir)

    test_results = OrderedDict()
    test_results["psnr"] = []
    test_results["ssim"] = []
    test_results["psnr_y"] = []
    test_results["ssim_y"] = []
    test_results["lpips"] = []
    test_times = []

    for i, test_data in enumerate(test_loader):
        single_img_psnr = []
        single_img_ssim = []
        single_img_psnr_y = []
        single_img_ssim_y = []

        need_GT = False if test_loader.dataset.opt["dataroot_X_GT"] is None else True # vis
        img_path = test_data["X_GT_path"][0] if need_GT else test_data["X_LQ_path"][0] # ir
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        #### input dataset_LQ
        X_LQ, X_GT = test_data["X_LQ"], test_data["X_GT"] # vis vis
        Y_LQ, Y_GT = test_data["Y_LQ"], test_data["Y_GT"]# ir ir

        model.feed_data(X_LQ, Y_LQ, X_GT, Y_GT)
        tic = time.time()
        model.test()
        toc = time.time()
        test_times.append(toc - tic)

        visuals = model.get_current_visuals()
        # 获取结果 (保持原有逻辑)
        Get_X_fake_gt = visuals["X_fake_gt"]
        Get_X_fake_lq = visuals["X_fake_lq"]
        Get_Y_fake_gt = visuals["Y_fake_gt"]
        Get_Y_fake_lq = visuals["Y_fake_lq"]
        Get_X_fake_lq_by_Ylq = visuals["X_fake_lq_by_Ylq"]
        Get_X_fake_lq_by_Ygt = visuals["X_fake_lq_by_Ygt"]
        Get_X_fake_gt_by_Ylq = visuals["X_fake_gt_by_Ylq"]
        Get_X_fake_gt_by_Ygt = visuals["X_fake_gt_by_Ygt"]
        Get_Y_fake_lq_by_Xlq = visuals["Y_fake_lq_by_Xlq"]
        Get_Y_fake_lq_by_Xgt = visuals["Y_fake_lq_by_Xgt"]
        Get_Y_fake_gt_by_Xlq = visuals["Y_fake_gt_by_Xlq"]
        Get_Y_fake_gt_by_Xgt = visuals["Y_fake_gt_by_Xgt"]

        # Tensor 转 Img
        output_X_fake_gt = util.tensor2img(Get_X_fake_gt.squeeze())
        output_X_fake_lq = util.tensor2img(Get_X_fake_lq.squeeze())
        output_Y_fake_gt = util.tensor2img(Get_Y_fake_gt.squeeze())
        output_Y_fake_lq = util.tensor2img(Get_Y_fake_lq.squeeze())
        output_X_fake_lq_by_Ylq = util.tensor2img(Get_X_fake_lq_by_Ylq.squeeze())
        output_X_fake_lq_by_Ygt = util.tensor2img(Get_X_fake_lq_by_Ygt.squeeze())
        output_X_fake_gt_by_Ylq = util.tensor2img(Get_X_fake_gt_by_Ylq.squeeze())
        output_X_fake_gt_by_Ygt = util.tensor2img(Get_X_fake_gt_by_Ygt.squeeze())
        output_Y_fake_lq_by_Xlq = util.tensor2img(Get_Y_fake_lq_by_Xlq.squeeze())
        output_Y_fake_lq_by_Xgt = util.tensor2img(Get_Y_fake_lq_by_Xgt.squeeze())
        output_Y_fake_gt_by_Xlq = util.tensor2img(Get_Y_fake_gt_by_Xlq.squeeze())
        output_Y_fake_gt_by_Xgt = util.tensor2img(Get_Y_fake_gt_by_Xgt.squeeze())

        X_LQ_ = util.tensor2img(visuals["X_LQ"].squeeze())
        X_GT_ = util.tensor2img(visuals["X_GT"].squeeze())
        Y_LQ_ = util.tensor2img(visuals["Y_LQ"].squeeze())
        Y_GT_ = util.tensor2img(visuals["Y_GT"].squeeze())

        suffix = opt["suffix"]
        # 为了简洁，这里保持原有文件名生成逻辑
        # 建议使用 f-string 或 os.path.join 确保路径在 Win 下正常，原代码使用 os.path.join 是正确的
        if suffix:
            save_img_path_X_fake_gt = os.path.join(dataset_dir, img_name + suffix + "_X_fake_gt.png")
            save_img_path_X_fake_lq = os.path.join(dataset_dir, img_name + suffix + "_X_fake_lq.png")
            save_img_path_Y_fake_gt = os.path.join(dataset_dir, img_name + suffix + "_Y_fake_gt.png")
            save_img_path_Y_fake_lq = os.path.join(dataset_dir, img_name + suffix + "_Y_fake_lq.png")
            save_img_path_X_fake_lq_by_Ylq = os.path.join(dataset_dir, img_name + suffix + "_X_fake_lq_by_Ylq.png")
            save_img_path_X_fake_lq_by_Ygt = os.path.join(dataset_dir, img_name + suffix + "_X_fake_lq_by_Ygt.png")
            save_img_path_X_fake_gt_by_Ylq = os.path.join(dataset_dir, img_name + suffix + "_X_fake_gt_by_Ylq.png")
            save_img_path_X_fake_gt_by_Ygt = os.path.join(dataset_dir, img_name + suffix + "_X_fake_gt_by_Ygt.png")
            save_img_path_Y_fake_lq_by_Xlq = os.path.join(dataset_dir, img_name + suffix + "_Y_fake_lq_by_Xlq.png")
            save_img_path_Y_fake_lq_by_Xgt = os.path.join(dataset_dir, img_name + suffix + "_Y_fake_lq_by_Xgt.png")
            save_img_path_Y_fake_gt_by_Xlq = os.path.join(dataset_dir, img_name + suffix + "_Y_fake_gt_by_Xlq.png")
            save_img_path_Y_fake_gt_by_Xgt = os.path.join(dataset_dir, img_name + suffix + "_Y_fake_gt_by_Xgt.png")

        else:
            save_img_path_X_fake_gt = os.path.join(dataset_dir, img_name + "_X_fake_gt.png")
            save_img_path_X_fake_lq = os.path.join(dataset_dir, img_name + "_X_fake_lq.png")
            save_img_path_Y_fake_gt = os.path.join(dataset_dir, img_name + "_Y_fake_gt.png")
            save_img_path_Y_fake_lq = os.path.join(dataset_dir, img_name + "_Y_fake_lq.png")
            save_img_path_X_fake_lq_by_Ylq = os.path.join(dataset_dir, img_name + "_X_fake_lq_by_Ylq.png")
            save_img_path_X_fake_lq_by_Ygt = os.path.join(dataset_dir, img_name + "_X_fake_lq_by_Ygt.png")
            save_img_path_X_fake_gt_by_Ylq = os.path.join(dataset_dir, img_name + "_X_fake_gt_by_Ylq.png")
            save_img_path_X_fake_gt_by_Ygt = os.path.join(dataset_dir, img_name + "_X_fake_gt_by_Ygt.png")
            save_img_path_Y_fake_lq_by_Xlq = os.path.join(dataset_dir, img_name + "_Y_fake_lq_by_Xlq.png")
            save_img_path_Y_fake_lq_by_Xgt = os.path.join(dataset_dir, img_name + "_Y_fake_lq_by_Xgt.png")
            save_img_path_Y_fake_gt_by_Xlq = os.path.join(dataset_dir, img_name + "_Y_fake_gt_by_Xlq.png")
            save_img_path_Y_fake_gt_by_Xgt = os.path.join(dataset_dir, img_name + "_Y_fake_gt_by_Xgt.png")

        util.save_img(output_X_fake_gt, save_img_path_X_fake_gt)
        util.save_img(output_X_fake_lq, save_img_path_X_fake_lq)
        util.save_img(output_Y_fake_gt, save_img_path_Y_fake_gt)
        util.save_img(output_Y_fake_lq, save_img_path_Y_fake_lq)
        util.save_img(output_X_fake_lq_by_Ylq, save_img_path_X_fake_lq_by_Ylq)
        util.save_img(output_X_fake_lq_by_Ygt, save_img_path_X_fake_lq_by_Ygt)
        util.save_img(output_X_fake_gt_by_Ylq, save_img_path_X_fake_gt_by_Ylq)
        util.save_img(output_X_fake_gt_by_Ygt, save_img_path_X_fake_gt_by_Ygt)
        util.save_img(output_Y_fake_lq_by_Xlq, save_img_path_Y_fake_lq_by_Xlq)
        util.save_img(output_Y_fake_lq_by_Xgt, save_img_path_Y_fake_lq_by_Xgt)
        util.save_img(output_Y_fake_gt_by_Xlq, save_img_path_Y_fake_gt_by_Xlq)
        util.save_img(output_Y_fake_gt_by_Xgt, save_img_path_Y_fake_gt_by_Xgt)

        # remove it if you only want to save output images
        X_LQ_img_path = os.path.join(dataset_dir, img_name + "_X_LQ.png")
        X_GT_img_path = os.path.join(dataset_dir, img_name + "_X_HQ.png")
        Y_LQ_img_path = os.path.join(dataset_dir, img_name + "_Y_LQ.png")
        Y_GT_img_path = os.path.join(dataset_dir, img_name + "_Y_HQ.png")
        util.save_img(X_LQ_, X_LQ_img_path)
        util.save_img(X_GT_, X_GT_img_path)
        util.save_img(Y_LQ_, Y_LQ_img_path)
        util.save_img(Y_GT_, Y_GT_img_path)

        if need_GT:
            gt_img = X_GT_ / 255.0
            sr_img = output_X_fake_gt_by_Ylq / 255.0

            crop_border = opt["crop_border"] if opt["crop_border"] else 0
            if crop_border == 0:
                cropped_sr_img = sr_img
                cropped_gt_img = gt_img
            else:
                cropped_sr_img = sr_img[
                                 crop_border:-crop_border, crop_border:-crop_border
                                 ]
                cropped_gt_img = gt_img[
                                 crop_border:-crop_border, crop_border:-crop_border
                                 ]

            psnr = util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
            ssim = util.calculate_ssim(cropped_sr_img * 255, cropped_gt_img * 255)

            test_results["psnr"].append(psnr)
            test_results["ssim"].append(ssim)

            if len(gt_img.shape) == 3:
                if gt_img.shape[2] == 3:
                    sr_img_y = bgr2ycbcr(sr_img, only_y=True)
                    gt_img_y = bgr2ycbcr(gt_img, only_y=True)
                    if crop_border == 0:
                        cropped_sr_img_y = sr_img_y
                        cropped_gt_img_y = gt_img_y
                    else:
                        cropped_sr_img_y = sr_img_y[
                                           crop_border:-crop_border, crop_border:-crop_border
                                           ]
                        cropped_gt_img_y = gt_img_y[
                                           crop_border:-crop_border, crop_border:-crop_border
                                           ]
                    psnr_y = util.calculate_psnr(
                        cropped_sr_img_y * 255, cropped_gt_img_y * 255
                    )
                    ssim_y = util.calculate_ssim(
                        cropped_sr_img_y * 255, cropped_gt_img_y * 255
                    )

                    test_results["psnr_y"].append(psnr_y)
                    test_results["ssim_y"].append(ssim_y)

                    logger.info(
                        "img{:3d}:{:15s} - PSNR: {:.6f} dB; SSIM: {:.6f};  PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.".format(
                            i, img_name, psnr, ssim, psnr_y, ssim_y
                        )
                    )

            else:
                logger.info(
                    "img:{:15s} - PSNR: {:.6f} dB; SSIM: {:.6f}.".format(
                        img_name, psnr, ssim
                    )
                )

                test_results["psnr_y"].append(psnr)
                test_results["ssim_y"].append(ssim)
        else:
            logger.info(img_name)

    ave_psnr = sum(test_results["psnr"]) / len(test_results["psnr"])
    ave_ssim = sum(test_results["ssim"]) / len(test_results["ssim"])
    logger.info(
        "----Average PSNR/SSIM results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}\n".format(
            test_set_name, ave_psnr, ave_ssim
        )
    )
    if test_results["psnr_y"] and test_results["ssim_y"]:
        ave_psnr_y = sum(test_results["psnr_y"]) / len(test_results["psnr_y"])
        ave_ssim_y = sum(test_results["ssim_y"]) / len(test_results["ssim_y"])
        logger.info(
            "----Y channel, average PSNR/SSIM----\n\tPSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}\n".format(
                ave_psnr_y, ave_ssim_y
            )
        )

    print(f"average test time: {np.mean(test_times):.4f}")