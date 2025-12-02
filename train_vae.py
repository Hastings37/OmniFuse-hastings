import argparse
import logging
import math
import os
import random
import sys
import copy
import gc
import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import options as option
from models import create_model

import utils as util
from data import create_dataloader, create_dataset
from data.data_sampler import DistIterSampler

from data.util import bgr2ycbcr


def val_data(train_loader,val_loader):
    train_data = next(iter(train_loader))
    val_data = next(iter(val_loader))
    vis_lq, ir_lq, vis_gt, ir_gt, full_train, text_train, name_train = train_data
    vis_lq_val, ir_lq_val, vis_gt_val, ir_gt_val, full_val, text_val, name_val = val_data
    print(
        f'vis_lq.shape:{vis_lq.shape}, ir_lq.shape:{ir_lq.shape}, vis_gt.shape:{vis_gt.shape}, ir_gt.shape:{ir_gt.shape}')
    print(f'full_train.shape:{full_train.shape}, text_train.shape:{text_train}, name_train.shape:{name_train}')
    print(
        f'vis_lq_val.shape:{vis_lq_val.shape}, ir_lq_val.shape:{ir_lq_val.shape}, vis_gt_val.shape:{vis_gt_val.shape}, ir_gt_val.shape:{ir_gt_val.shape}')
    print(f'full_val.shape:{full_val.shape}, text_val.shape:{text_val}, name_val.shape:{name_val}')

    return

def val_model(model):
    pass

def init_dist(backend="nccl", **kwargs):
    if (
            mp.get_start_method(allow_none=True) != "spawn"
    ):
        mp.set_start_method("spawn", force=True)
    rank = int(os.environ["RANK"])  # system env process ranks
    num_gpus = torch.cuda.device_count()  # Returns the number of GPUs available
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(
        backend=backend, **kwargs
    )  # Initializes the default distributed process group


def main():
    #### setup options of three networks
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt",default='./options/train/train_vae.yaml' ,type=str, help="Path to option YMAL file.")
    parser.add_argument(
        "--launcher", choices=["none", "pytorch"], default="none", help="job launcher"
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    # choose small opt for SFTMD test, fill path of pre-trained model_F
    #### set random seed
    seed = opt["train"]["manual_seed"]

    #### distributed training settings
    if args.launcher == "none":  # disabled distributed training
        opt["dist"] = False
        opt["dist"] = False
        rank = -1
        print("Disabled distributed training.")
    else:
        opt["dist"] = True
        opt["dist"] = True
        init_dist()
        world_size = (
            torch.distributed.get_world_size()
        )  # Returns the number of processes in the current process group
        rank = torch.distributed.get_rank()  # Returns the rank of current process group
        # util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True

    ###### Predictor&Corrector train ######

    #### loading resume state if exists
    if opt["path"].get("resume_state", None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt["path"]["resume_state"],
            map_location=lambda storage, loc: storage.cuda(device_id),
        )
        option.check_resume(opt, resume_state["iter"])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0-7)
        if resume_state is None:
            # Predictor path
            util.mkdir_and_rename(
                opt["path"]["experiments_root"]
            )  # rename experiment folder if exists
            util.mkdirs(
                (
                    path
                    for key, path in opt["path"].items()
                    if not key == "experiments_root"
                       and "pretrain_model" not in key
                       and "resume" not in key
                )
            )
            import shutil
            import os
            log_dir = "./log"

            # 删除已有 log 文件夹
            if os.path.exists(log_dir):
                shutil.rmtree(log_dir)

            # 找到 experiments_root 的上一级目录
            src = os.path.abspath(os.path.join(opt["path"]["experiments_root"], ".."))

            # 拷贝整个目录到 log
            shutil.copytree(src, log_dir)

        # config loggers. Before it, the log will not work
        util.setup_logger(
            "base",
            opt["path"]["log"],
            "train_" + opt["name"],
            level=logging.INFO,
            screen=False,
            tofile=True,
        )
        util.setup_logger(
            "val",
            opt["path"]["log"],
            "val_" + opt["name"],
            level=logging.INFO,
            screen=False,
            tofile=True,
        )
        logger = logging.getLogger("base")
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt["use_tb_logger"] and "debug" not in opt["name"]:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    "You are using PyTorch {}. Tensorboard will use [tensorboardX]".format(
                        version
                    )
                )
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir="log/{}/tb_logger/".format(opt["name"]))
    else:
        util.setup_logger(
            "base", opt["path"]["log"], "train", level=logging.INFO, screen=False
        )
        logger = logging.getLogger("base")

    #### create train and val dataloader
    dataset_ratio = 1  # enlarge the size of each epoch
    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt["batch_size"]))
            total_iters = int(opt["train"]["niter"])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt["dist"]:
                train_sampler = DistIterSampler(
                    train_set, world_size, rank, dataset_ratio
                )
                total_epochs = int(
                    math.ceil(total_iters / (train_size * dataset_ratio))
                )
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info(
                    "Number of train images: {:,d}, iters: {:,d}".format(
                        len(train_set), train_size
                    )
                )
                logger.info(
                    "Total epochs needed: {:d} for iters {:,d}".format(
                        total_epochs, total_iters
                    )
                )
        elif phase == "val":
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info(
                    "Number of val images in [{:s}]: {:d}".format(
                        dataset_opt["name"], len(val_set)
                    )
                )
        else:
            raise NotImplementedError("Phase [{:s}] is not recognized.".format(phase))
    assert train_loader is not None
    assert val_loader is not None


    #### create model
    model = create_model(opt)
    device = model.device

    #### resume training
    if resume_state:
        logger.info(
            "Resuming training from epoch: {}, iter: {}.".format(
                resume_state["epoch"], resume_state["iter"]
            )
        )

        start_epoch = resume_state["epoch"]
        current_step = resume_state["iter"]
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    #### training
    logger.info(
        "Start training from epoch: {:d}, iter: {:d}".format(start_epoch, current_step)
    )

    best_psnr = 0.0
    best_iter = 0
    error = mp.Value('b', False)

    for epoch in range(start_epoch, total_epochs + 1):
        if opt["dist"]:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            current_step += 1

            if current_step > total_iters:
                break

            vis,ir,vis_gt,ir_gt,full,text,name= train_data
            X_LQ=vis.to(device)
            Y_LQ=ir.to(device)
            X_GT=vis_gt.to(device)
            Y_GT=ir_gt.to(device)
            model.feed_data(X_LQ, Y_LQ, X_GT, Y_GT)  # xt, mu, x0
            model.optimize_parameters(current_step)
            model.update_learning_rate(
                current_step, warmup_iter=opt["train"]["warmup_iter"]
            )

            if current_step % opt["logger"]["print_freq"] == 0:
                logs = model.get_current_log()
                message = "<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> ".format(
                    epoch, current_step, model.get_current_learning_rate()
                )
                for k, v in logs.items():
                    message += "{:s}: {:.4e} ".format(k, v)
                    # tensorboard logger
                    if opt["use_tb_logger"] and "debug" not in opt["name"]:
                        if rank <= 0:
                            tb_logger.add_scalar(k, v, current_step)
                if rank <= 0:
                    logger.info(message)

            # validation
            if current_step % opt["train"]["val_freq"] == 0 and rank <= 0:
                avg_psnr = 0.0
                val_count = 0

                max_iter = 8

                from tqdm import tqdm
                pbar = tqdm(val_loader, desc=f"[VAL] iter={current_step}", total=max_iter)

                for v_i, val_data in enumerate(pbar):
                    if v_i >= max_iter: # 0-N-1
                        break  # 限制验证数量

                    vis_val, ir_val, vis_gt_val, ir_gt_val, full_val, text_val, name_val = val_data
                    X_LQ = vis_val.to(device)
                    Y_LQ = ir_val.to(device)
                    X_GT = vis_gt_val.to(device)
                    Y_GT = ir_gt_val.to(device)

                    model.feed_data(X_LQ, Y_LQ, X_GT, Y_GT)
                    model.test()
                    visuals = model.get_current_visuals()

                    X_fake_gt_img = util.tensor2img(visuals["X_fake_gt"].squeeze())
                    X_GT_img = util.tensor2img(visuals["X_GT"].squeeze())
                    Y_fake_gt_img = util.tensor2img(visuals["Y_fake_gt"].squeeze())
                    Y_GT_img = util.tensor2img(visuals["Y_GT"].squeeze())

                    X_fake_gt_by_Ylq_img = util.tensor2img(visuals["X_fake_gt_by_Ylq"].squeeze())
                    X_fake_gt_by_Ygt_img = util.tensor2img(visuals["X_fake_gt_by_Ygt"].squeeze())
                    Y_fake_gt_by_Xlq_img = util.tensor2img(visuals["Y_fake_gt_by_Xlq"].squeeze())
                    Y_fake_gt_by_Xgt_img = util.tensor2img(visuals["Y_fake_gt_by_Xgt"].squeeze())

                    psnr = (
                                   util.calculate_psnr(X_fake_gt_img, X_GT_img) +
                                   util.calculate_psnr(Y_fake_gt_img, Y_GT_img) +
                                   util.calculate_psnr(X_fake_gt_by_Ylq_img, X_GT_img) +
                                   util.calculate_psnr(X_fake_gt_by_Ygt_img, X_GT_img) +
                                   util.calculate_psnr(Y_fake_gt_by_Xlq_img, Y_GT_img) +
                                   util.calculate_psnr(Y_fake_gt_by_Xgt_img, Y_GT_img)
                           ) / 6.0 # 自身的从低分辨率到

                    avg_psnr += psnr
                    val_count += 1

                    # tqdm 更新动态信息
                    pbar.set_postfix({
                        "psnr": f"{psnr:.4f}",
                        "img": str(name_val[0])
                    })

                    # 避免显存累积
                    gc.collect()
                    torch.cuda.empty_cache()

                # 真实的验证均值
                avg_psnr /= val_count
                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    best_iter = current_step

                # log
                logger.info(
                    "# Validation # PSNR: {:.6f}, Best PSNR: {:.6f}| Iter: {}".format(avg_psnr, best_psnr, best_iter))
                logger_val = logging.getLogger("val")  # validation logger
                logger_val.info(
                    "<epoch:{:3d}, iter:{:8,d}, psnr: {:.6f}".format(
                        epoch, current_step, avg_psnr
                    )
                )
                print("<epoch:{:3d}, iter:{:8,d}, psnr: {:.6f}".format(
                    epoch, current_step, avg_psnr
                ))
                # tensorboard logger
                if opt["use_tb_logger"] and "debug" not in opt["name"]:
                    tb_logger.add_scalar("psnr", avg_psnr, current_step)

            if error.value:
                sys.exit(0)
            #### save models and training states
            if current_step % opt["logger"]["save_checkpoint_freq"] == 0:
                if rank <= 0:
                    logger.info("Saving models and training states.")
                    model.save(current_step)
                    model.save_training_state(epoch, current_step)

    if rank <= 0:
        logger.info("Saving the final model.")
        model.save("latest")
        logger.info("End of Predictor and Corrector training.")
    tb_logger.close()


if __name__ == "__main__":
    main()