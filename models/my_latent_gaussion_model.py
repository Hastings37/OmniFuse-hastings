import logging
from collections import OrderedDict
import os
from email.policy import strict

import numpy as np

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torchvision.utils as tvutils
from tqdm import tqdm

import models.lr_scheduler as lr_scheduler
import models.networks as networks
from models.optimizer import Lion
from models.modules.loss import MatchingLoss, PerceptualMatchingLoss

from .base_model import BaseModel

logger = logging.getLogger("base")

'''
将数据内容通入到encoder 之后得到潜在的特征内容 
随后将其通过这里的对应网络映射到 标准的高斯分布状态中去；
'''
from models.modules.ae_arch import TransformerUNet  # 这里的内容也是带有归一化的操作的；
from models.modules.latent_to_gaussion import LatentVAE


class LatentModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)

        if opt["dist"]:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt["train"]

        # define network and load pretrained models
        # 按照默认的设置定义起来；
        self.latent_model = TransformerUNet()
        self.latent_model.load_state_dict(torch.load(opt['path']['pretrain_model_G']), strict=True)
        self.latent_model.eval().to(self.device)
        for param in self.latent_model.parameters():
            param.requires_grad = False

        # 这里的in_channels 就是上面的Latent 对应的维度内容了；
        self.model = LatentVAE().to(self.device)
        self.model.train()

        if opt["dist"]:
            self.model = DistributedDataParallel(
                self.model, device_ids=[torch.cuda.current_device()]
            )
        else:
            self.model = DataParallel(self.model)
        self.print_network()
        # self.load()

        if self.is_train:
            self.model.train()

            is_weighted = opt['train']['is_weighted']
            loss_type = opt['train']['loss_type']
            self.loss_fn = MatchingLoss(loss_type, is_weighted).to(self.device)
            # self.loss_fn = PerceptualMatchingLoss(loss_type, is_weighted).to(self.device)
            self.weight = opt['train']['weight']

            # optimizers
            wd_G = train_opt["weight_decay_G"] if train_opt["weight_decay_G"] else 0
            optim_params = []
            for (
                    k,
                    v,
            ) in self.model.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning("Params [{:s}] will not optimize.".format(k))

            if train_opt['optimizer'] == 'Adam':
                self.optimizer = torch.optim.Adam(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            elif train_opt['optimizer'] == 'AdamW':
                self.optimizer = torch.optim.AdamW(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            elif train_opt['optimizer'] == 'Lion':
                self.optimizer = Lion(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            else:
                print('Not implemented optimizer, default using Adam!')
                self.optimizer = torch.optim.Adam(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )

            self.optimizers.append(self.optimizer)

            # schedulers
            if train_opt["lr_scheme"] == "MultiStepLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(
                            optimizer,
                            train_opt["lr_steps"],
                            restarts=train_opt["restarts"],
                            weights=train_opt["restart_weights"],
                            gamma=train_opt["lr_gamma"],
                            clear_state=train_opt["clear_state"],
                        )
                    )
            elif train_opt["lr_scheme"] == "CosineAnnealingLR_Restart":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer,
                            train_opt["T_period"],
                            eta_min=train_opt["eta_min"],
                            restarts=train_opt["restarts"],
                            weights=train_opt["restart_weights"],
                        )
                    )
            elif train_opt["lr_scheme"] == "TrueCosineAnnealingLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer,
                            T_max=train_opt["niter"],  # 超出的情况就确定为 eta_min 的状态了；
                            eta_min=train_opt["eta_min"])
                    )
            else:
                raise NotImplementedError("MultiStepLR learning rate scheme is enough.")

            # self.ema = EMA(self.model, beta=0.995, update_every=10).to(self.device)
            self.log_dict = OrderedDict()

    def feed_data(self, X_LQ, Y_LQ, X_GT=None, Y_GT=None):
        self.X_LQ = X_LQ.to(self.device)  # LQ
        self.X_GT = X_GT.to(self.device) if X_GT is not None else None
        self.Y_LQ = Y_LQ.to(self.device)  # LQ
        self.Y_GT = Y_GT.to(self.device) if Y_GT is not None else None

    def optimize_latent_vae(self, step):
        # =========================================================
        # 0. 准备工作
        # =========================================================
        # 确保 TransformerUNet 处于评估模式且不计算梯度
        # self.latent_model  and self.model
        self.optimizer.zero_grad()  # 这是一个新的优化器，专门优化 LatentVAE

        kl_weight = 0.001  # 这里的 KL 权重可以稍微大一点，因为 Latent 空间很容易对齐

        with torch.no_grad():
            # 1. 从冻结的 TransformerUNet 中提取 Latent (作为 Ground Truth)
            # 注意：这里我们使用 encode 返回的第一个值（如果是确定性AE，直接用；如果是VAE结构，用mean）
            # 假设你的 encode 返回: latent, skips

            # 提取 X 的特征
            target_X_lq, _ = self.model.encode(self.X_LQ)
            target_X_gt, _ = self.model.encode(self.X_GT)

            # 提取 Y 的特征
            target_Y_lq, _ = self.model.encode(self.Y_LQ)
            target_Y_gt, _ = self.model.encode(self.Y_GT)

        # =========================================================
        # 1. Latent VAE 前向传播
        # =========================================================
        # 我们将 4 组 Latent 拼接在一起批量处理，效率更高
        # targets shape: [4B, 16, 64, 64]
        targets = torch.cat([target_X_lq, target_X_gt, target_Y_lq, target_Y_gt], dim=0)

        # 输入给小 VAE
        recon_latents, mean, logvar, z_sample = self.model(targets)

        # =========================================================
        # 2. 计算损失
        # =========================================================

        # (A) Latent 重建损失 (MSE): 保证 z 能还原回 Transformer 的特征
        # 这一步保证了语义信息不丢失
        loss_rec = F.mse_loss(recon_latents, targets)

        # (B) KL 散度损失: 保证 z 服从标准正态分布
        # 这一步保证了扩散模型有一个完美的训练目标
        def calc_kl(mean, logvar):
            return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

        loss_kl = calc_kl(mean, logvar) / targets.numel()  # 归一化

        # 总损失
        loss = loss_rec + (kl_weight * loss_kl)

        # =========================================================
        # 3. 反向传播
        # =========================================================
        loss.backward()
        self.optimizer.step()

        # 记录日志
        self.log_dict["vae_loss_rec"] = loss_rec.item()
        self.log_dict["vae_loss_kl"] = loss_kl.item()

    def optimize_parameters(self, step):
        self.optimizer.zero_grad()

        if self.opt["dist"]:
            encode_fn = self.model.module.encode
            decode_fn = self.model.module.decode
        else:
            encode_fn = self.model.encode
            decode_fn = self.model.decode

        X_L_lq, X_H_lq = encode_fn(self.X_LQ)
        X_L_gt, X_H_gt = encode_fn(self.X_GT)
        Y_L_lq, Y_H_lq = encode_fn(self.Y_LQ)
        Y_L_gt, Y_H_gt = encode_fn(self.Y_GT)

        X_rec_llq_hlq = decode_fn(X_L_lq, X_H_lq)  ## obtain X true LQ
        X_rec_llq_hgt = decode_fn(X_L_lq, X_H_gt)  ## obtain X fake LQ

        X_rec_lgt_hgt = decode_fn(X_L_gt, X_H_gt)  ## obtain X true GT
        X_rec_lgt_hlq = decode_fn(X_L_gt, X_H_lq)  ## obtain X fake GT

        Y_rec_llq_hlq = decode_fn(Y_L_lq, Y_H_lq)  ## obtain Y true LQ
        Y_rec_llq_hgt = decode_fn(Y_L_lq, Y_H_gt)  ## obtain Y fake LQ

        Y_rec_lgt_hgt = decode_fn(Y_L_gt, Y_H_gt)  ## obtain Y true GT
        Y_rec_lgt_hlq = decode_fn(Y_L_gt, Y_H_lq)  ## obtain Y fake GT

        rec_X_llq_Y_hlq = decode_fn(X_L_lq, Y_H_lq)  ## obtain X fake LQ
        rec_X_llq_Y_hgt = decode_fn(X_L_lq, Y_H_gt)  ## obtain X fake LQ

        rec_X_lgt_Y_hlq = decode_fn(X_L_gt, Y_H_lq)  ## obtain X fake GT
        rec_X_lgt_Y_hgt = decode_fn(X_L_gt, Y_H_gt)  ## obtain X fake GT

        rec_Y_llq_X_hlq = decode_fn(Y_L_lq, X_H_lq)  ## obtain Y fake LQ
        rec_Y_llq_X_hgt = decode_fn(Y_L_lq, X_H_gt)  ## obtain Y fake LQ

        rec_Y_lgt_X_hlq = decode_fn(Y_L_gt, X_H_lq)  ## obtain Y fake GT
        rec_Y_lgt_X_hgt = decode_fn(Y_L_gt, X_H_gt)  ## obtain Y fake GT

        X_loss_rec = self.loss_fn(X_rec_llq_hlq, self.X_LQ) + self.loss_fn(X_rec_lgt_hgt, self.X_GT) + self.loss_fn(
            X_rec_lgt_hlq, self.X_GT) + self.loss_fn(X_rec_llq_hgt, self.X_LQ)
        Y_loss_rec = self.loss_fn(Y_rec_llq_hlq, self.Y_LQ) + self.loss_fn(Y_rec_lgt_hgt, self.Y_GT) + self.loss_fn(
            Y_rec_lgt_hlq, self.Y_GT) + self.loss_fn(Y_rec_llq_hgt, self.Y_LQ)
        X_Y_loss_rec = self.loss_fn(rec_X_llq_Y_hlq, self.X_LQ) + self.loss_fn(rec_X_lgt_Y_hlq,
                                                                               self.X_GT) + self.loss_fn(
            rec_X_lgt_Y_hgt, self.X_GT) + self.loss_fn(rec_X_llq_Y_hgt, self.X_LQ)
        Y_X_loss_rec = self.loss_fn(rec_Y_llq_X_hlq, self.Y_LQ) + self.loss_fn(rec_Y_lgt_X_hlq,
                                                                               self.Y_GT) + self.loss_fn(
            rec_Y_lgt_X_hgt, self.Y_GT) + self.loss_fn(rec_Y_llq_X_hgt, self.Y_LQ)

        loss = X_loss_rec + Y_loss_rec + X_Y_loss_rec + Y_X_loss_rec  # + loss_reg * 0.001
        loss.backward()
        self.optimizer.step()

        # set log
        self.log_dict["X_loss_rec"] = X_loss_rec.item()
        self.log_dict["Y_loss_rec"] = Y_loss_rec.item()
        self.log_dict["X_Y_loss_rec"] = X_Y_loss_rec.item()
        self.log_dict["Y_X_loss_rec"] = Y_X_loss_rec.item()

    def test(self):
        # 1. 切换模式
        self.model.eval()  # 正在训练的小 VAE
        self.latent_model.eval()  # 冻结的大 TransformerUNet

        # 2. 获取大模型的编解码函数句柄
        if self.opt["dist"]:
            encode_fn = self.latent_model.module.encode
            decode_fn = self.latent_model.module.decode
        else:
            encode_fn = self.latent_model.encode
            decode_fn = self.latent_model.decode

        with torch.no_grad():
            # =================================================================
            # Step 1: 使用 self.latent_model 提取原始特征 (Latent + Skips)
            # =================================================================
            # X 模态
            X_lat_lq, X_skips_lq = encode_fn(self.X_LQ)
            X_lat_gt, X_skips_gt = encode_fn(self.X_GT)

            # Y 模态
            Y_lat_lq, Y_skips_lq = encode_fn(self.Y_LQ)
            Y_lat_gt, Y_skips_gt = encode_fn(self.Y_GT)

            # =================================================================
            # Step 2: 生成 [原始] 重建图像 (Baseline, 不经过 VAE)
            # =================================================================

            # --- 同模态 (Intra-modal) ---
            self.X_real_lq = decode_fn(X_lat_lq, X_skips_lq)  # X_LQ 自重建
            self.X_fake_gt = decode_fn(X_lat_gt, X_skips_lq)  # X_GT 内容 + X_LQ 细节 (复原)
            self.X_fake_lq = decode_fn(X_lat_lq, X_skips_gt)  # X_LQ 内容 + X_GT 细节 (退化)
            self.X_real_gt = decode_fn(X_lat_gt, X_skips_gt)  # X_GT 自重建

            self.Y_real_lq = decode_fn(Y_lat_lq, Y_skips_lq)
            self.Y_fake_gt = decode_fn(Y_lat_gt, Y_skips_lq)
            self.Y_fake_lq = decode_fn(Y_lat_lq, Y_skips_gt)
            self.Y_real_gt = decode_fn(Y_lat_gt, Y_skips_gt)

            # --- 跨模态 (Cross-modal) ---
            # X 内容, Y 细节
            self.X_fake_lq_by_Ylq = decode_fn(X_lat_lq, Y_skips_lq)
            self.X_fake_lq_by_Ygt = decode_fn(X_lat_lq, Y_skips_gt)
            self.X_fake_gt_by_Ylq = decode_fn(X_lat_gt, Y_skips_lq)
            self.X_fake_gt_by_Ygt = decode_fn(X_lat_gt, Y_skips_gt)

            # Y 内容, X 细节
            self.Y_fake_lq_by_Xlq = decode_fn(Y_lat_lq, X_skips_lq)
            self.Y_fake_lq_by_Xgt = decode_fn(Y_lat_lq, X_skips_gt)
            self.Y_fake_gt_by_Xlq = decode_fn(Y_lat_gt, X_skips_lq)
            self.Y_fake_gt_by_Xgt = decode_fn(Y_lat_gt, X_skips_gt)

            # =================================================================
            # Step 3: 生成 [VAE修正] 重建图像 (主要关注 GT Latent 的修正效果)
            # =================================================================

            # 1. 运行 VAE 矫正 Latent
            # Latent VAE forward 返回: recon_x, mean, logvar, z
            # 我们取 recon_x (矫正后的特征)
            X_lat_gt_vae, _, _, _ = self.model(X_lat_gt)
            Y_lat_gt_vae, _, _, _ = self.model(Y_lat_gt)

            # 2. 解码 (添加 _vae 后缀)
            # 对应您需要的 6 种组合
            self.X_fake_gt_vae = decode_fn(X_lat_gt_vae, X_skips_lq)
            self.Y_fake_gt_vae = decode_fn(Y_lat_gt_vae, Y_skips_lq)

            self.X_fake_gt_by_Ylq_vae = decode_fn(X_lat_gt_vae, Y_skips_lq)
            self.X_fake_gt_by_Ygt_vae = decode_fn(X_lat_gt_vae, Y_skips_gt)

            self.Y_fake_gt_by_Xlq_vae = decode_fn(Y_lat_gt_vae, X_skips_lq)
            self.Y_fake_gt_by_Xgt_vae = decode_fn(Y_lat_gt_vae, X_skips_gt)

        # 恢复训练模式
        self.model.train()

        # =================================================================
        # Step 4: 保存图像
        # =================================================================
        test_folder = './image/'
        if not os.path.exists(test_folder):
            os.makedirs(test_folder)

        # 1. 保存 GT
        tvutils.save_image(self.X_GT.data, f'image/X_GT.png', normalize=False)
        tvutils.save_image(self.Y_GT.data, f'image/Y_GT.png', normalize=False)
        tvutils.save_image(self.X_LQ.data, f'image/X_LQ.png', normalize=False)
        tvutils.save_image(self.Y_LQ.data, f'image/Y_LQ.png', normalize=False)

        # 2. 保存原始重建 (Baseline)
        tvutils.save_image(self.X_fake_gt.data, f'image/X_GT_fake.png', normalize=False)
        tvutils.save_image(self.X_fake_lq.data, f'image/X_LQ_fake.png', normalize=False)
        tvutils.save_image(self.X_real_gt.data, f'image/X_GT_real.png', normalize=False)
        tvutils.save_image(self.X_real_lq.data, f'image/X_LQ_real.png', normalize=False)

        tvutils.save_image(self.Y_fake_gt.data, f'image/Y_GT_fake.png', normalize=False)
        tvutils.save_image(self.Y_fake_lq.data, f'image/Y_LQ_fake.png', normalize=False)
        tvutils.save_image(self.Y_real_gt.data, f'image/Y_GT_real.png', normalize=False)
        tvutils.save_image(self.Y_real_lq.data, f'image/Y_LQ_real.png', normalize=False)

        # 跨模态原始
        tvutils.save_image(self.X_fake_lq_by_Ylq.data, f'image/X_LQ_fake_by_Ylq.png', normalize=False)
        tvutils.save_image(self.X_fake_lq_by_Ygt.data, f'image/X_LQ_fake_by_Ygt.png', normalize=False)
        tvutils.save_image(self.X_fake_gt_by_Ylq.data, f'image/X_GT_fake_by_Ylq.png', normalize=False)
        tvutils.save_image(self.X_fake_gt_by_Ygt.data, f'image/X_GT_fake_by_Ygt.png', normalize=False)

        tvutils.save_image(self.Y_fake_lq_by_Xlq.data, f'image/Y_LQ_fake_by_Xlq.png', normalize=False)
        tvutils.save_image(self.Y_fake_lq_by_Xgt.data, f'image/Y_LQ_fake_by_Xgt.png', normalize=False)
        tvutils.save_image(self.Y_fake_gt_by_Xlq.data, f'image/Y_GT_fake_by_Xlq.png', normalize=False)
        tvutils.save_image(self.Y_fake_gt_by_Xgt.data, f'image/Y_GT_fake_by_Xgt.png', normalize=False)

        # 3. 保存 VAE 重建 (Corrected)
        tvutils.save_image(self.X_fake_gt_vae.data, f'image/X_fake_gt_vae.png', normalize=False)
        tvutils.save_image(self.Y_fake_gt_vae.data, f'image/Y_fake_gt_vae.png', normalize=False)

        tvutils.save_image(self.X_fake_gt_by_Ylq_vae.data, f'image/X_fake_gt_by_Ylq_vae.png', normalize=False)
        tvutils.save_image(self.X_fake_gt_by_Ygt_vae.data, f'image/X_fake_gt_by_Ygt_vae.png', normalize=False)

        tvutils.save_image(self.Y_fake_gt_by_Xlq_vae.data, f'image/Y_fake_gt_by_Xlq_vae.png', normalize=False)
        tvutils.save_image(self.Y_fake_gt_by_Xgt_vae.data, f'image/Y_fake_gt_by_Xgt_vae.png', normalize=False)

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        # 原始输入
        out_dict["X_LQ"] = self.X_LQ.detach()[0].float().cpu()
        out_dict["Y_LQ"] = self.Y_LQ.detach()[0].float().cpu()

        # --- 1. 原始重建结果 (Baseline) ---
        out_dict["X_fake_gt"] = self.X_fake_gt.detach()[0].float().cpu()
        out_dict["X_fake_lq"] = self.X_fake_lq.detach()[0].float().cpu()
        out_dict["X_real_gt"] = self.X_real_gt.detach()[0].float().cpu()  # 也可以不加，因为基本等于GT
        out_dict["X_real_lq"] = self.X_real_lq.detach()[0].float().cpu()

        out_dict["Y_fake_gt"] = self.Y_fake_gt.detach()[0].float().cpu()
        out_dict["Y_fake_lq"] = self.Y_fake_lq.detach()[0].float().cpu()
        out_dict["Y_real_gt"] = self.Y_real_gt.detach()[0].float().cpu()
        out_dict["Y_real_lq"] = self.Y_real_lq.detach()[0].float().cpu()

        out_dict["X_fake_lq_by_Ylq"] = self.X_fake_lq_by_Ylq.detach()[0].float().cpu()
        out_dict["X_fake_lq_by_Ygt"] = self.X_fake_lq_by_Ygt.detach()[0].float().cpu()
        out_dict["X_fake_gt_by_Ylq"] = self.X_fake_gt_by_Ylq.detach()[0].float().cpu()
        out_dict["X_fake_gt_by_Ygt"] = self.X_fake_gt_by_Ygt.detach()[0].float().cpu()

        out_dict["Y_fake_lq_by_Xlq"] = self.Y_fake_lq_by_Xlq.detach()[0].float().cpu()
        out_dict["Y_fake_lq_by_Xgt"] = self.Y_fake_lq_by_Xgt.detach()[0].float().cpu()
        out_dict["Y_fake_gt_by_Xlq"] = self.Y_fake_gt_by_Xlq.detach()[0].float().cpu()
        out_dict["Y_fake_gt_by_Xgt"] = self.Y_fake_gt_by_Xgt.detach()[0].float().cpu()

        # --- 2. VAE 重建结果 (Corrected) ---
        out_dict["X_fake_gt_vae"] = self.X_fake_gt_vae.detach()[0].float().cpu()
        out_dict["Y_fake_gt_vae"] = self.Y_fake_gt_vae.detach()[0].float().cpu()

        out_dict["X_fake_gt_by_Ylq_vae"] = self.X_fake_gt_by_Ylq_vae.detach()[0].float().cpu()
        out_dict["X_fake_gt_by_Ygt_vae"] = self.X_fake_gt_by_Ygt_vae.detach()[0].float().cpu()

        out_dict["Y_fake_gt_by_Xlq_vae"] = self.Y_fake_gt_by_Xlq_vae.detach()[0].float().cpu()
        out_dict["Y_fake_gt_by_Xgt_vae"] = self.Y_fake_gt_by_Xgt_vae.detach()[0].float().cpu()

        if need_GT:
            out_dict["X_GT"] = self.X_GT.detach()[0].float().cpu()
            out_dict["Y_GT"] = self.Y_GT.detach()[0].float().cpu()
        return out_dict

    def get_current_log(self):
        return self.log_dict


    def print_network(self):
        s, n = self.get_network_description(self.model)
        if isinstance(self.model, nn.DataParallel) or isinstance(
                self.model, DistributedDataParallel
        ):
            net_struc_str = "{} - {}".format(
                self.model.__class__.__name__, self.model.module.__class__.__name__
            )
        else:
            net_struc_str = "{}".format(self.model.__class__.__name__)
        if self.rank <= 0:
            logger.info(
                "Network G structure: {}, with parameters: {:,d}".format(
                    net_struc_str, n
                )
            )
            logger.info(s)

    def load(self):
        load_path_G = self.opt["path"]["pretrain_model_AE"]
        if load_path_G is not None:
            logger.info("Loading model for G [{:s}] ...".format(load_path_G))
            self.load_network(load_path_G, self.model, self.opt["path"]["strict_load"])

    def save(self, iter_label):
        self.save_network(self.model, "C", iter_label) # 名为cast的操作；
