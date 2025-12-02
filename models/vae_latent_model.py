import logging
from collections import OrderedDict
import os
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


class LatentModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)

        if opt["dist"]:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt["train"]

        # define network and load pretrained models
        self.model = networks.define_AE(opt).to(self.device)
        if opt["dist"]:
            self.model = DistributedDataParallel(
                self.model, device_ids=[torch.cuda.current_device()]
            )
        # else:
        #     self.model = DataParallel(self.model)
        # print network
        # self.print_network()
        self.load()

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


    def optimize_parameters(self, step):
        self.optimizer.zero_grad()

        # 1. 获取 KL 权重 (防止 KL loss 淹没重建 loss)
        # 建议在 yaml配置中添加 kl_weight: 0.00001
        kl_weight = self.opt.get("kl_weight", 1e-6)

        if self.opt["dist"]:
            encode_fn = self.model.module.encode
            decode_fn = self.model.module.decode
        else:
            encode_fn = self.model.encode
            decode_fn = self.model.decode

        # =====================================================================
        # 2. Encode 阶段 (输出变更为: z, (mean, logvar), skips)
        # =====================================================================
        # X_LQ
        X_z_lq, (X_mean_lq, X_logvar_lq), X_H_lq = encode_fn(self.X_LQ)
        # X_GT
        X_z_gt, (X_mean_gt, X_logvar_gt), X_H_gt = encode_fn(self.X_GT)
        # Y_LQ
        Y_z_lq, (Y_mean_lq, Y_logvar_lq), Y_H_lq = encode_fn(self.Y_LQ)
        # Y_GT
        Y_z_gt, (Y_mean_gt, Y_logvar_gt), Y_H_gt = encode_fn(self.Y_GT)

        # =====================================================================
        # 3. 计算 KL Loss (对 4 个 Latent 分布进行正则化)
        # =====================================================================
        def calc_kl(mean, logvar):
            # 公式: -0.5 * sum(1 + log(var) - mean^2 - var)
            # 这里的 sum 包含了 batch, channel, h, w
            return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

        # 计算总的 KL loss (除以 batch_size 进行归一化是常见做法，取决于你的 loss_fn 定义)
        # 这里的 total_kl 是所有像素的 loss 之和
        total_kl = calc_kl(X_mean_lq, X_logvar_lq) + \
                   calc_kl(X_mean_gt, X_logvar_gt) + \
                   calc_kl(Y_mean_lq, Y_logvar_lq) + \
                   calc_kl(Y_mean_gt, Y_logvar_gt)

        # 归一化: 通常除以 (batch_size * C * H * W) 或者仅除以 batch_size
        # 这里为了与 L1 Loss (通常是 mean reduction) 匹配，建议除以总元素数量
        num_elements = X_z_lq.numel()  # B * C * H * W
        kl_loss_mean = total_kl / num_elements

        # =====================================================================
        # 4. Decode 阶段 (必须使用采样的 z 进行重建)
        # =====================================================================

        # --- 同模态重建 ---
        X_rec_llq_hlq = decode_fn(X_z_lq, X_H_lq)  # X true LQ
        X_rec_llq_hgt = decode_fn(X_z_lq, X_H_gt)  # X fake LQ (Content:LQ, Style:GT)

        X_rec_lgt_hgt = decode_fn(X_z_gt, X_H_gt)  # X true GT
        X_rec_lgt_hlq = decode_fn(X_z_gt, X_H_lq)  # X fake GT

        Y_rec_llq_hlq = decode_fn(Y_z_lq, Y_H_lq)  # Y true LQ
        Y_rec_llq_hgt = decode_fn(Y_z_lq, Y_H_gt)  # Y fake LQ

        Y_rec_lgt_hgt = decode_fn(Y_z_gt, Y_H_gt)  # Y true GT
        Y_rec_lgt_hlq = decode_fn(Y_z_gt, Y_H_lq)  # Y fake GT

        # --- 跨模态重建 (Latent来自X/Y, Skips来自Y/X) ---
        rec_X_llq_Y_hlq = decode_fn(X_z_lq, Y_H_lq)
        rec_X_llq_Y_hgt = decode_fn(X_z_lq, Y_H_gt)

        rec_X_lgt_Y_hlq = decode_fn(X_z_gt, Y_H_lq)
        rec_X_lgt_Y_hgt = decode_fn(X_z_gt, Y_H_gt)

        rec_Y_llq_X_hlq = decode_fn(Y_z_lq, X_H_lq)
        rec_Y_llq_X_hgt = decode_fn(Y_z_lq, X_H_gt)

        rec_Y_lgt_X_hlq = decode_fn(Y_z_gt, X_H_lq)
        rec_Y_lgt_X_hgt = decode_fn(Y_z_gt, X_H_gt)

        # =====================================================================
        # 5. 计算重建 Loss (保持原有逻辑)
        # =====================================================================
        X_loss_rec = self.loss_fn(X_rec_llq_hlq, self.X_LQ) + \
                     self.loss_fn(X_rec_lgt_hgt, self.X_GT) + \
                     self.loss_fn(X_rec_lgt_hlq, self.X_GT) + \
                     self.loss_fn(X_rec_llq_hgt, self.X_LQ)

        Y_loss_rec = self.loss_fn(Y_rec_llq_hlq, self.Y_LQ) + \
                     self.loss_fn(Y_rec_lgt_hgt, self.Y_GT) + \
                     self.loss_fn(Y_rec_lgt_hlq, self.Y_GT) + \
                     self.loss_fn(Y_rec_llq_hgt, self.Y_LQ)

        X_Y_loss_rec = self.loss_fn(rec_X_llq_Y_hlq, self.X_LQ) + \
                       self.loss_fn(rec_X_lgt_Y_hlq, self.X_GT) + \
                       self.loss_fn(rec_X_lgt_Y_hgt, self.X_GT) + \
                       self.loss_fn(rec_X_llq_Y_hgt, self.X_LQ)

        Y_X_loss_rec = self.loss_fn(rec_Y_llq_X_hlq, self.Y_LQ) + \
                       self.loss_fn(rec_Y_lgt_X_hlq, self.Y_GT) + \
                       self.loss_fn(rec_Y_lgt_X_hgt, self.Y_GT) + \
                       self.loss_fn(rec_Y_llq_X_hgt, self.Y_LQ)

        # 总重建损失
        total_rec_loss = X_loss_rec + Y_loss_rec + X_Y_loss_rec + Y_X_loss_rec

        # =====================================================================
        # 6. 总 Loss (重建 + KL)
        # =====================================================================
        loss = total_rec_loss + (kl_weight * kl_loss_mean)

        loss.backward()
        self.optimizer.step()

        # set log
        self.log_dict["X_loss_rec"] = X_loss_rec.item()
        self.log_dict["Y_loss_rec"] = Y_loss_rec.item()
        self.log_dict["X_Y_loss_rec"] = X_Y_loss_rec.item()
        self.log_dict["Y_X_loss_rec"] = Y_X_loss_rec.item()
        # [新增] 记录 KL Loss
        self.log_dict["kl_loss"] = kl_loss_mean.item()

    def optimize_latent_vae(self, step):
        # =========================================================
        # 0. 准备工作
        # =========================================================
        # 确保 TransformerUNet 处于评估模式且不计算梯度
        self.model.eval()
        self.latent_vae.train()
        self.optimizer_vae.zero_grad()  # 这是一个新的优化器，专门优化 LatentVAE

        kl_weight = 0.0001  # 这里的 KL 权重可以稍微大一点，因为 Latent 空间很容易对齐

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
        recon_latents, mean, logvar, z_sample = self.latent_vae(targets)

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
        self.optimizer_vae.step()

        # 记录日志
        self.log_dict["vae_loss_rec"] = loss_rec.item()
        self.log_dict["vae_loss_kl"] = loss_kl.item()

    def test(self):
        self.model.eval()

        if self.opt["dist"]:
            encode_fn = self.model.module.encode
            decode_fn = self.model.module.decode
        else:
            encode_fn = self.model.encode
            decode_fn = self.model.decode

        with torch.no_grad():
            # =================================================================
            # 1. Encode: 解包三个返回值 (z, dist_params, skips)
            # =================================================================
            # X_LQ
            X_z_lq, (X_mean_lq, X_logvar_lq), X_H_lq = encode_fn(self.X_LQ)
            # X_GT
            X_z_gt, (X_mean_gt, X_logvar_gt), X_H_gt = encode_fn(self.X_GT)
            # Y_LQ
            Y_z_lq, (Y_mean_lq, Y_logvar_lq), Y_H_lq = encode_fn(self.Y_LQ)
            # Y_GT
            Y_z_gt, (Y_mean_gt, Y_logvar_gt), Y_H_gt = encode_fn(self.Y_GT)

            # 保存用于可视化的方差图 (计算标准差并对通道取平均，得到空间热力图)
            # shape: [B, 1, H/16, W/16]
            self.X_std_lq = torch.exp(0.5 * X_logvar_lq).mean(dim=1, keepdim=True)
            self.Y_std_lq = torch.exp(0.5 * Y_logvar_lq).mean(dim=1, keepdim=True)

            # =================================================================
            # 2. Decode: 使用均值 z 进行确定性重建
            # =================================================================

            # 同模态
            self.X_real_lq = decode_fn(X_z_lq, X_H_lq)  # latent LQ, hidden LQ
            self.X_fake_gt = decode_fn(X_z_gt, X_H_lq)  # latent GT, hidden LQ

            self.X_fake_lq = decode_fn(X_z_lq, X_H_gt)  # latent LQ, hidden GT
            self.X_real_gt = decode_fn(X_z_gt, X_H_gt)  # latent GT, hidden GT

            self.Y_real_lq = decode_fn(Y_z_lq, Y_H_lq)  # latent LQ, hidden LQ
            self.Y_fake_gt = decode_fn(Y_z_gt, Y_H_lq)  # latent GT, hidden LQ

            self.Y_fake_lq = decode_fn(Y_z_lq, Y_H_gt)  # latent LQ, hidden GT
            self.Y_real_gt = decode_fn(Y_z_gt, Y_H_gt)  # latent GT, hidden GT

            # 跨模态 (使用 X 的 Latent 和 Y 的 Skips，反之亦然)
            self.X_fake_lq_by_Ylq = decode_fn(X_z_lq, Y_H_lq)
            self.X_fake_lq_by_Ygt = decode_fn(X_z_lq, Y_H_gt)

            self.X_fake_gt_by_Ylq = decode_fn(X_z_gt, Y_H_lq)
            self.X_fake_gt_by_Ygt = decode_fn(X_z_gt, Y_H_gt)

            self.Y_fake_lq_by_Xlq = decode_fn(Y_z_lq, X_H_lq)
            self.Y_fake_lq_by_Xgt = decode_fn(Y_z_lq, X_H_gt)

            self.Y_fake_gt_by_Xlq = decode_fn(Y_z_gt, X_H_lq)
            self.Y_fake_gt_by_Xgt = decode_fn(Y_z_gt, X_H_gt)

        self.model.train()

        # =================================================================
        # 3. 保存图像 (新增方差图保存)
        # =================================================================
        test_folder = './image/'
        if not os.path.exists(test_folder):
            os.makedirs(test_folder)

        # 保存原始输入
        tvutils.save_image(self.X_LQ.data, f'image/X_LQ.png', normalize=False)
        tvutils.save_image(self.X_GT.data, f'image/X_GT.png', normalize=False)
        tvutils.save_image(self.Y_LQ.data, f'image/Y_LQ.png', normalize=False)
        tvutils.save_image(self.Y_GT.data, f'image/Y_GT.png', normalize=False)

        # [新增] 保存方差/不确定性图 (建议 normalize=True 以便观察相对强弱)
        tvutils.save_image(self.X_std_lq.data, f'image/X_LQ_std.png', normalize=True)
        tvutils.save_image(self.Y_std_lq.data, f'image/Y_LQ_std.png', normalize=True)

        # 保存重建结果
        tvutils.save_image(self.X_fake_gt.data, f'image/X_GT_fake.png', normalize=False)
        tvutils.save_image(self.X_fake_lq.data, f'image/X_LQ_fake.png', normalize=False)
        tvutils.save_image(self.X_real_gt.data, f'image/X_GT_real.png', normalize=False)
        tvutils.save_image(self.X_real_lq.data, f'image/X_LQ_real.png', normalize=False)

        tvutils.save_image(self.Y_fake_gt.data, f'image/Y_GT_fake.png', normalize=False)
        tvutils.save_image(self.Y_fake_lq.data, f'image/Y_LQ_fake.png', normalize=False)
        tvutils.save_image(self.Y_real_gt.data, f'image/Y_GT_real.png', normalize=False)
        tvutils.save_image(self.Y_real_lq.data, f'image/Y_LQ_real.png', normalize=False)

        tvutils.save_image(self.X_fake_lq_by_Ylq.data, f'image/X_LQ_fake_by_Ylq.png', normalize=False)
        tvutils.save_image(self.X_fake_lq_by_Ygt.data, f'image/X_LQ_fake_by_Ygt.png', normalize=False)
        tvutils.save_image(self.X_fake_gt_by_Ylq.data, f'image/X_GT_fake_by_Ylq.png', normalize=False)
        tvutils.save_image(self.X_fake_gt_by_Ygt.data, f'image/X_GT_fake_by_Ygt.png', normalize=False)

        tvutils.save_image(self.Y_fake_lq_by_Xlq.data, f'image/Y_LQ_fake_by_Xlq.png', normalize=False)
        tvutils.save_image(self.Y_fake_lq_by_Xgt.data, f'image/Y_LQ_fake_by_Xgt.png', normalize=False)
        tvutils.save_image(self.Y_fake_gt_by_Xlq.data, f'image/Y_GT_fake_by_Xlq.png', normalize=False)
        tvutils.save_image(self.Y_fake_gt_by_Xgt.data, f'image/Y_GT_fake_by_Xgt.png', normalize=False)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict["X_LQ"] = self.X_LQ.detach()[0].float().cpu()
        out_dict["Y_LQ"] = self.Y_LQ.detach()[0].float().cpu()

        # [新增] 将方差图加入 visuals 字典
        out_dict["X_std_lq"] = self.X_std_lq.detach()[0].float().cpu()
        out_dict["Y_std_lq"] = self.Y_std_lq.detach()[0].float().cpu()

        out_dict["X_fake_gt"] = self.X_fake_gt.detach()[0].float().cpu()
        out_dict["X_fake_lq"] = self.X_fake_lq.detach()[0].float().cpu()
        out_dict["Y_fake_gt"] = self.Y_fake_gt.detach()[0].float().cpu()
        out_dict["Y_fake_lq"] = self.Y_fake_lq.detach()[0].float().cpu()
        out_dict["X_fake_lq_by_Ylq"] = self.X_fake_lq_by_Ylq.detach()[0].float().cpu()
        out_dict["X_fake_lq_by_Ygt"] = self.X_fake_lq_by_Ygt.detach()[0].float().cpu()
        out_dict["X_fake_gt_by_Ylq"] = self.X_fake_gt_by_Ylq.detach()[0].float().cpu()
        out_dict["X_fake_gt_by_Ygt"] = self.X_fake_gt_by_Ygt.detach()[0].float().cpu()
        out_dict["Y_fake_lq_by_Xlq"] = self.Y_fake_lq_by_Xlq.detach()[0].float().cpu()
        out_dict["Y_fake_lq_by_Xgt"] = self.Y_fake_lq_by_Xgt.detach()[0].float().cpu()
        out_dict["Y_fake_gt_by_Xlq"] = self.Y_fake_gt_by_Xlq.detach()[0].float().cpu()
        out_dict["Y_fake_gt_by_Xgt"] = self.Y_fake_gt_by_Xgt.detach()[0].float().cpu()

        if need_GT:
            out_dict["X_GT"] = self.X_GT.detach()[0].float().cpu()
            out_dict["Y_GT"] = self.Y_GT.detach()[0].float().cpu()
        return out_dict

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
        self.save_network(self.model, "G", iter_label)
