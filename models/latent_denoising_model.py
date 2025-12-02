import logging
from collections import OrderedDict
import os
import numpy as np

import math
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torchvision.utils as tvutils
from tqdm import tqdm

import copy

class EMA(nn.Module):
    """
    一个简易的指数移动平均 (Exponential Moving Average) 实现。
    用于在训练过程中维护一个模型参数的滑动平均版本，通常能获得更好的泛化能力。
    """

    def __init__(self, model, beta=0.995, update_every=10):
        super().__init__()
        self.beta = beta
        self.update_every = update_every
        self.step = 0

        # 保存原模型的引用，用于在 update 时获取最新的参数
        # 注意：这只是引用，不会占用额外的显存，且能随原模型实时变化
        self.source_model = model

        # 1. 深拷贝一个模型作为 EMA 模型 (影子模型)
        self.ema_model = copy.deepcopy(model)

        # 2. 冻结 EMA 模型的所有参数
        # 我们不希望 EMA 模型参与反向传播，它只通过 update 方法更新
        for param in self.ema_model.parameters():
            param.requires_grad = False

        # 3. 初始化为 eval 模式 (这对于 BatchNorm 等层很重要)
        self.ema_model.eval()

    def update(self):
        """
        需要在训练循环的每个 step 调用此方法。
        它会根据 update_every 决定是否执行参数平滑更新。
        """
        self.step += 1
        if self.step % self.update_every == 0:
            self._update_moving_average()

    def _update_moving_average(self):
        with torch.no_grad():
            # 更新模型参数 (Parameters)
            # 公式: ema_param = beta * ema_param + (1 - beta) * current_param
            for ema_param, current_param in zip(self.ema_model.parameters(), self.source_model.parameters()):
                ema_param.data.mul_(self.beta).add_(current_param.data, alpha=1.0 - self.beta)

            # 更新缓冲区 (Buffers)
            # 例如 BatchNorm 的 running_mean 和 running_var，通常直接复制原模型的 buffer
            for ema_buffer, current_buffer in zip(self.ema_model.buffers(), self.source_model.buffers()):
                ema_buffer.data.copy_(current_buffer.data)

    def forward(self, *args, **kwargs):
        """
        允许直接像使用原模型一样使用 ema 实例：
        output = self.ema(input)
        """
        return self.ema_model(*args, **kwargs)

# from ema_pytorch import EMA

import models.lr_scheduler as lr_scheduler
import models.networks as networks
from models.optimizer import Lion

from models.modules.loss import MatchingLoss

from .base_model import BaseModel

logger = logging.getLogger("base")


class DenoisingModel(BaseModel):
    def __init__(self, opt):
        super(DenoisingModel, self).__init__(opt)

        os.makedirs('image', exist_ok=True) # 用来存储中间生成的图片内容；

        if opt["dist"]:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt["train"]

        # define network and load pretrained models
        self.model = networks.define_Diff(opt).to(self.device) # 图像维度原始情况下应该是 480
        # 30 * 30
        self.latent_model = networks.define_AE(opt).to(self.device)

        for param in self.latent_model.parameters():
                param.requires_grad = False

        if opt["dist"]:
            self.model = DistributedDataParallel(self.model, device_ids=[torch.cuda.current_device()])

        self.load()

        self.encode = self.latent_model.encode
        self.decode = self.latent_model.decode

        if self.is_train:
            self.model.train()

            is_weighted = opt['train']['is_weighted']
            loss_type = opt['train']['loss_type']
            self.loss_fn = MatchingLoss(loss_type, is_weighted).to(self.device)
            self.weight = opt['train']['weight'] # None 指定的权重信息；

            # optimizers
            wd_G = train_opt["weight_decay_G"] if train_opt["weight_decay_G"] else 0
            optim_params = []
            for (
                k,
                v,
            ) in self.model.named_parameters():  # can optimize for a part of the model
                #if 'NAFBlock' in k:
                #    v.requires_grad = False
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
                            T_max=train_opt["niter"],
                            eta_min=train_opt["eta_min"])
                    ) 
            else:
                raise NotImplementedError("MultiStepLR learning rate scheme is enough.")

            self.ema = EMA(self.model, beta=0.995, update_every=10).to(self.device)
            self.log_dict = OrderedDict()

    def feed_data(self, state, LQ, GT=None):
        self.state = state.to(self.device)    # noisy_state
        self.condition = LQ.to(self.device)  # LQ
        if GT is not None:
            self.state_0 = GT.to(self.device)  # GT
        else:
            self.state_0 = None

    def optimize_parameters(self, step, timesteps, sde=None):
        sde.set_mu(self.condition)

        self.optimizer.zero_grad()

        timesteps = timesteps.to(self.device)

        # Get noise and score
        noise = sde.noise_fn(self.state, timesteps.squeeze())
        score = sde.get_score_from_noise(noise, timesteps)

        # Learning the maximum likelihood objective for state x_{t-1}
        xt_1_expection = sde.reverse_sde_step_mean(self.state, score, timesteps)
        xt_1_optimum = sde.reverse_optimum_step(self.state, self.state_0, timesteps)
        loss = self.weight * self.loss_fn(xt_1_expection, xt_1_optimum)

        loss.backward()
        self.optimizer.step()
        self.ema.update()

        # set log
        self.log_dict["loss"] = loss.item()

    def test(self, sde=None, hidden=None, perform_ode=False, save_states=False):
        sde.set_mu(self.condition)

        self.model.eval()
        with torch.no_grad():
            if not perform_ode:
                # for SDE
                latent = sde.reverse_sde(self.state, save_states=save_states)
            else:
                # if perform Denoising ODE
                latent = sde.reverse_ode(self.state, save_states=save_states)

            self.output = self.decode(latent, hidden)

        self.model.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict["Input"] = self.condition.detach()[0].float().cpu()
        out_dict["Output"] = self.output.detach()[0].float().cpu()
        if need_GT:
            out_dict["GT"] = self.state_0.detach()[0].float().cpu()
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
        load_path_G = self.opt["path"]["pretrain_model_Diff"]
        if load_path_G is not None:
            logger.info("Loading model for G [{:s}] ...".format(load_path_G))
            self.load_network(load_path_G, self.model, self.opt["path"]["strict_load"])

        load_path_L = self.opt["path"]["pretrain_model_AE"]
        if load_path_L is not None:
            logger.info("Loading model for L [{:s}] ...".format(load_path_L))
            self.load_network(load_path_L, self.latent_model, self.opt["path"]["strict_load"])

    def save(self, iter_label):
        self.save_network(self.model, "G", iter_label)
        self.save_network(self.ema.ema_model, "EMA", 'lastest')

