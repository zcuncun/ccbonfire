#! /usr/bin/python
# -*- coding: utf-8 -*-
# Create by zhoubaicun@bytedance.com AT 2022/01/27 15:37:29
"""
base_runner.py
"""

import torch
import time
from .hook import Hooks, TrainHooks


class TrainRunner:
    """ 训练 Runner, 流程设定如下
    准备: 初始化, 加载 ckpts 等,  -> hooks["before_run"]
    epoch 循环, 训练集上训练一轮 验证集验证一轮:
        一个 epoch 训练开始前的处理 -> hooks["before_train_epoch"]
        训练 batch 循环:
            进行一步训练前的处理 -> hooks["before_train_step"]
            模型进行一步训练 self.train_output = self.model.train_step(data)
            进行一步训练后的处理  -> hooks["after_train_step"]
        一个 epoch 训练结束后的处理 -> hooks["after_train_epoch"]
       
        验证开始前的处理  -> hooks["before_val"] 
        验证 batch 循环:
            一个 batch 验证开始前的处理 -> hooks["before_val_step"] 
            模型进行一个 batch 验证 self.val_output = self.model.val_step(data)
            一个 batch 验证结束后的处理, 计算损失,精度 等等 -> hooks["after_val_step"] 
        验证结束后的处理, 存储 ckpt, 比较最优 ckpt -> hooks["after_val"]

    训练结束 -> hooks["after_run"] 

    Args:
        model : 模型
        train_dataloader: 训练数据 loader
        val_dataloader: 验证数据 loader
        loss_fn: 损失函数
        loss_collate_fn: 收集多个 batch 的 loss, 统计在验证数据上的 loss
        eval_fn: 指标计算函数
        eval_collate_fn: 收集多个 batch 的指标, 统计在验证数据上的指标
        hooks: 用于在一些阶段执行一些处理, 如读存 ckpt
        meta: 存储一些元信息, 基本不会随着训练过程变化
    """

    def __init__(
        self,
        model,
        optimizer,
        train_dataloader,
        val_dataloader,
        hooks={},
        meta={}
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.hooks = TrainHooks()
        self.hooks.update(hooks)
        self.record = {
            "epoch": 0
        }
        self.meta = {
            "max_epochs":0
        }
        self.meta.update(meta)

    def call_hook(self, stage):
        for hook in self.hooks[stage]:
            hook(self)

    def run(self):
        self.call_hook("before_run")
        while self.record["epoch"] < self.meta["max_epochs"]:
            self.call_hook("before_train_epoch")
            for data in self.train_dataloader:
                self.call_hook("before_train_step")
                self.train_output = self.model.train_step(data)
                self.call_hook("after_train_step")
            self.call_hook("after_train_epoch")

            self.call_hook("before_val")
            with torch.no_grad():
                for data in self.val_dataloader:
                    self.call_hook("before_val_step")
                    self.val_output = self.model.eval_step(data)
                    self.call_hook("after_val_step")
            # 经过一个 epoch 迭代后调用
            self.call_hook("after_val")

        # 运行完成前调用
        self.call_hook("after_run")


if __name__ == "__main__":
    from ccbonfire import backbone, neck, head, model
    
