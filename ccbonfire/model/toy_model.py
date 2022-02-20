#! /usr/bin/python
# -*- coding: utf-8 -*-
# Create by zhoubaicun@bytedance.com AT 2022/01/27 15:57:14
"""
toy_model.py
"""

import torch
import torch.nn as nn


class ToyModel(nn.Module):
    """A example"""

    def __init__(self, backbone, neck, head, loss_fn, eval_fn):
        super(ToyModel, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.loss_fn = loss_fn
        self.eval_fn = eval_fn

    def forward(self, x) -> torch.Tensor:
        feature = self.backbone(x)
        feature = self.neck(feature)
        output = self.head(feature)
        return output

    def train_step(self, batch, optimizers):
        for optimizer in optimizers:
            optimizer.zero_grad()
        batch_input, batch_label = batch["input"], batch["label"]
        batch_output = self.forward(batch_input)
        batch_loss = self.loss_fn(batch_output, batch_label)
        batch_loss.backward()
        for optimizer in optimizers:
            optimizer.step()
        return {"batch_loss": batch_loss}

    def eval_step(self, batch):
        batch_input, batch_label = batch["input"], batch["label"]
        batch_output = self.forward(batch_input)
        batch_loss = self.loss_fn(batch_output, batch_label)
        batch_eval = self.eval_fn(batch_output, batch_label)
        return {"batch_loss": batch_loss, "batch_eval": batch_eval}
