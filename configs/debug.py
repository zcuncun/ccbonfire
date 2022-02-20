#! /usr/bin/python
# -*- coding: utf-8 -*-
# Create by zhoubaicun@bytedance.com AT 2022/02/17 15:02:13
"""
debug.py
"""
from ccbonfire.utils import build


import torch
from ccbonfire import backbone, neck, head, model, runner


def main():
    my_model = model.ToyModel(
        backbone=backbone.ToyBackbone(),
        neck=neck.ToyNeck(),
        head=head.ToyHead()
    )
    loss_fn = torch.nn.functional.mse_loss

    test_input = torch.randn(2, 3, 24, 24)
    test_label = torch.rand(2)
    output = my_model(test_input)
    loss = loss_fn(output, test_label)
    loss.backward()

    
if __name__ == '__main__':
    main()
