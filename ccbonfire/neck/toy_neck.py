#! /usr/bin/python
# -*- coding: utf-8 -*-
# Create by zhoubaicun@bytedance.com AT 2022/01/27 15:12:36
"""
toy_neck.py
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyNeck(nn.Module):
    def __init__(self):
        super(ToyNeck, self).__init__()
        self.op_0 = nn.AdaptiveAvgPool2d([1,1])

    def forward(self, x) -> torch.Tensor:
        x = self.op_0(x)
        x = x.reshape([x.shape[0],-1])
        return x


def test():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    neck = ToyNeck()
    logging.debug(f"neck ==> {neck}")
    input_tensor = torch.zeros([6, 20, 4, 4])
    logging.debug(f"input_tensor shape ==> {input_tensor.shape}")
    output_tensor = neck(input_tensor)
    logging.debug(f"output_tensor shape ==> {output_tensor.shape}")


if __name__ == '__main__':
    test()
