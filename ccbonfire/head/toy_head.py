#! /usr/bin/python
# -*- coding: utf-8 -*-
# Create by zhoubaicun@bytedance.com AT 2022/01/27 15:27:11
"""
toy_head.py
"""


import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyHead(nn.Module):
    def __init__(self):
        super(ToyHead, self).__init__()
        self.fc = nn.Linear(20, 1)

    def forward(self, x) -> torch.Tensor:
        x = self.fc(x)
        return x


def test():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    head = ToyHead()
    logging.debug(f"head ==> {head}")
    input_tensor = torch.zeros([6, 20])
    logging.debug(f"input_tensor shape ==> {input_tensor.shape}")
    output_tensor = head(input_tensor)
    logging.debug(f"output_tensor shape ==> {output_tensor.shape}")


if __name__ == '__main__':
    test()
