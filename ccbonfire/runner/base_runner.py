#! /usr/bin/python
# -*- coding: utf-8 -*-
# Create by zhoubaicun@bytedance.com AT 2022/01/27 15:37:29
"""
base_runner.py
"""

import argparse, logging
import torch


class BaseRunner():
    def __init__(self, cfg):
        is_gpu = torch.cuda.is_available()
        model = cfg["model"]["type"](cfg["model"]["args"])

    def run():
        


        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OCRTrain launch')
    parser.add_argument('--cfg', required=True, type=str, help='config path')
    args = parser.parse_args()
    main(args)
