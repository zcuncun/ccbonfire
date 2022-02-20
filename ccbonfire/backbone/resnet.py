#! /usr/bin/python
# -*- coding: utf-8 -*-
# Create by zhoubaicun@bytedance.com AT 2022/02/17 14:48:13
"""
resnet.py
Using create_feature_extractor to get backbone

"""

import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.models.feature_extraction import create_feature_extractor


class Resnet50(nn.Module):
    """ResNet-50 model 
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        output_node (list): node to output, default nodes are 4 last layer of each layers
    """
    def __init__(self, pretrained: bool = False, output_node=['layer1.2.relu_2','layer2.3.relu_2','layer3.5.relu_2','layer4.2.relu_2'], **kwargs) -> None:
        super().__init__()
        # Extract 4 main layers
        self.output_node = output_node
        m = resnet50(pretrained, **kwargs)
        self.body = create_feature_extractor(
            m, return_nodes={node: node for node in output_node})

    def forward(self, x):
        features = self.body(x)
        return features


if __name__ == "__main__":
    bb = Resnet50()
    with torch.no_grad():
        test_input = torch.randn(2, 3, 224, 224)
        print(bb(test_input))
