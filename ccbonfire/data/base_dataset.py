#! /usr/bin/python
# -*- coding: utf-8 -*-
# Create by zhoubaicun AT 2022/02/17 20:13:35
"""
base_dataset.py
"""

from abc import abstractmethod, ABCMeta
from torch.utils.data import Dataset
import pickle
from ccbonfire.utils import logging


class BaseDataset(Dataset, metaclass=ABCMeta):
    def __init__(self, preprocess=[]) -> None:
        """lazy init, 避免初始化了无法 pickle 的东西, 子类需要同样避免
        实际初始化通过 init() 实现
        Args:
            preprocess (list): preprocess for sample
        """
        super().__init__()
        self.kv_map = None  # KV-map
        self.keys = []
        self.initialized = False
        self.preprocess = preprocess

    def __len__(self):
        if not self.initialized:
            self.init()
        return len(self.keys)

    def __getitem__(self, idx):
        if not self.initialized:
            self.init()
        key = self.keys[idx]
        value = self.kv_map[key]
        sample = self.get_sample(key, value)
        return sample

    def get_sample(self, key, value):
        """获取数据并预处理, 失败返回权重为 0 的默认样本"""
        is_bad = False
        try:
            sample = self.parse_sample(key, value)
        except Exception as e:
            logging.error(e)
            is_bad = True
            sample = self.fallback_sample()
        for process in self.preprocess:
            sample = process(sample)
        if is_bad:
            sample["weight"] = 0
        return sample

    @abstractmethod
    def init(self):
        """初始化
        必须填充 self.kv_map, self.keys, self.initialized
        数据读取将根据 self.kv_map[self.keys[idx]] 进行处理
        需要考虑不同进程的 self.keys 一致
        """
        self.initialized = True

    @abstractmethod
    def fallback_sample(self):
        """获取数据失败时返回 默认样本"""
        return {"key": 0, "value": 0, "weight": 0}

    @abstractmethod
    def parse_sample(self, key, value):
        """从 key, value 解析样本"""
        return {"key": key, "value": value, "weight": 1}


class ToyDataset(BaseDataset):
    def __init__(self, kv_map_path, preprocess=[]) -> None:
        super().__init__(preprocess)
        self.kv_map_path = kv_map_path

    def init(self):
        print("wandanle")
        with open(self.kv_map_path, "rb") as f:
            self.kv_map = pickle.load(f)
            self.keys = sorted(list(self.kv_map.keys()))
        self.initialized = True

    def fallback_sample(self):
        return {"key": 0, "label": 0, "weight": 0}

    def parse_sample(self, key, value):
        return {"key": key, "label": value, "weight": 1}


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = ToyDataset("toy_data.pkl")
    train_loader = DataLoader(dataset, batch_size=100, shuffle=True)
    # for epoch in range(3):
    #     c = 0
    #     for data in train_loader:
    #         c += 1
    #         if c == 1:
    #             print(data)
    #     print(epoch)
