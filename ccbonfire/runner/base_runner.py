#! /usr/bin/python
# -*- coding: utf-8 -*-
# Create by zhoubaicun@bytedance.com AT 2022/01/27 15:37:29
"""
base_runner.py
"""

import torch
import time


class TrainRunner:
    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader,
        loss_fn,
        loss_collate_fn,
        eval_fn,
        eval_collate_fn,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.loss_fn = loss_fn
        self.loss_collate_fn = loss_collate_fn
        self.eval_fn = eval_fn
        self.eval_collate_fn = eval_collate_fn
        self.hooks = {
            "before_run": [],
            "before_train_epoch": [],
            "after_train_epoch": [],
            "before_train_iter": [],
            "after_train_iter": [],
            "before_val_epoch": [],
            "after_val_epoch": [],
        }
        self.record = {
            "epoch": 0,
            "step": 0,
            "loss": [],
            "val_loss": [],
            "val_eval": [],
            "best_eval": None,
        }
        self.meta = {}

    def call_hook(self, stage):
        for hook in self.hooks[stage]:
            hook(self)

    def run(
        self,
    ):
        self.call_hook("before_run")
        while self.record["epoch"] < self._max_epochs:
            self.call_hook("before_train_epoch")
            for data in self.train_dataloader:
                self.call_hook("before_train_forward")
                output = self.model(data)
                self.loss = self.loss_fn(output, data)
                self.call_hook("after_train_forward")
            self.call_hook("after_train_epoch")

            self.call_hook("before_val_epoch")
            self.model.eval()
            val_losses = []
            val_evals = []
            for data in self.val_dataloader:
                output = self.model(data)
                val_loss = self.loss_fn(output, data)
                val_losses.append(val_loss)
                val_eval = self.eval_fn(output, data)
                val_evals.append(val_eval)
            self.record["val_loss"].append(self.loss_collate_fn(val_losses))
            self.record["val_eval"].append(self.eval_collate_fn(val_evals))

            # 经过一个 epoch 迭代后调用
            self.call_hook("after_val_epoch")

        # 运行完成前调用
        self.call_hook("after_run")


class EvalRunner:
    def __init__(self, model, eval_data, eval_fn, eval_collate_fn):
        """Runner for testing.
        Args:
            model : model to evaluate.
            test_data : data to evaluate.
            eval_fn: evaluate function for batch
            eval_collate_fn : collate eval result from batch outputs
        """
        self.model = model
        self.eval_data = eval_data
        self.eval_fn = eval_fn
        self.eval_collate_fn = eval_collate_fn

    def run(self):
        self.model.eval()
        eval_reses = []
        for data in self.eval_data:
            output = self.model(data["input"])
            eval_res = self.eval_fn(output, data["label"])
            eval_reses.append(eval_res)
        final_res = self.eval_collate_fn(eval_reses)
        return final_res


class TestRunner:
    def __init__(self, model, test_data, output_collate_fn):
        """Runner for testing.
        Args:
            model : model to test.
            test_data : data to test.
            collate_fn : collate output from batch outputs
        """
        self.model = model
        self.test_data = test_data
        self.output_collate_fn = output_collate_fn

    def run(self):
        self.model.eval()
        outputs = []
        for data in self.test_data:
            output = self.model(data["input"])
            outputs.append(output)
        final_output = self.output_collate_fn(outputs)
        return final_output


if __name__ == "__main__":
    pass
