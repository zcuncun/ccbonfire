#! /usr/bin/python
# -*- coding: utf-8 -*-
# Create by zhoubaicun AT 2022/02/18 11:12:40
"""
hook.py
"""


class Hooks(dict):
    """Hooks 包含有多个阶段, 每个阶段有各自会顺序执行的函数
    这些函数尽量互不干扰or依赖, 如果有依赖关系应保证执行顺序
    Args:
        stages (list): 定义这个 hooks 包含哪些 stages
    """

    def __init__(self, stages) -> None:
        super().__init__({stage: [] for stage in stages})

    def update(self, update_dict):
        """添加各个 stage 的函数"""
        for stage in update_dict:
            if stage not in self:
                continue
            else:
                self[stage] += update_dict[stage]


class TrainHooks(Hooks):
    def __init__(self) -> None:
        train_stages = [
            "before_run",
            "before_train_epoch",
            "before_train_step",
            "after_train_step",
            "after_train_epoch",
            "before_val",
            "before_val_step",
            "after_val_step",
            "after_val",
            "after_run",
        ]
        super().__init__(train_stages)
        self.update(
            {
                "before_run": [self.before_run],
                "before_train_epoch": [self.before_train_epoch],
                "before_train_step": [self.before_train_step],
                "after_train_step": [self.after_train_step],
                "after_train_epoch": [self.after_train_epoch],
                "before_val": [self.before_val],
                "before_val_step": [self.before_val_step],
                "after_val_step": [self.after_val_step],
                "after_val": [self.after_val],
                "after_run": [self.after_run],
            }
        )

    def before_run(self, runner):
        # TODO load ckpt
        pass

    def before_train_epoch(self, runner):
        runner.model.train()
        runner.record["train_losses"] = []

    def before_train_step(self, runner):
        pass

    def after_train_step(self, runner):
        # TODO write summury
        pass

    def after_train_epoch(self, runner):
        # TODO write summury
        pass

    def before_val(self, runner):
        runner.model.eval()
        runner.record["val_output"] = []


    def before_val_step(self, runner):
        pass

    def after_val_step(self, runner):
        pass
        # TODO write summury

    def after_val(self, runner):
        runner.record["val_loss"].append(runner.loss_collate_fn(runner.record["val_losses"]))
        runner.record["val_eval"].append(runner.eval_collate_fn(runner.record["val_evals"]))
        # TODO 比较 val 指标, 存储 ckpt, write summury

    def after_run(self, runner):
        pass


if __name__ == "__main__":
    my_hook = Hooks(["s_a", "s_b"])
    print(my_hook)
    my_hook.update({"s_a": [1]})
    print(my_hook)
    new_hooks = Hooks(["s_a", "s_b", "s_c"])
    new_hooks.update({"s_b": [2]})
    my_hook.update(new_hooks)
    print(my_hook)
