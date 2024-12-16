from enum import auto, Enum
import math

class LRSchedule(Enum):
    Constant = auto()
    Cosine = auto()

class Scheduler:
    def __init__(
        self,
        schedule: str,
        base_lr: float,
        epochs: int,
        optimizer,
    ):
        self.schedule = schedule
        self.base_lr = base_lr
        self.epochs = epochs
        self.optimizer = optimizer
        self.current_step = 0
        self.max_steps = None
        self.warmup_steps = None

    def set_total_steps(self, total_steps):
        # 如果需要根据数据集大小设置学习率调度，可以在训练开始后调用本函数设定max_steps
        self.max_steps = total_steps
        self.warmup_steps = int(0.1 * self.max_steps)

    def adjust_learning_rate(self, step: int):
        # 如果用户没有调用 set_total_steps，不进行动态调整，仅使用base_lr
        if self.schedule == LRSchedule.Constant or self.max_steps is None:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.base_lr
            return self.base_lr
        else:
            # Cosine衰减策略
            if step < self.warmup_steps:
                lr = self.base_lr * step / self.warmup_steps
            else:
                step_adj = step - self.warmup_steps
                max_steps_adj = self.max_steps - self.warmup_steps
                q = 0.5 * (1 + math.cos(math.pi * step_adj / max_steps_adj))
                end_lr = self.base_lr * 0.001
                lr = self.base_lr * q + end_lr * (1 - q)

            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
            return lr
