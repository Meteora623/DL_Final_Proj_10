from enum import auto, Enum
import math


class LRSchedule(Enum):
    Constant = auto()
    Cosine = auto()


class Scheduler:
    def __init__(self, schedule, base_lr, batch_size, epochs, optimizer):
        self.schedule = schedule
        self.base_lr = base_lr
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size

    def adjust_learning_rate(self, step, max_steps):
        if self.schedule == LRSchedule.Constant:
            lr = self.base_lr
        elif self.schedule == LRSchedule.Cosine:
            q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
            end_lr = self.base_lr * 0.001
            lr = self.base_lr * q + end_lr * (1 - q)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr
