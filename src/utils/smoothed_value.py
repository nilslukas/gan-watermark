from collections import deque

import torch


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size: int = 10, fmt: str = '{avg:.3f}'):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value: float, n: int = 1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def update_list(self, value_list: list[float]):
        for value in value_list:
            self.deque.append(value)
            self.total += value
        self.count += len(value_list)

    def reset(self):
        self.deque = deque(maxlen=self.deque.maxlen)
        self.count = 0
        self.total = 0.0

    @property
    def median(self) -> float:
        try:
            d = torch.tensor(list(self.deque))
            return d.median().item()
        except Exception:
            return 0.0

    @property
    def avg(self) -> float:
        try:
            d = torch.tensor(list(self.deque), dtype=torch.float32)
            if len(d) == 0:
                return 0.0
            return d.mean().item()
        except Exception:
            return 0.0

    @property
    def global_avg(self) -> float:
        try:
            return self.total / self.count
        except Exception:
            return 0.0

    @property
    def max(self) -> float:
        try:
            return max(self.deque)
        except Exception:
            return 0.0

    @property
    def min(self) -> float:
        try:
            return min(self.deque)
        except Exception:
            return 0.0

    @property
    def value(self) -> float:
        try:
            return self.avg
        except Exception:
            return 0.0

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            min=self.min,
            max=self.max,
            value=self.value)

    def __format__(self, format_spec: str) -> str:
        return self.__str__()

    def __repr__(self) -> str:
        return self.__str__()
