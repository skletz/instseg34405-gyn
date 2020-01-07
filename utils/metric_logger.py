from torch import Tensor
from collections import defaultdict
from utils.running_average import RunningAverage


class MetricLogger:

    def __init__(self):
        self.meters = defaultdict(RunningAverage)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def get_dictionary(self):
        d = {}
        for key, meter in self.meters.items():
            d[key] = meter()
        return d

    def __str__(self):

        d = self.get_dictionary()
        string = "; ".join("{}: {:05.5f}".format(k, v) for k, v in d.items())
        return string
