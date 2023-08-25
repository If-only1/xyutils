__all__ = ['format_time', 'TimeRecord']

import time
from collections import OrderedDict


def format_time(elapse):
    elapse = int(elapse)
    hour = elapse // 3600
    minute = elapse % 3600 // 60
    seconds = elapse % 60
    return '{:02d}:{:02d}:{:02d}'.format(hour, minute, seconds)


class TimeRecord:
    def __init__(self, enable=True, print_fn=print, print_inter=1):
        self.print_i = None
        self.cost = None
        self.cur_t = None
        self.last_t = None
        self.total_cost = None
        self.enable = enable
        if not self.enable:
            return
        self.print_fn = print_fn
        self.print_inter = print_inter
        self.reset()

    def reset(self):
        self.last_t = time.perf_counter()
        self.cur_t = self.last_t
        self.cost = OrderedDict()
        self.print_i = 0

    def record(self, name):
        if not self.enable:
            return
        self.cur_t = time.perf_counter()
        cost = self.cur_t - self.last_t
        self.last_t = self.cur_t
        if name not in self.cost:
            self.cost[name] = cost
        else:
            self.cost[name] += cost

    def print_cost(self):
        if not self.enable:
            return
        self.print_i += 1
        if self.print_i % self.print_inter == 0:
            total_cost = sum(self.cost.values())
            self.total_cost = total_cost
            s = ''
            for k in self.cost:
                s += f'{k}={self.cost[k]:.2f}({self.cost[k] / total_cost:.0%}), '

            s += f'total={total_cost:.2f}'
            self.print_fn(s)
            self.reset()
