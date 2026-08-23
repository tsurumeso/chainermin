import numpy

from chainermin import optimizer
from chainermin import backend


class MomentumSGD(optimizer.Optimizer):

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum

    def init_state(self, param, state):
        xp = backend.get_array_module(param.data)
        state['v'] = xp.zeros_like(param.data)

    def update_one(self, param, state):
        v = state['v']
        v *= self.momentum
        v -= self.lr * param.grad
        param.data += v
