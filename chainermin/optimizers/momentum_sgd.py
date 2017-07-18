import numpy

from chainermin import optimizer


class MomentumSGD(optimizer.Optimizer):

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum

    def init_state(self, param, state):
        state['v'] = numpy.zeros_like(param.data)

    def update_one(self, param, state):
        v = state['v']
        v *= self.momentum
        v -= self.lr * param.grad
        param.data += v
