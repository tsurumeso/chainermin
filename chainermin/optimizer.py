import math
import numpy


class Optimizer(object):

    def setup(self, link):
        self.target = link
        self.t = 0
        self.states = {}
        self.prepare()

    def prepare(self):
        for name, param in self.target.namedparams():
            if name not in self.states:
                state = {}
                self.init_state(param, state)
                self.states[name] = state

    def init_state(self, param, state):
        pass

    def update(self):
        self.prepare()
        self.t += 1
        for name, param in self.target.namedparams():
            self.update_one(param, self.states[name])

    def update_one(self, param, state):
        raise NotImplementedError()


class SGD(Optimizer):

    def __init__(self, lr=0.01):
        self.lr = lr

    def update_one(self, param, state):
        param.data -= self.lr * param.grad


class MomentumSGD(Optimizer):

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


class Adam(Optimizer):

    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def init_state(self, param, state):
        state['m'] = numpy.zeros_like(param.data)
        state['v'] = numpy.zeros_like(param.data)

    def update_one(self, param, state):
        m, v = state['m'], state['v']
        grad = param.grad

        m += (1 - self.beta1) * (grad - m)
        v += (1 - self.beta2) * (grad * grad - v)
        param.data -= self.lr * m / (numpy.sqrt(v) + self.eps)

    @property
    def lr(self):
        fix1 = 1. - self.beta1 ** self.t
        fix2 = 1. - self.beta2 ** self.t
        return self.alpha * math.sqrt(fix2) / fix1
