import math

from chainermin import backend, optimizer


class Adam(optimizer.Optimizer):
    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def init_state(self, param, state):
        xp = backend.get_array_module(param.data)
        state["m"] = xp.zeros_like(param.data)
        state["v"] = xp.zeros_like(param.data)

    def update_one(self, param, state):
        xp = backend.get_array_module(param.data)
        m, v = state["m"], state["v"]
        grad = param.grad

        m += (1 - self.beta1) * (grad - m)
        v += (1 - self.beta2) * (grad * grad - v)
        param.data -= self.lr * m / (xp.sqrt(v) + self.eps)

    @property
    def lr(self):
        fix1 = 1.0 - self.beta1**self.t
        fix2 = 1.0 - self.beta2**self.t
        return self.alpha * math.sqrt(fix2) / fix1
