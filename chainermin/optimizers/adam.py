import math

import numpy

from chainermin import backend, optimizer


class Adam(optimizer.Optimizer):
    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8, loss_scale=1.0):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.loss_scale = loss_scale

    def init_state(self, param, state):
        xp = backend.get_array_module(param.data)
        state["m"] = xp.zeros(param.data.shape, dtype=numpy.float32)
        state["v"] = xp.zeros(param.data.shape, dtype=numpy.float32)
        if param.data.dtype != numpy.float32:
            state["master_param"] = param.data.astype(numpy.float32)

    def update_one(self, param, state):
        xp = backend.get_array_module(param.data)
        m, v = state["m"], state["v"]
        grad = param.grad.astype(numpy.float32, copy=False)
        if not xp.all(xp.isfinite(grad)):
            return
        if self.loss_scale != 1.0:
            grad = grad / self.loss_scale

        target = state.get("master_param", param.data)

        m += (1 - self.beta1) * (grad - m)
        v += (1 - self.beta2) * (grad * grad - v)
        target -= self.lr * m / (xp.sqrt(v) + self.eps)

        if "master_param" in state:
            param.data = target.astype(param.data.dtype)

    @property
    def lr(self):
        fix1 = 1.0 - self.beta1**self.t
        fix2 = 1.0 - self.beta2**self.t
        return self.alpha * math.sqrt(fix2) / fix1
