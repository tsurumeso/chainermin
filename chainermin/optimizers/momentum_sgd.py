import numpy

from chainermin import backend, optimizer


class MomentumSGD(optimizer.Optimizer):
    def __init__(self, lr=0.01, momentum=0.9, loss_scale=1.0):
        self.lr = lr
        self.momentum = momentum
        self.loss_scale = loss_scale

    def init_state(self, param, state):
        xp = backend.get_array_module(param.data)
        state["v"] = xp.zeros(param.data.shape, dtype=numpy.float32)
        if param.data.dtype != numpy.float32:
            state["master_param"] = param.data.astype(numpy.float32)

    def update_one(self, param, state):
        xp = backend.get_array_module(param.data)
        v = state["v"]
        grad = param.grad.astype(numpy.float32, copy=False)
        if not xp.all(xp.isfinite(grad)):
            return
        if self.loss_scale != 1.0:
            grad = grad / self.loss_scale

        target = state.get("master_param", param.data)

        v *= self.momentum
        v -= self.lr * grad
        target += v

        if "master_param" in state:
            param.data = target.astype(param.data.dtype)
