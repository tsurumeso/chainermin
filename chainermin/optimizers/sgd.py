import numpy

from chainermin import backend, optimizer


class SGD(optimizer.Optimizer):
    def __init__(self, lr=0.01, loss_scale=1.0):
        self.lr = lr
        self.loss_scale = loss_scale

    def init_state(self, param, state):
        if param.data.dtype != numpy.float32:
            state["master_param"] = param.data.astype(numpy.float32)

    def update_one(self, param, state):
        xp = backend.get_array_module(param.data)
        grad = param.grad.astype(numpy.float32, copy=False)
        if not xp.all(xp.isfinite(grad)):
            return
        if self.loss_scale != 1.0:
            grad = grad / self.loss_scale

        target = state.get("master_param", param.data)
        target -= self.lr * grad

        if "master_param" in state:
            param.data = target.astype(param.data.dtype)
