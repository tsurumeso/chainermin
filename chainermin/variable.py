import queue

import chainermin
from chainermin import backend


class Variable(object):

    def __init__(self, data, grad=None):
        self.data = data
        self.creator = None
        self.grad = grad
        self.rank = 0

    def set_creator(self, gen_func):
        self.creator = gen_func
        self.rank = gen_func.rank + 1

    def backward(self, retain_grad=False):
        if self.creator is None:
            return

        xp = backend.get_array_module(self.data)
        if self.data.size == 1 and self.grad is None:
            self.grad = xp.ones_like(self.data)

        cand_funcs = []
        seen_set = set()

        def add_cand(cand):
            if cand is not None and id(cand) not in seen_set:
                cand_funcs.append(cand)
                seen_set.add(id(cand))
                cand_funcs.sort(key=lambda x: x.rank)

        add_cand(self.creator)

        while cand_funcs:
            func = cand_funcs.pop()
            in_data = [x.data for x in func.inputs]
            out_grad = [y().grad for y in func.outputs]
            gxs = func.backward(in_data, out_grad)

            for x, gx in zip(func.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad += gx

                if x.creator is not None:
                    add_cand(x.creator)

            if not retain_grad:
                for y in func.outputs:
                    y().grad = None

    def to_cpu(self):
        self.data = backend.to_cpu(self.data)
        if self.grad is not None:
            self.grad = backend.to_cpu(self.grad)

    def to_gpu(self):
        self.data = backend.to_gpu(self.data)
        if self.grad is not None:
            self.grad = backend.to_gpu(self.grad)

    def zerograd(self):
        self.grad.fill(0)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return chainermin.functions.reshape(self, shape)

    def transpose(self, *axes):
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1 and (isinstance(axes[0], (tuple, list)) or
                                 axes[0] is None):
            axes = axes[0]
        return chainermin.functions.transpose(self, axes)

    @property
    def T(self):
        return chainermin.functions.transpose(self)

    @property
    def shape(self):
        raw_shape = self.data.shape
        return raw_shape

    @property
    def ndim(self):
        return self.data.ndim
