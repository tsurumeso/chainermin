import numpy
import queue

import chainermin


class Variable(object):

    def __init__(self, data, grad=None):
        self.data = data
        self.creator = None
        self.grad = grad

    def set_creator(self, gen_func):
        self.creator = gen_func

    def backward(self):
        if self.creator is None:
            return

        if self.data.size == 1 and self.grad is None:
            self.grad = numpy.ones_like(self.data)

        cand_funcs = queue.Queue()
        seen_set = set()

        def add_cand(cand):
            if cand is not None and id(cand) not in seen_set:
                cand_funcs.put(cand)
                seen_set.add(id(cand))

        add_cand(self.creator)

        while not cand_funcs.empty():
            func = cand_funcs.get()
            in_data = [x.data for x in func.inputs]
            out_grad = [y.grad for y in func.outputs]
            gxs = func.backward(in_data, out_grad)

            for x, gx in zip(func.inputs, gxs):
                if x.creator is None:
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad += gx
                else:
                    add_cand(x.creator)
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad += gx

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
