import weakref

import chainermin
from chainermin import backend


class VariableNode:
    def __init__(self, variable):
        self._variable = weakref.ref(variable)
        self._data = None
        self._creator_node = None
        self._rank = 0

    @property
    def creator_node(self):
        return self._creator_node

    @creator_node.setter
    def creator_node(self, func):
        self._creator_node = func
        self._rank = func.rank + 1

    @property
    def data(self):
        return self._data

    @property
    def variable(self):
        var = self._variable()
        return var

    @property
    def rank(self):
        return self._rank

    @property
    def grad(self):
        var = self._variable()
        if var is not None:
            return var.grad
        else:
            return None

    def retain_data(self):
        var = self._variable()
        if var is not None:
            self._data = var.data
        else:
            raise RuntimeError(
                "cannot retain variable data: the variable has been already released"
            )


class Variable:
    def __init__(self, data, grad=None):
        self._data = data
        self._node = VariableNode(self)
        self._grad = grad

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def node(self):
        return self._node

    @property
    def creator_node(self):
        return self._node.creator_node

    @creator_node.setter
    def creator_node(self, func):
        self._node.creator_node = func

    @property
    def grad(self):
        return self._grad

    @grad.setter
    def grad(self, value):
        self._grad = value

    @property
    def rank(self):
        return self._node.rank

    def backward(self, retain_grad=False):
        if self.creator_node is None:
            return

        cand_funcs = []
        seen_set = set()
        grads = {}

        xp = backend.get_array_module(self.data)
        if self.data.size == 1 and self.grad is None:
            self.grad = xp.ones_like(self.data)
        grads[self._node] = self.grad

        def add_cand(cand):
            if cand is not None and id(cand) not in seen_set:
                cand_funcs.append(cand)
                seen_set.add(id(cand))
                cand_funcs.sort(key=lambda x: x.rank)

        add_cand(self.creator_node)

        while cand_funcs:
            func = cand_funcs.pop()
            inputs = func.inputs
            outputs = [y() for y in func.outputs]
            in_data = [x.data for x in inputs]
            out_grad = [grads[y] for y in outputs]
            gxs = func.backward(in_data, out_grad)

            for x, gx in zip(inputs, gxs):
                if x not in grads:
                    if x.grad is not None:
                        grads[x] = x.grad + gx
                    else:
                        grads[x] = gx
                else:
                    grads[x] += gx

                if x.variable is not None:
                    x.variable.grad = grads[x]

                if x.creator_node is not None:
                    add_cand(x.creator_node)

            if not retain_grad:
                for y in outputs:
                    grads[y] = None

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
        elif len(axes) == 1 and (isinstance(axes[0], (tuple, list)) or axes[0] is None):
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
