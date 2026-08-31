import numpy

import chainermin
from chainermin import backend
from chainermin import function


class Where(function.FunctionNode):

    def __init__(self, condition):
        self.condition = condition

    def forward(self, inputs):
        xp = backend.get_array_module(*inputs)
        x, y = inputs
        condition = self.condition
        return xp.where(condition, x, y),

    def backward(self, inputs, grad_outputs):
        xp = backend.get_array_module(*inputs)
        gx = xp.where(self.condition, grad_outputs[0], 0)
        gy = xp.where(self.condition, 0, grad_outputs[0])
        return gx, gy


def where(condition, x, y):
    """Choose elements depending on condition.
    """
    return Where(condition)(x, y)
