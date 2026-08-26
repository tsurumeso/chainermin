from chainermin import config
from chainermin import function
from chainermin import backend


class Dropout(function.FunctionNode):

    def __init__(self, dropout_ratio):
        self.dropout_ratio = dropout_ratio

    def forward(self, inputs):
        xp = backend.get_array_module(*inputs)
        x = inputs[0]

        if not hasattr(self, 'mask'):
            scale = x.dtype.type(1. / (1 - self.dropout_ratio))
            flag = xp.random.rand(*x.shape) >= self.dropout_ratio
            self.mask = scale * flag
        return x * self.mask,

    def backward(self, inputs, grad_outputs):
        gy = grad_outputs[0]
        return gy * self.mask,


def dropout(x, ratio=0.5):
    if config._inference_mode:
        return x
    return Dropout(ratio)(x)
