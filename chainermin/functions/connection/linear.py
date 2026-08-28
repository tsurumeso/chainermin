import math

from chainermin import function


class LinearFunction(function.FunctionNode):

    def forward(self, inputs):
        self.retain_inputs()
        x = inputs[0]
        W = inputs[1]
        y = x.dot(W.T)
        if len(inputs) == 3:
            b = inputs[2]
            y += b
        return y,

    def backward(self, inputs, grad_outputs):
        x, W = inputs[:2]
        gy = grad_outputs[0]
        gx = gy.dot(W).reshape(x.shape)
        gW = gy.T.dot(x)
        if len(inputs) == 3:
            gb = gy.sum(axis=0)
            return gx, gW, gb
        return gx, gW


def linear(x, W, b=None, n_batch_axes=1):
    if n_batch_axes <= 0:
        raise ValueError('n_batch_axes should be greater than 0.')
    if n_batch_axes > 1:
        batch_shape = x.shape[:n_batch_axes]
        batch_size = math.prod(batch_shape)
        x = x.reshape(batch_size, -1)
    elif x.ndim > 2:
        x = x.reshape(x.shape[0], -1)

    if b is None:
        args = x, W
    else:
        args = x, W, b

    y = LinearFunction()(*args)
    if n_batch_axes > 1:
        y = y.reshape(batch_shape + (-1,))
    return y
