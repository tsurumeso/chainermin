import math

from chainermin import function


class LinearFunction(function.FunctionNode):

    def forward(self, inputs):
        self.retain_inputs()
        x, W, b = inputs[:3]
        y = x.dot(W.T) + b
        return y,

    def backward(self, inputs, grad_outputs):
        x, W = inputs[:2]
        gy = grad_outputs[0]
        gx = gy.dot(W).reshape(x.shape)
        gW = gy.T.dot(x)
        gb = gy.sum(axis=0)
        return gx, gW, gb


def linear(x, W, b, n_batch_axes=1):
    if n_batch_axes <= 0:
        raise ValueError('n_batch_axes should be greater than 0.')
    if n_batch_axes > 1:
        batch_shape = x.shape[:n_batch_axes]
        batch_size = math.prod(batch_shape)
        x = x.reshape(batch_size, -1)
    elif x.ndim > 2:
        x = x.reshape(x.shape[0], -1)

    y = LinearFunction()(x, W, b)
    if n_batch_axes > 1:
        y = y.reshape(batch_shape + (-1,))
    return y
