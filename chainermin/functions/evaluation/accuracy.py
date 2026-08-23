from chainermin import function
from chainermin import backend


class Accuracy(function.Function):

    def forward(self, inputs):
        xp = backend.get_array_module(*inputs)
        y, t = inputs
        pred = y.argmax(axis=1).reshape(t.shape)
        return xp.asarray((pred == t).mean(dtype=y.dtype)),


def accuracy(y, t):
    return Accuracy()(y, t)
