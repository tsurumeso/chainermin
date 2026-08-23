import numpy

from chainermin import function


class EmbedIDFunction(function.Function):

    def __init__(self, ignore_label=None):
        self.ignore_label = ignore_label

    def forward(self, inputs):
        x, W = inputs
        self._w_shape = W.shape

        if self.ignore_label is not None:
            mask = (x == self.ignore_label)
            return numpy.where(mask[..., None], 0, W[numpy.where(mask, 0, x)]),

        return W[x],

    def backward(self, inputs, grad_outputs):
        x = inputs[0]
        gy = grad_outputs[0]
        gW = numpy.zeros(self._w_shape, dtype=gy.dtype)

        # It is equivalent to `numpy.add.at(gW, x, gy)` but ufunc.at is
        # too slow.
        for ix, igy in zip(x.ravel(), gy.reshape(x.size, -1)):
            if ix == self.ignore_label:
                continue
            gW[ix] += igy

        return gW,


def embed_id(x, W, ignore_label=None):
    """Efficient linear function for one-hot input.
    """
    return EmbedIDFunction(ignore_label=ignore_label)(x, W)
