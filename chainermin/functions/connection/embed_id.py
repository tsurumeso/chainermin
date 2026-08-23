from chainermin import function
from chainermin import backend


class EmbedIDFunction(function.Function):

    def forward(self, inputs):
        x, W = inputs
        self._w_shape = W.shape

        return W[x],

    def backward(self, inputs, grad_outputs):
        xp = backend.get_array_module(*inputs, *grad_outputs)
        x = inputs[0]
        gy = grad_outputs[0]
        gW = xp.zeros(self._w_shape, dtype=gy.dtype)

        # It is equivalent to `numpy.add.at(gW, x, gy)` but ufunc.at is
        # too slow.
        for ix, igy in zip(x.ravel(), gy.reshape(x.size, -1)):
            gW[ix] += igy

        return gW,


def embed_id(x, W):
    """Efficient linear function for one-hot input.
    """
    return EmbedIDFunction()(x, W)
