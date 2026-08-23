from chainermin import function


def _count_unknown_dims(shape):
    cnt = 0
    for dim in shape:
        cnt += dim < 0
    return cnt


class Reshape(function.Function):

    def __init__(self, shape):
        self.shape = shape
        self._cnt = _count_unknown_dims(shape)
        assert self._cnt <= 1

    def forward(self, inputs):
        x, = inputs
        self._shape = x.shape
        return x.reshape(self.shape),

    def backward(self, indexes, grad_outputs):
        gx, = grad_outputs
        return gx.reshape(self._shape),


def reshape(x, shape):
    """Reshapes an input variable without copy.
    """
    return Reshape(shape)(x)
