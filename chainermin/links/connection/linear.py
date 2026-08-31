import math

from chainermin import initializers, link
from chainermin.functions.connection import linear


class Linear(link.Link):
    def __init__(self, in_size, out_size, nobias=False, wscale=1):
        super().__init__()
        W_initializer = initializers.HeNormal(math.sqrt(wscale))
        bias_initializer = initializers.Constant(0)
        self.add_param("W", (out_size, in_size), initializer=W_initializer)
        if nobias:
            self.b = None
        else:
            self.add_param("b", out_size, initializer=bias_initializer)

    def __call__(self, x, n_batch_axes=1):
        return linear.linear(x, self.W, self.b, n_batch_axes=n_batch_axes)
