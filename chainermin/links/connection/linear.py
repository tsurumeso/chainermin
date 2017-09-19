import math

from chainermin.functions.connection import linear
from chainermin import initializers
from chainermin import link


class Linear(link.Link):

    def __init__(self, in_size, out_size, wscale=1):
        super(Linear, self).__init__()
        W_initializer = initializers.HeNormal(math.sqrt(wscale))
        bias_initializer = initializers.Constant(0)
        self.add_param('W', (out_size, in_size), initializer=W_initializer)
        self.add_param('b', out_size, initializer=bias_initializer)

    def __call__(self, x):
        return linear.linear(x, self.W, self.b)
