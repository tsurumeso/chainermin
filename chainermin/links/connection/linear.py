import numpy

from chainermin.functions.connection import linear
from chainermin import initializers
from chainermin import link


class Linear(link.Link):

    def __init__(self, in_size, out_size):
        super(Linear, self).__init__()
        W_initializer = initializers.Normal(numpy.sqrt(2. / in_size))
        bias_initializer = initializers.Constant(0)
        self.add_param('W', (out_size, in_size), initializer=W_initializer)
        self.add_param('b', out_size, initializer=bias_initializer)

    def __call__(self, x):
        return linear.linear(x, self.W, self.b)
