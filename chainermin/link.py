import numpy
from chainermin import variable
from chainermin.initializer import Normal
from chainermin.initializer import Constant
from chainermin.function import linear


class Link(object):

    def __init__(self, **params):
        self._params = []
        for name, value in params.items():
            self.add_param(name, value.shape)

    def add_param(self, name, shape, dtype=numpy.float32, initializer=None):
        if initializer is None:
            data = numpy.full(shape, 0, dtype=dtype)
        else:
            data = initializer(shape)
        grad = numpy.zeros_like(data)
        var = variable.Variable(data, grad)
        self._params.append(name)
        self.__dict__[name] = var

    def params(self):
        for name in self._params:
            yield self.__dict__[name]

    def namedparams(self):
        for name in self._params:
            yield '/' + name, self.__dict__[name]

    def zerograds(self):
        for param in self.params():
            param.zerograd()


class Chain(Link):

    def __init__(self, **links):
        super(Chain, self).__init__()
        self._children = []
        for name, link in links.items():
            self._children.append(name)
            self.__dict__[name] = link

    def params(self):
        for name in self._children:
            for param in self.__dict__[name].params():
                yield param

    def namedparams(self):
        for name in self._children:
            prefix = '/' + name
            for path, param in self.__dict__[name].namedparams():
                yield prefix + path, param


class Linear(Link):

    def __init__(self, in_size, out_size):
        super(Linear, self).__init__()
        wscale = numpy.sqrt(2. / in_size)
        self.add_param('W', (out_size, in_size), initializer=Normal(wscale))
        self.add_param('b', out_size, initializer=Constant(0))

    def __call__(self, x):
        return linear(x, self.W, self.b)
