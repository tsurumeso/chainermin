import numpy

from chainermin import variable


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
