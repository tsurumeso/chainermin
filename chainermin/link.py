import numpy

from chainermin import backend, initializers, variable


class Link:
    def __init__(self, **params):
        self._params = []
        self._xp = numpy
        for name, value in params.items():
            self.add_param(name, value.shape)

    @property
    def xp(self):
        return self._xp

    def add_param(self, name, shape, dtype=numpy.float32, initializer=None):
        if initializer is None:
            data = self._xp.full(shape, 0, dtype=dtype)
        else:
            data = initializers.generate_array(initializer, shape)
        grad = self._xp.zeros_like(data)
        var = variable.Variable(data, grad)
        self._params.append(name)
        self.__dict__[name] = var

    def params(self):
        for name in self._params:
            yield self.__dict__[name]

    def namedparams(self):
        for name in self._params:
            yield "/" + name, self.__dict__[name]

    def zerograds(self):
        for param in self.params():
            param.zerograd()

    def to_cpu(self):
        if self._xp is numpy:
            return
        for param in self.params():
            param.to_cpu()
        self._xp = numpy

    def to_gpu(self):
        if self._xp is backend.cupy:
            return
        for param in self.params():
            param.to_gpu()
        self._xp = backend.cupy


class Chain(Link):
    def __init__(self, **links):
        super().__init__()
        self._children = []
        for name, link in links.items():
            self._children.append(name)
            self.__dict__[name] = link

    def __setattr__(self, name, value):
        if isinstance(value, Link):
            value.name = name
            self._children.append(name)
        super().__setattr__(name, value)

    def params(self):
        for name in self._children:
            yield from self.__dict__[name].params()

    def namedparams(self):
        for name in self._children:
            prefix = "/" + name
            for path, param in self.__dict__[name].namedparams():
                yield prefix + path, param

    def zerograds(self):
        super().zerograds()
        for name in self._children:
            self.__dict__[name].zerograds()

    def to_cpu(self):
        super().to_cpu()
        for name in self._children:
            self.__dict__[name].to_cpu()

    def to_gpu(self):
        super().to_gpu()
        for name in self._children:
            self.__dict__[name].to_gpu()

    def save_npz(self, filename):
        params_dict = {}
        for path, param in self.namedparams():
            key = path.lstrip("/")
            data = param.data
            if hasattr(data, "get"):
                data = data.get()
            params_dict[key] = data

        numpy.savez(filename, **params_dict)

    def load_npz(self, filename):
        with numpy.load(filename) as f:
            for path, param in self.namedparams():
                key = path.lstrip("/")
                if key in f:
                    loaded_data = f[key]
                    param.data = self._xp.array(loaded_data)
                else:
                    print(f"Warning: {key} not found in {filename}")
