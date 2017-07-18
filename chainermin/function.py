import numpy
from chainermin import variable


class Function(object):

    def __call__(self, *inputs):
        inputs = [x if isinstance(x, variable.Variable) 
                  else variable.Variable(x)
                  for x in inputs]
        in_data = [x.data for x in inputs]
        outputs = self.forward(in_data)
        ret = [variable.Variable(y) for y in outputs]
        for y in ret:
            y.set_creator(self)
        self.inputs = inputs
        self.outputs = ret
        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def forward(self, inputs):
        raise NotImplementedError()

    def backward(self, inputs, grad_outputs):
        raise NotImplementedError()


class LinearFunction(Function):

    def forward(self, inputs):
        x = inputs[0]
        W = inputs[1]
        b = inputs[2]
        y = x.dot(W.T) + b
        return y,

    def backward(self, inputs, grad_outputs):
        x = inputs[0]
        W = inputs[1]
        gy = grad_outputs[0]

        gx = gy.dot(W).reshape(x.shape)
        gW = gy.T.dot(x)
        gb = gy.sum(axis=0)
        return gx, gW, gb


def linear(x, W, b):
    return LinearFunction()(x, W, b)


class Dropout(Function):

    def __init__(self, dropout_ratio):
        self.dropout_ratio = dropout_ratio

    def forward(self, inputs):
        x = inputs[0]
        if not hasattr(self, 'mask'):
            scale = x.dtype.type(1. / (1 - self.dropout_ratio))
            flag = numpy.random.rand(*x.shape) >= self.dropout_ratio
            self.mask = scale * flag
        return x * self.mask,

    def backward(self, inputs, grad_outputs):
        gy = grad_outputs[0]
        return gy * self.mask,


def dropout(x, ratio=0.5, train=True):
    if train:
        return Dropout(ratio)(x)
    return x


class ReLU(Function):

    def forward(self, x):
        return numpy.maximum(x[0], 0),

    def backward(self, x, gy):
        return gy[0] * (x[0] > 0),


def relu(x):
    return ReLU()(x)


class Sigmoid(Function):

    def forward(self, x):
        self.y = numpy.tanh(x[0] * 0.5) * 0.5 + 0.5
        return self.y,

    def backward(self, x, gy):
        return gy[0] * self.y * (1 - self.y),


def sigmoid(x):
    return Sigmoid()(x)


class MeanSquaredError(Function):

    def forward(self, inputs):
        y, t = inputs
        self.diff = y - t
        diff = self.diff.ravel()
        return diff.dot(diff) / diff.size,

    def backward(self, inputs, grad_outputs):
        gy = grad_outputs[0]
        coeff = gy * (2. / self.diff.size)
        gx = coeff * self.diff
        return gx, -gx

        
def mean_squared_error(y, t):
    return MeanSquaredError()(y, t)


class Accuracy(Function):
    
    def forward(self, inputs):
        y, t = inputs
        pred = y.argmax(axis=1).reshape(t.shape)
        return numpy.asarray((pred == t).mean(dtype=y.dtype)),


def accuracy(y, t):
    return Accuracy()(y, t)
