import numpy

from chainermin import function
from chainermin import utils
from chainermin import variable


class Add(function.Function):

    def forward(self, x):
        y = utils.force_array(x[0] + x[1])
        return y,

    def backward(self, x, gy):
        return gy[0], gy[0]


class AddConstant(function.Function):

    def __init__(self, value):
        self.value = value

    def forward(self, x):
        y = utils.force_array(x[0] + self.value)
        return y,

    def backward(self, x, gy):
        return gy[0],


def add(lhs, rhs):
    if isinstance(rhs, variable.Variable):
        return Add()(lhs, rhs)
    return AddConstant(rhs)(lhs)


class Div(function.Function):

    def forward(self, x):
        y = utils.force_array(x[0] / x[1])
        return y,

    def backward(self, x, gy):
        gx0 = gy[0] / x[1]
        return gx0, -x[0] * gx0 / x[1]


def div(lhs, rhs):
    if isinstance(rhs, variable.Variable):
        return Div()(lhs, rhs)
    return MulConstant(1. / rhs)(lhs)


class DivFromConstant(function.Function):

    def __init__(self, value):
        self.value = value

    def forward(self, x):
        y = utils.force_array(self.value / x[0])
        return y,

    def backward(self, x, gy):
        return -self.value * gy[0] / (x[0] ** 2)


def rdiv(lhs, rhs):
    if isinstance(rhs, variable.Variable):
        return Div()(rhs, lhs)
    return DivFromConstant(rhs)(lhs)


class Mul(function.Function):

    def forward(self, x):
        y = utils.force_array(x[0] * x[1])
        return y,

    def backward(self, x, gy):
        return x[1] * gy[0], x[0] * gy[0]


class MulConstant(function.Function):

    def __init__(self, value):
        self.value = value

    def forward(self, x):
        y = utils.force_array(x[0] * self.value)
        return y,

    def backward(self, x, gy):
        return self.value * gy[0],


def mul(lhs, rhs):
    if isinstance(rhs, variable.Variable):
        return Mul()(lhs, rhs)
    return MulConstant(rhs)(lhs)


class PowVarVar(function.Function):

    def forward(self, x):
        y = utils.force_array(x[0] ** x[1])
        return y,

    def backward(self, x, gy):
        gx0 = gy[0] * x[1] * x[0] ** (x[1] - 1)
        gx1 = gy[1] * numpy.log(x[0]) * x[0] ** x[1]
        return gx0, gx1


class PowVarConst(function.Function):

    def __init__(self, value):
        self.value = value

    def forward(self, x):
        y = utils.force_array(x[0] ** self.value)
        return y,

    def backward(self, x, gy):
        gx0 = gy[0] * self.value * x[0] ** (self.value - 1)
        return gx0,


def pow(lhs, rhs):
    if numpy.isscalar(rhs):
        return PowVarConst(rhs)(lhs)
    return PowVarVar()(lhs, rhs)


class PowConstVar(function.Function):

    def __init__(self, value):
        self.value = value

    def forward(self, x):
        y = utils.force_array(self.value ** x[0])
        return y,

    def backward(self, x, gy):
        gx0 = gy[1] * numpy.log(self.value) * self.value ** x[0]
        return gx0,


def rpow(lhs, rhs):
    if numpy.isscalar(rhs):
        return PowConstVar(rhs)(lhs)
    return PowVarVar()(rhs, lhs)


class Sub(function.Function):

    def forward(self, x):
        y = utils.force_array(x[0] - x[1])
        return y,

    def backward(self, x, gy):
        return gy[0], -gy[0]


class SubConstant(function.Function):

    def __init__(self, value):
        self.value = value

    def forward(self, x):
        y = utils.force_array(x[0] - self.value)
        return y,

    def backward(self, x, gy):
        return gy[0],


def sub(lhs, rhs):
    if isinstance(rhs, variable.Variable):
        return Sub()(lhs, rhs)
    return SubConstant(rhs)(lhs)


def install_variable_arithmetics():
    variable.Variable.__add__ = add
    variable.Variable.__radd__ = add
    variable.Variable.__div__ = div
    variable.Variable.__truediv__ = div
    variable.Variable.__rdiv__ = rdiv
    variable.Variable.__rtruediv__ = rdiv
    variable.Variable.__mul__ = mul
    variable.Variable.__rmul__ = mul
    variable.Variable.__pow__ = pow
    variable.Variable.__rpow__ = rpow
    variable.Variable.__sub__ = sub
    variable.Variable.__rsub__ = sub
