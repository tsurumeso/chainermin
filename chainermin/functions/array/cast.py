from chainermin import function


class Cast(function.FunctionNode):

    def __init__(self, dtype):
        self.type = dtype

    def forward(self, x):
        self._in_type = x[0].dtype.type
        return x[0].astype(self.type, copy=False),

    def backward(self, inputs, g):
        return g[0].astype(self._in_type, copy=False),


def cast(x, dtype):
    """Cast an input variable to a given type.
    """
    return Cast(dtype)(x)
