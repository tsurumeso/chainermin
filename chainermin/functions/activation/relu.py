from chainermin import backend, function


class ReLU(function.FunctionNode):
    def forward(self, x):
        xp = backend.get_array_module(*x)
        self.mask = x[0] > 0
        return (xp.maximum(x[0], 0),)

    def backward(self, x, gy):
        return (gy[0] * self.mask,)


def relu(x):
    return ReLU()(x)
