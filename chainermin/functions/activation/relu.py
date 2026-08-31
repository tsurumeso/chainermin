from chainermin import backend, function


class ReLU(function.FunctionNode):
    def forward(self, x):
        self.retain_inputs()
        xp = backend.get_array_module(*x)
        return (xp.maximum(x[0], 0),)

    def backward(self, x, gy):
        return (gy[0] * (x[0] > 0),)


def relu(x):
    return ReLU()(x)
