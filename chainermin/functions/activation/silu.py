from chainermin import backend, function


class SiLU(function.FunctionNode):
    def forward(self, x):
        self.retain_inputs()
        xp = backend.get_array_module(*x)
        s = xp.tanh(x[0] * 0.5) * 0.5 + 0.5
        return (x[0] * s,)

    def backward(self, x, gy):
        xp = backend.get_array_module(*x)
        s = xp.tanh(x[0] * 0.5) * 0.5 + 0.5
        return (gy[0] * s * (1 + x[0] * (1 - s)),)


def silu(x):
    return SiLU()(x)
