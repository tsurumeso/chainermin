from chainermin import backend, function


class MeanSquaredError(function.FunctionNode):
    def forward(self, inputs):
        xp = backend.get_array_module(*inputs)
        y, t = inputs
        self.diff = y - t
        return (xp.mean(xp.square(self.diff)),)

    def backward(self, inputs, grad_outputs):
        gy = grad_outputs[0]
        coeff = gy * (2.0 / self.diff.size)
        gx = coeff * self.diff
        return gx, -gx


def mean_squared_error(y, t):
    return MeanSquaredError()(y, t)
