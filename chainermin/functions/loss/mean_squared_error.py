from chainermin import function


class MeanSquaredError(function.Function):

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