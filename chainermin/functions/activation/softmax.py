from chainermin import backend, function


class Softmax(function.FunctionNode):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        xp = backend.get_array_module(*x)
        self.y = x[0] - x[0].max(axis=self.axis, keepdims=True)
        xp.exp(self.y, out=self.y)
        self.y /= self.y.sum(axis=self.axis, keepdims=True)
        return (self.y,)

    def backward(self, x, gy):
        gx = self.y * gy[0]
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx -= self.y * sumdx
        return (gx,)


def softmax(x, axis=1):
    return Softmax(axis=axis)(x)
