from chainermin import function


class LinearFunction(function.Function):

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
