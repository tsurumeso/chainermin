from chainermin import backend, function


def _log_softmax(x):
    xp = backend.get_array_module(x)
    m = x.max(axis=1, keepdims=True)
    y = x - m
    z = xp.log(xp.exp(y).sum(axis=1, keepdims=True))
    y -= z
    return y


class SoftmaxCrossEntropy(function.FunctionNode):
    def forward(self, inputs):
        self.retain_inputs((1,))
        xp = backend.get_array_module(*inputs)
        x, t = inputs
        n_batches = x.shape[0]
        self.log_p = _log_softmax(x)
        log_p = self.log_p[xp.arange(n_batches), t.ravel()]
        y = -xp.sum(log_p) / n_batches
        return (y.reshape(()),)

    def backward(self, inputs, grad_outputs):
        xp = backend.get_array_module(*grad_outputs)
        _, t = inputs
        gy = grad_outputs[0]
        n_batches = self.log_p.shape[0]
        gx = xp.exp(self.log_p, out=self.log_p)
        gx[xp.arange(n_batches), t.ravel()] -= 1
        gx *= gy / n_batches
        return gx, None


def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)
