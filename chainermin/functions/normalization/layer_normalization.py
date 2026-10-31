import math

from chainermin import backend, function


class LayerNormalization(function.FunctionNode):
    def __init__(self, eps=1e-5):
        self.eps = eps

    def forward(self, inputs):
        self.retain_inputs((1,))
        xp = backend.get_array_module(*inputs)
        x, gamma, beta = inputs
        mu = xp.mean(x, axis=1, keepdims=True)
        x_mu = x - mu
        var = xp.mean(xp.square(x_mu), axis=1, keepdims=True)
        self.inv_std = 1.0 / xp.sqrt(var + self.eps)
        self.x_hat = x_mu * self.inv_std
        scaled_x = self.x_hat * gamma[None,]
        shifted_x = scaled_x + beta[None,]
        return (shifted_x,)

    def backward(self, inputs, gy):
        xp = backend.get_array_module(*gy)
        _, gamma, _ = inputs
        gy = gy[0]

        g_beta = gy.sum(axis=0)
        g_gamma = xp.sum(gy * self.x_hat, axis=0)
        g_x_hat = gy * gamma[None,]

        n_units = g_x_hat.shape[1]
        g_x = (self.inv_std / n_units) * (
            n_units * g_x_hat
            - xp.sum(g_x_hat, axis=1, keepdims=True)
            - self.x_hat * xp.sum(g_x_hat * self.x_hat, axis=1, keepdims=True)
        )

        return (
            g_x,
            g_gamma,
            g_beta,
        )


def layer_normalization(x, gamma, beta, eps=1e-5, n_batch_axes=1):
    """Layer normalization.
    See: `Layer Normalization <https://arxiv.org/abs/1607.06450>`_
    """

    if n_batch_axes <= 0:
        raise ValueError("n_batch_axes should be greater than 0.")
    if n_batch_axes > 1:
        batch_shape = x.shape[:n_batch_axes]
        batch_size = math.prod(batch_shape)
        x = x.reshape(batch_size, -1)
    elif x.ndim > 2:
        x = x.reshape(x.shape[0], -1)

    y = LayerNormalization(eps)(x, gamma, beta)
    if n_batch_axes > 1:
        y = y.reshape(batch_shape + (-1,))
    return y
