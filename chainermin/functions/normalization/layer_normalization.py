import math
import numpy

from chainermin import function


class LayerNormalization(function.Function):

    def __init__(self, eps=1e-5):
        self.eps = eps

    def forward(self, inputs):
        x, gamma, beta = inputs
        mu = numpy.mean(x, axis=1, keepdims=True)
        self.x_mu = x - mu
        self.squ_x_mu = numpy.square(self.x_mu)
        self.var = numpy.mean(self.squ_x_mu, axis=1, keepdims=True)
        std = numpy.sqrt(self.var + self.eps)
        self.inv_std = 1. / std
        self.x_hat = self.x_mu * self.inv_std
        scaled_x = self.x_hat * gamma[None, ]
        shifted_x = scaled_x + beta[None, ]
        return shifted_x,

    def backward(self, inputs, gy):
        x, gamma, beta = inputs
        gy = gy[0]

        g_beta = gy.sum(axis=0)
        g_scaled_x = gy

        g_gamma = numpy.sum(g_scaled_x * self.x_hat, axis=0)
        g_x_hat = g_scaled_x * gamma[None, ]

        g_inv_std = numpy.sum(g_x_hat * self.x_mu, axis=1, keepdims=True)
        g_x_mu_1 = g_x_hat * self.inv_std

        g_std = g_inv_std * (- 1. / self.var)
        # = g_inv_std * (- 1. / (self.std ** 2))

        g_var = g_std * 0.5 * self.inv_std
        # = g_std * 0.5 * 1. / xp.sqrt(self.var + self.eps)

        n_units = x.shape[1]
        g_squ_x_mu = numpy.broadcast_to(g_var * 1. / n_units, x.shape)
        g_x_mu_2 = g_squ_x_mu * 2 * self.x_mu

        g_x_1 = g_x_mu_1 + g_x_mu_2
        g_mu = numpy.sum(g_x_1, axis=1, keepdims=True) * (- 1.)
        # = numpy.sum(g_x_mu_1 + g_x_mu_2, axis=1, keepdims=True) * (- 1.)

        g_x_2 = numpy.broadcast_to(g_mu * 1. / n_units, x.shape)

        g_x = g_x_1 + g_x_2

        return g_x, g_gamma, g_beta,


def layer_normalization(x, gamma, beta, eps=1e-5, n_batch_axes=1):
    """Layer normalization.
    See: `Layer Normalization <https://arxiv.org/abs/1607.06450>`_
    """

    if n_batch_axes <= 0:
        raise ValueError('n_batch_axes should be greater than 0.')
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
