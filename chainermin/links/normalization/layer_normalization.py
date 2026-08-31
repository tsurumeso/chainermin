from chainermin import initializers, link
from chainermin.functions.normalization import layer_normalization


class LayerNormalization(link.Link):
    """Layer normalization layer on outputs of linear functions.
    See: `Layer Normalization <https://arxiv.org/abs/1607.06450>`_
    """

    def __init__(self, size, eps=1e-6, initial_gamma=None, initial_beta=None):
        super().__init__()
        if initial_gamma is None:
            initial_gamma = initializers.Constant(1)
        self._gamma_initializer = initial_gamma
        if initial_beta is None:
            initial_beta = initializers.Constant(0)
        self._beta_initializer = initial_beta
        self.eps = eps

        self.add_param("gamma", size, initializer=self._gamma_initializer)
        self.add_param("beta", size, initializer=self._beta_initializer)

    def __call__(self, x, n_batch_axes=1):
        return layer_normalization.layer_normalization(
            x, self.gamma, self.beta, self.eps, n_batch_axes=n_batch_axes
        )
