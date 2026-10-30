from chainermin import initializers, link
from chainermin.functions.connection import embed_id


class EmbedID(link.Link):
    """Efficient linear layer for one-hot input."""

    def __init__(self, in_size, out_size, wscale=1):
        super().__init__()

        W_initializer = initializers.Normal(wscale)
        self.add_param("W", (in_size, out_size), initializer=W_initializer)

    def __call__(self, x):
        return embed_id.embed_id(x, self.W)
