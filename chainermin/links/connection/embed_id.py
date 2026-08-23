from chainermin.functions.connection import embed_id
from chainermin.initializers import normal
from chainermin import link


class EmbedID(link.Link):

    """Efficient linear layer for one-hot input.
    """

    ignore_label = None

    def __init__(self, in_size, out_size, ignore_label=None):
        super(EmbedID, self).__init__()
        self.ignore_label = ignore_label

        W_initializer = normal.Normal(1.0)
        self.add_param('W', (in_size, out_size), initializer=W_initializer)

    def __call__(self, x):
        return embed_id.embed_id(x, self.W, ignore_label=self.ignore_label)
