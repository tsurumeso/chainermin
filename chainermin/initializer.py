class Initializer(object):

    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, shape):
        raise NotImplementedError()
