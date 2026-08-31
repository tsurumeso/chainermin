from chainermin import backend, function


class Concat(function.FunctionNode):
    # concat along the channel dimension by default
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, xs):
        xp = backend.get_array_module(*xs)
        return (xp.concatenate(xs, axis=self.axis),)

    def backward(self, xs, gy):
        self.retain_inputs()
        xp = backend.get_array_module(*xs, *gy)
        sizes = xp.array([x.shape[self.axis] for x in xs[:-1]]).cumsum()
        return xp.split(gy[0], sizes, axis=self.axis)


def concat(xs, axis=1):
    """Concatenates given variables along an axis."""
    return Concat(axis=axis)(*xs)
