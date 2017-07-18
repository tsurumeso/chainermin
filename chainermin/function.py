from chainermin import variable


class Function(object):

    def __call__(self, *inputs):
        inputs = [x if isinstance(x, variable.Variable)
                  else variable.Variable(x)
                  for x in inputs]
        in_data = [x.data for x in inputs]
        outputs = self.forward(in_data)
        ret = [variable.Variable(y) for y in outputs]
        for y in ret:
            y.set_creator(self)
        self.inputs = inputs
        self.outputs = ret
        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def forward(self, inputs):
        raise NotImplementedError()

    def backward(self, inputs, grad_outputs):
        raise NotImplementedError()
