import weakref

from chainermin import config
from chainermin import variable


class Function(object):

    def __call__(self, *inputs):
        inputs = [x if isinstance(x, variable.Variable)
                  else variable.Variable(x)
                  for x in inputs]
        in_data = [x.data for x in inputs]
        outputs = self.forward(in_data)
        ret = [variable.Variable(y) for y in outputs]
        if not config._inference_mode:
            self.rank = max([x.rank for x in inputs])
            for y in ret:
                y.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(y) for y in ret]
        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def forward(self, inputs):
        raise NotImplementedError()

    def backward(self, inputs, grad_outputs):
        raise NotImplementedError()
