import weakref

from chainermin import config, variable


class FunctionNode:
    def __call__(self, *inputs):
        inputs = [x if isinstance(x, variable.Variable) else variable.Variable(x) for x in inputs]
        in_data = [x.data for x in inputs]

        self._retain_inputs = None
        outputs = self.forward(in_data)

        ret = [variable.Variable(y) for y in outputs]

        if not config._inference_mode:
            self.rank = max([x.rank for x in inputs])
            for y in ret:
                y.creator_node = self
            self.inputs = [x.node for x in inputs]
            self.outputs = [weakref.ref(y.node) for y in ret]

            if self._retain_inputs is True:
                for x in inputs:
                    x.node.retain_data()
            elif self._retain_inputs is not None:
                for idx in self._retain_inputs:
                    inputs[idx].node.retain_data()

        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def retain_inputs(self, indexes=None):
        if indexes is None:
            self._retain_inputs = True
        else:
            self._retain_inputs = indexes

    def forward(self, inputs):
        raise NotImplementedError()

    def backward(self, inputs, grad_outputs):
        raise NotImplementedError()
