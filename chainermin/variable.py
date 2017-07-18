import numpy
import queue


class Variable(object):

    def __init__(self, data, grad=None):
        self.data = data
        self.creator = None
        self.grad = grad

    def set_creator(self, gen_func):
        self.creator = gen_func

    def backward(self):
        if self.creator is None:
            return

        if self.data.size == 1 and self.grad is None:
            self.grad = numpy.ones_like(self.data)

        cand_funcs = queue.Queue()
        cand_funcs.put(self.creator)

        while not cand_funcs.empty():
            func = cand_funcs.get()
            in_data = [x.data for x in func.inputs]
            out_grad = [y.grad for y in func.outputs]
            gxs = func.backward(in_data, out_grad)

            for x, gx in zip(func.inputs, gxs):
                if x.creator is None:
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad += gx
                else:
                    cand_funcs.put(x.creator)
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad += gx

    def zerograd(self):
        self.grad.fill(0)
