import numpy


class Optimizer(object):

    def setup(self, link):
        self.target = link
        self.t = 0
        self.states = {}
        self.prepare()

    def prepare(self):
        for name, param in self.target.namedparams():
            if name not in self.states:
                state = {}
                self.init_state(param, state)
                self.states[name] = state

    def init_state(self, param, state):
        pass

    def update(self):
        self.prepare()
        self.t += 1
        for name, param in self.target.namedparams():
            self.update_one(param, self.states[name])

    def update_one(self, param, state):
        raise NotImplementedError()
