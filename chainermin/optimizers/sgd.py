from chainermin import optimizer


class SGD(optimizer.Optimizer):

    def __init__(self, lr=0.01):
        self.lr = lr

    def update_one(self, param, state):
        param.data -= self.lr * param.grad
