import numpy as np

import chainermin
import chainermin.functions as F
import chainermin.links as L
from chainermin import optimizers

from sklearn.datasets import fetch_mldata


class MLP(chainermin.Chain):

    def __init__(self, n_units):
        super(MLP, self).__init__(
            l1=L.Linear(784, n_units),
            l2=L.Linear(n_units, n_units),
            l3=L.Linear(n_units, 10)
        )

    def __call__(self, x, t, train=True):
        h = F.dropout(F.relu(self.l1(x)), train=train)
        h = F.dropout(F.relu(self.l2(h)), train=train)
        y = self.l3(h)
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)


if __name__ == '__main__':
    N = 60000
    batch_size = 100
    hidden_units = 1000

    mnist = fetch_mldata('MNIST original', data_home='.')
    X = mnist.data.astype(np.float32)
    X /= 255
    y = mnist.target.astype(np.int32)

    x_train, x_test = np.split(X, [N])
    y_train, y_test = np.split(y, [N])
    N_test = y_test.size

    model = MLP(hidden_units)
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    for epoch in range(100):
        perm = np.random.permutation(N)
        sum_loss = 0
        sum_acc = 0
        for i in range(0, N, batch_size):
            batch_x = x_train[perm[i:i + batch_size]]
            batch_y = y_train[perm[i:i + batch_size]]
            model.zerograds()
            loss, acc = model(batch_x, batch_y)
            loss.backward()
            optimizer.update()

            sum_loss += loss.data * batch_size
            sum_acc += acc.data * batch_size

        print('train mean loss={}, accuracy={}'.format(
            sum_loss / N, sum_acc / N))

        sum_loss = 0
        sum_acc = 0
        for i in range(0, N_test, batch_size):
            batch_x = x_test[i:i + batch_size]
            batch_y = y_test[i:i + batch_size]
            loss, acc = model(batch_x, batch_y, train=False)

            sum_loss += loss.data * batch_size
            sum_acc += acc.data * batch_size

        print('test  mean loss={}, accuracy={}'.format(
            sum_loss / N_test, sum_acc / N_test))
