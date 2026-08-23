import numpy

from chainermin import function


class MatMul(function.Function):

    def forward(self, x):
        a, b = x
        y = numpy.matmul(a, b)
        return y,

    def backward(self, inputs, grad_outputs):
        a, b = inputs
        gy, = grad_outputs

        # dL/dA = G @ B^T -> (M, P) @ (P, N) -> (M, N)
        ga = numpy.matmul(gy, b.T)
        # dL/dB = A^T @ G -> (N, M) @ (M, P) -> (N, P)
        gb = numpy.matmul(a.T, gy)

        return ga, gb


def matmul(a, b):
    """Computes the matrix multiplication of two arrays.
    """
    return MatMul()(a, b)


class BatchMatMul(function.Function):

    def forward(self, x):
        a, b = x
        # np.matmul はバッチ次元に対して行列積を実行
        # a: (B, M, N), b: (B, N, P) -> y: (B, M, P)
        y = numpy.matmul(a, b)
        return y,

    def backward(self, inputs, grad_outputs):
        a, b = inputs
        gy, = grad_outputs

        # dL/da = gy @ b^T -> (B, M, P) @ (B, P, N) -> (B, M, N)
        ga = numpy.matmul(gy, b.transpose(0, 2, 1))
        
        # dL/db = a^T @ gy -> (B, N, M) @ (B, M, P) -> (B, N, P)
        gb = numpy.matmul(a.transpose(0, 2, 1), gy)

        return ga, gb


def batch_matmul(a, b):
    """Computes the batch matrix multiplication of two arrays.
    """

    return BatchMatMul()(a, b)
