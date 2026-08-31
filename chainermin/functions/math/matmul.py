from chainermin import backend, function


class MatMul(function.FunctionNode):
    def forward(self, x):
        self.retain_inputs()
        xp = backend.get_array_module(*x)
        a, b = x
        y = xp.matmul(a, b)
        return (y,)

    def backward(self, inputs, grad_outputs):
        xp = backend.get_array_module(*inputs, *grad_outputs)
        a, b = inputs
        (gy,) = grad_outputs

        # dL/dA = G @ B^T -> (M, P) @ (P, N) -> (M, N)
        ga = xp.matmul(gy, b.T)
        # dL/dB = A^T @ G -> (N, M) @ (M, P) -> (N, P)
        gb = xp.matmul(a.T, gy)

        return ga, gb


def matmul(a, b):
    """Computes the matrix multiplication of two arrays."""
    return MatMul()(a, b)


class BatchMatMul(function.FunctionNode):
    def forward(self, x):
        self.retain_inputs()
        xp = backend.get_array_module(*x)
        a, b = x
        # np.matmul はバッチ次元に対して行列積を実行
        # a: (B, M, N), b: (B, N, P) -> y: (B, M, P)
        y = xp.matmul(a, b)
        return (y,)

    def backward(self, inputs, grad_outputs):
        xp = backend.get_array_module(*inputs, *grad_outputs)
        a, b = inputs
        (gy,) = grad_outputs

        # dL/da = gy @ b^T -> (B, M, P) @ (B, P, N) -> (B, M, N)
        ga = xp.matmul(gy, xp.swapaxes(b, -1, -2))

        # dL/db = a^T @ gy -> (B, N, M) @ (B, M, P) -> (B, N, P)
        gb = xp.matmul(xp.swapaxes(a, -1, -2), gy)

        return ga, gb


def batch_matmul(a, b):
    """Computes the batch matrix multiplication of two arrays."""

    return BatchMatMul()(a, b)
