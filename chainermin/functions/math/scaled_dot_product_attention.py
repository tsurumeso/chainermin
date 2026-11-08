import math
from pathlib import Path

import numpy as np

from chainermin import backend, config, function

_CU_PATH = Path(__file__).with_name("scaled_dot_product_attention.cu")
_raw_module = None
_kernels = {}


def _get_kernel(name):
    global _raw_module
    if name not in _kernels:
        if _raw_module is None:
            _raw_module = backend.cupy.RawModule(code=_CU_PATH.read_text(encoding="utf-8"))
        _kernels[name] = _raw_module.get_function(name)
    return _kernels[name]


class ScaledDotProductAttention(function.FunctionNode):
    """Scaled Dot-Product Attention (SDPA)."""

    def __init__(self, is_causal=False, scale=None):
        self.is_causal = bool(is_causal)
        self.scale = scale
        self._probs = None

    def _forward_cpu(self, xp, q, k, v, scale):
        scores = xp.matmul(q, xp.swapaxes(k, -1, -2)) * xp.asarray(scale, dtype=q.dtype)
        if self.is_causal:
            S_q, S_kv = q.shape[-2], k.shape[-2]
            mask = xp.triu(xp.ones((S_q, S_kv), dtype=bool), k=1)
            scores = xp.where(mask, -xp.inf, scores)
        probs = scores - scores.max(axis=-1, keepdims=True)
        xp.exp(probs, out=probs)
        probs /= probs.sum(axis=-1, keepdims=True)
        if not config._inference_mode:
            self._probs = probs
        out = xp.matmul(probs, v)
        return (out,)

    def _forward_gpu(self, xp, q, k, v, scale):
        scores = xp.matmul(q, xp.swapaxes(k, -1, -2))
        if not scores.flags.c_contiguous:
            scores = xp.ascontiguousarray(scores)
        S_q, S_kv = scores.shape[-2], scores.shape[-1]
        n_rows = scores.size // S_kv
        kernel_name = "sdpa_softmax_fwd_fp16" if q.dtype == np.float16 else "sdpa_softmax_fwd_fp32"
        kernel = _get_kernel(kernel_name)
        kernel(
            (n_rows,),
            (128,),
            (
                scores,
                np.int32(S_q),
                np.int32(S_kv),
                np.float32(scale),
                np.int32(1 if self.is_causal else 0),
            ),
        )
        if not config._inference_mode:
            self._probs = scores
        out = xp.matmul(scores, v)
        return (out,)

    def _backward_cpu(self, xp, q, k, v, gy, scale):
        probs = self._probs
        self._probs = None
        dv = xp.matmul(xp.swapaxes(probs, -1, -2), gy)
        dp = xp.matmul(gy, xp.swapaxes(v, -1, -2))
        ds = (
            probs
            * (dp - xp.sum(probs * dp, axis=-1, keepdims=True))
            * xp.asarray(scale, dtype=q.dtype)
        )
        if self.is_causal:
            S_q, S_kv = q.shape[-2], k.shape[-2]
            mask = xp.triu(xp.ones((S_q, S_kv), dtype=bool), k=1)
            ds = xp.where(mask, xp.asarray(0, dtype=ds.dtype), ds)
        dq = xp.matmul(ds, k)
        dk = xp.matmul(xp.swapaxes(ds, -1, -2), q)
        return dq, dk, dv

    def _backward_gpu(self, xp, q, k, v, gy, scale):
        probs = self._probs
        self._probs = None
        dv = xp.matmul(xp.swapaxes(probs, -1, -2), gy)
        dp = xp.matmul(gy, xp.swapaxes(v, -1, -2))
        if not dp.flags.c_contiguous:
            dp = xp.ascontiguousarray(dp)
        S_q, S_kv = dp.shape[-2], dp.shape[-1]
        n_rows = dp.size // S_kv
        kernel_name = "sdpa_softmax_bwd_fp16" if q.dtype == np.float16 else "sdpa_softmax_bwd_fp32"
        kernel = _get_kernel(kernel_name)
        kernel(
            (n_rows,),
            (128,),
            (
                probs,
                dp,
                np.int32(S_q),
                np.int32(S_kv),
                np.float32(scale),
                np.int32(1 if self.is_causal else 0),
            ),
        )
        del probs
        dq = xp.matmul(dp, k)
        dk = xp.matmul(xp.swapaxes(dp, -1, -2), q)
        return dq, dk, dv

    def forward(self, inputs):
        self.retain_inputs()
        q, k, v = inputs
        xp = backend.get_array_module(q, k, v)
        scale = float(1.0 / math.sqrt(q.shape[-1])) if self.scale is None else float(self.scale)

        if xp is backend.cupy and q.dtype in (np.float16, np.float32):
            return self._forward_gpu(xp, q, k, v, scale)
        return self._forward_cpu(xp, q, k, v, scale)

    def backward(self, inputs, grad_outputs):
        q, k, v = inputs
        (gy,) = grad_outputs
        xp = backend.get_array_module(q, k, v, gy)
        scale = float(1.0 / math.sqrt(q.shape[-1])) if self.scale is None else float(self.scale)

        if xp is backend.cupy and q.dtype in (np.float16, np.float32):
            return self._backward_gpu(xp, q, k, v, gy, scale)
        return self._backward_cpu(xp, q, k, v, gy, scale)


def scaled_dot_product_attention(q, k, v, is_causal=False, scale=None):
    """Computes scaled dot-product attention with a fused scale-mask-softmax CUDA kernel."""
    return ScaledDotProductAttention(is_causal=is_causal, scale=scale)(q, k, v)
