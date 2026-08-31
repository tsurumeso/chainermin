_inference_mode = False


class inference_mode:
    """Context manager to disable stochastic ops (dropout) during inference.

    Usage:
        with chainermin.inference_mode():
            y = model(x)
    """

    def __enter__(self):
        global _inference_mode
        _inference_mode = True

    def __exit__(self, *exc):
        global _inference_mode
        _inference_mode = False
