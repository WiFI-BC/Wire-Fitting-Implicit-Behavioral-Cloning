"""Stub for IBC's block_pushing.metrics module.

The original uses tf_agents.metrics.py_metrics which we don't have. The env
calls `block_pushing_metrics.AverageFinalGoalDistance` / `AverageSuccessMetric`
only inside `get_metrics()` — a method we don't invoke when running via our
Gymnasium wrapper. We expose no-op classes with the same names so any code
path that touches `block_pushing_metrics.X` still imports cleanly.
"""


class _NoopMetric:
    def __init__(self, *args, **kwargs):
        self._values: list = []

    def __call__(self, *args, **kwargs):
        return None

    def call(self, *args, **kwargs):
        return None

    def reset(self):
        self._values.clear()

    def result(self):
        return float("nan")


class AverageFinalGoalDistance(_NoopMetric):
    pass


class AverageSuccessMetric(_NoopMetric):
    pass
