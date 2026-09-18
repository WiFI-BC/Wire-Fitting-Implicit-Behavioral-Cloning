"""WiFI-BC — Wire-Fitting Implicit Behavioral Cloning.

The algorithm itself: the control-point generator and Q estimator (`models`),
the training objectives (`loss`), observation/Q-value normalization
(`normalizations`), the action samplers (`sampling`), and the plug-and-play
`WiFIBC` policy wrapper (`policy`) for use in your own stack.
"""

from .policy import WiFIBC

__all__ = ["WiFIBC"]
