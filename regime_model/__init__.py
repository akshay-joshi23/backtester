"""regime_model package init.

Enables JAX 64-bit precision globally — required for numerical agreement
between the JAX HMM forward/backward pass and the numpy reference, and to
keep smoothed posteriors row-summing to 1.0 within tight tolerances.
"""

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)
