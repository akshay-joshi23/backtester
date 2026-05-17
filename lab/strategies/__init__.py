"""Reference strategies for Strategy Lab.

These exist for three reasons:
  1. Sanity checks during framework development.
  2. Few-shot examples for the LLM strategy-generator.
  3. Baselines to compare LLM-generated strategies against.
"""

from lab.strategies.baselines import EqualWeight, FixedMix
from lab.strategies.momentum import CrossSectionalMomentum, LongShortMomentum

# Bayesian regime model — imports JAX/NumPyro lazily, so the bare import is cheap.
from lab.strategies.bayesian_regime import BayesianRegime

__all__ = [
    "EqualWeight",
    "FixedMix",
    "CrossSectionalMomentum",
    "LongShortMomentum",
    "BayesianRegime",
]
