"""Reference strategies for Strategy Lab.

These exist for three reasons:
  1. Sanity checks during framework development.
  2. Few-shot examples for the LLM strategy-generator.
  3. Baselines to compare LLM-generated strategies against.
"""

from lab.strategies.baselines import EqualWeight, FixedMix
from lab.strategies.momentum import CrossSectionalMomentum

__all__ = ["EqualWeight", "FixedMix", "CrossSectionalMomentum"]
