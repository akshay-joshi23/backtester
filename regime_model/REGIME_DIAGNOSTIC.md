# Regime-Value Diagnostic — Why Regime Detection Doesn't Earn Its Keep

**Run date:** 2026-05-16
**Inputs:** VI fit on training window 2006-03-28 .. 2010-12-31 (1,203 obs, K=3), then smoothed posterior applied to full 2006–2026 sample. Strategy config: `risk_aversion=50`, `max_weight=0.4`, `target_annual_vol=10%`.

---

## Verdict

The regime model **correctly identifies high-vol vs low-vol periods**. Per-regime MVO weights are **meaningfully different** (L1 distance 0.43–0.53). The posterior is **highly decisive** (99.3% of training days have `max(p) > 0.9`). All the machinery works.

**The failure is in the conditional-mean estimator, not the regime detection.**

The strategy uses **gamma-weighted historical returns** as `μ_k`. Most of those historical returns came from 2008. So the strategy believes "stress = SPY loses 25% annualized." This was true in-sample. **Out-of-sample, the opposite is true:** stress-labeled days precede +38%/yr SPY rallies (post-2009 buy-the-dip regime). The strategy persistently sells the dip and buys the top.

This is **not** a regime-detection failure. It's a **conditional-return-estimator failure**. The model identifies the right state; it draws the wrong forecast from it.

---

## Evidence

### 1. Regimes are well-separated and decisive

| Stat | Value |
|------|-------|
| Mean posterior max-prob | 0.998 |
| Median posterior max-prob | 1.000 |
| Share decisive (max_p > 0.9) | 99.3% |
| Mean entropy | 0.008 bits |
| Pairwise L1 distance between regimes (features) | 3.15 / 3.56 / 4.55 |

Transition matrix is highly persistent but not pathological:

| | to stress | to normal | to calm |
|---|---|---|---|
| from stress | 0.9815 | 0.0056 | 0.0129 |
| from normal | 0.0221 | 0.9597 | 0.0182 |
| from calm | 0.0047 | 0.0084 | 0.9870 |

Implied dwell: stress 54d, normal 25d, calm 77d.

### 2. Per-regime MVO weights *are* meaningfully different

| Regime | SPY | TLT | GLD | UUP | HYG | sum (pre-vol-target) |
|--------|-----|-----|-----|-----|-----|----------------------|
| stress | 0.00 | 0.09 | 0.05 | 0.05 | 0.00 | **0.18** |
| normal | 0.05 | 0.29 | 0.03 | 0.03 | 0.14 | **0.54** |
| calm   | 0.03 | 0.02 | 0.10 | 0.01 | 0.29 | **0.46** |

Stress goes 80%+ cash (pre-scaling). Normal loads on TLT. Calm loads on HYG. **The framework is using the regime info — it's not collapsing to a single allocation.**

### 3. In-sample, forward returns *match* the conditional means

| Regime | in-sample SPY mean | fwd 1d | fwd 5d | fwd 21d |
|--------|--------------------|--------|--------|---------|
| stress | -25.0% | -28.7% | -27.8% | -25.9% |
| normal |  -2.7% |  -2.6% |  -6.8% |  -1.7% |
| calm   | +21.2% | +25.0% | +25.5% | +21.9% |

This *looks* like the strategy should work. It would, if the world looked like 2006–2010.

### 4. Out-of-sample, the relationship inverts — **the punchline**

Same VI parameters (frozen at 2010), smoothed posterior applied to OOS 2011–2026:

| Regime | in-sample SPY ret | **OOS 21d-fwd SPY ret** | shift | OOS day count |
|--------|-------------------|--------------------------|-------|----------------|
| stress | **-25.0%** | **+37.7%** | **+62.7pp** | 415 |
| normal | -2.7% | +10.6% | +13.3pp | 426 |
| calm | +21.2% | +11.6% | -9.6pp | 3,024 |

OOS unconditional SPY: +14%/yr. So OOS:
- "stress" days **outperform** unconditional by +24pp
- "calm" days **underperform** unconditional by -2pp

**The strategy's conditioning is inverted.** It de-risks (goes 0% SPY) when forward SPY returns are highest, and risks-on (3-5% SPY) when they're average.

### 5. This contaminates the whole stack

- **vs HMM-EM (Sharpe 0.93):** EM uses the same gamma-weighted historical means → same misforecast → similar performance. The Bayesian machinery isn't *broken*; both methods are downstream of the same broken estimator.
- **vs RandomRegime (Sharpe 0.86):** Random regimes get nearly identical *unconditional* moments → MVO produces near-unconditional weights → strategy holds basically a static portfolio. RandomRegime is essentially "the strategy without the regime conditioning" — and it's only 0.09 Sharpe behind. That's the value the regime conditioning is adding: 0.09 Sharpe, which is barely above noise on 15 years of data.
- **vs Risk Parity (Sharpe 1.18):** RP doesn't condition on regime at all, so it can't be *wrong* about regimes. It just allocates risk equally across uncorrelated assets. The fact that it wins is the strongest signal that **the regime conditioning, as currently specified, is net-negative** compared to no conditioning at all.

---

## Why this happened (root cause)

1. **The training window 2006–2010 is one regime episode.** 2008 dominates everything labeled "stress." Conditional means inherit 2008's signature (continued losses).
2. **Gamma-weighted historical means decay slowly.** Even after refitting each year, the historical pool of "stress" days remains anchored in 2008-style stress until thousands of new stress days accumulate — which never happens because OOS stress episodes are short (415 days over 15 years).
3. **The 2009-onward regime change is structural.** Post-QE / post-GFC, vol spikes get bought, not faded. The 2006–2010 model has no concept of "buy-the-dip stress."

So the model is doing exactly what it was specified to do — it just was specified to fit 2008.

---

## What this implies for fixes

In rough order of expected impact:

1. **Don't condition WEIGHTS on regime — condition SIZING (vol target) on regime.** Sizing is robust to the conditional-mean error: reducing exposure in high-vol regimes is *always* correct on a vol-budgeting basis, regardless of forward direction. Currently the strategy targets 10% vol *regardless of regime*, which is exactly backwards.

2. **Use a different conditional-mean estimator.** Options:
   - Shrink `μ_k` heavily toward unconditional (or even toward zero). Black-Litterman with mild views is a natural fit.
   - Use a *forward-window* estimator: when in regime k, use the average forward N-day return on days that *entered* this regime (transition-conditional, not state-conditional).
   - Recognize that mu estimation from 5 years of training data is hopeless and just go to **min-variance** (drop μ entirely).

3. **Train on a much longer history.** 1990–present has multiple stress episodes (1998 LTCM, 2000 dot-com, 2008, 2011 EU, 2020 COVID, 2022 inflation). With a more diverse stress pool, the conditional moments wouldn't be 2008-only.

4. **Use risk parity *conditional on regime*.** Take what's already winning (risk parity at Sharpe 1.18) and let regime info modulate the risk budget per asset (e.g. equity risk budget shrinks in stress, bond risk budget grows). This sidesteps the mean-estimation problem entirely.

5. **Invert the trade in stress.** Trivially: if "stress = +38% forward SPY" empirically OOS, then the model literally has the sign backwards. Buying SPY when the filter says stress is a coherent (if uncomfortable) strategy that matches the OOS data. This would be a research finding, not a product, but it's worth backtesting as a sanity check.

---

## What this is NOT

- **NOT** a regime-detection failure. The model finds clean, separated, persistent regimes.
- **NOT** a Bayesian-vs-classical issue. HMM-EM has the same problem.
- **NOT** a particle-filter / online-inference issue. The smoothed (full-hindsight) labels show the same inversion OOS.
- **NOT** a tuning issue (λ, K, vol target). Those would all fail to produce a coherent edge given the underlying mean-estimator bias.

The infrastructure is sound. The single broken component is the conditional-mean estimator's lack of generalization from 2008 to 2009–present.

---

## Artifacts

- `regime_model/outputs/diag_regime_value.md` — full numerical output from the diagnostic run
- `regime_model/notebooks/diagnose_regime_value.py` — reproducible script
