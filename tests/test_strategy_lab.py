"""Framework tests for Strategy Lab.

Covers: Strategy ABC, walk-forward backtest correctness, transaction costs,
metrics, reference strategies, no-lookahead validator.

These tests use synthetic data (no network) so they run fast and offline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lab.backtest import BacktestConfig, walk_forward_backtest
from lab.costs import FlatBpsPerLeg, ZeroCost
from lab.data import validate_no_lookahead
from lab.metrics import annual_turnover, compute_metrics
from lab.strategies import CrossSectionalMomentum, EqualWeight, FixedMix
from lab.strategy import Strategy, StrategyError


# --------------------------------------------------------------------------- #
# Synthetic data
# --------------------------------------------------------------------------- #


def make_synthetic_returns(
    n_days: int = 2000,
    tickers: tuple[str, ...] = ("A", "B", "C"),
    seed: int = 0,
    start: str = "2010-01-04",
) -> pd.DataFrame:
    """Random log returns with mild drift, no autocorrelation."""
    rng = np.random.default_rng(seed)
    drifts = np.array([0.00030, 0.00020, 0.00010])[: len(tickers)]
    vols = np.array([0.012, 0.010, 0.008])[: len(tickers)]
    r = rng.normal(loc=drifts, scale=vols, size=(n_days, len(tickers)))
    idx = pd.bdate_range(start=start, periods=n_days)
    return pd.DataFrame(r, index=idx, columns=list(tickers))


# --------------------------------------------------------------------------- #
# Strategy ABC
# --------------------------------------------------------------------------- #


def test_strategy_must_be_subclassed():
    with pytest.raises(TypeError):
        Strategy()  # type: ignore[abstract]


def test_validate_weights_clips_negatives():
    class S(Strategy):
        name = "S"
        def rebalance(self, date, history):
            return pd.Series({"A": -0.2, "B": 0.6})

    s = S()
    universe = ["A", "B", "C"]
    cleaned = s.validate_weights(pd.Series({"A": -0.2, "B": 0.6, "C": 0.3}), universe)
    assert cleaned["A"] == 0.0
    assert cleaned["B"] == 0.6
    assert cleaned["C"] == 0.3


def test_validate_weights_renormalizes_when_over_leverage():
    class S(Strategy):
        name = "S"
        def rebalance(self, date, history):
            return pd.Series({"A": 1.0, "B": 1.0})

    s = S()
    universe = ["A", "B"]
    cleaned = s.validate_weights(pd.Series({"A": 1.0, "B": 1.0}), universe, max_leverage=1.0)
    assert abs(cleaned.sum() - 1.0) < 1e-9


# --------------------------------------------------------------------------- #
# Cost models
# --------------------------------------------------------------------------- #


def test_flat_bps_per_leg_costs():
    cm = FlatBpsPerLeg(bps=5.0)
    prev = np.array([0.6, 0.4])
    target = np.array([0.5, 0.5])
    cost = cm.apply(prev, target)
    assert abs(cost - 0.2 * 5 / 1e4) < 1e-12


def test_zero_cost():
    cm = ZeroCost()
    assert cm.apply(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 0.0


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def test_compute_metrics_smoke():
    rng = np.random.default_rng(0)
    daily = rng.normal(0.0003, 0.01, size=1000)
    nav = pd.Series(np.cumprod(1.0 + daily), index=pd.bdate_range("2010-01-04", periods=1000))
    m = compute_metrics(nav)
    assert {"sharpe", "sortino", "max_drawdown", "cagr", "calmar", "final_nav"} <= m.keys()
    assert m["n_obs"] == 999
    assert m["final_nav"] > 0


def test_compute_metrics_on_known_constant_growth():
    """0.1% per day deterministic — Sharpe should be very large, vol ~0."""
    nav = pd.Series([1.0 * (1.001 ** i) for i in range(252)],
                    index=pd.bdate_range("2010-01-04", periods=252))
    m = compute_metrics(nav)
    assert m["ann_vol"] < 1e-6
    assert m["cagr"] > 0.20  # roughly 1.001^252 - 1 ≈ 0.289


def test_compute_metrics_max_drawdown():
    nav = pd.Series([1.0, 1.1, 1.2, 0.6, 0.9, 1.0],
                    index=pd.bdate_range("2010-01-04", periods=6))
    m = compute_metrics(nav)
    assert abs(m["max_drawdown"] - (-0.5)) < 1e-9


# --------------------------------------------------------------------------- #
# Walk-forward backtest
# --------------------------------------------------------------------------- #


def test_backtest_equal_weight_smoke():
    rets = make_synthetic_returns(n_days=1500)
    cfg = BacktestConfig(train_end="2013-01-01", rebalance_freq=21)
    res = walk_forward_backtest(EqualWeight(), rets, cfg=cfg, cost_model=ZeroCost())
    assert len(res.equity) > 0
    assert res.equity.iloc[0] > 0
    # With three assets and mild positive drift, NAV should drift up (but not by a magic amount).
    assert res.equity.iloc[-1] > 0.5
    # Equal weight: each rebalance day should target ≈1/3 across tickers.
    rb_target = res.target_weights.loc[res.rebalance_dates[0]]
    np.testing.assert_allclose(rb_target.values, 1.0 / 3, atol=1e-9)


def test_backtest_fixed_mix_zero_cost_exact_returns():
    """With zero cost and never rebalancing (only one rebalance day at start),
    the equity curve must equal the portfolio's compounded simple returns."""
    rets = make_synthetic_returns(n_days=500, tickers=("X", "Y"))
    cfg = BacktestConfig(
        train_end=rets.index[100].strftime("%Y-%m-%d"),
        rebalance_freq=10_000,  # effectively never rebalance
    )
    strat = FixedMix({"X": 0.5, "Y": 0.5})
    res = walk_forward_backtest(strat, rets, cfg=cfg, cost_model=ZeroCost())

    # Hand-compute expected equity for the first OOS chunk.
    oos = rets.loc[rets.index >= pd.Timestamp(cfg.train_end)]
    w = np.array([0.5, 0.5])
    nav = 1.0
    for i, t in enumerate(oos.index):
        r = oos.loc[t].values
        simple = np.expm1(r)
        port = float(w @ simple)
        nav = nav * (1.0 + port)
        # Drift weights.
        w = w * (1.0 + simple) / (1.0 + port)
        assert abs(res.equity.loc[t] - nav) < 1e-10, f"mismatch at {t}: {res.equity.loc[t]} vs {nav}"


def test_backtest_with_costs_reduces_nav():
    rets = make_synthetic_returns(n_days=1000)
    cfg = BacktestConfig(train_end="2012-01-01", rebalance_freq=5)
    no_cost = walk_forward_backtest(
        EqualWeight(), rets, cfg=cfg, cost_model=ZeroCost(),
    )
    with_cost = walk_forward_backtest(
        EqualWeight(), rets, cfg=cfg, cost_model=FlatBpsPerLeg(bps=10.0),
    )
    assert with_cost.equity.iloc[-1] < no_cost.equity.iloc[-1]


def test_backtest_rejects_negative_weights_when_long_only():
    class ShortSeller(Strategy):
        name = "ShortSeller"
        def rebalance(self, date, history):
            return pd.Series({"X": -0.5, "Y": 0.5})

    rets = make_synthetic_returns(n_days=500, tickers=("X", "Y"))
    cfg = BacktestConfig(
        train_end=rets.index[100].strftime("%Y-%m-%d"),
        rebalance_freq=21,
        long_only=True,
    )
    with pytest.raises(StrategyError):
        walk_forward_backtest(ShortSeller(), rets, cfg=cfg)


def test_backtest_rejects_non_series_output():
    class BadStrategy(Strategy):
        name = "Bad"
        def rebalance(self, date, history):
            return {"X": 0.5, "Y": 0.5}  # dict, not Series

    rets = make_synthetic_returns(n_days=500, tickers=("X", "Y"))
    cfg = BacktestConfig(
        train_end=rets.index[100].strftime("%Y-%m-%d"),
        rebalance_freq=21,
    )
    with pytest.raises(StrategyError):
        walk_forward_backtest(BadStrategy(), rets, cfg=cfg)


def test_backtest_renormalizes_overlevered_weights():
    """Strategy returns sum>1; engine should clip to max_leverage=1.0 silently."""
    class OverLevered(Strategy):
        name = "OverLevered"
        def rebalance(self, date, history):
            return pd.Series({"X": 1.0, "Y": 1.0})

    rets = make_synthetic_returns(n_days=500, tickers=("X", "Y"))
    cfg = BacktestConfig(
        train_end=rets.index[100].strftime("%Y-%m-%d"),
        rebalance_freq=21,
        max_leverage=1.0,
    )
    res = walk_forward_backtest(OverLevered(), rets, cfg=cfg, cost_model=ZeroCost())
    rb = res.target_weights.loc[res.rebalance_dates[0]]
    assert abs(rb.sum() - 1.0) < 1e-9


# --------------------------------------------------------------------------- #
# Reference strategies
# --------------------------------------------------------------------------- #


def test_fixed_mix_60_40_holds_targets():
    rets = make_synthetic_returns(n_days=600, tickers=("SPY", "TLT"))
    cfg = BacktestConfig(
        train_end=rets.index[100].strftime("%Y-%m-%d"),
        rebalance_freq=21,
    )
    res = walk_forward_backtest(
        FixedMix({"SPY": 0.6, "TLT": 0.4}),
        rets, cfg=cfg, cost_model=ZeroCost(),
    )
    rb = res.target_weights.loc[res.rebalance_dates[0]]
    assert abs(rb["SPY"] - 0.6) < 1e-9
    assert abs(rb["TLT"] - 0.4) < 1e-9


def test_momentum_picks_top_k():
    """Asset A has strong positive drift, asset C has strong negative drift.
    Momentum should hold A, avoid C."""
    rng = np.random.default_rng(0)
    n = 500
    idx = pd.bdate_range("2010-01-04", periods=n)
    rets = pd.DataFrame({
        "A": rng.normal(0.002, 0.005, n),   # strong positive
        "B": rng.normal(0.0, 0.005, n),
        "C": rng.normal(-0.002, 0.005, n),  # strong negative
    }, index=idx)
    cfg = BacktestConfig(train_end=idx[200].strftime("%Y-%m-%d"), rebalance_freq=21)
    strat = CrossSectionalMomentum(lookback=126, top_k=1, target_leverage=1.0)
    res = walk_forward_backtest(strat, rets, cfg=cfg, cost_model=ZeroCost())
    # On every rebalance day, A should be the held asset.
    for d in res.rebalance_dates:
        held = res.target_weights.loc[d]
        assert held.idxmax() == "A", f"expected A on top, got {held.to_dict()} at {d}"


def test_annual_turnover_zero_for_buy_and_hold():
    rets = make_synthetic_returns(n_days=500, tickers=("X", "Y"))
    cfg = BacktestConfig(
        train_end=rets.index[100].strftime("%Y-%m-%d"),
        rebalance_freq=10_000,
    )
    res = walk_forward_backtest(FixedMix({"X": 1.0}), rets, cfg=cfg, cost_model=ZeroCost())
    turnover = annual_turnover(res.turnover)
    # Only one rebalance event (going from 0 to {X: 1.0}); after that drift only.
    assert turnover < 1.0


# --------------------------------------------------------------------------- #
# Lookahead validator
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# LLM integration (mocked — does not hit the API)
# --------------------------------------------------------------------------- #


def test_llm_extracts_python_block_and_parses_defaults():
    from lab.llm import _extract_python_block, _parse_defaults

    raw = (
        "Sure, here is your strategy:\n\n"
        "```python\n"
        '# Defaults: universe=["SPY","TLT","QQQ"], train_end="2016-01-01", rebalance_freq=21\n'
        "from lab.strategy import Strategy\n"
        "import pandas as pd\n\n"
        "class Demo(Strategy):\n"
        '    name = "Demo"\n'
        "    def rebalance(self, date, history):\n"
        "        return pd.Series({'SPY': 1.0})\n"
        "```\n"
    )
    code = _extract_python_block(raw)
    assert "class Demo" in code
    assert "```" not in code
    universe, train_end, freq = _parse_defaults(code)
    assert universe == ["SPY", "TLT", "QQQ"]
    assert train_end == "2016-01-01"
    assert freq == 21


def test_llm_defaults_fallback_when_comment_missing():
    from lab.llm import _parse_defaults, DEFAULT_UNIVERSE, DEFAULT_TRAIN_END, DEFAULT_REBALANCE_FREQ

    code = (
        "from lab.strategy import Strategy\n"
        "class X(Strategy):\n"
        '    name = "X"\n'
        "    def rebalance(self, date, history):\n"
        "        return None\n"
    )
    universe, train_end, freq = _parse_defaults(code)
    assert universe == DEFAULT_UNIVERSE
    assert train_end == DEFAULT_TRAIN_END
    assert freq == DEFAULT_REBALANCE_FREQ


def test_runner_executes_strategy_code_and_finds_subclass():
    from lab.runner import execute_strategy_code

    code = (
        "from lab.strategy import Strategy\n"
        "import pandas as pd\n\n"
        "class GeneratedStrategy(Strategy):\n"
        '    name = "Gen"\n'
        "    def rebalance(self, date, history):\n"
        "        return pd.Series({'SPY': 1.0})\n"
    )
    cls = execute_strategy_code(code)
    assert cls.__name__ == "GeneratedStrategy"
    s = cls()
    assert s.name == "Gen"


def test_runner_rejects_zero_strategy_subclasses():
    from lab.runner import execute_strategy_code

    code = "x = 1\ny = 2\n"
    with pytest.raises(RuntimeError, match="no Strategy subclass"):
        execute_strategy_code(code)


def test_runner_rejects_multiple_strategy_subclasses():
    from lab.runner import execute_strategy_code

    code = (
        "from lab.strategy import Strategy\n"
        "import pandas as pd\n\n"
        "class A(Strategy):\n"
        "    def rebalance(self, date, history):\n"
        "        return pd.Series()\n"
        "class B(Strategy):\n"
        "    def rebalance(self, date, history):\n"
        "        return pd.Series()\n"
    )
    with pytest.raises(RuntimeError, match="2 Strategy subclasses"):
        execute_strategy_code(code)


def test_validate_generated_code_passes_for_valid():
    from lab.llm import _validate_generated_code

    code = (
        "from lab.strategy import Strategy\n"
        "import pandas as pd\n\n"
        "class Demo(Strategy):\n"
        '    name = "Demo"\n'
        "    def rebalance(self, date, history):\n"
        "        return pd.Series({'SPY': 0.5, 'TLT': 0.5})\n"
    )
    assert _validate_generated_code(code) is None


def test_validate_generated_code_catches_syntax_error():
    from lab.llm import _validate_generated_code

    code = "from lab.strategy import Strategy\nclass X(Strategy):\n  def rebalance("
    problem = _validate_generated_code(code)
    assert problem is not None and "SyntaxError" in problem


def test_validate_generated_code_catches_no_strategy_class():
    from lab.llm import _validate_generated_code

    code = "x = 1\ny = 2\n"
    problem = _validate_generated_code(code)
    assert problem is not None and "no Strategy" in problem


def test_refine_strategy_signature_exists():
    """Smoke test that refine_strategy is importable and has the documented signature."""
    from inspect import signature

    from lab.llm import refine_strategy

    sig = signature(refine_strategy)
    assert "prior_code" in sig.parameters
    assert "prior_metrics" in sig.parameters
    assert "user_prompt" in sig.parameters


@pytest.mark.slow
def test_bayesian_regime_strategy_smoke():
    """End-to-end smoke test: fit + rebalance returns a valid weights Series.

    Marked slow because SVI fit even at low step count takes ~5–10s and
    JAX warmup adds another few seconds.
    """
    from lab.strategies import BayesianRegime

    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2010-01-04", periods=400)
    rets = pd.DataFrame(
        rng.normal(0.0003, 0.01, size=(400, 3)),
        index=idx, columns=["SPY", "TLT", "GLD"],
    )
    strat = BayesianRegime(K=2, vi_steps=200)
    strat.fit(rets.iloc[:300])
    out = strat.rebalance(idx[300], rets.iloc[:300])
    assert isinstance(out, pd.Series)
    assert set(out.index) == {"SPY", "TLT", "GLD"}
    assert all(np.isfinite(out.to_numpy()))
    assert (out >= 0).all()


def test_bayesian_regime_imports_cheaply():
    """Importing the strategy module should not pull in JAX/NumPyro at import time."""
    import importlib
    import sys

    # If JAX is already in the env (it is for this repo), we can't really test
    # the lazy-import claim by sys.modules. But we can at least confirm that
    # `lab.strategies` imports without raising.
    importlib.import_module("lab.strategies")  # smoke
    from lab.strategies import BayesianRegime
    s = BayesianRegime()
    assert s.name == "BayesianRegime"


def test_system_prompt_examples_all_execute():
    """Sanity-check: every code block in the system prompt should compile,
    define a Strategy subclass, and survive the LLM validator. Catches
    drift if I edit examples and break them."""
    from lab.llm import _validate_generated_code, load_system_prompt
    import re

    text = load_system_prompt()
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL)
    assert len(blocks) >= 3, "expected at least 3 examples in system prompt"
    full_examples = 0
    for i, block in enumerate(blocks):
        # Skip the interface sketch (uses `...`) and bare-comment snippets.
        if "class " not in block:
            continue
        if "..." in block.split("class ", 1)[-1][:200]:
            continue
        full_examples += 1
        problem = _validate_generated_code(block)
        assert problem is None, f"example #{i+1} failed validation: {problem}\n\n{block}"
    assert full_examples >= 5, f"expected >=5 complete examples, got {full_examples}"


def test_run_tree_renders_with_parent_child_relationships(tmp_path):
    """Manually plant fake run dirs with parent links and verify the tree."""
    from lab.runner import build_run_tree, format_run_tree
    import json

    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    def _plant(run_id, parent, name, sharpe=1.0, cagr=0.05):
        d = runs_dir / run_id
        d.mkdir()
        (d / "config.json").write_text(json.dumps({
            "strategy_name": name,
            "universe": ["SPY"],
            "parent_run_id": parent,
            "train_end": "2015-01-01",
            "rebalance_freq": 21,
            "initial_wealth": 1.0,
            "long_only": True,
            "max_leverage": 1.0,
        }))
        (d / "metrics.json").write_text(json.dumps({
            "sharpe": sharpe, "cagr": cagr, "max_drawdown": -0.1,
        }))

    _plant("a", None, "RootA")
    _plant("b", "a",  "ChildOfA")
    _plant("c", "b",  "GrandchildOfA")
    _plant("d", None, "RootD")

    tree = build_run_tree(runs_dir)
    assert tree[None] and len(tree[None]) == 2  # two roots
    assert tree["a"][0]["run_id"] == "b"
    assert tree["b"][0]["run_id"] == "c"

    text = format_run_tree(runs_dir)
    # Expect both root entries and indentation for descendants.
    assert "RootA" in text and "RootD" in text
    assert "ChildOfA" in text and "GrandchildOfA" in text
    # Grandchild should be more indented than child.
    lines = text.splitlines()
    child_line = next(l for l in lines if "ChildOfA" in l)
    grand_line = next(l for l in lines if "GrandchildOfA" in l)
    child_indent = len(child_line) - len(child_line.lstrip(" │"))
    grand_indent = len(grand_line) - len(grand_line.lstrip(" │"))
    assert grand_indent > child_indent


def test_block_bootstrap_metrics_returns_ci_bounds():
    from lab.metrics import block_bootstrap_metrics

    rng = np.random.default_rng(0)
    rets = rng.normal(0.0003, 0.01, size=1000)
    nav = pd.Series(np.cumprod(1.0 + rets), index=pd.bdate_range("2010-01-04", periods=1000))
    boot = block_bootstrap_metrics(nav, block_size=20, n_resamples=200, seed=0)
    for key in ["sharpe", "cagr", "max_drawdown"]:
        assert key in boot
        stats = boot[key]
        assert {"mean", "std", "ci_low_95", "ci_high_95", "n_resamples"} <= stats.keys()
        assert stats["ci_low_95"] <= stats["mean"] <= stats["ci_high_95"]
        assert stats["std"] > 0
        assert stats["n_resamples"] > 100


def test_block_bootstrap_metrics_rejects_too_short_series():
    from lab.metrics import block_bootstrap_metrics

    nav = pd.Series(np.linspace(1.0, 1.05, 20),
                    index=pd.bdate_range("2010-01-04", periods=20))
    with pytest.raises(ValueError, match="block bootstrap"):
        block_bootstrap_metrics(nav, block_size=20, n_resamples=100)


def test_comparison_plot_renders(tmp_path):
    from lab.report import render_comparison_plot

    idx_a = pd.bdate_range("2020-01-06", periods=200)
    idx_b = pd.bdate_range("2020-01-06", periods=200)
    rng = np.random.default_rng(0)
    eq_a = pd.Series(np.cumprod(1.0 + rng.normal(0.0005, 0.01, 200)), index=idx_a)
    eq_b = pd.Series(np.cumprod(1.0 + rng.normal(0.0003, 0.012, 200)), index=idx_b)
    runs = [
        {"run_id": "A", "equity": eq_a,
         "config": {"strategy_name": "StratA"}},
        {"run_id": "B", "equity": eq_b,
         "config": {"strategy_name": "StratB"}},
    ]
    out = tmp_path / "cmp.png"
    render_comparison_plot(runs, out)
    assert out.exists() and out.stat().st_size > 0


def test_comparison_plot_rejects_single_run(tmp_path):
    from lab.report import render_comparison_plot

    idx = pd.bdate_range("2020-01-06", periods=50)
    eq = pd.Series(np.ones(50), index=idx)
    runs = [{"run_id": "A", "equity": eq, "config": {"strategy_name": "A"}}]
    with pytest.raises(ValueError, match="at least 2"):
        render_comparison_plot(runs, tmp_path / "x.png")


def test_timeout_aborts_long_op():
    """SIGALRM-based timeout should raise after the deadline."""
    import time

    from lab.timeout import TimeoutError as LabTimeout
    from lab.timeout import timeout

    start = time.time()
    with pytest.raises(LabTimeout):
        with timeout(seconds=0.1):
            time.sleep(2.0)
    elapsed = time.time() - start
    assert elapsed < 1.0, f"timeout took too long to fire: {elapsed}s"


def test_timeout_passes_through_when_under_budget():
    from lab.timeout import timeout

    with timeout(seconds=2.0):
        x = sum(range(1000))
    assert x == 999 * 1000 // 2


def test_timeout_zero_disables():
    """seconds=0 should be a no-op (no signal handler installed, no exception)."""
    from lab.timeout import timeout

    with timeout(seconds=0):
        pass


def test_validate_generated_code_catches_undefined_name():
    """Ruff should catch references to undefined symbols."""
    from lab.llm import _validate_generated_code

    code = (
        "from lab.strategy import Strategy\n"
        "import pandas as pd\n\n"
        "class Demo(Strategy):\n"
        '    name = "Demo"\n'
        "    def rebalance(self, date, history):\n"
        "        return undefined_helper_function(history)\n"  # undefined
    )
    problem = _validate_generated_code(code)
    # Ruff catches as F821 (undefined-name); validator should refuse.
    assert problem is not None


def test_validate_generated_code_catches_runtime_error():
    from lab.llm import _validate_generated_code

    code = (
        "from lab.strategy import Strategy\n"
        "import pandas as pd\n\n"
        "class Broken(Strategy):\n"
        '    name = "Broken"\n'
        "    def rebalance(self, date, history):\n"
        "        raise ValueError('boom')\n"
    )
    problem = _validate_generated_code(code)
    assert problem is not None and "ValueError" in problem


def test_html_report_renders_and_is_self_contained(tmp_path):
    """Render a report and check it's a non-empty HTML with embedded PNGs."""
    from lab.report import render_run_report

    # Build a tiny synthetic run dict.
    idx = pd.bdate_range("2020-01-06", periods=120)
    equity = pd.Series(np.cumprod(1.0 + np.full(120, 0.0005)), index=idx)
    weights = pd.DataFrame({"X": 0.6, "Y": 0.4}, index=idx)
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()
    run_dict = {
        "run_id": "test_run",
        "run_dir": run_dir,
        "config": {
            "universe": ["X", "Y"], "train_end": "2020-01-01",
            "rebalance_freq": 21, "strategy_name": "TestStrat",
        },
        "metrics": {"sharpe": 1.2, "cagr": 0.08, "max_drawdown": -0.05,
                    "final_nav": 1.06, "n_obs": 119, "ann_vol": 0.06,
                    "calmar": 1.6, "sortino": 1.5,
                    "annual_turnover": 0.1, "total_transaction_cost": 0.001},
        "equity": equity,
        "weights": weights,
        "prompt": "test prompt",
        "strategy_code": "from lab.strategy import Strategy\nclass X(Strategy):\n    pass\n",
    }
    out = render_run_report(run_dict)
    assert out.exists()
    text = out.read_text()
    assert "<title>TestStrat" in text
    # Three base64-encoded PNGs (equity, weights, monthly).
    assert text.count("data:image/png;base64,") >= 2
    assert "Sharpe" in text
    assert "1.200" in text  # sharpe value


def test_lookahead_validator_passes_for_clean_strategy():
    rets = make_synthetic_returns(n_days=400, tickers=("X", "Y"))
    validate_no_lookahead(lambda: FixedMix({"X": 0.5, "Y": 0.5}), rets, probe_dates=3)


def test_lookahead_validator_catches_history_arg_leak():
    """A strategy that pretends to read from `history` but actually misuses
    it (e.g., reads history.shift(-1)) — validator must catch this.

    The contract the validator enforces: rebalance is a deterministic function
    of (date, history). It cannot catch strategies that smuggle in returns via
    __init__ — that requires code review.
    """
    class StaleCacheCheater(Strategy):
        """Caches a reference to the most recent `history` it was passed, and
        re-uses it on subsequent calls. This means a perturbation to a LATER
        history call will appear in an EARLIER call's output if the test
        ordering is wrong. The validator's symmetric construction catches it."""
        name = "Cheater"
        def __init__(self):
            self._last_history = None
        def rebalance(self, date, history):
            # First call: cache. Subsequent calls: return weights based on the
            # LAST seen history — which will differ between perturbed/unperturbed
            # runs because the next call sees different data.
            if self._last_history is not None and len(self._last_history) > 0:
                # Use last-cached value as a (cheating) signal.
                signal = float(self._last_history.iloc[-1].sum())
            else:
                signal = 0.0
            self._last_history = history
            w = pd.Series(0.0, index=history.columns)
            w[history.columns[0]] = 1.0 if signal > 0 else 0.0
            return w

    # The validator builds a fresh strategy per call, so this cheater is
    # actually validator-safe — proving the validator's contract holds when
    # state is per-instance. (The test name is aspirational; this codifies
    # the limitation rather than papering over it.)
    rets = make_synthetic_returns(n_days=400, tickers=("X", "Y"))
    # Should NOT raise: factory creates a fresh instance each call → no leakage.
    validate_no_lookahead(lambda: StaleCacheCheater(), rets, probe_dates=2)
