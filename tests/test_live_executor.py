"""Tests for the lab.live paper-trading executor.

All tests use FakeBroker so CI doesn't need network. The few tests that
exercise the real Alpaca SDK are gated behind @pytest.mark.live and the
RUN_LIVE_TESTS=1 environment variable; they're skipped by default.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lab.live.broker import (
    AlpacaAdapter,
    BrokerAdapter,
    BrokerError,
    FakeBroker,
    FakeBrokerConfig,
    OrderRequest,
    OrderSide,
    OrderState,
    OrderType,
    Position,
    Quote,
)
from lab.live.executor import ExecutorConfig, PaperExecutor
from lab.live.safety import (
    KillSwitchTriggered,
    SafetyConfig,
    check_safety,
    halt,
    is_halted,
    read_halt_file,
    remove_halt_file,
    write_halt_file,
)
from lab.live.state import (
    ExecutorState,
    append_trade_log,
    initialize_state,
    load_state,
    read_trade_log,
    save_state,
    state_exists,
    state_path,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _plant_minimal_run(runs_dir: Path, run_id: str = "test_run",
                       universe: tuple[str, ...] = ("SPY", "TLT"),
                       weights: dict | None = None) -> Path:
    """Plant a fake saved-run directory under runs_dir for the executor to load."""
    weights = weights or {"SPY": 0.6, "TLT": 0.4}
    run_dir = runs_dir / run_id
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps({
        "universe": list(universe),
        "train_end": "2015-01-01",
        "rebalance_freq": 21,
        "initial_wealth": 1.0,
        "long_only": True,
        "max_leverage": 1.0,
        "strategy_name": "TestFixedMix",
        "parent_run_id": None,
    }))
    (run_dir / "metrics.json").write_text(json.dumps({
        "sharpe": 0.7, "cagr": 0.08, "max_drawdown": -0.20,
    }))
    weight_dict = ", ".join(f"'{k}': {v}" for k, v in weights.items())
    (run_dir / "strategy.py").write_text(
        "from lab.strategy import Strategy\n"
        "import pandas as pd\n\n"
        "class TestFixedMix(Strategy):\n"
        "    name = 'TestFixedMix'\n"
        "    def rebalance(self, date, history):\n"
        f"        return pd.Series({{{weight_dict}}})\n"
    )
    # equity.csv is loaded by load_run; needs index + nav col.
    idx = pd.bdate_range("2020-01-06", periods=10)
    equity = pd.Series(np.linspace(1.0, 1.05, 10), index=idx, name="nav")
    equity.to_csv(run_dir / "equity.csv", header=["nav"])
    weights_df = pd.DataFrame(0.5, index=idx, columns=list(universe))
    weights_df.to_csv(run_dir / "weights.csv")
    (run_dir / "prompt.txt").write_text("test")
    return run_dir


def _make_executor(tmp_path: Path, *, dry_run: bool = False,
                   safety: SafetyConfig | None = None,
                   broker: BrokerAdapter | None = None,
                   weights: dict | None = None,
                   universe: tuple[str, ...] = ("SPY", "TLT")) -> PaperExecutor:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    run_dir = _plant_minimal_run(runs_dir, universe=universe, weights=weights)
    if broker is None:
        broker = FakeBroker()
        for t in universe:
            broker.set_quote(t, 100.0)
    cfg = ExecutorConfig(
        run_id=run_dir.name, broker_name="fake",
        dry_run=dry_run,
        safety=safety or SafetyConfig().loosen_for_tests(),
    )
    return PaperExecutor(cfg, broker, run_dir=run_dir)


# --------------------------------------------------------------------------- #
# Broker adapter contract
# --------------------------------------------------------------------------- #


def test_fake_broker_satisfies_abc():
    fb = FakeBroker()
    assert isinstance(fb, BrokerAdapter)


def test_alpaca_adapter_subclass_of_abc():
    assert issubclass(AlpacaAdapter, BrokerAdapter)


def test_fake_broker_round_trip_submit_get_cancel():
    fb = FakeBroker(FakeBrokerConfig(fill_mode="instant"))
    fb.set_quote("SPY", 500.0)
    o = fb.submit_order(OrderRequest(ticker="SPY", shares=10,
                                      side=OrderSide.BUY))
    status = fb.get_order(o.order_id)
    assert status.state == OrderState.FILLED
    assert status.filled_shares == 10
    # Filling a buy at the ask.
    assert status.average_fill_price > 500.0

    # Submit + cancel.
    fb2 = FakeBroker(FakeBrokerConfig(fill_mode="delayed",
                                       fill_delay_seconds=999))
    fb2.set_quote("SPY", 500.0)
    o2 = fb2.submit_order(OrderRequest(ticker="SPY", shares=5,
                                        side=OrderSide.BUY))
    fb2.cancel_order(o2.order_id)
    assert fb2.get_order(o2.order_id).state == OrderState.CANCELLED


def test_alpaca_adapter_raises_without_env(monkeypatch):
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="credentials"):
        AlpacaAdapter()


def test_alpaca_adapter_refuses_live(monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "x")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "y")
    with pytest.raises(NotImplementedError, match="Live trading is not enabled"):
        AlpacaAdapter(live=True)


def test_alpaca_adapter_refuses_live_base_url(monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "x")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "y")
    monkeypatch.setenv("APCA_BASE_URL", "https://api.alpaca.markets")  # live URL
    with pytest.raises(NotImplementedError, match="non-paper endpoint"):
        AlpacaAdapter()


def test_fake_broker_rejection_mode():
    fb = FakeBroker(FakeBrokerConfig(fill_mode="reject"))
    fb.set_quote("SPY", 100.0)
    o = fb.submit_order(OrderRequest(ticker="SPY", shares=1, side=OrderSide.BUY))
    assert fb.get_order(o.order_id).state == OrderState.REJECTED


def test_fake_broker_error_mode_raises():
    fb = FakeBroker(FakeBrokerConfig(fill_mode="error"))
    fb.set_quote("SPY", 100.0)
    with pytest.raises(BrokerError):
        fb.submit_order(OrderRequest(ticker="SPY", shares=1,
                                      side=OrderSide.BUY))


def test_fake_broker_partial_fill_position_math():
    fb = FakeBroker(FakeBrokerConfig(fill_mode="partial",
                                      partial_fill_fraction=0.5))
    fb.set_quote("SPY", 100.0)
    o = fb.submit_order(OrderRequest(ticker="SPY", shares=10,
                                      side=OrderSide.BUY))
    status = fb.get_order(o.order_id)
    assert status.state == OrderState.PARTIAL
    assert abs(status.filled_shares - 5.0) < 1e-9
    assert abs(fb.get_positions()["SPY"].shares - 5.0) < 1e-9


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #


def test_state_save_load_roundtrip(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    initial = ExecutorState(run_id="r", broker_name="fake",
                             starting_nav=100_000.0, peak_nav=100_000.0,
                             total_orders=3, total_fills=2,
                             intended_positions={"SPY": 10.0, "TLT": 20.0})
    save_state(run_dir, initial)
    loaded = load_state(run_dir)
    assert loaded.run_id == "r"
    assert loaded.broker_name == "fake"
    assert loaded.total_orders == 3
    assert loaded.intended_positions == {"SPY": 10.0, "TLT": 20.0}


def test_state_atomic_write_no_partial_files(tmp_path):
    """state.json should never coexist with a half-written file."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    s = ExecutorState(run_id="r", broker_name="fake", starting_nav=100.0)
    save_state(run_dir, s)
    files = list((run_dir / "live_state").iterdir())
    # Just state.json + a possible lock file should exist; no .tmp leftovers.
    names = {f.name for f in files}
    assert "state.json" in names
    # tempfile mkstemp prefix is 'state-' — should be gone.
    leftover = [n for n in names if n.startswith("state-") and n != "state.json"]
    assert leftover == [], f"leftover tmp files: {leftover}"


def test_state_initialize_refuses_overwrite(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    initialize_state(run_dir, run_id="r", broker_name="fake",
                     starting_nav=100.0)
    with pytest.raises(FileExistsError):
        initialize_state(run_dir, run_id="r", broker_name="fake",
                         starting_nav=100.0)


def test_trade_log_append_and_read(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    append_trade_log(run_dir, "kind1", {"a": 1})
    append_trade_log(run_dir, "kind2", {"b": 2})
    log = read_trade_log(run_dir)
    assert len(log) == 2
    assert log[0]["kind"] == "kind1"
    assert log[1]["payload"]["b"] == 2


def test_trade_log_limit(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for i in range(10):
        append_trade_log(run_dir, "k", {"i": i})
    last3 = read_trade_log(run_dir, limit=3)
    assert len(last3) == 3
    assert last3[-1]["payload"]["i"] == 9


def test_trade_log_skips_malformed_lines(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    append_trade_log(run_dir, "ok", {"x": 1})
    # Corrupt one line.
    from lab.live.state import trade_log_path
    with open(trade_log_path(run_dir), "a") as f:
        f.write("this is not json\n")
    append_trade_log(run_dir, "ok2", {"x": 2})
    log = read_trade_log(run_dir)
    assert len(log) == 2  # malformed was skipped


# --------------------------------------------------------------------------- #
# Safety
# --------------------------------------------------------------------------- #


def test_halt_file_creates_and_removes(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert read_halt_file(run_dir) is None
    write_halt_file(run_dir, "boom")
    assert read_halt_file(run_dir) is not None
    assert "boom" in read_halt_file(run_dir)
    remove_halt_file(run_dir)
    assert read_halt_file(run_dir) is None


def test_is_halted_triggers_on_file_or_state(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    state = ExecutorState(run_id="r")
    assert not is_halted(run_dir, state)
    state.halt_reason = "x"
    assert is_halted(run_dir, state)
    state.halt_reason = None
    write_halt_file(run_dir, "y")
    assert is_halted(run_dir, state)


def test_safety_blocks_oversize_order(tmp_path):
    fb = FakeBroker(FakeBrokerConfig(starting_cash=100_000))
    fb.set_quote("SPY", 100.0)
    state = ExecutorState(run_id="r", starting_nav=100_000, peak_nav=100_000)
    cfg = SafetyConfig(max_order_size_pct=0.10, raise_on_violation=True)
    # Try to buy $50k worth (50% of NAV) when limit is 10%.
    orders = [OrderRequest(ticker="SPY", shares=500, side=OrderSide.BUY)]
    with pytest.raises(KillSwitchTriggered, match="max_order_size"):
        check_safety(broker=fb, state=state, target_orders=orders, cfg=cfg)


def test_safety_blocks_oversize_post_trade_position(tmp_path):
    fb = FakeBroker()
    fb.set_quote("SPY", 100.0)
    state = ExecutorState(run_id="r", starting_nav=100_000, peak_nav=100_000)
    cfg = SafetyConfig(max_position_pct=0.30, max_order_size_pct=1.0,
                       raise_on_violation=True)
    # Buy $40k (40% of NAV) — order size OK at 100%, but position cap 30%.
    orders = [OrderRequest(ticker="SPY", shares=400, side=OrderSide.BUY)]
    with pytest.raises(KillSwitchTriggered, match="max_position"):
        check_safety(broker=fb, state=state, target_orders=orders, cfg=cfg)


def test_safety_blocks_when_daily_loss_exceeded(tmp_path):
    fb = FakeBroker(FakeBrokerConfig(starting_cash=90_000))  # 10% loss
    state = ExecutorState(run_id="r", starting_nav=100_000, peak_nav=100_000)
    cfg = SafetyConfig(max_daily_loss_pct=0.05, raise_on_violation=True)
    with pytest.raises(KillSwitchTriggered, match="daily loss"):
        check_safety(broker=fb, state=state, target_orders=[], cfg=cfg)


def test_safety_blocks_on_drawdown(tmp_path):
    fb = FakeBroker(FakeBrokerConfig(starting_cash=70_000))  # 30% below peak
    state = ExecutorState(run_id="r", starting_nav=100_000, peak_nav=100_000)
    cfg = SafetyConfig(max_total_drawdown_pct=0.20, raise_on_violation=True,
                       max_daily_loss_pct=1.0)  # disable daily-loss guard for this test
    with pytest.raises(KillSwitchTriggered, match="drawdown"):
        check_safety(broker=fb, state=state, target_orders=[], cfg=cfg)


def test_safety_blocks_when_orders_per_day_exceeded():
    fb = FakeBroker()
    fb.set_quote("SPY", 100.0)
    state = ExecutorState(run_id="r", starting_nav=100_000, peak_nav=100_000)
    cfg = SafetyConfig(max_orders_per_day=5, raise_on_violation=True)
    orders = [OrderRequest(ticker="SPY", shares=1, side=OrderSide.BUY)
              for _ in range(3)]
    with pytest.raises(KillSwitchTriggered, match="max_orders_per_day"):
        check_safety(broker=fb, state=state, target_orders=orders, cfg=cfg,
                     orders_today=4)


def test_safety_require_paper_blocks_live_request():
    fb = FakeBroker()
    state = ExecutorState(run_id="r", starting_nav=100_000, peak_nav=100_000)
    cfg = SafetyConfig(require_paper=True, raise_on_violation=True)
    with pytest.raises(KillSwitchTriggered, match="live trading was requested"):
        check_safety(broker=fb, state=state, target_orders=[], cfg=cfg,
                     is_live_requested=True)


def test_safety_market_closed_is_a_skip_reason():
    fb = FakeBroker(FakeBrokerConfig(market_open=False))
    state = ExecutorState(run_id="r", starting_nav=100_000, peak_nav=100_000)
    cfg = SafetyConfig(require_market_open=True, raise_on_violation=True)
    with pytest.raises(KillSwitchTriggered, match="market is not open"):
        check_safety(broker=fb, state=state, target_orders=[], cfg=cfg)


# --------------------------------------------------------------------------- #
# Executor logic
# --------------------------------------------------------------------------- #


def test_executor_first_step_buys_to_targets(tmp_path, monkeypatch):
    fb = FakeBroker()
    fb.set_quote("SPY", 100.0)
    fb.set_quote("TLT", 100.0)
    # Patch load_universe to return our quotes — avoids yfinance fetch.
    _patch_load_universe(monkeypatch, ["SPY", "TLT"])
    ex = _make_executor(tmp_path, broker=fb,
                         weights={"SPY": 0.6, "TLT": 0.4})
    result = ex.step()
    assert not result.halted
    # Should target ~0.6 * 100000 / 100 = 600 SPY shares and 400 TLT.
    target = result.target_shares
    assert abs(target["SPY"] - 600) < 5
    assert abs(target["TLT"] - 400) < 5
    pos = fb.get_positions()
    # Allow some slack for ask-side fills (mid plus half-spread).
    assert abs(pos["SPY"].shares - target["SPY"]) < 1e-6
    assert abs(pos["TLT"].shares - target["TLT"]) < 1e-6


def test_executor_step_is_idempotent_within_drift(tmp_path, monkeypatch):
    fb = FakeBroker()
    fb.set_quote("SPY", 100.0); fb.set_quote("TLT", 100.0)
    _patch_load_universe(monkeypatch, ["SPY", "TLT"])
    ex = _make_executor(tmp_path, broker=fb,
                         weights={"SPY": 0.6, "TLT": 0.4})
    ex.step()
    # Second step should produce zero new orders.
    r2 = ex.step()
    assert len(r2.orders_submitted) == 0


def test_executor_zero_weights_liquidates_positions(tmp_path, monkeypatch):
    fb = FakeBroker()
    fb.set_quote("SPY", 100.0); fb.set_quote("TLT", 100.0)
    _patch_load_universe(monkeypatch, ["SPY", "TLT"])
    # First create positions via a 50/50 strategy.
    ex = _make_executor(tmp_path, broker=fb,
                         weights={"SPY": 0.5, "TLT": 0.5})
    ex.step()
    pos_before = fb.get_positions()
    assert pos_before["SPY"].shares > 0

    # Now rewrite the strategy on disk to zero-weights and start a fresh
    # executor on the same run_dir.
    run_dir = ex.run_dir
    (run_dir / "strategy.py").write_text(
        "from lab.strategy import Strategy\nimport pandas as pd\n\n"
        "class ZeroStrat(Strategy):\n"
        "    name = 'ZeroStrat'\n"
        "    def rebalance(self, date, history):\n"
        "        return pd.Series(0.0, index=history.columns)\n"
    )
    cfg2 = ExecutorConfig(run_id=run_dir.name, broker_name="fake",
                          safety=SafetyConfig().loosen_for_tests())
    ex2 = PaperExecutor(cfg2, fb, run_dir=run_dir)
    r2 = ex2.step()
    # Sells should have been issued for both legs.
    assert len(r2.orders_submitted) >= 1
    # Positions cleared.
    pos_after = fb.get_positions()
    for t in ("SPY", "TLT"):
        if t in pos_after:
            assert abs(pos_after[t].shares) < 1e-6


def test_executor_dry_run_does_not_submit(tmp_path, monkeypatch):
    fb = FakeBroker()
    fb.set_quote("SPY", 100.0); fb.set_quote("TLT", 100.0)
    _patch_load_universe(monkeypatch, ["SPY", "TLT"])
    ex = _make_executor(tmp_path, dry_run=True, broker=fb,
                         weights={"SPY": 0.6, "TLT": 0.4})
    result = ex.step()
    # Orders are recorded but broker has zero positions.
    assert len(result.orders_submitted) > 0
    assert not fb.get_positions()


def test_executor_halt_short_circuits(tmp_path, monkeypatch):
    fb = FakeBroker()
    fb.set_quote("SPY", 100.0); fb.set_quote("TLT", 100.0)
    _patch_load_universe(monkeypatch, ["SPY", "TLT"])
    ex = _make_executor(tmp_path, broker=fb)
    write_halt_file(ex.run_dir, "manual stop")
    result = ex.step()
    assert result.halted
    assert "manual stop" in (result.halt_reason or "")
    assert not fb.get_positions()


def test_executor_market_closed_skips_cleanly(tmp_path, monkeypatch):
    fb = FakeBroker(FakeBrokerConfig(market_open=False))
    fb.set_quote("SPY", 100.0); fb.set_quote("TLT", 100.0)
    _patch_load_universe(monkeypatch, ["SPY", "TLT"])
    safety = SafetyConfig().loosen_for_tests()
    # Re-enable require_market_open while keeping other guards relaxed.
    safety = SafetyConfig(
        max_daily_loss_pct=1.0, max_total_drawdown_pct=1.0,
        max_position_pct=1.0, max_order_size_pct=1.0,
        max_orders_per_day=10_000,
        require_market_open=True, require_paper=True,
    )
    ex = _make_executor(tmp_path, broker=fb, safety=safety)
    result = ex.step()
    assert result.skipped_market_closed
    assert not fb.get_positions()


def test_executor_records_failed_order_without_crashing(tmp_path, monkeypatch):
    fb = FakeBroker(FakeBrokerConfig(fill_mode="error"))
    fb.set_quote("SPY", 100.0); fb.set_quote("TLT", 100.0)
    _patch_load_universe(monkeypatch, ["SPY", "TLT"])
    ex = _make_executor(tmp_path, broker=fb)
    result = ex.step()
    # Should record failures, not raise.
    assert len(result.orders_failed) >= 1


def test_executor_reconcile_reports_drift(tmp_path, monkeypatch):
    fb = FakeBroker()
    fb.set_quote("SPY", 100.0); fb.set_quote("TLT", 100.0)
    _patch_load_universe(monkeypatch, ["SPY", "TLT"])
    ex = _make_executor(tmp_path, broker=fb,
                         weights={"SPY": 0.6, "TLT": 0.4})
    ex.step()  # establish positions
    # Manually nudge actual positions so they diverge.
    fb._positions["SPY"] = Position(ticker="SPY", shares=100.0,
                                     avg_entry_price=100.0,
                                     market_value=10_000.0)
    result = ex.reconcile()
    assert result.drift["SPY"] > 1e-3  # huge drift


def test_executor_constructor_raises_when_live_true(tmp_path, monkeypatch):
    _patch_load_universe(monkeypatch, ["SPY", "TLT"])
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    run_dir = _plant_minimal_run(runs_dir)
    cfg = ExecutorConfig(run_id="test_run", broker_name="fake", live=True)
    with pytest.raises(NotImplementedError, match="Live trading is not enabled"):
        PaperExecutor(cfg, FakeBroker(), run_dir=run_dir)


# --------------------------------------------------------------------------- #
# Backtest-live parity (the load-bearing test)
# --------------------------------------------------------------------------- #


def test_backtest_live_parity_within_one_percent(tmp_path, monkeypatch):
    """Run the executor for ~100 days using a FakeBroker that fills at the
    saved equity-curve's daily prices. The executor's NAV should match the
    backtest's equity to within 1%."""
    from lab.backtest import BacktestConfig, walk_forward_backtest
    from lab.costs import ZeroCost
    from lab.data import UniverseBundle
    from lab.strategies.baselines import FixedMix
    from lab.runner import save_run
    from lab.llm import GenerationResult

    # 1. Synthetic universe.
    rng = np.random.default_rng(0)
    n = 250
    idx = pd.bdate_range("2020-01-06", periods=n)
    rets = pd.DataFrame({
        "SPY": rng.normal(0.0005, 0.012, n),
        "TLT": rng.normal(0.0001, 0.008, n),
    }, index=idx)
    prices = pd.DataFrame(np.exp(rets.cumsum()) * 100.0, index=idx,
                           columns=rets.columns)
    bundle = UniverseBundle(
        prices=prices, returns=rets, tickers=("SPY", "TLT"),
        start="2020", end=None, frequency="D",
    )
    # Patch load_universe everywhere it's used.
    import lab.runner as runner_mod
    import lab.live.executor as live_exec_mod
    monkeypatch.setattr(runner_mod, "load_universe",
                         lambda *_a, **_k: bundle, raising=False)
    # The executor imports load_universe locally, so we patch lab.data too.
    import lab.data as data_mod
    monkeypatch.setattr(data_mod, "load_universe",
                         lambda *_a, **_k: bundle, raising=False)

    # 2. Save a backtest run for FixedMix 60/40.
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    monkeypatch.setattr(runner_mod, "RUNS_DIR", runs_dir, raising=False)

    strategy = FixedMix({"SPY": 0.6, "TLT": 0.4}, name="60_40")
    cfg = BacktestConfig(train_end=idx[20].strftime("%Y-%m-%d"),
                          rebalance_freq=21)
    res = walk_forward_backtest(strategy, rets[["SPY", "TLT"]],
                                  cfg=cfg, cost_model=ZeroCost())
    fake_gen = GenerationResult(
        code=(
            "from lab.strategy import Strategy\n"
            "import pandas as pd\n\n"
            "class SixtyForty(Strategy):\n"
            "    name = '60_40'\n"
            "    def rebalance(self, date, history):\n"
            "        return pd.Series({'SPY': 0.6, 'TLT': 0.4})\n"
        ),
        raw_response="", universe=["SPY", "TLT"], train_end="2020-01-31",
        rebalance_freq=21, usage=None,
    )
    artifacts = save_run(
        prompt="60/40", generation=fake_gen,
        equity=res.equity, weights=res.realized_weights,
        metrics={"sharpe": 0.5}, cfg=cfg,
        strategy_name="60_40", universe=["SPY", "TLT"],
        runs_dir=runs_dir,
    )
    run_id = artifacts.run_id

    # 3. Build a FakeBroker that fills at the saved daily prices.
    fb = FakeBroker(FakeBrokerConfig(starting_cash=1.0, spread_bps=0.0,
                                       allow_fractional=True))
    # Disable safety for parity (we want the executor to trade unhindered).
    safety = SafetyConfig().loosen_for_tests()

    # Step through each backtest day.
    exec_cfg = ExecutorConfig(
        run_id=run_id, broker_name="fake", drift_threshold_pct=0.0,
        safety=safety,
    )
    fb.set_quote("SPY", float(prices["SPY"].iloc[0]))
    fb.set_quote("TLT", float(prices["TLT"].iloc[0]))
    executor = PaperExecutor(exec_cfg, fb, run_dir=runs_dir / run_id)

    # We'll mimic the backtest's calendar: step once per rebalance period only.
    # The backtest rebalances every 21 days; we do the same here.
    nav_series: list[float] = []
    for i, t in enumerate(res.equity.index):
        # Update FakeBroker prices to today's close.
        fb.set_quote("SPY", float(prices["SPY"].loc[t]))
        fb.set_quote("TLT", float(prices["TLT"].loc[t]))
        # On rebalance days, run step(); always record NAV.
        if i % 21 == 0:
            # Monkeypatch executor's notion of "today" so its history reflects
            # what the backtest had at this point.
            class _FakeDT:
                @staticmethod
                def now(tz=None):
                    return datetime.combine(t.to_pydatetime().date(),
                                             datetime.min.time(),
                                             tzinfo=timezone.utc)
            monkeypatch.setattr(live_exec_mod, "datetime", _FakeDT,
                                  raising=False)
            executor.step()
        nav_series.append(fb.get_account().equity)

    # 4. Compare equity curves.
    live_nav = pd.Series(nav_series, index=res.equity.index)
    # Both should be NAV trajectories starting at 1.0.
    # Backtest equity starts at 1.0 (initial_wealth); FakeBroker starts at $1.
    # Normalize.
    live_norm = live_nav / live_nav.iloc[0]
    bt_norm = res.equity / res.equity.iloc[0]
    max_diff = float((live_norm - bt_norm).abs().max())
    assert max_diff < 0.01, (
        f"backtest-live parity failure: max NAV diff {max_diff:.4f} > 1%"
    )


# --------------------------------------------------------------------------- #
# CLI smoke tests
# --------------------------------------------------------------------------- #


def test_cli_halt_creates_file(tmp_path, monkeypatch):
    import lab.live.cli as live_cli
    import lab.runner as runner_mod

    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    _plant_minimal_run(runs_dir, run_id="my_run")
    monkeypatch.setattr(runner_mod, "RUNS_DIR", runs_dir, raising=False)
    import argparse
    ns = argparse.Namespace(run_id="my_run", reason="testing")
    assert live_cli.cmd_halt(ns) == 0
    assert (runs_dir / "my_run" / "HALT").exists()


def test_cli_resume_removes_file(tmp_path, monkeypatch):
    import lab.live.cli as live_cli
    import lab.runner as runner_mod

    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    run_dir = _plant_minimal_run(runs_dir, run_id="my_run")
    write_halt_file(run_dir, "test")
    monkeypatch.setattr(runner_mod, "RUNS_DIR", runs_dir, raising=False)
    import argparse
    ns = argparse.Namespace(run_id="my_run")
    assert live_cli.cmd_resume(ns) == 0
    assert not (runs_dir / "my_run" / "HALT").exists()


def test_cli_status_handles_no_state(tmp_path, monkeypatch, capsys):
    import lab.live.cli as live_cli
    import lab.runner as runner_mod

    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    _plant_minimal_run(runs_dir, run_id="r")
    monkeypatch.setattr(runner_mod, "RUNS_DIR", runs_dir, raising=False)
    import argparse
    ns = argparse.Namespace(run_id="r")
    assert live_cli.cmd_status(ns) == 0
    out = capsys.readouterr().out
    assert "no live state" in out


def test_cli_live_trade_always_errors():
    import lab.live.cli as live_cli
    import argparse
    ns = argparse.Namespace(run_id="anything")
    assert live_cli.cmd_live_trade(ns) == 3


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _patch_load_universe(monkeypatch, tickers: list[str]):
    """Avoid yfinance fetches by handing back synthetic data."""
    from lab.data import UniverseBundle
    import lab.data as data_mod

    n = 300
    idx = pd.bdate_range("2018-01-04", periods=n)
    rng = np.random.default_rng(0)
    rets = pd.DataFrame(
        rng.normal(0.0003, 0.012, size=(n, len(tickers))),
        index=idx, columns=tickers,
    )
    prices = pd.DataFrame(np.exp(rets.cumsum()) * 100.0, index=idx,
                           columns=tickers)
    bundle = UniverseBundle(
        prices=prices, returns=rets, tickers=tuple(tickers),
        start="2018", end=None, frequency="D",
    )
    monkeypatch.setattr(data_mod, "load_universe",
                         lambda *_a, **_k: bundle, raising=False)
