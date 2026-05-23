"""FastAPI server for the lab web UI.

Endpoints:
  GET  /                        — single-page app
  GET  /static/*                — js, css, etc.
  POST /api/run                 — kick off a backtest (background task)
  GET  /api/tasks/{task_id}     — poll status
  GET  /api/runs                — list saved runs
  GET  /api/runs/{run_id}       — single run details
  GET  /api/runs/{run_id}/report — serve the HTML report
  GET  /api/health              — sanity check
  POST /api/reference           — run a built-in reference strategy (no LLM)

API key flow: passed in request body, kept in memory only for the duration
of the request. NOT persisted to disk anywhere. Each request must include
the key — no session storage on the server.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from lab.web.tasks import TaskState, registry

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


# --------------------------------------------------------------------------- #
# Request / response models
# --------------------------------------------------------------------------- #


class BacktestRequest(BaseModel):
    prompt: str
    universe: list[str] = Field(default_factory=lambda: ["SPY", "TLT"])
    rebalance_freq: int | None = None
    frequency: str = "D"               # D | W | M
    realistic_costs: bool = False
    cost_bps: float = 5.0
    long_short: bool = False
    max_leverage: float = 1.0
    refine: bool = False
    agent: bool = False
    data_aware: bool = False
    provider: str | None = None        # anthropic | openai | None (auto)
    model: str | None = None
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    parent_run_id: str | None = None   # for fork


class ReferenceRequest(BaseModel):
    strategy: str                      # 60_40 | equal_weight | momentum_top2_6m | long_short_momentum | bayesian_regime
    universe: list[str] = Field(default_factory=lambda: ["SPY", "TLT"])
    start: str = "2010-01-01"
    train_end: str = "2015-01-01"
    rebalance_freq: int = 21
    cost_bps: float = 5.0
    max_leverage: float = 1.0


# --------------------------------------------------------------------------- #
# Background-task implementations
# --------------------------------------------------------------------------- #


def _run_backtest_task(req: BacktestRequest, state: TaskState) -> dict:
    """Background work: generate the strategy, run the backtest, save the run."""
    from lab.runner import run_backtest, save_run, load_run
    from lab.llm import build_universe_brief, generate_strategy, refine_strategy

    # Temporary env-var injection scoped to this request. We restore afterwards.
    saved = {}
    if req.anthropic_api_key:
        saved["ANTHROPIC_API_KEY"] = os.environ.get("ANTHROPIC_API_KEY")
        os.environ["ANTHROPIC_API_KEY"] = req.anthropic_api_key
    if req.openai_api_key:
        saved["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = req.openai_api_key

    try:
        state.log.append("composing prompt")
        full_prompt = req.prompt
        if req.parent_run_id:
            parent = load_run(req.parent_run_id)
            full_prompt = (
                f"Previously you wrote this strategy (run {req.parent_run_id}, "
                f"named '{parent['config'].get('strategy_name','?')}'):\n\n"
                "```python\n" + parent["strategy_code"] + "```\n\n"
                f"Now revise it for the following follow-up:\n\n{req.prompt}"
            )

        universe_brief = None
        if req.data_aware:
            state.log.append("building universe brief")
            universe_brief = build_universe_brief(req.universe, sample_days=30)

        if req.agent:
            from lab.agent import run_agent
            state.log.append("running agent loop (max 5 iterations)")
            generation = run_agent(
                full_prompt + ("\n\n" + universe_brief if universe_brief else ""),
                model=req.model, provider=req.provider,
            )
        else:
            state.log.append("generating strategy code")
            generation = generate_strategy(
                full_prompt, model=req.model, universe_brief=universe_brief,
                provider=req.provider,
            )

        universe = req.universe or generation.universe
        rebalance_freq = req.rebalance_freq or generation.rebalance_freq
        train_end = generation.train_end

        state.log.append(f"running backtest on {universe}")
        equity, weights, metrics, cfg, strategy_name = run_backtest(
            generation.code,
            universe=universe, train_end=train_end,
            rebalance_freq=rebalance_freq,
            cost_bps=req.cost_bps,
            long_only=not req.long_short,
            max_leverage=req.max_leverage,
            frequency=req.frequency,
            realistic_costs=req.realistic_costs,
        )

        if req.refine:
            state.log.append("refinement pass")
            refined = refine_strategy(
                generation.code, metrics, req.prompt,
                model=req.model, provider=req.provider,
            )
            if refined.code.strip() != generation.code.strip():
                state.log.append("refinement changed code; re-running backtest")
                generation = refined
                equity, weights, metrics, cfg, strategy_name = run_backtest(
                    generation.code,
                    universe=universe, train_end=train_end,
                    rebalance_freq=rebalance_freq,
                    cost_bps=req.cost_bps,
                    long_only=not req.long_short,
                    max_leverage=req.max_leverage,
                    frequency=req.frequency,
                    realistic_costs=req.realistic_costs,
                )

        state.log.append("saving run")
        artifacts = save_run(
            prompt=req.prompt, generation=generation,
            equity=equity, weights=weights, metrics=metrics,
            cfg=cfg, strategy_name=strategy_name, universe=universe,
            parent_run_id=req.parent_run_id,
        )
        return {
            "run_id": artifacts.run_id,
            "strategy_name": strategy_name,
            "metrics": metrics,
            "universe": universe,
            "code": generation.code,
            "report_url": f"/api/runs/{artifacts.run_id}/report",
        }

    finally:
        # Restore env vars to their prior state. Don't leak the request's key
        # into the shared process env beyond the request lifetime.
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _run_reference_task(req: ReferenceRequest, state: TaskState) -> dict:
    """Run a built-in reference strategy — no LLM, no API key required."""
    from lab.backtest import BacktestConfig, walk_forward_backtest
    from lab.costs import FlatBpsPerLeg
    from lab.data import load_universe
    from lab.llm import GenerationResult
    from lab.metrics import annual_turnover, compute_metrics
    from lab.runner import save_run

    REFERENCE_STRATEGIES = {
        "60_40": lambda: __import__(
            "lab.strategies", fromlist=["FixedMix"]
        ).FixedMix({"SPY": 0.6, "TLT": 0.4}, name="60/40 SPY-TLT"),
        "equal_weight": lambda: __import__(
            "lab.strategies", fromlist=["EqualWeight"]
        ).EqualWeight(),
        "momentum_top2_6m": lambda: __import__(
            "lab.strategies", fromlist=["CrossSectionalMomentum"]
        ).CrossSectionalMomentum(lookback=126, top_k=2),
        "long_short_momentum": lambda: __import__(
            "lab.strategies", fromlist=["LongShortMomentum"]
        ).LongShortMomentum(lookback=126, top_k=2),
        "bayesian_regime": lambda: __import__(
            "lab.strategies", fromlist=["BayesianRegime"]
        ).BayesianRegime(K=3, vi_steps=3000),
    }
    LONG_SHORT = {"long_short_momentum"}
    if req.strategy not in REFERENCE_STRATEGIES:
        raise ValueError(
            f"unknown reference: {req.strategy}. "
            f"options: {sorted(REFERENCE_STRATEGIES)}"
        )

    state.log.append(f"loading data for {req.universe}")
    strategy = REFERENCE_STRATEGIES[req.strategy]()
    bundle = load_universe(req.universe, start=req.start)
    cfg = BacktestConfig(
        train_end=req.train_end, rebalance_freq=req.rebalance_freq,
        long_only=req.strategy not in LONG_SHORT, max_leverage=req.max_leverage,
    )
    state.log.append("running backtest")
    res = walk_forward_backtest(
        strategy, bundle.returns[req.universe], cfg=cfg,
        cost_model=FlatBpsPerLeg(req.cost_bps),
    )
    metrics = compute_metrics(res.equity)
    metrics["annual_turnover"] = annual_turnover(res.turnover)
    metrics["total_transaction_cost"] = float(res.transaction_costs.sum())

    state.log.append("saving run")
    fake_gen = GenerationResult(
        code=f"# Reference strategy: {req.strategy}\n",
        raw_response="reference strategy — not LLM-generated\n",
        universe=req.universe, train_end=req.train_end,
        rebalance_freq=req.rebalance_freq, usage=None,
    )
    artifacts = save_run(
        prompt=f"[reference] {req.strategy}", generation=fake_gen,
        equity=res.equity, weights=res.realized_weights,
        metrics=metrics, cfg=cfg, strategy_name=strategy.name,
        universe=req.universe,
    )
    return {
        "run_id": artifacts.run_id,
        "strategy_name": strategy.name,
        "metrics": metrics,
        "universe": req.universe,
        "report_url": f"/api/runs/{artifacts.run_id}/report",
    }


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #


def create_app() -> FastAPI:
    app = FastAPI(
        title="Strategy Lab",
        description="Natural-language strategy backtester (local web UI)",
        version="0.3.0",
    )

    # --- API endpoints ---

    @app.get("/api/health")
    def health():
        from lab.llm.selection import list_available_providers
        return {
            "status": "ok",
            "providers_with_env_keys": list_available_providers(),
            "task_count": len(registry.list_recent(limit=10_000)),
        }

    @app.post("/api/run")
    def start_run(req: BacktestRequest):
        # Validate provider/key combination before kicking off the task so
        # we can return a 400 synchronously.
        has_key = bool(
            req.anthropic_api_key or req.openai_api_key
            or os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        if not has_key:
            raise HTTPException(
                status_code=400,
                detail="No API key provided. Set anthropic_api_key or openai_api_key "
                       "in the request, or export ANTHROPIC_API_KEY/OPENAI_API_KEY.",
            )
        state = registry.submit(lambda s: _run_backtest_task(req, s))
        return state.to_dict()

    @app.post("/api/reference")
    def start_reference(req: ReferenceRequest):
        state = registry.submit(lambda s: _run_reference_task(req, s))
        return state.to_dict()

    @app.get("/api/tasks/{task_id}")
    def task_status(task_id: str):
        state = registry.get(task_id)
        if state is None:
            raise HTTPException(status_code=404, detail="task not found")
        return state.to_dict()

    @app.get("/api/runs")
    def list_runs():
        from lab.runner import list_runs as _list_runs
        return {"runs": _list_runs()}

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        from lab.runner import load_run
        try:
            run = load_run(run_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"run {run_id} not found")
        return {
            "run_id": run["run_id"],
            "config": run["config"],
            "metrics": run["metrics"],
            "prompt": run["prompt"],
            "strategy_code": run["strategy_code"],
            "report_url": f"/api/runs/{run_id}/report",
        }

    @app.get("/api/runs/{run_id}/report")
    def get_report(run_id: str):
        from lab.runner import RUNS_DIR
        report_path = RUNS_DIR / run_id / "report.html"
        if not report_path.exists():
            # Try to render on demand if the saved run pre-dated the HTML feature.
            try:
                from lab.report import render_run_report
                from lab.runner import load_run
                render_run_report(load_run(run_id))
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"report unavailable and couldn't be regenerated: {e}",
                )
        return FileResponse(report_path, media_type="text/html")

    # --- Static frontend ---

    app.mount(
        "/static", StaticFiles(directory=str(STATIC_DIR)), name="static",
    )

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    return app


app = create_app()


# --------------------------------------------------------------------------- #
# Standalone launcher
# --------------------------------------------------------------------------- #


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True):
    import uvicorn
    if open_browser:
        # Open the browser shortly after uvicorn starts.
        import threading, time, webbrowser
        def _open():
            time.sleep(1.0)
            webbrowser.open(f"http://{host}:{port}/")
        threading.Thread(target=_open, daemon=True).start()
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    serve()
