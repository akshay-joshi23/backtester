// Strategy Lab web UI — vanilla JS, no framework.

const $ = (id) => document.getElementById(id);
const KEY_STORAGE = "stratlab_keys";

// ---------- API key handling (localStorage only, never to disk) ----------

function loadKeys() {
  try {
    const stored = JSON.parse(localStorage.getItem(KEY_STORAGE) || "{}");
    $("anthropic-key").value = stored.anthropic || "";
    $("openai-key").value = stored.openai || "";
    updateKeyStatus();
  } catch {}
}

function saveKeys() {
  const data = {
    anthropic: $("anthropic-key").value.trim(),
    openai: $("openai-key").value.trim(),
  };
  localStorage.setItem(KEY_STORAGE, JSON.stringify(data));
  updateKeyStatus();
  toast("Keys saved (in browser localStorage)");
}

function clearKeys() {
  localStorage.removeItem(KEY_STORAGE);
  $("anthropic-key").value = "";
  $("openai-key").value = "";
  updateKeyStatus();
  toast("Keys cleared");
}

function updateKeyStatus() {
  const a = $("anthropic-key").value.trim();
  const o = $("openai-key").value.trim();
  const parts = [];
  if (a) parts.push(`anthropic ✓ (${a.length})`);
  if (o) parts.push(`openai ✓ (${o.length})`);
  $("keys-status").textContent = parts.length ? parts.join(" · ") : "(none set)";
}

// ---------- Status display ----------

function setStatus(text, cls = "status-pending") {
  const box = $("status-box");
  box.textContent = text;
  box.className = cls;
}

function toast(msg) {
  // Lightweight toast — just temporarily replaces the status box.
  const prev = $("status-box").textContent;
  const prevCls = $("status-box").className;
  setStatus(msg, "status-done");
  setTimeout(() => { setStatus(prev, prevCls); }, 2000);
}

// ---------- Result rendering ----------

function fmtPct(v) { return v == null ? "—" : (v * 100).toFixed(2) + "%"; }
function fmtNum(v, p = 4) { return v == null ? "—" : Number(v).toFixed(p); }

function renderMetrics(metrics) {
  if (!metrics) return "<p class='muted'>(no metrics)</p>";
  const rows = [
    ["Sharpe", fmtNum(metrics.sharpe, 3), metrics.sharpe > 1 ? "pos-good" : (metrics.sharpe < 0 ? "neg-bad" : "")],
    ["Sortino", fmtNum(metrics.sortino, 3)],
    ["CAGR", fmtPct(metrics.cagr)],
    ["Ann. vol", fmtPct(metrics.ann_vol)],
    ["Max drawdown", fmtPct(metrics.max_drawdown), metrics.max_drawdown < -0.25 ? "neg-bad" : ""],
    ["Calmar", fmtNum(metrics.calmar, 3)],
    ["Final NAV", fmtNum(metrics.final_nav, 4)],
    ["Annual turnover", fmtNum(metrics.annual_turnover, 2)],
    ["Total TX cost", fmtNum(metrics.total_transaction_cost, 4)],
    ["Trading days", metrics.n_obs ?? "—"],
  ];
  return "<table class='metric-table'>" +
    rows.map(([k, v, cls]) => `<tr><td>${k}</td><td class="${cls || ''}">${v}</td></tr>`).join("") +
    "</table>";
}

function renderResult(result) {
  if (!result) {
    $("result-box").innerHTML = "<p class='muted'>(no result)</p>";
    return;
  }
  const code = result.code ? `<details><summary>Strategy code</summary><pre class="code">${escapeHtml(result.code)}</pre></details>` : "";
  $("result-box").innerHTML = `
    <div class="run-card">
      <div>
        <div class="name">${escapeHtml(result.strategy_name || "?")}</div>
        <div class="id">run_id: ${result.run_id}</div>
      </div>
      <div class="actions">
        <a href="${result.report_url}" target="_blank">open report ↗</a>
      </div>
    </div>
    ${renderMetrics(result.metrics)}
    ${code}
  `;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;",
  }[c]));
}

// ---------- Runs list ----------

async function loadRuns() {
  try {
    const r = await fetch("/api/runs");
    const { runs } = await r.json();
    if (!runs.length) {
      $("runs-list").innerHTML = "<p class='muted'>(no past runs)</p>";
      return;
    }
    $("runs-list").innerHTML = runs.slice().reverse().map((run) => `
      <div class="run-card">
        <div>
          <div class="name">${escapeHtml(run.strategy_name || "?")}</div>
          <div class="id">${run.run_id} · ${escapeHtml((run.universe || []).join(", "))}</div>
          <div class="metrics">
            Sharpe ${fmtNum(run.sharpe, 2)} · CAGR ${fmtPct(run.cagr)} · MaxDD ${fmtPct(run.max_drawdown)}
          </div>
        </div>
        <div class="actions">
          <a href="/api/runs/${run.run_id}/report" target="_blank">report ↗</a>
        </div>
      </div>
    `).join("");
  } catch (e) {
    $("runs-list").innerHTML = `<p class='muted'>error loading runs: ${escapeHtml(e.message)}</p>`;
  }
}

// ---------- Submission + polling ----------

async function startBacktest() {
  const keys = JSON.parse(localStorage.getItem(KEY_STORAGE) || "{}");
  const prompt = $("prompt").value.trim();
  if (!prompt) { alert("Please enter a strategy prompt."); return; }

  const universe = $("universe").value.trim().split(/\s+/).map((s) => s.toUpperCase()).filter(Boolean);
  if (!universe.length) { alert("Please enter at least one ticker."); return; }

  const rebalRaw = $("rebalance-freq").value.trim();
  const body = {
    prompt,
    universe,
    frequency: $("frequency").value,
    rebalance_freq: rebalRaw ? Number(rebalRaw) : null,
    refine: $("refine").checked,
    agent: $("agent").checked,
    data_aware: $("data-aware").checked,
    realistic_costs: $("realistic-costs").checked,
    long_short: $("long-short").checked,
    provider: $("provider").value || null,
    anthropic_api_key: keys.anthropic || null,
    openai_api_key: keys.openai || null,
  };

  $("run-btn").disabled = true;
  setStatus("submitting...", "status-running");

  try {
    const res = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || res.statusText);
    }
    const task = await res.json();
    await pollTask(task.task_id);
  } catch (e) {
    setStatus("error: " + e.message, "status-error");
  } finally {
    $("run-btn").disabled = false;
    loadRuns();
  }
}

async function startReference() {
  const universe = $("universe").value.trim().split(/\s+/).map((s) => s.toUpperCase()).filter(Boolean);
  if (!universe.length) { alert("Please enter at least one ticker."); return; }

  const body = {
    strategy: $("ref-strategy").value,
    universe,
  };

  $("ref-btn").disabled = true;
  setStatus("submitting reference...", "status-running");
  try {
    const res = await fetch("/api/reference", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || res.statusText);
    }
    const task = await res.json();
    await pollTask(task.task_id);
  } catch (e) {
    setStatus("error: " + e.message, "status-error");
  } finally {
    $("ref-btn").disabled = false;
    loadRuns();
  }
}

async function pollTask(taskId) {
  const start = Date.now();
  while (true) {
    const res = await fetch(`/api/tasks/${taskId}`);
    if (!res.ok) {
      setStatus(`error polling task ${taskId}`, "status-error");
      return;
    }
    const t = await res.json();
    const elapsed = Math.floor(t.elapsed_seconds || 0);
    const lastLog = (t.log || []).slice(-1)[0] || "";
    setStatus(`${t.status} · ${elapsed}s · ${lastLog}`, "status-" + t.status);

    if (t.status === "done") {
      renderResult(t.result);
      setStatus(`done · ${elapsed}s`, "status-done");
      return;
    }
    if (t.status === "error") {
      $("result-box").innerHTML = `<pre class="log">${escapeHtml(t.error || "(no error message)")}</pre>`;
      return;
    }
    if (Date.now() - start > 300_000) {  // 5min hard cap on polling
      setStatus("polling timed out (5min); the task may still be running", "status-error");
      return;
    }
    await new Promise((r) => setTimeout(r, 1000));
  }
}

// ---------- Wire up ----------

document.addEventListener("DOMContentLoaded", () => {
  loadKeys();
  loadRuns();
  $("save-keys").addEventListener("click", saveKeys);
  $("clear-keys").addEventListener("click", clearKeys);
  $("run-btn").addEventListener("click", startBacktest);
  $("ref-btn").addEventListener("click", startReference);
  $("anthropic-key").addEventListener("input", updateKeyStatus);
  $("openai-key").addEventListener("input", updateKeyStatus);
});
