import { useMemo, useState } from "react";
import data from "@data";
import { BarList, LineChart, Scatter } from "./charts.jsx";

const C = ["var(--series-1)", "var(--series-2)", "var(--series-3)", "var(--series-4)"];
const pct = (v, d = 1) => (v == null || !isFinite(v) ? "—" : `${(v * 100).toFixed(d)}%`);
const spct = (v, d = 1) => (v == null || !isFinite(v) ? "—" : `${v >= 0 ? "+" : ""}${(v * 100).toFixed(d)}%`);
const usd = (v) => (v == null ? "—" : v.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 }));
const num = (v, d = 2) => (v == null || !isFinite(v) ? "—" : Number(v).toFixed(d));
const BOOK_NAMES = { live: "Live paper book", "bench-SPY": "SPY buy & hold", "bench-QQQ": "QQQ buy & hold", "bench-EW": "Equal-weight universe" };

function Pill({ ok, children }) {
  if (ok == null) return <span className="pill na">{children ?? "n/a"}</span>;
  return <span className={`pill ${ok ? "pass" : "fail"}`}><span className="ic">{ok ? "✓" : "✕"}</span>{children ?? (ok ? "pass" : "fail")}</span>;
}

function Kpi({ label, value, sub }) {
  return <div className="kpi"><div className="label">{label}</div><div className="value">{value}</div>{sub && <div className="sub">{sub}</div>}</div>;
}

function Card({ title, hint, right, children, style }) {
  return (
    <section className="card" style={style}>
      <div className="card-head"><div><h2>{title}</h2>{hint && <p className="hint">{hint}</p>}</div>{right}</div>
      {children}
    </section>
  );
}

function curveStats(c) {
  if (!c || !c.equity.length) return null;
  const e = c.equity, first = e[0], last = e[e.length - 1];
  let peak = -Infinity, mdd = 0;
  e.forEach((v) => { peak = Math.max(peak, v); mdd = Math.max(mdd, 1 - v / peak); });
  return { ret: last / data.config.capital - 1, mdd, dd: c.drawdown?.[c.drawdown.length - 1] ?? 0, days: e.length, last, first };
}

// ------------------------------------------------------------------ Overview
function Overview() {
  const cv = data.curves;
  const live = curveStats(cv.live), spy = curveStats(cv["bench-SPY"]);
  const [range, setRange] = useState("all");
  const cut = (c) => {
    if (!c || range === "all") return c;
    const n = { "1m": 21, "3m": 63, "6m": 126 }[range];
    return { dates: c.dates.slice(-n), equity: c.equity.slice(-n), drawdown: c.drawdown.slice(-n) };
  };
  const ids = ["live", "bench-SPY", "bench-QQQ", "bench-EW"].filter((k) => cv[k]);
  const eqSeries = ids.map((k, i) => ({ name: BOOK_NAMES[k], short: k === "live" ? "Live" : k.replace("bench-", ""), color: C[i], dates: cut(cv[k]).dates, values: cut(cv[k]).equity }));
  const ddSeries = ["live", "bench-SPY"].filter((k) => cv[k]).map((k, i) => ({ name: BOOK_NAMES[k], short: k === "live" ? "Live" : "SPY", color: C[i], dates: cut(cv[k]).dates, values: cut(cv[k]).drawdown }));
  const L = data.live;
  const expo = L.equity ? 1 - (L.cash ?? 0) / L.equity : 0;
  const nCand = data.candidates.rows.length;
  return (
    <>
      <div className="kpis">
        <Kpi label="Live equity" value={usd(L.equity)} sub={live ? `${spct(live.ret)} since ${L.start_date}` : "awaiting first live day"} />
        <Kpi label="vs SPY" value={live && spy ? spct(live.ret - spy.ret) : "—"} sub={spy ? `SPY ${spct(spy.ret)}` : ""} />
        <Kpi label="Drawdown now" value={live ? pct(-live.dd) : "—"} sub={`breaker at ${pct(data.config.max_drawdown, 0)}`} />
        <Kpi label="Max drawdown" value={live ? pct(live.mdd) : "—"} sub={live ? `${live.days} trading days` : ""} />
        <Kpi label="Positions" value={L.positions.length} sub={`${pct(expo, 0)} invested`} />
        <Kpi label="Champion" value={data.champion.model_id} sub={data.champion.since ? `since ${data.champion.since}` : ""} />
        <Kpi label="Strategies tested" value={nCand.toLocaleString()} sub={`${data.challengers.length} in shadow`} />
      </div>
      <div className="grid g2">
        <Card title="Equity — live paper book vs benchmarks" hint="All books start with $100,000 on the first live day. Fills at the next open with modeled costs."
          right={<div className="seg">{["1m", "3m", "6m", "all"].map((r) => <button key={r} className={range === r ? "on" : ""} onClick={() => setRange(r)}>{r}</button>)}</div>}>
          <div className="legend">{eqSeries.map((s) => <span key={s.name}><i style={{ background: s.color }} />{s.name}</span>)}</div>
          <LineChart series={eqSeries} height={280} fmt={(v) => `$${(v / 1000).toFixed(1)}k`} baseline={data.config.capital} />
          <div className="legend" style={{ marginTop: 10 }}>{ddSeries.map((s) => <span key={s.name}><i style={{ background: s.color }} />{s.name} drawdown</span>)}</div>
          <LineChart series={ddSeries} height={130} fmt={(v) => pct(v, 0)} baseline={-data.config.max_drawdown} />
        </Card>
        <Card title="Current strategy" hint={data.live.label}>
          <dl className="kv" style={{ marginBottom: 10 }}>
            <dt>Model</dt><dd>{data.champion.model_id}</dd>
            <dt>Status</dt><dd>{L.halted ? "HALTED — cash" : "trading"}</dd>
            <dt>Last mark</dt><dd>{L.last_date ?? "—"}</dd>
            <dt>Cash</dt><dd>{usd(L.cash)}</dd>
          </dl>
          <h2 style={{ marginTop: 6 }}>Positions</h2>
          <div className="scroll" style={{ maxHeight: 300 }}>
            <table className="t"><thead><tr><th>Ticker</th><th className="num">Weight</th><th className="num">Value</th><th className="num">Return</th><th>Since</th></tr></thead>
              <tbody>{L.positions.map((p) => <tr key={p.ticker}><td className="mono">{p.ticker}</td><td className="num">{pct(p.weight)}</td><td className="num">{usd(p.value)}</td><td className="num">{spct(p.gross_return)}</td><td className="mono muted">{p.entry_date}</td></tr>)}
                {!L.positions.length && <tr><td colSpan={5} className="muted">No positions.</td></tr>}</tbody></table>
          </div>
          {L.pending && Object.keys(L.pending).length > 0 && <>
            <h2 style={{ marginTop: 12 }}>Orders for next open</h2>
            <div className="scroll" style={{ maxHeight: 180 }}>
              <table className="t"><thead><tr><th>Ticker</th><th className="num">Target weight</th></tr></thead>
                <tbody>{Object.entries(L.pending).sort((a, b) => b[1] - a[1]).map(([t, w]) => <tr key={t}><td className="mono">{t}</td><td className="num">{pct(w)}</td></tr>)}</tbody></table>
            </div></>}
        </Card>
      </div>
      <div className="grid g11" style={{ marginTop: 12 }}>
        <Card title="Recent fills" hint="Internal paper broker: next-open execution, slippage + modeled half-spread.">
          <Fills rows={data.fills.slice(-60).reverse()} />
        </Card>
        <Card title="System timeline" hint="Promotions, halts, research runs, reviews — from the hash-chained ledger.">
          <Timeline rows={data.timeline.slice(-40).reverse()} />
        </Card>
      </div>
    </>
  );
}

function Fills({ rows }) {
  return (
    <div className="scroll"><table className="t"><thead><tr><th>Date</th><th>Ticker</th><th>Side</th><th className="num">Dollars</th><th className="num">Price</th><th className="num">Cost</th></tr></thead>
      <tbody>{rows.map((f, i) => <tr key={i}><td className="mono muted">{f.date}</td><td className="mono">{f.ticker}</td><td>{f.side}</td><td className="num">{usd(Math.abs(f.dollars))}</td><td className="num">{f.price ? f.price.toFixed(2) : "—"}</td><td className="num">{num(f.cost)}</td></tr>)}
        {!rows.length && <tr><td colSpan={6} className="muted">No fills yet.</td></tr>}</tbody></table></div>
  );
}

function describeEvent(e) {
  const d = e.data;
  switch (e.type) {
    case "RESEARCH_RUN": return `${d.tested} tested · ${d.cv_passed} passed walk-forward · ${d.new_challengers.length} new challenger(s) · ${d.total_trials} trials total`;
    case "PROMOTION": return `${d.model_id} promoted (replaces ${d.replaces})`;
    case "CHALLENGER_ADDED": return `${d.label} · holdout CAGR ${pct(d.holdout_cagr)}`;
    case "CHALLENGER_RETIRED": return `${d.spec_hash} · ${d.reason}`;
    case "HALT": return `${d.reason} · equity ${usd(d.equity)}`;
    case "RESUME": return d.note;
    case "REVIEW": return d.findings?.length ? d.findings.join(" ") : "no anomalies";
    case "SYSTEM_INIT": return `holdout ${d.holdout.join(" → ")} · ${d.priors}`;
    case "HOLDOUT_ACCESS": return `spec ${d.spec_hash}`;
    case "CONFIG_CHANGE": return `config ${d.old} → ${d.new}`;
    default: return JSON.stringify(d).slice(0, 140);
  }
}

function Timeline({ rows }) {
  return (
    <div className="scroll"><ul className="timeline">{rows.map((e) => (
      <li key={e.seq}><span className="mono muted">{e.ts.slice(0, 10)}</span><span className="mono sec">{e.type}</span><span>{describeEvent(e)}</span></li>))}
      {!rows.length && <li className="muted">Nothing yet.</li>}</ul></div>
  );
}

// ------------------------------------------------------------------ Champion & challengers
function MetricsTable({ cols }) {
  const keys = [["cagr", "CAGR", pct], ["sharpe", "Sharpe", num], ["sortino", "Sortino", num], ["max_drawdown", "Max drawdown", pct],
    ["calmar", "Calmar", num], ["trades", "Trades", (v) => v ?? "—"], ["win_rate", "Win rate", pct], ["avg_win", "Avg win", pct],
    ["avg_loss", "Avg loss", pct], ["avg_hold_days", "Avg hold (days)", (v) => num(v, 0)], ["turnover_annual", "Turnover / yr", (v) => num(v, 1) + "×"],
    ["exposure", "Exposure", pct], ["alpha", "Alpha vs SPY", pct], ["beta", "Beta", num], ["cost_annual", "Costs / yr", pct]];
  return (
    <div className="scroll-x"><table className="t"><thead><tr><th>Metric</th>{cols.map((c) => <th key={c.name} className="num">{c.name}</th>)}</tr></thead>
      <tbody>{keys.map(([k, n, f]) => <tr key={k}><td className="sec">{n}</td>{cols.map((c) => <td key={c.name} className="num">{c.m ? f(c.m[k]) : "—"}</td>)}</tr>)}</tbody></table></div>
  );
}

function Champion() {
  const ch = data.champion, v = ch.validation;
  const cv = data.curves;
  const shadowIds = Object.keys(cv).filter((k) => k.startsWith("shadow-"));
  const shadowSeries = [["live", "Live (champion)"], ...shadowIds.map((k) => [k, data.challengers.find((c) => "shadow-" + c.hash === k)?.label || k])]
    .slice(0, 4).map(([k, n], i) => ({ name: n, short: k === "live" ? "Live" : k.slice(7, 13), color: C[i], dates: cv[k]?.dates || [], values: cv[k]?.equity || [] }));
  return (
    <>
      <div className="grid g2">
        <Card title={`Champion ${ch.model_id}`} hint={ch.label}>
          {v ? <p className="note">{v.hypothesis}</p> : <p className="note">The baseline champion: buy and hold SPY. Every learned strategy has to beat it through the full gate before it can trade.</p>}
          <MetricsTable cols={[{ name: "Walk-forward OOS", m: ch.cv_metrics || v?.cv?.metrics }, { name: "Locked holdout", m: ch.holdout_metrics },
            ...(v ? [{ name: "Predecessor OOS", m: v.champion_at_test.cv }] : [])]} />
          {ch.holdout_curve && <>
            <h2 style={{ marginTop: 14 }}>Holdout equity (single touch, growth of $1)</h2>
            <LineChart series={[{ name: "Champion on holdout", short: "Champ", color: C[0], dates: ch.holdout_curve.dates, values: ch.holdout_curve.equity }]} height={170} fmt={(x) => x.toFixed(2)} baseline={1} />
          </>}
        </Card>
        <Card title="Promotion gate" hint="All five walk-forward checks, then one holdout touch, then live shadow evidence.">
          {v ? <>
            {Object.entries(v.gate_cv.checks).map(([k, ok]) => <div key={k} style={{ display: "flex", justifyContent: "space-between", padding: "4px 0", borderBottom: "1px solid var(--grid)" }}><span className="sec">{k.replaceAll("_", " ")}</span><Pill ok={ok} /></div>)}
            <dl className="kv" style={{ marginTop: 10 }}>
              <dt>Deflated Sharpe</dt><dd>{num(v.gate_cv.dsr.dsr, 3)} over {v.gate_cv.dsr.n_trials} trials</dd>
              <dt>Holdout</dt><dd>{v.gate_holdout ? (v.gate_holdout.passed ? "beat predecessor" : "failed") : "—"}</dd>
            </dl>
            <h2 style={{ marginTop: 12 }}>Excess return by regime</h2>
            <table className="t"><thead><tr><th>Regime</th><th className="num">Days</th><th className="num">Ann. excess</th></tr></thead>
              <tbody>{Object.entries(v.gate_cv.regimes).map(([g, r]) => <tr key={g}><td>{g.replace("_", " ")}</td><td className="num">{r.days}</td><td className="num">{spct(r.ann_excess)}</td></tr>)}</tbody></table>
            <h2 style={{ marginTop: 12 }}>Walk-forward folds</h2>
            <table className="t"><thead><tr><th>Fold</th><th className="num">CAGR</th><th className="num">Max DD</th></tr></thead>
              <tbody>{v.cv.folds.map((f) => <tr key={f.start}><td className="mono muted">{f.start} → {f.end}</td><td className="num">{pct(f.cagr)}</td><td className="num">{pct(f.max_drawdown)}</td></tr>)}</tbody></table>
          </> : <p className="note">No learned champion yet.</p>}
          <h2 style={{ marginTop: 12 }}>Spec</h2>
          <div className="code">{JSON.stringify(ch.spec, null, 1)}</div>
        </Card>
      </div>
      <div className="grid g11" style={{ marginTop: 12 }}>
        <Card title="Challengers in shadow" hint={`Need ≥ ${data.config.gate.min_shadow_days} live days with z ≥ ${data.config.gate.shadow_z_floor} vs backtest expectation.`}>
          <div className="scroll-x"><table className="t"><thead><tr><th>Strategy</th><th className="num">OOS CAGR</th><th className="num">Holdout CAGR</th><th className="num">DSR</th><th className="num">Shadow days</th><th className="num">z</th><th>Status</th></tr></thead>
            <tbody>{data.challengers.map((c) => <tr key={c.hash}><td><div>{c.label}</div><div className="mono muted">{c.hash}</div></td><td className="num">{pct(c.cv.cagr)}</td><td className="num">{pct(c.holdout.cagr)}</td><td className="num">{num(c.dsr, 3)}</td><td className="num">{c.shadow?.days ?? 0}</td><td className="num">{num(c.shadow?.z)}</td><td>{c.shadow ? <Pill ok={c.shadow.ok}>{c.shadow.ready ? "ready" : "accumulating"}</Pill> : <Pill>new</Pill>}</td></tr>)}
              {!data.challengers.length && <tr><td colSpan={7} className="muted">No challengers. Candidates that clear the walk-forward gate and their single holdout touch appear here.</td></tr>}</tbody></table></div>
          {shadowSeries.length > 1 && <>
            <div className="legend" style={{ marginTop: 12 }}>{shadowSeries.map((s) => <span key={s.name}><i style={{ background: s.color }} />{s.name}</span>)}</div>
            <LineChart series={shadowSeries} height={200} fmt={(x) => `$${(x / 1000).toFixed(1)}k`} />
          </>}
        </Card>
        <Card title="Model versions" hint="Every promotion and retirement, with the commit and validation report that justified it.">
          <div className="scroll"><table className="t"><thead><tr><th>Event</th><th>Model</th><th>Strategy</th><th>Date</th><th>Commit</th></tr></thead>
            <tbody>{[...data.models].reverse().map((m, i) => <tr key={i}><td className="mono sec">{m.type.replace("MODEL_", "")}</td><td className="mono">{m.model_id}</td><td>{m.label || m.reason || ""}</td><td className="mono muted">{m.promoted_at || m.date || m.ts.slice(0, 10)}</td><td className="mono muted">{(m.git_commit || "").slice(0, 7)}</td></tr>)}</tbody></table></div>
        </Card>
      </div>
    </>
  );
}

// ------------------------------------------------------------------ Research
function Research() {
  const cols = data.candidates.columns;
  const rows = useMemo(() => data.candidates.rows.map((r) => Object.fromEntries(cols.map((c, i) => [c, r[i]]))), []);
  const [q, setQ] = useState("");
  const [status, setStatus] = useState("all");
  const [origin, setOrigin] = useState("all");
  const [sort, setSort] = useState(["seq", -1]);
  const [limit, setLimit] = useState(200);
  const champCagr = data.champion.cv_metrics?.cagr;
  const filtered = rows.filter((r) => (!q || (r.label || "").toLowerCase().includes(q.toLowerCase()) || r.hash.includes(q))
    && (origin === "all" || r.origin === origin)
    && (status === "all" || (status === "cv" && r.cv_pass) || (status === "holdout" && r.holdout_pass) || (status === "fail" && !r.cv_pass) || (status === "error" && r.error)));
  const sorted = [...filtered].sort((a, b) => { const [k, d] = sort; const x = a[k] ?? -Infinity, y = b[k] ?? -Infinity; return x < y ? -d : x > y ? d : 0; });
  const fails = {};
  rows.forEach((r) => (r.failed_checks || []).forEach((k) => (fails[k] = (fails[k] || 0) + 1)));
  const valid = rows.filter((r) => r.cv_cagr != null);
  const pts = valid.map((r) => ({ x: r.cv_maxdd, y: r.cv_cagr, label: r.label, group: r.holdout_pass ? "h" : r.cv_pass ? "c" : "f" }));
  const fstats = Object.entries(data.feature_stats || {}).map(([k, v]) => ({ name: k, value: v.lift, title: `${v.used} uses, ${v.in_top_decile} in top decile · ${v.status}` })).sort((a, b) => b.value - a.value);
  const th = (k, n, isNum) => <th className={`sort ${isNum ? "num" : ""}`} onClick={() => setSort([k, sort[0] === k ? -sort[1] : -1])}>{n}{sort[0] === k ? (sort[1] < 0 ? " ↓" : " ↑") : ""}</th>;
  const holdoutTouches = rows.filter((r) => r.holdout_pass != null).length;
  return (
    <>
      <div className="kpis">
        <Kpi label="Strategies tested" value={rows.length.toLocaleString()} sub={`${data.runs.length} research runs`} />
        <Kpi label="Passed walk-forward" value={rows.filter((r) => r.cv_pass).length} sub={pct(rows.filter((r) => r.cv_pass).length / Math.max(rows.length, 1), 2) + " of tested"} />
        <Kpi label="Holdout touches" value={holdoutTouches} sub={`${rows.filter((r) => r.holdout_pass).length} passed`} />
        <Kpi label="Best OOS CAGR" value={pct(Math.max(...valid.filter((r) => r.cv_maxdd < data.config.max_drawdown).map((r) => r.cv_cagr), -Infinity))} sub={`champion ${pct(champCagr)}`} />
        <Kpi label="Errors" value={rows.filter((r) => r.error).length} sub="candidates that failed to run" />
      </div>
      <div className="grid g2">
        <Card title="Every candidate: out-of-sample return vs drawdown" hint="Walk-forward folds 2..K. Dashed lines: champion's OOS CAGR and the 35% drawdown limit. Hover a point for its spec.">
          <div className="legend"><span><i style={{ background: "var(--text-muted)" }} />failed gate</span><span><i style={{ background: C[0] }} />passed walk-forward</span><span><i style={{ background: C[1] }} />passed holdout</span></div>
          <Scatter points={pts} height={340} xFmt={(v) => pct(v, 0)} yFmt={(v) => pct(v, 0)} xLabel="max drawdown" yLabel="OOS CAGR"
            refX={data.config.max_drawdown} refY={champCagr}
            groups={[{ key: "f", name: "failed gate", color: "var(--text-muted)", r: 3, opacity: 0.45 }, { key: "c", name: "passed walk-forward", color: C[0], r: 5, ring: true }, { key: "h", name: "passed holdout", color: C[1], r: 6, ring: true }]} />
        </Card>
        <Card title="Why candidates fail" hint="Count of candidates failing each check (a candidate can fail several).">
          <BarList rows={Object.entries(fails).map(([k, v]) => ({ name: k.replaceAll("_", " "), value: v })).sort((a, b) => b.value - a.value)} base={0} fmt={(v) => v.toFixed(0)} />
          <h2 style={{ marginTop: 16 }}>Learned feature evidence</h2>
          <p className="hint">Lift = share of top-decile candidates using a feature ÷ share of all candidates. Drives sampling weights; 1.0 = no evidence either way.</p>
          {fstats.length ? <div className="scroll" style={{ maxHeight: 330 }}><BarList rows={fstats} base={1} /></div> : <p className="muted">Appears after 30 valid candidates.</p>}
        </Card>
      </div>
      <Card title="Candidate log" hint="Every strategy ever tested, including failures — the honest multiple-testing count used by the deflated Sharpe test." style={{ marginTop: 12 }}>
        <div className="filters">
          <input className="search" placeholder="Search features or hash" value={q} onChange={(e) => setQ(e.target.value)} />
          <select className="sel" value={status} onChange={(e) => setStatus(e.target.value)}>
            <option value="all">All results</option><option value="cv">Passed walk-forward</option><option value="holdout">Passed holdout</option><option value="fail">Failed gate</option><option value="error">Errors</option>
          </select>
          <select className="sel" value={origin} onChange={(e) => setOrigin(e.target.value)}>
            <option value="all">All origins</option><option value="random">Random</option><option value="mutation">Mutation</option><option value="llm">LLM</option>
          </select>
          <span className="muted">{filtered.length.toLocaleString()} shown</span>
        </div>
        <div className="scroll" style={{ maxHeight: 520 }}><table className="t"><thead><tr>
          {th("seq", "#", true)}<th>Strategy</th>{th("origin", "Origin")}{th("cv_cagr", "OOS CAGR", true)}{th("cv_sharpe", "Sharpe", true)}{th("cv_maxdd", "Max DD", true)}{th("trades", "Trades", true)}{th("avg_hold", "Hold d", true)}{th("dsr", "DSR", true)}{th("regime_wins", "Regimes", true)}<th>Gate</th><th>Holdout</th>
        </tr></thead><tbody>{sorted.slice(0, limit).map((r) => <tr key={r.seq}>
          <td className="num muted">{r.seq}</td><td><div>{r.label}</div>{r.error ? <div className="muted">{r.error}</div> : r.failed_checks?.length ? <div className="muted">fails: {r.failed_checks.join(", ")}</div> : null}</td>
          <td className="muted">{r.origin}</td><td className="num">{pct(r.cv_cagr)}</td><td className="num">{num(r.cv_sharpe)}</td><td className="num">{pct(r.cv_maxdd)}</td><td className="num">{r.trades ?? "—"}</td><td className="num">{num(r.avg_hold, 0)}</td><td className="num">{num(r.dsr, 2)}</td><td className="num">{r.regime_wins ?? "—"}/4</td>
          <td><Pill ok={r.error ? null : r.cv_pass} /></td><td>{r.holdout_pass == null ? <span className="muted">—</span> : <Pill ok={r.holdout_pass}>{pct(r.holdout_cagr)}</Pill>}</td></tr>)}</tbody></table></div>
        {sorted.length > limit && <button className="seg" style={{ marginTop: 8, padding: "5px 10px", color: "var(--text-secondary)", background: "none" }} onClick={() => setLimit(limit + 500)}>Show 500 more</button>}
      </Card>
      <div className="grid g11" style={{ marginTop: 12 }}>
        <Card title="Research runs">
          <div className="scroll"><table className="t"><thead><tr><th>Run</th><th className="num">Tested</th><th className="num">Passed WF</th><th className="num">Holdout</th><th className="num">New chall.</th><th className="num">Trials total</th><th className="num">Secs</th></tr></thead>
            <tbody>{[...data.runs].reverse().map((r) => <tr key={r.run_id}><td className="mono">{r.run_id}</td><td className="num">{r.tested}</td><td className="num">{r.cv_passed}</td><td className="num">{r.holdout_touched}</td><td className="num">{r.new_challengers.length}</td><td className="num">{r.total_trials}</td><td className="num">{r.seconds}</td></tr>)}</tbody></table></div>
        </Card>
        <Card title="Monthly reviews" hint="Live results vs each strategy's backtest distribution (block bootstrap).">
          {data.reviews.length ? [...data.reviews].reverse().map((r) => <div key={r.date} style={{ marginBottom: 12 }}>
            <div className="mono sec">{r.date} · regime {r.regime_now}</div>
            <ul className="note">{(r.findings.length ? r.findings : ["No anomalies."]).map((f, i) => <li key={i}>{f}</li>)}</ul>
            {r.narrative && <p className="note">{r.narrative}</p>}
          </div>) : <p className="muted">First review runs at month end.</p>}
        </Card>
      </div>
    </>
  );
}

// ------------------------------------------------------------------ Ledger & data
function Ledger() {
  const g = data.data_gaps;
  return (
    <>
      <div className="grid g11">
        <Card title="Ledger integrity" hint="Append-only JSONL with a SHA-256 hash chain; any edited or deleted line breaks verification.">
          {Object.entries(data.verification).map(([k, v]) => <div key={k} style={{ display: "flex", justifyContent: "space-between", padding: "5px 0", borderBottom: "1px solid var(--grid)" }}><span className="mono">{k}.jsonl</span><span><span className="muted mono" style={{ marginRight: 8 }}>{v.detail}</span><Pill ok={v.ok}>{v.ok ? "verified" : "BROKEN"}</Pill></span></div>)}
          <h2 style={{ marginTop: 14 }}>Rules in force</h2>
          <dl className="kv">
            <dt>Objective</dt><dd>max OOS CAGR, DD &lt; {pct(data.config.max_drawdown, 0)}</dd>
            <dt>Capital</dt><dd>{usd(data.config.capital)} long-only, no leverage</dd>
            <dt>Holdout</dt><dd>{data.config.holdout.join(" → ")}</dd>
            <dt>Universe</dt><dd>price &gt; ${data.config.universe.min_price}, ADV &gt; ${(data.config.universe.min_dollar_volume / 1e6).toFixed(0)}M</dd>
            <dt>Costs</dt><dd>{data.config.costs.slippage_bps_per_side} bps slip + modeled half-spread</dd>
            <dt>Gate</dt><dd>≥{data.config.gate.min_oos_trades} trades, DSR ≥ {data.config.gate.dsr_threshold}, ≥{data.config.gate.min_regimes}/4 regimes, ≥{data.config.gate.min_shadow_days}d shadow</dd>
          </dl>
        </Card>
        <Card title="Data limitations" hint="What free data cannot do, stated up front.">
          <ul className="note">{(g?.known_limitations || []).map((x, i) => <li key={i}>{x}</li>)}</ul>
          {g && <dl className="kv" style={{ marginTop: 8 }}>
            {g.prices_missing_count != null && <><dt>Tickers unavailable</dt><dd>{g.prices_missing_count}</dd></>}
            {g.fundamentals_companies != null && <><dt>Fundamentals coverage</dt><dd>{g.fundamentals_companies} / {g.fundamentals_requested} companies</dd></>}
            <dt>Updated</dt><dd>{g.updated_at?.slice(0, 16)}</dd>
          </dl>}
        </Card>
      </div>
      <div className="grid g11" style={{ marginTop: 12 }}>
        <Card title="All live fills"><Fills rows={[...data.fills].reverse()} /></Card>
        <Card title="Event log"><Timeline rows={[...data.timeline].reverse()} /></Card>
      </div>
    </>
  );
}

export default function App() {
  const [tab, setTab] = useState("overview");
  const s = data.system;
  const verified = Object.values(data.verification).every((v) => v.ok);
  const synthetic = data.source === "synthetic";
  return (
    <div className="app">
      <header className="top">
        <h1>ADAPTIVE TRADER</h1>
        <span className="chip"><span className="dot" style={{ background: s.halted ? "var(--status-critical)" : "var(--status-good)" }} />{s.halted ? `Halted ${s.halted_at}` : "Trading"}</span>
        <span className="chip">Champion {data.champion.model_id}</span>
        <span className="chip"><span className="ic" style={{ color: verified ? "var(--status-good)" : "var(--status-critical)" }}>{verified ? "✓" : "✕"}</span>Ledger {verified ? "verified" : "broken"}</span>
        <span className="spacer" />
        <span className="muted mono">data through {data.live.last_date || "—"} · exported {data.generated_at.slice(0, 16).replace("T", " ")} UTC</span>
      </header>
      {synthetic && <div className="banner" style={{ marginTop: 12 }}><b>Synthetic market.</b> This build was generated from the offline test market, not real prices.</div>}
      {!data.curves.live && <div className="banner" style={{ marginTop: 12 }}><b>No live days yet.</b> The paper record starts on {data.config.init_date}; research and holdout results are already shown.</div>}
      <nav className="tabs">{[["overview", "Overview"], ["champion", "Champion & challengers"], ["research", "Research"], ["ledger", "Ledger & data"]].map(([k, n]) =>
        <button key={k} className={tab === k ? "on" : ""} onClick={() => setTab(k)}>{n}</button>)}</nav>
      {tab === "overview" && <Overview />}
      {tab === "champion" && <Champion />}
      {tab === "research" && <Research />}
      {tab === "ledger" && <Ledger />}
    </div>
  );
}
