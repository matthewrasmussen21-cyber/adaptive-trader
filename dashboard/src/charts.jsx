import { useEffect, useMemo, useRef, useState } from "react";

function useWidth() {
  const ref = useRef(null);
  const [w, setW] = useState(600);
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver((e) => setW(Math.max(260, e[0].contentRect.width)));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

function niceTicks(min, max, n = 5) {
  if (!isFinite(min) || !isFinite(max)) return [];
  if (min === max) { min -= 1; max += 1; }
  const span = max - min;
  const step0 = Math.pow(10, Math.floor(Math.log10(span / n)));
  const err = (n * step0) / span;
  const step = step0 * (err <= 0.15 ? 10 : err <= 0.35 ? 5 : err <= 0.75 ? 2 : 1);
  const out = [];
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-12; v += step) out.push(+v.toFixed(10));
  return out;
}

const PAD = { l: 56, r: 64, t: 8, b: 22 };
const MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
function tickLabel(d, dates) {
  if (!d) return "";
  const spanDays = (new Date(dates[dates.length - 1]) - new Date(dates[0])) / 864e5;
  const [y, m, day] = d.split("-");
  if (spanDays < 300) return `${MON[+m - 1]} ${+day}`;
  if (spanDays < 1100) return `${MON[+m - 1]} ${y.slice(2)}`;
  return y;
}

/** Multi-series line chart over dates with crosshair tooltip and end-of-line direct labels. */
export function LineChart({ series, height = 260, fmt = (v) => v.toFixed(2), baseline = null, area = false }) {
  const [ref, w] = useWidth();
  const [hover, setHover] = useState(null);
  const dates = useMemo(() => {
    const s = new Set();
    series.forEach((x) => x.dates.forEach((d) => s.add(d)));
    return [...s].sort();
  }, [series]);
  const idx = useMemo(() => series.map((x) => { const m = new Map(); x.dates.forEach((d, i) => m.set(d, x.values[i])); return m; }), [series]);
  if (!dates.length) return <div ref={ref} className="muted" style={{ padding: 20 }}>No data yet.</div>;
  const vals = series.flatMap((s) => s.values.filter((v) => v != null && isFinite(v)));
  let lo = Math.min(...vals, baseline ?? Infinity), hi = Math.max(...vals, baseline ?? -Infinity);
  const padv = (hi - lo) * 0.06 || 1;
  lo -= padv; hi += padv;
  const ticks = niceTicks(lo, hi, 5);
  const iw = w - PAD.l - PAD.r, ih = height - PAD.t - PAD.b;
  const x = (i) => PAD.l + (dates.length === 1 ? iw / 2 : (i / (dates.length - 1)) * iw);
  const y = (v) => PAD.t + (1 - (v - lo) / (hi - lo)) * ih;
  const xt = [];
  const nX = Math.max(2, Math.floor(iw / 110));
  for (let k = 0; k < nX; k++) xt.push(Math.round((k / (nX - 1)) * (dates.length - 1)));
  const path = (s, m) => {
    let d = "", pen = false;
    dates.forEach((dt, i) => {
      const v = m.get(dt);
      if (v == null || !isFinite(v)) { pen = false; return; }
      d += `${pen ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`;
      pen = true;
    });
    return d;
  };
  const onMove = (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - r.left;
    const i = Math.round(((px - PAD.l) / iw) * (dates.length - 1));
    setHover(Math.max(0, Math.min(dates.length - 1, i)));
  };
  // end labels, nudged apart vertically
  const ends = series.map((s, k) => {
    const last = [...s.values].reverse().find((v) => v != null && isFinite(v));
    return { k, name: s.short || s.name, color: s.color, y: last == null ? null : y(last) };
  }).filter((e) => e.y != null).sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 12) ends[i].y = ends[i - 1].y + 12;
  return (
    <div ref={ref} className="chart">
      <svg width={w} height={height} onPointerMove={onMove} onPointerLeave={() => setHover(null)}
        role="img" aria-label={series.map((s) => s.name).join(", ")}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={PAD.l} x2={w - PAD.r} y1={y(t)} y2={y(t)} stroke="var(--grid)" />
            <text x={PAD.l - 6} y={y(t) + 3.5} textAnchor="end">{fmt(t)}</text>
          </g>
        ))}
        {baseline != null && <line x1={PAD.l} x2={w - PAD.r} y1={y(baseline)} y2={y(baseline)} stroke="var(--text-muted)" strokeDasharray="3 3" />}
        {xt.map((i) => <text key={i} x={x(i)} y={height - 6} textAnchor="middle">{tickLabel(dates[i], dates)}</text>)}
        {area && series.map((s, k) => {
          const p = path(s, idx[k]);
          return p ? <path key={"a" + k} d={`${p}L${x(dates.length - 1)},${y(Math.min(0, hi))}L${x(0)},${y(Math.min(0, hi))}Z`} fill={s.color} opacity={0.18} /> : null;
        })}
        {series.map((s, k) => <path key={k} d={path(s, idx[k])} fill="none" stroke={s.color} strokeWidth={s.width || 2} strokeDasharray={s.dash} strokeLinejoin="round" />)}
        {ends.map((e) => <text key={e.k} x={w - PAD.r + 6} y={e.y + 3.5} style={{ fill: "var(--text-secondary)" }}>{e.name}</text>)}
        {hover != null && (
          <g>
            <line x1={x(hover)} x2={x(hover)} y1={PAD.t} y2={PAD.t + ih} stroke="var(--text-muted)" />
            {series.map((s, k) => { const v = idx[k].get(dates[hover]); return v == null ? null :
              <circle key={k} cx={x(hover)} cy={y(v)} r={4} fill={s.color} stroke="var(--surface-1)" strokeWidth={2} />; })}
          </g>
        )}
      </svg>
      {hover != null && (
        <div className="tip" style={{ left: Math.min(x(hover) + 12, w - 190), top: 8 }}>
          <div className="d">{dates[hover]}</div>
          {series.map((s, k) => { const v = idx[k].get(dates[hover]); return v == null ? null : (
            <div className="r" key={k}><i style={{ background: s.color }} /><b>{fmt(v)}</b><span>{s.name}</span></div>); })}
        </div>
      )}
    </div>
  );
}

/** Scatter with nearest-point hover. groups: [{key, name, color, r}] drawn in order (last on top). */
export function Scatter({ points, groups, height = 300, xFmt, yFmt, xLabel, yLabel, refX, refY, onPick }) {
  const [ref, w] = useWidth();
  const [hover, setHover] = useState(null);
  const pts = points.filter((p) => isFinite(p.x) && isFinite(p.y));
  if (!pts.length) return <div ref={ref} className="muted" style={{ padding: 20 }}>No candidates yet.</div>;
  const q = (arr, p) => { const s = [...arr].sort((a, b) => a - b); return s[Math.floor(p * (s.length - 1))]; };
  const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y);
  let x0 = Math.min(q(xs, 0.01), refX ?? Infinity), x1 = Math.max(q(xs, 0.995), refX ?? -Infinity);
  let y0 = Math.min(q(ys, 0.01), refY ?? Infinity), y1 = Math.max(q(ys, 0.995), refY ?? -Infinity);
  const px_ = (x1 - x0) * 0.05 || 0.01, py_ = (y1 - y0) * 0.05 || 0.01;
  x0 -= px_; x1 += px_; y0 -= py_; y1 += py_;
  const P = { l: 52, r: 14, t: 8, b: 34 };
  const iw = w - P.l - P.r, ih = height - P.t - P.b;
  const X = (v) => P.l + ((Math.min(Math.max(v, x0), x1) - x0) / (x1 - x0)) * iw;
  const Y = (v) => P.t + (1 - (Math.min(Math.max(v, y0), y1) - y0) / (y1 - y0)) * ih;
  const onMove = (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    let best = null, bd = 24 * 24;
    for (const p of pts) { const d = (X(p.x) - mx) ** 2 + (Y(p.y) - my) ** 2; if (d < bd) { bd = d; best = p; } }
    setHover(best);
  };
  const g = Object.fromEntries(groups.map((x) => [x.key, x]));
  return (
    <div ref={ref} className="chart">
      <svg width={w} height={height} onPointerMove={onMove} onPointerLeave={() => setHover(null)}
        onClick={() => hover && onPick && onPick(hover)} role="img" aria-label="candidate scatter">
        {niceTicks(y0, y1, 5).map((t) => <g key={"y" + t}><line x1={P.l} x2={w - P.r} y1={Y(t)} y2={Y(t)} stroke="var(--grid)" /><text x={P.l - 6} y={Y(t) + 3.5} textAnchor="end">{yFmt(t)}</text></g>)}
        {niceTicks(x0, x1, Math.max(3, Math.floor(iw / 90))).map((t) => <text key={"x" + t} x={X(t)} y={P.t + ih + 14} textAnchor="middle">{xFmt(t)}</text>)}
        <text x={P.l + iw / 2} y={height - 3} textAnchor="middle">{xLabel}</text>
        <text x={12} y={P.t + ih / 2} textAnchor="middle" transform={`rotate(-90 12 ${P.t + ih / 2})`}>{yLabel}</text>
        {refX != null && <line x1={X(refX)} x2={X(refX)} y1={P.t} y2={P.t + ih} stroke="var(--text-muted)" strokeDasharray="3 3" />}
        {refY != null && <line x1={P.l} x2={w - P.r} y1={Y(refY)} y2={Y(refY)} stroke="var(--text-muted)" strokeDasharray="3 3" />}
        {groups.map((gr) => pts.filter((p) => p.group === gr.key).map((p, i) => (
          <circle key={gr.key + i} cx={X(p.x)} cy={Y(p.y)} r={gr.r || 4} fill={gr.color} fillOpacity={gr.opacity ?? 0.85}
            stroke="var(--surface-1)" strokeWidth={gr.ring ? 2 : 1} />)))}
        {hover && <circle cx={X(hover.x)} cy={Y(hover.y)} r={(g[hover.group]?.r || 4) + 3} fill="none" stroke="var(--text-primary)" strokeWidth={1.5} />}
      </svg>
      {hover && (
        <div className="tip" style={{ left: Math.min(X(hover.x) + 12, w - 280), top: Math.max(0, Y(hover.y) - 60), maxWidth: 270 }}>
          <div className="r"><b>{xFmt(hover.x)}</b><span>{xLabel}</span></div>
          <div className="r"><b>{yFmt(hover.y)}</b><span>{yLabel}</span></div>
          <div className="d" style={{ marginTop: 4, whiteSpace: "normal" }}>{hover.label}</div>
          <div className="r"><i style={{ background: g[hover.group]?.color }} /><span>{g[hover.group]?.name}</span></div>
        </div>
      )}
    </div>
  );
}

/** Horizontal bars around a baseline (e.g. lift = 1). */
export function BarList({ rows, base = 1, max, color = "var(--series-1)", fmt = (v) => v.toFixed(2) }) {
  const hi = max ?? Math.max(base * 1.5, ...rows.map((r) => r.value));
  const lo = Math.min(0, ...rows.map((r) => r.value));
  const pos = (v) => ((v - lo) / (hi - lo)) * 100;
  return (
    <div>
      {rows.map((r) => {
        const a = pos(Math.min(r.value, base)), b = pos(Math.max(r.value, base));
        return (
          <div className="barrow" key={r.name} title={r.title || ""}>
            <span className="mono sec" style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{r.name}</span>
            <div className="track">
              <div className="fill" style={{ left: `${a}%`, width: `${Math.max(b - a, 0.5)}%`, background: r.value >= base ? color : "var(--text-muted)", borderRadius: r.value >= base ? "0 3px 3px 0" : "3px 0 0 3px" }} />
              <div className="base" style={{ left: `${pos(base)}%` }} />
            </div>
            <span className="mono" style={{ textAlign: "right" }}>{fmt(r.value)}</span>
          </div>
        );
      })}
    </div>
  );
}
