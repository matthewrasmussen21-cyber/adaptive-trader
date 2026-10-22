# Adaptive Trader

An autonomous, self-improving US equity research system that paper trades and keeps an auditable track record.

- **Objective:** maximize out-of-sample CAGR, subject to a hard 35% max drawdown. Long-only, no leverage, no shorting, $100,000 paper capital. Holding period is learned (2–252 trading days).
- **No human priors:** no named methodology, chart pattern, sector or ticker preference is encoded. The system starts from 43 generic cross-sectional measurements and 14 market-context series and discovers what works. A test (`tests/test_system.py::test_no_named_methodologies_in_code`) fails the build if one sneaks in.
- **Baseline champion:** buy-and-hold SPY. Every learned strategy has to beat it through the full gate before it can trade.

## How it learns

```
            ┌────────────── Research agent (weekly + monthly) ──────────────────┐
 random ──┐ │                                                                   │
 mutation ├─► candidate spec ─► walk-forward CV ─► gate ─► locked holdout ─► shadow challenger
 LLM*  ───┘ │  (logged, even      (folds 2..K,     5 checks   (one touch per      (≥ 42 live days,
            │   if it fails)       purged)                     spec, logged)       z ≥ −2 vs backtest)
            └───────────────────────────────────────────────────────────────────┘
                                                                     │ weekly
                                                                     ▼
                     live paper book ◄──── promotion (new model version, commit, report)
                           │ daily: fill at next open, mark at close, decide
                           ▼
                 append-only hash-chained ledger  ──►  Review agent (monthly): live vs backtest
```
*LLM proposals are optional (set `ANTHROPIC_API_KEY`); the loop runs without them.

**Promotion gate** (all required; `config/system.toml`):

| Stage | Check |
|---|---|
| Walk-forward | beats the champion's out-of-sample CAGR (folds 2..5 of the research period) |
| | max drawdown < 35% and the breaker never tripped |
| | ≥ 100 out-of-sample trades |
| | deflated Sharpe ≥ 0.95 on returns *in excess of the champion*, deflated by the total number of strategies ever tested |
| | beats the champion in ≥ 3 of 4 market regimes (SPY trend × volatility) |
| Holdout | the 2 years before init, never seen by research; each spec may touch it once (logged); must beat the champion's holdout CAGR within the drawdown limit. Near-duplicates (CV return correlation > 0.9 with a spec that already touched it) are refused a look |
| Shadow | ≥ 42 live paper days, live mean return not more than 2 standard errors below backtest expectation, no drawdown breach |

**Feature learning:** sampling weights over features are updated from the candidate log — features over-represented among the top decile of out-of-sample results get sampled more; features never there decay toward a floor (never zero). Constructed features (`diff`, `ratio`, `prod`, `neg`) are documented in `state/registry/constructed_features.jsonl` when first created.

**What the review agent can't do:** edit the live champion. Its only outputs are a report and a note to the next research run. All changes go through the gate. No rule is changed to fit recent losses.

**Drawdown breaker:** if the live book falls 35% from its peak it goes to cash, the champion is demoted to the SPY reference, and trading resumes only when a candidate passes the full gate including a fresh shadow period.

## Timing and costs (no lookahead)

- Decide at the close of day *t* using data through *t*; fundamentals and FRED macro are lagged one more day.
- Execute at the open of *t+1*: 10 bps slippage per side plus a modeled half-spread of `1 + 50/sqrt(ADV $M)` bps (capped at 25).
- Positions are tracked as dollar values updated with total-return ratios, so dividends and split back-adjustments never distort the book.
- `tests/test_leakage.py` recomputes every feature on truncated history and fails if any value at date *t* changes when data after *t* is removed.

## Track record

- `state/ledger/events.jsonl` — every order, fill, daily mark, promotion, halt, holdout touch and review. Append-only JSONL with a SHA-256 hash chain; editing or deleting any past line breaks `python -m atl verify`, which every workflow runs first. Git history is a second audit trail.
- `state/registry/candidates.jsonl` — every strategy ever tested, including failures (the honest multiple-testing count).
- `state/registry/models.jsonl` — model versions: ID, spec hash, git commit, promotion date, validation report.
- `state/reports/validation/<hash>.md` — the report that justified each holdout touch.
- Benchmarks marked alongside: SPY, QQQ, equal-weight universe.

## Dashboard

Every workflow rebuilds a self-contained HTML dashboard and attaches it to the **`dashboard` release** (`dashboard.html`) and to the run as an artifact. Download it and open it in a browser — dark, data-dense, four tabs: Overview, Champion & challengers, Research (every candidate), Ledger & data.

Local build: `python -m atl export && cd dashboard && npm ci && npx vite build` → `dashboard/dist/index.html`.

## Schedules (GitHub Actions)

| Workflow | When (UTC) | Does |
|---|---|---|
| `bootstrap` | once (push to `.github/bootstrap.txt` or manual) | full backfill, init, first research run, first marks |
| `daily` | 22:47 Mon–Fri | prices + macro, features, fills/marks/orders for every book |
| `weekly` | Sat 14:23 | SEC fundamentals refresh, shadow evaluation, promotions, 300-candidate research run |
| `monthly` | 1st, 15:17 | listings refresh, review agent, research run |
| `research` | manual | extra research run |
| `CI` | on push | test suite on a synthetic market with a planted signal |

Market data (~hundreds of MB) lives in the `data-store` release, not in git.

## Configuration and secrets

All optional:

| Name | Kind | Effect |
|---|---|---|
| `SEC_USER_AGENT` | repository variable | overrides the SEC contact string set in `config/system.toml` |
| `ANTHROPIC_API_KEY` | secret | enables LLM-proposed candidates and the review narrative |
| `ALPACA_API_KEY_ID`, `ALPACA_API_SECRET_KEY` | secrets | also mirrors the live book's orders to an Alpaca paper account as market-on-open orders. The internal broker remains the system of record |

Changing `config/system.toml` is logged as a `CONFIG_CHANGE` event on the next run.

## Data sources and known limitations

Free sources only: nasdaqtrader symbol directory, SEC EDGAR (tickers, SIC codes, XBRL company facts), Yahoo Finance via `yfinance` (Stooq fallback), FRED.

- **Survivorship bias.** Free listings contain only current securities, so the research and holdout periods exclude companies that were delisted before init. Backtest returns are biased upward. From init onward delisted names are retained, so the live record is clean.
- **Spreads are modeled**, not observed — no free historical quote data.
- **Fundamentals** are first-reported values (no restatements), available the day after the filing date; refreshed weekly in live, so live fundamentals can lag a filing by up to a week.
- Yahoo Finance is unofficial and can rate-limit; tickers it can't serve are listed in `state/reports/data_gaps.json`.
- Not available free, not used: point-in-time index membership, short interest, estimates, options, intraday data.
- Cash earns 0% in the paper book.

## Commands

```
python -m atl data {listings|prices|fundamentals|macro|all}
python -m atl features            # rebuild the feature store from raw data
python -m atl bootstrap           # one-time init
python -m atl research -n 300     # research run
python -m atl daily | weekly | review | export | verify
python -m atl synthetic           # offline synthetic market for development
python -m pytest -q
```
