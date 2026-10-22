# state/

Written only by the system (GitHub Actions). Do not edit by hand: the ledgers are hash-chained and `python -m atl verify` fails if any past record changes.

- `ledger/events.jsonl` — trading ledger (orders, fills, marks, promotions, halts, holdout touches, reviews)
- `registry/` — candidates, model versions, holdout access log, champion/challengers, feature evidence, stored return series
- `books/` — working snapshots of each book (reconciled against the ledger every run)
- `reports/` — validation reports, research runs, monthly reviews, data gaps
- `dashboard/data.json` — dashboard export
