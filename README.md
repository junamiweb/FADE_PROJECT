# FADE — Financial Atom Discovery Engine (v0.1)

A minimal, research-grade Python system that tests a single hypothesis:

> *Do financial time series contain stable atomic event structures that generalise across time without overfitting?*

No machine learning. No deep learning. No external APIs. Just vectorised
pandas/numpy, frequency-based rule mining, walk-forward validation, and a
two-layer persistent memory (positive rules + negative anti-patterns).

## Install & run

```bash
pip install -r requirements.txt

# Use your own BTC 1H OHLCV CSV (timestamp, open, high, low, close, volume):
python -m fade.pipeline.main path/to/btc_1h.csv

# Quick calibrated forecast (uses positive memory from a prior main run):
python -m fade.pipeline.forecast path/to/btc_1h.csv
python -m fade.pipeline.forecast path/to/btc_1h.csv --json

# Or run on reproducible synthetic data (no CSV needed):
python -m fade.pipeline.main
```

## How it works

| Stage | Module | What it does |
|-------|--------|--------------|
| Data | `core/data_loader.py` | Load & validate OHLCV (single asset, 1H). |
| Atoms | `core/atoms.py` | Strict 5-atom set: `return_1h`, `return_6h`, `volatility` (24h std), `volume_zscore` (24h), `trend_slope` (12h OLS). |
| Events | `core/events.py` | Discretise atoms into `LOW/MID/HIGH` (quantiles), combine 2–4 atoms into hashable event keys, drop rare events. |
| Rules | `core/rules.py` | Map event → 4h-forward outcome distribution with support, confidence, decay (30d half-life), last-seen. |
| Backtest | `pipeline/backtest.py` | Expanding-window **walk-forward** (sequential splits only). Tracks each rule's **out-of-sample** hit-rate across folds. |
| Evaluator | `core/evaluator.py` | Confidence-weighted rule vote vs **random** and **momentum** baselines; reports predictive lift. |
| Memory | `memory/store.py` | `positive_rules.json` (stable, OOS-validated rules) and `negative_patterns.json` (repeatedly-failing anti-patterns, skipped **before** mining). |
| Loop | `pipeline/main.py` | Full learning loop + printed report. |

## Anti-overfitting design

1. **Minimum support** — rules need ≥ `min_support` occurrences.
2. **Event frequency filter** — rare events are discarded as noise.
3. **Rule decay** — older evidence is exponentially down-weighted.
4. **Stability tracking** — a rule is promoted to positive memory only if it
   beats the coin-flip **out-of-sample** in most folds. Training-set direction
   agreement is *not* enough.
5. **Negative memory** — patterns that fail repeatedly are blacklisted and
   skipped before any future rule generation.

All thresholds live in `fade/config.py`.

## Interpreting the verdict

The report ends with an honest verdict. On near-random data it will correctly
say *"No generalising structure"* — the system is built to avoid fooling
itself, not to manufacture signal.
