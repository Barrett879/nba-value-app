# scripts/

CLI tools that aren't part of the live Streamlit app. Use these for
validation, weight tuning, and methodology experiments.

## Run from repo root, not from this directory:

```
python scripts/analyze_accuracy.py
```

Each script imports `utils`, which is at the repo root.

## What's here

| Script | Purpose |
|---|---|
| `analyze_accuracy.py` | Cap-relative error analysis on historical signings. Use `--verbose` for per-pair breakdown. Answers "how close was the projection to the actual signing?" across the full validation set. |
| `optimize_weights.py` | Coordinate-descent search for the optimal Barrett-Score component weights (random search + hand-rolled descent, no scipy dependency). |
| `test_contract_model.py` | Manual spot-check on a curated set of recent signings — quick sanity check after model changes. |
| `test_age_position.py` | Calibrates the age + position multipliers from historical data. |
| `test_phase1_gains.py` | Phase-1 validation experiment (trajectory smoothing + playoff blend). Result: ≤0.5pp improvement, not shipped. |
| `test_phase2_regression.py` | Phase-2 validation (OLS / Ridge with service-time + interaction terms). Concluded the model is at structural ceiling for box-score inputs. |

## What's not here

Live page logic, data scrapers, and ranking pipelines live in `utils.py`
and `pages/*.py`. These scripts are for development/validation only —
the Streamlit app doesn't import them.
