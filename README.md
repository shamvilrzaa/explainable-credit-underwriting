# Explainable Credit Underwriting

[![CI](https://github.com/shamvilrzaa/explainable-credit-underwriting/actions/workflows/ci.yml/badge.svg)](https://github.com/shamvilrzaa/explainable-credit-underwriting/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

> An EU AI Act-aware credit underwriting engine: probability-of-default model, adverse-action reason codes, fairness audit, and a production-style scoring API.

**Status:** work in progress. See the roadmap below.

## Why this project

Under the EU AI Act, AI systems used to assess creditworthiness are classified as **high-risk** (Annex III). Lenders must be able to explain decisions, monitor for bias and document their models. This project builds a credit model the way a regulated lender would have to, not just the way a Kaggle leaderboard rewards.

## Data

| Mode | When | Use |
|---|---|---|
| **Real** | `data/raw/application_train.csv` exists | Training and reported metrics |
| **Synthetic** | No file present | Tests, CI, demos. Same schema, no personal data |

To use the real data, download `application_train.csv` from the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data) competition and put it in `data/raw/`. Raw data is git-ignored and never committed.

Protected attributes (gender, age) are **excluded from the model features** and used only for the fairness audit.

## Features

| Feature | Definition | Expected effect on risk |
|---|---|---|
| `debt_to_income` | Credit amount / annual income | ↑ |
| `annuity_to_income` | Annual repayment / annual income | ↑ |
| `credit_to_goods` | Credit amount / price of financed goods | ↑ |
| `employment_years` | Years in current employment (capped at 45) | ↓ |
| `ext_source_mean`, `ext_source_min` | Mean / minimum of external bureau scores | ↓ |
| `income_log`, `credit_log` | Log of income and credit amount | – |
| `cnt_children` | Number of children | – |

The expected directions are enforced as monotonic constraints in the gradient boosting model, so the model cannot learn counter-intuitive relationships such as "more debt lowers risk".

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
make install
make test
make train
```

| Command | Output |
|---|---|
| `make train` | Fits the scorecard and writes model, metrics, calibration table and coefficients to `artifacts/` |
| `make report` | Trains all models on the same split, writes `artifacts/comparison.md` and the charts in `docs/images/` |

## Models

| Model | Description |
|---|---|
| Scorecard | Logistic regression on standardised features, median imputation with missing-value indicators |
| Gradient boosting | LightGBM with monotonic constraints on every feature with a known risk direction |

Monotonic constraints guarantee that, all else equal, a higher repayment burden or a weaker bureau score never lowers the predicted risk. Without them, correlated features can produce counter-intuitive effects that are hard to defend to a regulator or a rejected applicant. The constraints are verified in `tests/test_gbm.py`.

## Results

Results below are on synthetic data and will be replaced with Home Credit results.

| model | Gini | AUC | KS | Brier |
|---|---|---|---|---|
| Scorecard | 0.707 | 0.853 | 0.564 | 0.085 |
| LightGBM (monotonic) | 0.725 | 0.862 | 0.589 | 0.085 |

<p float="left">
  <img src="docs/images/roc.png" width="45%" />
  <img src="docs/images/calibration.png" width="45%" />
</p>

## Evaluation

| Metric | Meaning |
|---|---|
| Gini | `2 × AUC − 1`, the standard ranking measure in retail credit |
| KS | Maximum separation between the cumulative distributions of good and bad applicants |
| Brier score | Mean squared error of the predicted default probability |
| Calibration | Predicted vs. observed default rate per risk decile |

Probabilities of default are mapped to a points score (600 points at 50:1 good:bad odds, 20 points to double the odds), the format credit decisions are usually communicated in.

## Roadmap

- [x] Data layer: real/synthetic loader, single schema, tests, CI
- [x] Feature engineering (affordability ratios, bureau scores)
- [x] Baseline logistic scorecard with points scaling and evaluation (Gini, KS, Brier, calibration)
- [x] Monotonic LightGBM and model comparison
- [ ] SHAP-based adverse-action reason codes
- [ ] Fairness audit across gender and age bands
- [ ] FastAPI scoring service + Docker
- [ ] Streamlit underwriter dashboard
- [ ] Model card and AI Act technical documentation

## Kurzfassung (Deutsch)

Ein erklärbares Kreditscoring-Modell nach den Anforderungen des EU AI Act: Ausfallwahrscheinlichkeit, nachvollziehbare Ablehnungsgründe, Fairness-Prüfung und eine produktionsnahe Scoring-API. Sensible Merkmale (Geschlecht, Alter) fließen nicht ins Modell ein und dienen nur der Fairness-Analyse.
