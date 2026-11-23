# Explainable Credit Underwriting

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

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
make install
make test
```

## Roadmap

- [x] Data layer: real/synthetic loader, single schema, tests, CI
- [ ] Feature engineering (affordability ratios, bureau scores)
- [ ] Baseline logistic scorecard vs. monotonic LightGBM
- [ ] SHAP-based adverse-action reason codes
- [ ] Fairness audit across gender and age bands
- [ ] FastAPI scoring service + Docker
- [ ] Streamlit underwriter dashboard
- [ ] Model card and AI Act technical documentation

## Kurzfassung (Deutsch)

Ein erklärbares Kreditscoring-Modell nach den Anforderungen des EU AI Act: Ausfallwahrscheinlichkeit, nachvollziehbare Ablehnungsgründe, Fairness-Prüfung und eine produktionsnahe Scoring-API. Sensible Merkmale (Geschlecht, Alter) fließen nicht ins Modell ein und dienen nur der Fairness-Analyse.
