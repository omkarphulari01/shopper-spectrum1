# 🛒 Shopper Spectrum

**Customer segmentation (RFM + clustering) and product recommendations (item-based collaborative filtering) for e-commerce — packaged as an installable Python library with a config-driven pipeline, a CLI, tests, CI, Docker, and a multipage Streamlit app.**

[![CI](https://github.com/omkarphulari01/shopper-spectrum/actions/workflows/ci.yml/badge.svg)](https://github.com/omkarphulari01/shopper-spectrum/actions)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Highlights

- **Config-driven pipeline** — every path and hyper-parameter lives in `config/config.yaml`; no magic numbers in code.
- **Multi-algorithm clustering** — compares **KMeans, Hierarchical, and Gaussian Mixture** across k=2–10 using **silhouette, Calinski-Harabasz, and Davies-Bouldin**.
- **Classic + ML segmentation** — RFM 1–5 quartile scoring *and* unsupervised clustering.
- **Evaluated recommender** — item-based cosine similarity with a **leave-one-out hit-rate** metric.
- **Installable package** with a console command: `shopper-spectrum train` / `recommend`.
- **Tested & linted** — pytest suite + ruff, wired into GitHub Actions CI.
- **Reproducible** — `make train` regenerates every artifact and figure.
- **Deployable** — Dockerfile + multipage Streamlit app.

---

## Results on the bundled dataset

~540K raw transactions → **392,688 clean rows** across **4,338 customers**; recommender covers **3,152 products**.

| Segment | Avg Recency | Avg Frequency | Avg Monetary | Customers |
|---|---|---|---|---|
| 🟢 **High-Value** | 19.8 | 15.8 | 9,823 | 571 |
| 🔵 **Regular** | 46.1 | 4.2 | 1,644 | 1,452 |
| 🟡 **Occasional** | 58.0 | 1.5 | 384 | 1,377 |
| 🔴 **At-Risk** | 259.4 | 1.4 | 385 | 938 |

Final model: **KMeans, k=4** (silhouette 0.38). Recommender hit-rate@5 ≈ 9× the random baseline.

---

## Project structure

```
shopper-spectrum/
├── config/config.yaml              # single source of truth for paths & params
├── src/shopper_spectrum/           # installable package
│   ├── config.py                   # config loader + logging
│   ├── data/                       # loader.py, preprocessing.py
│   ├── features/rfm.py             # RFM + quartile scoring
│   ├── models/                     # clustering.py, recommender.py
│   ├── evaluation/metrics.py       # cluster validation metrics
│   ├── visualization/plots.py      # all figures
│   ├── pipeline.py                 # end-to-end orchestrator
│   └── cli.py                      # `shopper-spectrum` command
├── app/                            # multipage Streamlit app
│   ├── app.py
│   └── pages/                      # Recommendation · Segmentation · Dashboard
├── notebooks/                      # 01_eda · 02_rfm_clustering · 03_recommender
├── tests/                          # pytest suite
├── models/                         # generated .pkl artifacts
├── reports/figures/                # generated plots
├── .github/workflows/ci.yml        # lint + test on push/PR
├── Dockerfile · Makefile · pyproject.toml · requirements*.txt
```

---

## Quickstart

```bash
# 1. Install (editable, with dev tools)
pip install -e ".[dev]"

# 2. Put the dataset at data/raw/online_retail.csv  (see data/README.md)

# 3. Train — cleans data, clusters, builds recommender, saves all artifacts
shopper-spectrum train          # or: make train

# 4. Try the recommender from the terminal
shopper-spectrum recommend "WHITE HANGING HEART T-LIGHT HOLDER"

# 5. Launch the app
streamlit run app/app.py        # or: make app
```

`make help` lists every available command.

---

## The Streamlit app

Three pages (sidebar navigation):

1. **🎯 Product Recommendation** — type a product name → 5 similar products (card view).
2. **🔍 Customer Segmentation** — enter Recency / Frequency / Monetary → predicted segment.
3. **📊 Analytics Dashboard** — EDA charts, segment profiles, algorithm comparison, 3D RFM plot, and the product-similarity heatmap.

---

## Methodology

**Cleaning** — drop missing `CustomerID`, exclude cancelled invoices (`InvoiceNo` starting with `C`), remove non-positive quantity/price, drop duplicates. All toggleable in config.

**RFM** — Recency (days since last purchase), Frequency (distinct invoices), Monetary (total spend), plus 1–5 quartile R/F/M scores.

**Clustering** — Frequency & Monetary are right-skewed, so they're `log1p`-transformed before `StandardScaler`. Three algorithms are compared on three internal metrics; **k=4** is chosen to match the four business segments. Clusters are labeled by ranking mean RFM.

**Recommender** — customer×product quantity matrix → cosine similarity between products → top-20 neighbours stored per product (compact, fast to load). Evaluated with leave-one-out hit-rate.

---

## Testing & quality

```bash
make test     # pytest
make lint     # ruff
```

CI runs both on Python 3.9 and 3.11 for every push and PR.

---

## Docker

```bash
make docker-build
make docker-run      # app on http://localhost:8501
```

## License

MIT — see [LICENSE](LICENSE).
