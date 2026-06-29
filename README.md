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

| Segment | Avg Recency | Avg Frequency | Avg Monetary | Customers | Revenue |
|---|---|---|---|---|---|
| ⭐ **High-Value** | 7.4 | 82.5 | 127,188 | 13 | £1.65M |
| ✅ **Regular** | 15.5 | 22.3 | 12,690 | 204 | £2.59M |
| 🔔 **Occasional** | 43.7 | 3.7 | 1,354 | 3,054 | £4.13M |
| ⚠️ **At-Risk** | 248.1 | 1.6 | 479 | 1,067 | £0.51M |

Final model: **KMeans, k=4** (silhouette **0.6162**). Headline insight: the 13
High-Value customers are just **0.3%** of the base but drive **18.6%** of revenue.

> Note: clustering runs on raw RFM by default (`config.yaml → rfm.log_transform: []`),
> which yields the sharp, business-friendly segmentation above. Setting
> `log_transform: [Frequency, Monetary]` instead produces more balanced cluster
> sizes (a valid alternative trade-off) — switch it in config, no code changes.

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

A dark-themed analytics dashboard with sidebar navigation across **10 sections**:

1. **Executive Dashboard** — KPI cards (customers, products, revenue, transactions, countries) + revenue trend + top products.
2. **Sales Analytics** — monthly revenue, best-sellers, peak/lowest/avg month metrics.
3. **Country Analysis** — revenue and transactions by country.
4. **RFM Analysis** — RFM distributions, cluster-profile heatmap, segment averages.
5. **Elbow Method** — elbow curve + silhouette, optimal-K insights.
6. **Customer Segmentation** — segment KPIs, **PCA cluster scatter**, distribution & revenue bars, descriptions.
7. **Similarity Matrix** — cosine-similarity heatmap for the top 15 products.
8. **Product Recommendation** — dropdown / search → top-N similar products.
9. **Customer Prediction** — enter R/F/M → predicted segment.
10. **Business Insights** — key findings + recommended actions by segment.

All charts are interactive (Plotly). Data is precomputed by the pipeline into a
`dashboard_data.pkl` bundle so the app loads instantly.

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
