# 🛒 Shopper Spectrum — Customer Segmentation & Product Recommendations

Customer segmentation (RFM + KMeans) and an item-based collaborative-filtering
product recommender for an online retail dataset, served through an interactive
**Streamlit** app.

**Domain:** E-Commerce & Retail Analytics
**Problem types:** Unsupervised clustering · Collaborative filtering

---

## ✨ What this project does

1. **Cleans** ~540K raw transactions down to ~393K valid rows (4,338 customers).
2. **Engineers RFM features** — Recency, Frequency, Monetary — per customer.
3. **Segments customers** into four actionable groups with KMeans.
4. **Recommends products** using item-based cosine similarity.
5. **Serves both** through a two-tab Streamlit app with real-time outputs.

### Customer segments (from this dataset)

| Segment | Avg Recency (days) | Avg Frequency | Avg Monetary | Customers |
|---|---|---|---|---|
| **High-Value** | 19.8 | 15.8 | 9,823 | 571 |
| **Regular** | 46.1 | 4.2 | 1,644 | 1,452 |
| **Occasional** | 58.0 | 1.5 | 384 | 1,377 |
| **At-Risk** | 259.4 | 1.4 | 385 | 938 |

---

## 📁 Project structure

```
shopper-spectrum/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── data/
│   ├── README.md              # dataset notes (CSV gitignored by default)
│   └── online_retail.csv      # <- place dataset here
├── notebooks/
│   └── shopper_spectrum_analysis.ipynb   # full EDA + modeling walkthrough
├── src/
│   ├── preprocessing.py       # load & clean
│   ├── rfm.py                 # RFM feature engineering
│   ├── clustering.py          # scaling, k-selection, KMeans, labeling
│   ├── recommender.py         # item-based collaborative filtering
│   └── train_pipeline.py      # end-to-end: trains & saves all artifacts
├── app/
│   └── app.py                 # Streamlit web application
├── models/                    # generated .pkl artifacts (after training)
└── reports/
    ├── figures/               # generated EDA & clustering plots
    ├── segment_profiles.csv
    ├── cluster_metrics.csv
    └── rfm_segments.csv
```

---

## 🚀 Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add the dataset

Put `online_retail.csv` in the `data/` folder (see `data/README.md` for the
expected columns).

### 3. Train the models

```bash
python src/train_pipeline.py
```

This cleans the data, runs RFM + clustering, builds the recommender, and writes
all artifacts to `models/` and figures to `reports/figures/`.

### 4. Launch the app

```bash
streamlit run app/app.py
```

---

## 📱 Streamlit app

**🎯 Product Recommendation** — type a product name, get 5 similar products
(case-insensitive with substring matching).

**🔍 Customer Segmentation** — enter Recency, Frequency, and Monetary values to
predict the customer's segment (High-Value / Regular / Occasional / At-Risk).

---

## 🔬 Methodology

**Cleaning** — drop missing `CustomerID`, exclude cancelled invoices (`InvoiceNo`
starting with `C`), remove non-positive quantities/prices, drop duplicates.

**RFM** — Recency = days since last purchase; Frequency = distinct invoices;
Monetary = total spend.

**Clustering** — Frequency and Monetary are heavily right-skewed, so they are
`log1p`-transformed before `StandardScaler`. Without this, KMeans isolates a tiny
outlier cluster. `k` is evaluated from 2–10 via the elbow method and silhouette
score; **k = 4** is used to match the four business segments. Clusters are
labeled by ranking each cluster's mean RFM (low recency + high frequency + high
monetary = best customers).

**Recommender** — a Customer × Product quantity matrix feeds cosine similarity
between products. Products bought by fewer than 5 distinct customers are dropped
to keep similarities meaningful. The saved artifact stores only the top-20
neighbors per product for a compact, fast-loading app.

---

## 🛠 Tech stack

Pandas · NumPy · scikit-learn · SciPy · Matplotlib · Seaborn · Streamlit · Joblib

## 📄 License

MIT — see [LICENSE](LICENSE).
