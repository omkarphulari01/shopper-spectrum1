"""
End-to-end training pipeline.

Loads raw data -> cleans -> builds RFM -> compares clustering algorithms ->
fits final model -> labels segments -> builds & evaluates recommender ->
saves all artifacts, figures and reports.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from .config import Config, load_config, setup_logging
from .data.loader import load_raw
from .data.preprocessing import clean
from .features.rfm import build_rfm
from .models import clustering as clu
from .models import recommender as rec
from .visualization import plots


def run(config_path: str | None = None) -> dict:
    """Run the full pipeline. Returns a summary dict."""
    logger = setup_logging()
    cfg: Config = load_config(config_path)

    models_dir = cfg.path("models_dir")
    fig_dir = cfg.path("figures_dir")
    reports_dir = cfg.path("reports_dir")
    for d in (models_dir, fig_dir, reports_dir, cfg.path("processed_data").parent):
        Path(d).mkdir(parents=True, exist_ok=True)

    # 1. Load & clean
    logger.info("STEP 1/6  Load & clean")
    df = clean(load_raw(cfg.path("raw_data")), cfg["preprocessing"])
    df.to_parquet(cfg.path("processed_data"))

    # 2. EDA figures
    logger.info("STEP 2/6  EDA figures")
    plots.eda_figures(df, fig_dir)

    # 3. RFM
    logger.info("STEP 3/6  RFM features")
    rfm = build_rfm(df, cfg["rfm"])
    plots.rfm_distributions(rfm, fig_dir)

    # 4. Compare algorithms & choose k
    logger.info("STEP 4/6  Clustering comparison")
    log_cols = cfg["rfm"]["log_transform"]
    X, scaler = clu.scale_features(rfm, log_cols)
    k_range = range(cfg["clustering"]["k_min"], cfg["clustering"]["k_max"] + 1)
    metrics = clu.compare_algorithms(
        X, cfg["clustering"]["algorithms"], k_range, cfg["random_state"])
    metrics.to_csv(Path(reports_dir) / "cluster_metrics.csv", index=False)
    plots.elbow_and_metrics(metrics, fig_dir)
    plots.algorithm_comparison(metrics, fig_dir)

    # 5. Final clustering + labels
    logger.info("STEP 5/6  Final clustering")
    k = cfg["clustering"]["chosen_k"]
    algo = cfg["clustering"]["final_algorithm"]
    labels, model = clu.fit_final(X, algo, k, cfg["random_state"])
    rfm["Cluster"] = labels
    label_map = clu.label_segments(rfm, cfg["clustering"]["segment_labels"])
    rfm["Segment"] = rfm["Cluster"].map(label_map)
    rfm.to_parquet(cfg.path("rfm_data"))

    profile = rfm.groupby("Segment")[clu.RFM_COLS].mean().round(1)
    profile["Customers"] = rfm["Segment"].value_counts()
    profile.to_csv(Path(reports_dir) / "segment_profiles.csv")
    logger.info("Segment profiles:\n%s", profile)
    plots.cluster_scatter(rfm, fig_dir)
    plots.cluster_scatter_3d(rfm, fig_dir)

    # 6. Recommender + evaluation
    logger.info("STEP 6/6  Recommender")
    rc = cfg["recommender"]
    matrix, code2name = rec.build_customer_product_matrix(
        df, rc["min_customers_per_product"])
    sim_df = rec.build_similarity(matrix)
    neighbors = rec.build_top_neighbors(sim_df, rc["top_n_neighbors"])
    name2code = rec.build_name_index(code2name, sim_df.index)
    plots.similarity_heatmap(sim_df, code2name, df, fig_dir)
    hit_rate = rec.evaluate_hit_rate(
        matrix, neighbors, rc["evaluation_sample"], rc["top_n_recommend"],
        cfg["random_state"])

    # Save artifacts
    joblib.dump(model, Path(models_dir) / "cluster_model.pkl")
    joblib.dump(scaler, Path(models_dir) / "scaler.pkl")
    joblib.dump(label_map, Path(models_dir) / "segment_labels.pkl")
    joblib.dump({"log_transform": log_cols, "rfm_cols": clu.RFM_COLS},
                Path(models_dir) / "feature_config.pkl")
    joblib.dump({"neighbors": neighbors, "code2name": code2name,
                 "name2code": name2code}, Path(models_dir) / "recommender.pkl")

    summary = {
        "rows_clean": int(len(df)),
        "customers": int(rfm["CustomerID"].nunique()),
        "products_in_recommender": int(sim_df.shape[0]),
        "final_algorithm": algo,
        "k": int(k),
        "segments": {s: int(c) for s, c in rfm["Segment"].value_counts().items()},
        "recommender_hit_rate": round(float(hit_rate), 4),
    }
    with open(Path(reports_dir) / "run_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Done. Summary: %s", summary)
    return summary


if __name__ == "__main__":
    run()
