"""Command-line interface for the Shopper Spectrum pipeline."""

from __future__ import annotations

import argparse

from .config import load_config
from .pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="shopper-spectrum",
        description="Customer segmentation & product recommendation pipeline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Run the full training pipeline.")
    p_train.add_argument("--config", default=None, help="Path to config.yaml")

    p_rec = sub.add_parser("recommend", help="Get recommendations for a product.")
    p_rec.add_argument("product", help="Product name to query")
    p_rec.add_argument("-n", type=int, default=5, help="Number of recommendations")
    p_rec.add_argument("--config", default=None)

    args = parser.parse_args()

    if args.command == "train":
        run(args.config)
    elif args.command == "recommend":
        import joblib

        from .models.recommender import recommend

        cfg = load_config(args.config)
        data = joblib.load(cfg.path("models_dir") / "recommender.pkl")
        results = recommend(args.product, data["neighbors"], data["code2name"],
                            data["name2code"], args.n)
        if results is None:
            print(f"No product matching '{args.product}' found.")
        elif not results:
            print("No similarity data for that product.")
        else:
            print(f"Top {len(results)} products similar to '{args.product}':")
            for r in results:
                print(f"  {r['product']:<40} {r['score']:.3f}")


if __name__ == "__main__":
    main()
