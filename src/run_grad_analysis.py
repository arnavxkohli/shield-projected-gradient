#!/usr/bin/env python

import os
import sys
import argparse
from pathlib import Path

import torch
import pandas as pd

from setup import setup_logging, get_device, set_seed, load_data, parse_numpy_data, make_dataloaders
from models import DeepMLP, ShieldedMLP, ShieldWithProjGrad
from gradient_analyzer import GradientAnalyzer

os.environ["PYTHONHASHSEED"] = "0"
torch.use_deterministic_algorithms(True)

def main(args):
    device = get_device()
    out_dir = Path("out") / args.data_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(out_dir / "gradient_analysis_log.txt")
    logger.info(f"Experiment Configuration:")
    logger.info(f"  Python: {sys.version}")
    logger.info(f"  PyTorch: {torch.__version__}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Arguments: {args}")

    if args.numpy_data:
        logger.info("Loading data from numpy files...")
        numpy_data_dir = None if not args.numpy_data else Path("data") / args.data_dir
        df = parse_numpy_data(logger, numpy_data_dir, args.data_list)
        if df.empty:
            logger.error("No data loaded from numpy files. Exiting.")
            sys.exit(1)
    else:
        logger.info("Loading data from CSV file...")
        df = pd.read_csv(Path("data") / args.data_dir / "data.csv")
        if df.empty:
            logger.error("No data loaded from CSV file. Exiting.")
            sys.exit(1)

    set_seed(args.seed)

    data = load_data(
        df,
        Path("data") / args.data_dir / "constraints.txt",
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
        scale_targets=args.scale_targets,
        train_fraction=args.train_fraction,
    )

    (train_d, val_d, test_d, constraints, in_dim, out_dim, scaler) = data
    loaders = make_dataloaders((train_d, val_d, test_d), args.batch_size)

    analyzer = GradientAnalyzer(device)

    baseline_model = DeepMLP(in_dim, out_dim).to(device)
    shielded_model = ShieldedMLP(DeepMLP(in_dim, out_dim), out_dim, Path("data") / args.data_dir / "constraints.txt").to(device)
    masked_model = ShieldedMLP(DeepMLP(in_dim, out_dim), out_dim, Path("data") / args.data_dir / "constraints.txt").to(device)
    proj_model = ShieldWithProjGrad(DeepMLP(in_dim, out_dim), out_dim, Path("data") / args.data_dir / "constraints.txt").to(device)

    models = {
        "baseline": baseline_model,
        "shielded": shielded_model,
        "masked": masked_model,
        "proj": proj_model
    }

    logger.info("Starting gradient analysis...")
    results_df = analyzer.analyze_dataset(models, loaders["test"], constraints, max_batches=args.max_batches)

    if not args.summary_only:
        output_file = out_dir / "gradient_analysis_results.csv"
        results_df.to_csv(output_file, index=False)
        logger.info(f"Gradient analysis results saved to {output_file}")

    if "batch_index" in results_df.columns:
        results_df = results_df.drop(columns=["batch_index"])
    
    summary = results_df.agg(['mean', 'std']).T

    summary = summary.round(4)

    summary_file = out_dir / "gradient_analysis_summary.csv"
    summary.to_csv(summary_file)
    logger.info(f"Gradient analysis summary (mean ± std) saved to {summary_file}")

    logger.info("\nSummary Statistics (mean ± std):\n%s", summary.to_string())

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="url")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--val-size", type=float, default=0.125)
    p.add_argument("--scale-targets", action="store_true")
    p.add_argument("--train-fraction", type=float, default=1.0)
    p.add_argument("--numpy-data", action="store_true")
    p.add_argument("--data-list", nargs="+", 
                   default=["X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy"])
    p.add_argument("--max-batches", type=int, default=100,
                   help="Number of batches to analyze (default: 100)")
    p.add_argument("--summary-only", action="store_true",
                   help="Skip saving per-batch CSV; save only summary statistics")
    main(p.parse_args())
