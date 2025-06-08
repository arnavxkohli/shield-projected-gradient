#!/usr/bin/env python

import os
import logging
from pathlib import Path

import torch
import argparse, sys, time
import pandas as pd
import torch.optim as optim

from train_eval import validate_constraints, train_epoch, eval_epoch
from setup import setup_logging, get_device, set_seed, load_data, parse_numpy_data, make_dataloaders
from models import ShallowMLP, DeepMLP, ShieldedMLP, ShieldWithProjGrad

os.environ["PYTHONHASHSEED"] = "0"
torch.use_deterministic_algorithms(True)


ARCH_MAP = {
    "shallow": ShallowMLP,
    "deep": DeepMLP,
}

def main(args):
    device = get_device()
    out_dir = Path("out") / args.data_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mask_method:
        log_file_name = f"{args.base_arch}_mask_log.txt"
    else:
        log_file_name = f"{args.base_arch}_log.txt"

    logger = setup_logging(out_dir / log_file_name)
    logger.info(f"Experiment Configuration:")
    logger.info(f"  Python: {sys.version}")
    logger.info(f"  PyTorch: {torch.__version__}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Arguments: {args}")
    logger.info(f"Masking Method Enabled: {args.mask_method}")

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

    if args.mask_method:
        results_file_name = f"final_rmses_masked_{args.base_arch}.csv"
    else:
        results_file_name = f"final_rmses_{args.base_arch}.csv"

    results_csv = out_dir / results_file_name
    results_csv.write_text("model,trial,test_rmse,test_sat,total_time\n")

    base_cls = ARCH_MAP[args.base_arch]

    for trial in range(args.trials):
        seed = args.seed + trial
        set_seed(seed)
        logger.info(f"\n=== Trial {trial + 1}/{args.trials} | Seed={seed} ===")

        data = load_data(
            df,
            Path("data") / args.data_dir / "constraints.txt",
            test_size=args.test_size,
            val_size=args.val_size,
            seed=seed,
            scale_targets=args.scale_targets,
            train_fraction=args.train_fraction,
        )
        (train_d, val_d, test_d, constraints, in_dim, out_dim, scaler) = data
        loaders = make_dataloaders((train_d, val_d, test_d), args.batch_size)
        
        model_list = [
            (base_cls.__name__.capitalize(), lambda: base_cls(in_dim, out_dim)),
            ("ShieldedMLP", lambda: ShieldedMLP(base_cls(in_dim, out_dim), out_dim, Path("data") / args.data_dir / "constraints.txt")),
        ]
        
        if args.mask_method:
            model_list.append(
                ("ShieldedMLPMasked", lambda: ShieldedMLP(base_cls(in_dim, out_dim), out_dim, Path("data") / args.data_dir / "constraints.txt"))
            )
        else:
            model_list.append(
                ("ShieldWithProjGrad", lambda: ShieldWithProjGrad(base_cls(in_dim, out_dim), out_dim, Path("data") / args.data_dir / "constraints.txt"))
            )

        for name, ctor in model_list:
            total_start = time.time()

            model = ctor().to(device)
            opt = (optim.Adam if args.optimizer == "adam" else optim.SGD)(model.parameters(), lr=args.lr)

            for epoch in range(1, args.epochs + 1):
                epoch_start = time.time()

                tloss, tsat, _ = train_epoch(model, loaders["train"], opt, constraints, device, 
                                             logger, name == "ShieldedMLPMasked")
                vrmse, vsat, _ = eval_epoch(model, loaders["val"], constraints, scaler, device, 
                                            logger)

                epoch_total = time.time() - epoch_start
                logger.info(f"{name} | Epoch {epoch:02d}/{args.epochs} | "
                            f"loss {tloss:.4f} | val RMSE {vrmse:.4f} | val sat {vsat}"
                            f" | Epoch Time: {epoch_total:.2f}s")

            trmse, tsat, _ = eval_epoch(model, loaders["test"], constraints, scaler, device, 
                                        logger)
            logger.info(f"{name} | TEST RMSE {trmse:.4f} | test sat {tsat}")

            total_time = time.time() - total_start
            with results_csv.open("a") as f:
                f.write(f"{name},{trial},{trmse:.4f},{tsat},{total_time:.2f}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="url")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--val-size", type=float, default=0.125)
    p.add_argument("--scale-targets", action="store_true")
    p.add_argument("--train-fraction", type=float, default=1.0)
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    p.add_argument("--base-arch", choices=["shallow", "deep"], default="shallow")
    p.add_argument("--mask-method", action="store_true")
    p.add_argument("--numpy-data", action="store_true")
    p.add_argument("--data-list", nargs="+", 
                   default=["X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy"])
    main(p.parse_args())
