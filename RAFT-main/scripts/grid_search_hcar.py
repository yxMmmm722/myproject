#!/usr/bin/env python3
import argparse
import csv
import itertools
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path


DATASET_FILES = {
    "ETTh1": ("./data/ETT", "ETTh1.csv"),
    "ETTh2": ("./data/ETT", "ETTh2.csv"),
    "ETTm1": ("./data/ETT", "ETTm1.csv"),
    "ETTm2": ("./data/ETT", "ETTm2.csv"),
    "electricity": ("./data/electricity", "electricity.csv"),
    "exchange_rate": ("./data/exchange_rate", "exchange_rate.csv"),
    "illness": ("./data/illness", "national_illness.csv"),
    "traffic": ("./data/traffic", "traffic.csv"),
    "weather": ("./data/weather", "weather.csv"),
}


def parse_csv_list(text, cast_fn=str):
    values = [v.strip() for v in text.split(",") if v.strip()]
    return [cast_fn(v) for v in values]


def infer_channels(csv_path: Path):
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
    # one timestamp column + N feature columns
    return max(1, len(header) - 1)


def parse_metrics(log_text: str):
    pattern = re.compile(r"mse:([\-0-9eE\.]+), mae:([\-0-9eE\.]+), dtw:([\-0-9eE\.]+)")
    matches = pattern.findall(log_text)
    if not matches:
        return None
    mse, mae, dtw = matches[-1]
    return {
        "mse": float(mse),
        "mae": float(mae),
        "dtw": float(dtw),
    }


def parse_vali_summary(log_text: str):
    pattern = re.compile(
        r"Epoch:\s*(\d+),\s*Steps:\s*(\d+)\s*\|\s*Train Loss:\s*([\-0-9eE\.]+)\s*Vali Loss:\s*([\-0-9eE\.]+)\s*Test Loss:\s*([\-0-9eE\.]+)"
    )
    matches = pattern.findall(log_text)
    if not matches:
        return None

    best = None
    for epoch, _steps, train_loss, vali_loss, test_loss in matches:
        row = {
            "best_vali_epoch": int(epoch),
            "vali_loss_min": float(vali_loss),
            "train_loss_at_best_vali": float(train_loss),
            "test_loss_at_best_vali": float(test_loss),
        }
        if best is None or row["vali_loss_min"] < best["vali_loss_min"]:
            best = row
    return best


def run_command(cmd, workdir: Path, log_file: Path):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        cmd,
        cwd=str(workdir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    lines = []
    with log_file.open("w", encoding="utf-8") as f:
        for line in process.stdout:
            sys.stdout.write(line)
            f.write(line)
            lines.append(line)

    return_code = process.wait()
    merged_output = "".join(lines)
    return return_code, merged_output


def _result_score(row, selection_metric: str):
    if row["return_code"] != 0:
        return math.inf
    if selection_metric == "mse":
        mse = row.get("mse")
        return float(mse) if mse is not None else math.inf
    # default: choose by validation set
    val = row.get("vali_loss_min")
    return float(val) if val is not None else math.inf


def main():
    parser = argparse.ArgumentParser(description="Grid search for HCAR-RAFT.")
    parser.add_argument("--python_bin", type=str, default="python3")
    parser.add_argument("--run_file", type=str, default="run.py")
    parser.add_argument("--datasets", type=str,
                        default="ETTh1,ETTh2,ETTm1,ETTm2,electricity,exchange_rate,illness,traffic,weather")
    parser.add_argument("--pred_lens", type=str, default="96")
    parser.add_argument("--lookback_grid", type=str, default="336,720,960",
                        help="comma-separated lookback window candidates; empty means use --seq_len")
    parser.add_argument("--learning_rate_grid", type=str, default="1e-4,3e-4,1e-3",
                        help="comma-separated learning-rate candidates; empty means use --learning_rate")
    parser.add_argument("--topm_grid", type=str, default="1,5,10,20")
    parser.add_argument("--coarse_k_grid", type=str, default="20,40,80,160")
    parser.add_argument("--n_period_grid", type=str, default="2,3")
    parser.add_argument("--context_dim_grid", type=str, default="32,64,128")
    parser.add_argument("--retrieval_alpha_grid", type=str, default="0.7",
                        help="deprecated: kept only for CLI/result compatibility; retrieval alpha is not searched")

    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--model_prefix", type=str, default="HCARGrid")
    parser.add_argument("--text_encoder_name", type=str, default="./models/bert-base-uncased")
    parser.add_argument("--retrieval_cache_device", type=str, default="gpu", choices=["cpu", "gpu"])
    parser.add_argument("--text_cache_device", type=str, default="gpu", choices=["cpu", "gpu"])
    parser.add_argument("--traffic_cache_device", type=str, default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--traffic_text_cache_device", type=str, default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--max_trials", type=int, default=0, help="0 means no limit.")

    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--require_text_encoder", action="store_true")
    parser.add_argument("--save_meta_texts", action="store_true")
    parser.add_argument("--freeze_context_encoder", action="store_true")
    parser.add_argument("--no_refresh_context_each_epoch", action="store_true")
    parser.add_argument("--selection_metric", type=str, default="vali_loss_min", choices=["vali_loss_min", "mse"],
                        help="metric used to choose best config per dataset/pred_len")

    parser.add_argument("--output_dir", type=str, default="")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "logs" / f"grid_search_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = parse_csv_list(args.datasets, str)
    pred_lens = parse_csv_list(args.pred_lens, int)
    lookback_grid = parse_csv_list(args.lookback_grid, int) if args.lookback_grid.strip() else [args.seq_len]
    lr_grid = parse_csv_list(args.learning_rate_grid, float) if args.learning_rate_grid.strip() else [args.learning_rate]
    topm_grid = parse_csv_list(args.topm_grid, int)
    coarse_k_grid = parse_csv_list(args.coarse_k_grid, int)
    n_period_grid = parse_csv_list(args.n_period_grid, int)
    context_dim_grid = parse_csv_list(args.context_dim_grid, int)
    alpha_grid = parse_csv_list(args.retrieval_alpha_grid, float)
    fixed_retrieval_alpha = alpha_grid[0] if alpha_grid else 0.7

    # Retrieval alpha is no longer part of the search space. We keep a fixed value in the
    # output schema for backward compatibility with downstream scripts.
    raw_grid = itertools.product(
        lookback_grid,
        lr_grid,
        topm_grid,
        coarse_k_grid,
        n_period_grid,
        context_dim_grid,
    )
    # Prune redundant configs where coarse_k < topm (effective value would be max(topm, coarse_k)).
    dedup_set = set()
    grid = []
    for seq_len, learning_rate, topm, coarse_k, n_period, context_dim in raw_grid:
        eff_coarse_k = max(topm, coarse_k)
        key = (seq_len, learning_rate, topm, eff_coarse_k, n_period, context_dim)
        if key in dedup_set:
            continue
        dedup_set.add(key)
        grid.append(key)

    if args.max_trials > 0:
        grid = grid[:args.max_trials]

    print(f"[GridSearch] project_root={project_root}")
    print(f"[GridSearch] output_dir={output_dir}")
    print(f"[GridSearch] total combinations per dataset/pred_len={len(grid)}")
    if len(alpha_grid) > 1:
        print(
            f"[GridSearch] retrieval_alpha_grid={args.retrieval_alpha_grid} is deprecated and ignored. "
            f"Using fixed retrieval_alpha={fixed_retrieval_alpha}."
        )

    results = []
    trial_id = 0

    for dataset in datasets:
        if dataset not in DATASET_FILES:
            print(f"[Skip] dataset={dataset} not in DATASET_FILES mapping.")
            continue

        root_path, data_file = DATASET_FILES[dataset]
        csv_path = project_root / root_path / data_file
        if not csv_path.exists():
            print(f"[Skip] dataset={dataset} missing file: {csv_path}")
            continue

        channels = infer_channels(csv_path)

        for pred_len in pred_lens:
            for seq_len, learning_rate, topm, coarse_k, n_period, context_dim in grid:
                trial_id += 1
                retrieval_cache_device_run = args.retrieval_cache_device
                text_cache_device_run = args.text_cache_device
                if dataset == "traffic":
                    retrieval_cache_device_run = args.traffic_cache_device
                    text_cache_device_run = args.traffic_text_cache_device
                model_id = (
                    f"{args.model_prefix}_{dataset}_pl{pred_len}"
                    f"_lb{seq_len}_lr{learning_rate:.0e}"
                    f"_m{topm}_k{coarse_k}_p{n_period}_cd{context_dim}"
                )
                log_file = output_dir / f"{trial_id:04d}_{model_id}.log"

                cmd = [
                    args.python_bin, args.run_file,
                    "--data", dataset,
                    "--root_path", root_path,
                    "--data_path", data_file,
                    "--model_id", model_id,
                    "--is_training", "1",
                    "--seq_len", str(seq_len),
                    "--label_len", str(args.label_len),
                    "--pred_len", str(pred_len),
                    "--enc_in", str(channels),
                    "--dec_in", str(channels),
                    "--c_out", str(channels),
                    "--n_period", str(n_period),
                    "--topm", str(topm),
                    "--retrieval_coarse_k", str(coarse_k),
                    "--retrieval_alpha", str(fixed_retrieval_alpha),
                    "--context_dim", str(context_dim),
                    "--retrieval_cache_device", retrieval_cache_device_run,
                    "--text_cache_device", text_cache_device_run,
                    "--batch_size", str(args.batch_size),
                    "--train_epochs", str(args.train_epochs),
                    "--learning_rate", str(learning_rate),
                    "--lradj", "cosine",
                    "--num_workers", str(args.num_workers),
                    "--text_encoder_name", args.text_encoder_name,
                    "--online_retrieval",
                ]

                if args.use_amp:
                    cmd.append("--use_amp")
                if args.require_text_encoder:
                    cmd.append("--require_text_encoder")
                if args.save_meta_texts:
                    cmd.append("--save_meta_texts")
                if args.freeze_context_encoder:
                    cmd.append("--freeze_context_encoder")
                if args.no_refresh_context_each_epoch:
                    cmd.append("--no_refresh_context_each_epoch")

                print("=" * 100)
                print(
                    f"[Trial {trial_id}] dataset={dataset} pred_len={pred_len} "
                    f"lookback={seq_len} lr={learning_rate:.0e} topm={topm} coarse_k={coarse_k} "
                    f"n_period={n_period} context_dim={context_dim} retrieval_alpha={fixed_retrieval_alpha:.2f} "
                    f"cache={retrieval_cache_device_run}/{text_cache_device_run}"
                )
                print(f"[Trial {trial_id}] log={log_file}")
                print("=" * 100)

                start = time.time()
                return_code, output = run_command(cmd, project_root, log_file)
                duration = time.time() - start
                metrics = parse_metrics(output)
                vali_summary = parse_vali_summary(output)

                record = {
                    "trial_id": trial_id,
                    "dataset": dataset,
                    "pred_len": pred_len,
                    "seq_len": seq_len,
                    "learning_rate": learning_rate,
                    "topm": topm,
                    "coarse_k": coarse_k,
                    "n_period": n_period,
                    "context_dim": context_dim,
                    "retrieval_alpha": fixed_retrieval_alpha,
                    "channels": channels,
                    "return_code": return_code,
                    "duration_sec": round(duration, 2),
                    "log_file": str(log_file),
                    "mse": None,
                    "mae": None,
                    "dtw": None,
                    "best_vali_epoch": None,
                    "vali_loss_min": None,
                    "train_loss_at_best_vali": None,
                    "test_loss_at_best_vali": None,
                }
                if metrics is not None:
                    record.update(metrics)
                if vali_summary is not None:
                    record.update(vali_summary)

                results.append(record)

                if return_code == 0:
                    print(
                        f"[Trial {trial_id}] done. vali_min={record['vali_loss_min']} "
                        f"(epoch={record['best_vali_epoch']}), mse={record['mse']} "
                        f"mae={record['mae']} duration={duration:.1f}s"
                    )
                else:
                    print(f"[Trial {trial_id}] failed with return_code={return_code} duration={duration:.1f}s")

    json_path = output_dir / "grid_results.json"
    csv_path = output_dir / "grid_results.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "trial_id", "dataset", "pred_len", "seq_len", "learning_rate", "topm", "coarse_k", "n_period",
        "context_dim", "retrieval_alpha", "channels", "return_code", "duration_sec",
        "best_vali_epoch", "vali_loss_min", "train_loss_at_best_vali", "test_loss_at_best_vali",
        "mse", "mae", "dtw", "log_file"
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    # Chosen parameter values by validation-set grid search.
    chosen_by_val = {}
    for row in results:
        key = (row["dataset"], row["pred_len"])
        score = _result_score(row, args.selection_metric)
        if not math.isfinite(score):
            continue
        prev = chosen_by_val.get(key)
        if prev is None or score < _result_score(prev, args.selection_metric):
            chosen_by_val[key] = row

    chosen_json_path = output_dir / "chosen_parameters_by_val.json"
    chosen_csv_path = output_dir / "chosen_parameters_by_val.csv"

    chosen_rows = []
    for key in sorted(chosen_by_val.keys()):
        row = chosen_by_val[key]
        chosen_rows.append({
            "dataset": row["dataset"],
            "pred_len": row["pred_len"],
            "lookback_window_size": row["seq_len"],
            "learning_rate": row["learning_rate"],
            "number_of_retrievals": row["topm"],
            "coarse_k": row["coarse_k"],
            "n_period": row["n_period"],
            "context_dim": row["context_dim"],
            "retrieval_alpha": row["retrieval_alpha"],
            "vali_loss_min": row["vali_loss_min"],
            "best_vali_epoch": row["best_vali_epoch"],
            "mse": row["mse"],
            "mae": row["mae"],
            "trial_id": row["trial_id"],
            "log_file": row["log_file"],
        })

    with chosen_json_path.open("w", encoding="utf-8") as f:
        json.dump(chosen_rows, f, ensure_ascii=False, indent=2)

    chosen_fieldnames = [
        "dataset", "pred_len", "lookback_window_size", "learning_rate", "number_of_retrievals",
        "coarse_k", "n_period", "context_dim", "retrieval_alpha",
        "vali_loss_min", "best_vali_epoch", "mse", "mae", "trial_id", "log_file"
    ]
    with chosen_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=chosen_fieldnames)
        writer.writeheader()
        for row in chosen_rows:
            writer.writerow(row)

    print("\n[GridSearch] Finished.")
    print(f"[GridSearch] json: {json_path}")
    print(f"[GridSearch] csv : {csv_path}")
    print(f"[GridSearch] chosen params json: {chosen_json_path}")
    print(f"[GridSearch] chosen params csv : {chosen_csv_path}")
    print(f"[GridSearch] selection metric : {args.selection_metric}")

    if chosen_rows:
        print("\n[GridSearch] Chosen parameter values of each setting via grid search over validation set:")
        for row in chosen_rows:
            print(
                f"  {row['dataset']} pl{row['pred_len']} -> "
                f"lookback={row['lookback_window_size']}, lr={row['learning_rate']}, "
                f"retrievals={row['number_of_retrievals']}, vali_min={row['vali_loss_min']}, "
                f"mse={row['mse']}"
            )
    else:
        print("\n[GridSearch] No successful trials with valid selection metric.")


if __name__ == "__main__":
    main()
