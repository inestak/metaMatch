#!/usr/bin/env python3
"""Finalize one expert using the independently computed HPO trial reports."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

import nine_expert_hpo_v3_worker as worker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", required=True, choices=worker.PAIRS)
    parser.add_argument("--expert", required=True, choices=worker.EXPERTS)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--trial-root", required=True)
    args = parser.parse_args()

    reports = Path(args.trial_root) / args.pair / args.expert

    def load_parallel_trials(train_df, valid_df, features, seed):
        expected = len(worker.hpo_candidates(seed))
        rows = []
        for trial in range(1, expected + 1):
            path = reports / f"trial_{trial:02d}.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing HPO result: {path}")
            row = json.loads(path.read_text())
            if row.get("pair") != args.pair or row.get("expert") != args.expert or row.get("trial") != trial:
                raise RuntimeError(f"Invalid HPO result identity: {path}")
            rows.append(row)

        best = max(
            rows,
            key=lambda row: (
                row["f1"], row["average_precision"], row["recall"],
                row["precision"], -row["n_estimators"],
            ),
        )
        parameter_names = {
            "n_estimators", "max_depth", "learning_rate", "min_child_weight",
            "subsample", "colsample_bytree", "gamma", "reg_alpha", "reg_lambda",
        }
        params = {name: best[name] for name in parameter_names}
        table = pd.DataFrame(rows).drop(columns=["pair", "expert", "hidden_gold_used"], errors="ignore")
        validation = {key: value for key, value in best.items() if key not in {"pair", "expert", "hidden_gold_used"}}
        print(f"Loaded {len(rows)} independent HPO trials; best trial={best['trial']}", flush=True)
        return params, validation, table

    worker.tune_on_train_valid = load_parallel_trials
    worker.N_JOBS = 1
    sys.argv = [
        "nine_expert_hpo_v3_worker.py",
        "--pair", args.pair,
        "--expert", args.expert,
        "--output-root", args.output_root,
    ]
    worker.main()


if __name__ == "__main__":
    main()
