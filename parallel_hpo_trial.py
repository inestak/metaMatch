#!/usr/bin/env python3
"""Run one independently scheduled HPO trial for one pair/MetaSpace expert."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score

import nine_expert_hpo_v3_worker as worker


MAX_TRIALS = 30


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", required=True, type=int)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    expert_slot, trial_zero = divmod(args.task_index, MAX_TRIALS)
    pair_index, expert_index = divmod(expert_slot, len(worker.EXPERTS))
    if pair_index >= len(worker.PAIRS):
        raise ValueError(f"Invalid task index: {args.task_index}")

    pair = worker.PAIRS[pair_index]
    expert = worker.EXPERTS[expert_index]
    seed = worker.SEED + 10000 * pair_index + 100 * expert_index
    candidates = worker.hpo_candidates(seed)
    trial = trial_zero + 1
    if trial > len(candidates):
        print(f"SKIP {pair}/{expert}: unused trial slot {trial}", flush=True)
        return

    outdir = Path(args.output_root) / pair / expert
    outdir.mkdir(parents=True, exist_ok=True)
    output = outdir / f"trial_{trial:02d}.json"
    if output.is_file() and output.stat().st_size:
        saved = json.loads(output.read_text())
        if saved.get("trial") == trial and saved.get("pair") == pair and saved.get("expert") == expert:
            print(f"REUSE {output}", flush=True)
            return

    archive = worker.ZIPS[pair]
    public_header = worker.member_header(archive, worker.PUBLIC_MEMBER)
    test_header = worker.member_header(archive, worker.TEST_MEMBER)
    features = worker.build_spaces(public_header)[expert]
    missing = [name for name in features if name not in test_header]
    if missing:
        raise RuntimeError(f"{pair}/{expert}: missing hidden features: {missing[:20]}")

    public = worker.read_zip_csv(
        archive,
        worker.PUBLIC_MEMBER,
        usecols=["src_iri", "tgt_iri"] + features,
    )
    public["src_iri"] = public["src_iri"].astype(str)
    public["tgt_iri"] = public["tgt_iri"].astype(str)
    public = public.drop_duplicates(["src_iri", "tgt_iri"]).reset_index(drop=True)
    train_gold = worker.read_gold(worker.GOLD_PATHS[pair]["train"])
    valid_gold = worker.read_gold(worker.GOLD_PATHS[pair]["valid"])
    train, valid = worker.public_split(public, train_gold, valid_gold)
    x_train, medians = worker.numeric_matrix(train, features)
    x_valid, _ = worker.numeric_matrix(valid, features, medians)
    y_train = train["label"].astype(int).to_numpy()
    y_valid = valid["label"].astype(int).to_numpy()

    params = candidates[trial_zero]
    model = worker.make_xgb(y_train, params, seed + trial)
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_valid)[:, 1]
    threshold = worker.best_f1_threshold(y_valid, probabilities)
    row = {
        "pair": pair,
        "expert": expert,
        "trial": trial,
        **params,
        "threshold": threshold["threshold"],
        "precision": threshold["precision"],
        "recall": threshold["recall"],
        "f1": threshold["f1"],
        "average_precision": float(average_precision_score(y_valid, probabilities)),
        "hidden_gold_used": False,
    }
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(row, indent=2))
    os.replace(temporary, output)
    print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
