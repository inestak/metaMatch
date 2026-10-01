#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PAIRS = ["NCIT-DOID", "SNOMED-FMA", "SNOMED-NCIT"]
EXPERTS = [
    "Global403", "Lexical", "Syntaxique", "Semantic", "ROUGE",
    "Retrieval", "Structural", "Spectral", "TDA",
]
GOLD_SIZE_REFERENCE = {
    "NCIT-DOID": 1551,
    "SNOMED-FMA": 1595,
    "SNOMED-NCIT": 8650,
}


def greedy_1to1(d):
    if d.empty:
        return pd.DataFrame(
            columns=[
                "SrcEntity", "TgtEntity", "Relation", "votes",
                "mean_score_rank", "max_score_rank"
            ]
        )

    order = d.sort_values(
        ["votes", "mean_score_rank", "max_score_rank"],
        ascending=[False, False, False],
        kind="mergesort",
    )

    used_src = set()
    used_tgt = set()
    rows = []

    for r in order.itertuples(index=False):
        s, t = str(r.SrcEntity), str(r.TgtEntity)
        if s in used_src or t in used_tgt:
            continue
        used_src.add(s)
        used_tgt.add(t)
        rows.append({
            "SrcEntity": s,
            "TgtEntity": t,
            "Relation": "=",
            "votes": int(r.votes),
            "mean_score_rank": float(r.mean_score_rank),
            "max_score_rank": float(r.max_score_rank),
        })

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, choices=PAIRS)
    ap.add_argument("--output-root", required=True)
    args = ap.parse_args()

    pair = args.pair
    root = Path(args.output_root)
    croot = root / pair / "consensus"
    croot.mkdir(parents=True, exist_ok=True)

    frames = []
    expert_rows = []

    for expert in EXPERTS:
        edir = root / pair / "experts" / expert
        p = edir / "HIDDEN_V3_PREDICTIONS.tsv"
        r = edir / "FINAL_REPORT.json"

        if not p.exists() or not r.exists():
            raise FileNotFoundError((p, r))

        rr = json.loads(r.read_text())
        d = pd.read_csv(p, sep="\t")

        if len(d):
            frames.append(
                pd.DataFrame({
                    "SrcEntity": d["src_iri"].astype(str),
                    "TgtEntity": d["tgt_iri"].astype(str),
                    "expert": expert,
                    "score_rank": pd.to_numeric(
                        d["score_rank"], errors="coerce"
                    ).fillna(0.0),
                }).drop_duplicates(["SrcEntity", "TgtEntity", "expert"])
            )

        expert_rows.append({
            "pair": pair,
            "expert": expert,
            "n_features": rr["n_features"],
            "hpo_valid_f1": rr["hpo_validation"]["f1"],
            "hpo_valid_precision": rr["hpo_validation"]["precision"],
            "hpo_valid_recall": rr["hpo_validation"]["recall"],
            "oof_f1": rr["frozen_oof_threshold"]["f1"],
            "v3_gate": rr["v3_config"]["gate"],
            "v3_q": rr["v3_config"]["q"],
            "v3_oof_f1": rr["v3_config"]["f1"],
            "hidden_base_predictions": rr["hidden_base_predictions"],
            "hidden_v3_predictions": rr["hidden_v3_predictions"],
            "best_hyperparameters": json.dumps(rr["best_hyperparameters"], sort_keys=True),
        })

    expert_summary = pd.DataFrame(expert_rows)
    expert_summary.to_csv(croot / "EXPERT_HPO_COUNTS.csv", index=False)

    if not frames:
        raise RuntimeError(f"{pair}: all 9 V3 sets are empty")

    all_votes = pd.concat(frames, ignore_index=True)

    agg = (
        all_votes
        .groupby(["SrcEntity", "TgtEntity"], as_index=False)
        .agg(
            votes=("expert", "nunique"),
            mean_score_rank=("score_rank", "mean"),
            max_score_rank=("score_rank", "max"),
        )
    )

    exact_hist = (
        agg["votes"].value_counts()
        .reindex(range(1, 10), fill_value=0)
    )

    rows = []

    for k in range(9, 0, -1):
        raw = agg[agg["votes"] >= k].copy()
        raw["Relation"] = "="
        raw = raw[
            [
                "SrcEntity", "TgtEntity", "Relation", "votes",
                "mean_score_rank", "max_score_rank",
            ]
        ]
        one = greedy_1to1(raw)

        raw_path = croot / f"vote_ge_{k}.tsv"
        one_path = croot / f"vote_ge_{k}_greedy1to1.tsv"

        raw.to_csv(raw_path, sep="\t", index=False)
        one.to_csv(one_path, sep="\t", index=False)

        rows.append({
            "pair": pair,
            "k": k,
            "rule": f"vote >= {k}/9",
            "exactly_k_votes": int(exact_hist.loc[k]),
            "raw_predictions": int(len(raw)),
            "greedy1to1_predictions": int(len(one)),
            "gold_size_reference": int(GOLD_SIZE_REFERENCE[pair]),
            "raw_over_gold": float(len(raw) / GOLD_SIZE_REFERENCE[pair]),
            "greedy1to1_over_gold": float(len(one) / GOLD_SIZE_REFERENCE[pair]),
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(croot / "CONSENSUS_COUNTS.csv", index=False)

    print("=" * 120)
    print(pair)
    print("=" * 120)
    print("\n9 HPO EXPERTS AFTER V3")
    print(
        expert_summary[
            [
                "expert", "n_features", "hpo_valid_f1", "oof_f1",
                "v3_gate", "v3_q", "hidden_base_predictions",
                "hidden_v3_predictions",
            ]
        ].to_string(index=False)
    )
    print("\nCONSENSUS 9 -> 1")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
