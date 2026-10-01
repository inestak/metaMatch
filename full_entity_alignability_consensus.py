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
TIERS = ["SAFE", "BALANCED", "AGGRESSIVE"]

GOLD_REFERENCE = {
    "NCIT-DOID": 1551,
    "SNOMED-FMA": 1595,
    "SNOMED-NCIT": 8650,
}


def greedy_1to1(d):
    if d.empty:
        return pd.DataFrame(
            columns=[
                "SrcEntity", "TgtEntity", "Relation",
                "votes", "mean_base_score",
                "mean_src_alignability_rank",
                "mean_tgt_alignability_rank",
            ]
        )

    order = d.sort_values(
        [
            "votes",
            "mean_src_alignability_rank",
            "mean_tgt_alignability_rank",
            "mean_base_score",
        ],
        ascending=[False, False, False, False],
        kind="mergesort",
    )

    used_src = set()
    used_tgt = set()
    rows = []

    for r in order.itertuples(index=False):
        s = str(r.SrcEntity)
        t = str(r.TgtEntity)

        if s in used_src or t in used_tgt:
            continue

        used_src.add(s)
        used_tgt.add(t)

        rows.append({
            "SrcEntity": s,
            "TgtEntity": t,
            "Relation": "=",
            "votes": int(r.votes),
            "mean_base_score": float(r.mean_base_score),
            "mean_src_alignability_rank": float(r.mean_src_alignability_rank),
            "mean_tgt_alignability_rank": float(r.mean_tgt_alignability_rank),
        })

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, choices=PAIRS)
    ap.add_argument("--input-root", required=True)
    args = ap.parse_args()

    pair = args.pair
    root = Path(args.input_root)
    outdir = root / pair / "consensus"
    outdir.mkdir(parents=True, exist_ok=True)

    all_summary = []
    expert_rows = []

    for expert in EXPERTS:
        r = root / pair / "experts" / expert / "FINAL_REPORT.json"
        if not r.exists():
            raise FileNotFoundError(r)
        rr = json.loads(r.read_text())

        row = {
            "pair": pair,
            "expert": expert,
            "hidden_base": rr["hidden_base_predictions"],
        }
        for tier in TIERS:
            row[tier] = rr["hidden_counts"][tier]
        expert_rows.append(row)

    pd.DataFrame(expert_rows).to_csv(
        outdir / "EXPERT_FILTER_COUNTS.csv",
        index=False,
    )

    for tier in TIERS:
        frames = []

        for expert in EXPERTS:
            p = (
                root
                / pair
                / "experts"
                / expert
                / f"HIDDEN_FILTERED_{tier}.tsv"
            )
            if not p.exists():
                raise FileNotFoundError(p)

            d = pd.read_csv(p, sep="\t")
            if len(d) == 0:
                continue

            frames.append(
                pd.DataFrame({
                    "SrcEntity": d["src_iri"].astype(str),
                    "TgtEntity": d["tgt_iri"].astype(str),
                    "expert": expert,
                    "base_score": pd.to_numeric(
                        d["score"],
                        errors="coerce",
                    ).fillna(0.0),
                    "src_alignability_rank": pd.to_numeric(
                        d["src_alignability_rank"],
                        errors="coerce",
                    ).fillna(0.0),
                    "tgt_alignability_rank": pd.to_numeric(
                        d["tgt_alignability_rank"],
                        errors="coerce",
                    ).fillna(0.0),
                })
            )

        if not frames:
            raise RuntimeError(f"{pair}/{tier}: no expert predictions")

        votes = pd.concat(frames, ignore_index=True)

        agg = (
            votes.groupby(
                ["SrcEntity", "TgtEntity"],
                as_index=False,
            )
            .agg(
                votes=("expert", "nunique"),
                mean_base_score=("base_score", "mean"),
                mean_src_alignability_rank=(
                    "src_alignability_rank",
                    "mean",
                ),
                mean_tgt_alignability_rank=(
                    "tgt_alignability_rank",
                    "mean",
                ),
            )
        )

        hist = (
            agg["votes"]
            .value_counts()
            .reindex(range(1, 10), fill_value=0)
        )

        rows = []

        tier_dir = outdir / tier
        tier_dir.mkdir(exist_ok=True)

        for k in range(9, 0, -1):
            raw = agg[agg["votes"] >= k].copy()
            raw["Relation"] = "="
            raw = raw[
                [
                    "SrcEntity",
                    "TgtEntity",
                    "Relation",
                    "votes",
                    "mean_base_score",
                    "mean_src_alignability_rank",
                    "mean_tgt_alignability_rank",
                ]
            ]

            one = greedy_1to1(raw)

            raw.to_csv(
                tier_dir / f"vote_ge_{k}.tsv",
                sep="\t",
                index=False,
            )
            one.to_csv(
                tier_dir / f"vote_ge_{k}_greedy1to1.tsv",
                sep="\t",
                index=False,
            )

            rows.append({
                "pair": pair,
                "tier": tier,
                "k": k,
                "rule": f"vote >= {k}/9",
                "exactly_k_votes": int(hist.loc[k]),
                "raw_predictions": int(len(raw)),
                "greedy1to1_predictions": int(len(one)),
                "gold_size_reference": int(GOLD_REFERENCE[pair]),
                "raw_over_gold": float(
                    len(raw) / GOLD_REFERENCE[pair]
                ),
                "greedy1to1_over_gold": float(
                    len(one) / GOLD_REFERENCE[pair]
                ),
            })

        summary = pd.DataFrame(rows)
        summary.to_csv(
            tier_dir / "CONSENSUS_COUNTS.csv",
            index=False,
        )
        all_summary.append(summary)

        print("=" * 120)
        print(pair, tier)
        print("=" * 120)
        print(summary.to_string(index=False))

    pd.concat(
        all_summary,
        ignore_index=True,
    ).to_csv(
        outdir / "CONSENSUS_COUNTS_ALL_TIERS.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
