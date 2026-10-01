#!/usr/bin/env python3
"""Select and apply a 2026 protocol with the official public valid.tsv.

Candidate generation and model fitting have already used ``train.tsv`` only.
The saved unlabeled prediction pool is split into:

* a public validation pool: either endpoint belongs to ``valid.tsv``;
* a hidden pool: neither endpoint belongs to ``train.tsv`` or ``valid.tsv``.

Model family, text view, threshold and post-filter are selected exclusively on
the public validation mappings.  The frozen decision is then applied to the
hidden pool.  No hidden ``test.tsv`` is read or expected.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from src.config import BIOML_DIR, OUTPUTS_DIR
from src.scripts.experiment_saved_metaspace import _fixed_decision_protocol
from src.scripts.run_token_overlap_pipeline import build_matches_with_postfilter


def _read_gold(path: Path) -> tuple[pd.DataFrame, set[tuple[str, str]]]:
    frame = pd.read_csv(path, sep="\t")
    required = {"SrcEntity", "TgtEntity"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Colonnes absentes dans {path}: {required - set(frame.columns)}")
    frame = frame.drop_duplicates(["SrcEntity", "TgtEntity"]).copy()
    frame["SrcEntity"] = frame["SrcEntity"].astype(str)
    frame["TgtEntity"] = frame["TgtEntity"].astype(str)
    return frame, set(zip(frame.SrcEntity, frame.TgtEntity))


def _read_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"src_iri", "tgt_iri", "score"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Colonnes absentes dans {path}: {required - set(frame.columns)}")
    frame = frame.drop_duplicates(["src_iri", "tgt_iri"], keep="first").copy()
    frame["src_iri"] = frame["src_iri"].astype(str)
    frame["tgt_iri"] = frame["tgt_iri"].astype(str)
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    return frame.dropna(subset=["score"]).reset_index(drop=True)


def _matches(frame: pd.DataFrame, threshold: float, mode: str) -> pd.DataFrame:
    return build_matches_with_postfilter(
        frame[["src_iri", "tgt_iri", "score"]], threshold, mode
    )


def _best_one_to_one_protocol(
    pairs: pd.DataFrame,
    scores,
    gold: set[tuple[str, str]],
) -> tuple[str, float, dict]:
    """Select only between post-filters that guarantee unique endpoints."""
    best = None
    for mode in ("mutual_best", "greedy_1to1"):
        selected_mode, threshold, metrics = _fixed_decision_protocol(
            pairs, scores, gold, mode, None
        )
        key = (metrics["f1"], metrics["recall"], metrics["precision"])
        if best is None or key > best[0]:
            best = (key, selected_mode, threshold, metrics)
    assert best is not None
    return best[1], best[2], best[3]


def _assert_one_to_one(matches: pd.DataFrame) -> None:
    if matches["SrcEntity"].duplicated().any():
        raise RuntimeError("Violation 1:1: une source apparaît plusieurs fois")
    if matches["TgtEntity"].duplicated().any():
        raise RuntimeError("Violation 1:1: une cible apparaît plusieurs fois")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--entity-mode",
        choices=["classes", "classes_and_properties"],
        default="classes",
    )
    parser.add_argument(
        "--family",
        choices=["auto", "direct_xgboost", "pairwise_reranker"],
        default="auto",
    )
    parser.add_argument("--min-valid-candidate-recall", type=float, default=0.80)
    parser.add_argument(
        "--submission-suffix",
        default="",
        help="Optional suffix inserted before _submission.tsv (for ablations).",
    )
    args = parser.parse_args()
    started = time.time()
    pair = args.pair.upper().replace("_", "-")
    pair_dir = BIOML_DIR / pair
    output_root = OUTPUTS_DIR / pair

    train_df, train_gold = _read_gold(pair_dir / "refs_equiv" / "train.tsv")
    valid_df, valid_gold = _read_gold(pair_dir / "refs_equiv" / "valid.tsv")
    train_src = set(train_df.SrcEntity)
    train_tgt = set(train_df.TgtEntity)
    valid_src = set(valid_df.SrcEntity)
    valid_tgt = set(valid_df.TgtEntity)
    overlap = {
        "source": len(train_src & valid_src),
        "target": len(train_tgt & valid_tgt),
        "pairs": len(train_gold & valid_gold),
    }
    if any(overlap.values()):
        raise ValueError(f"train.tsv et valid.tsv ne sont pas entity-disjoint: {overlap}")

    protocols: list[dict] = []
    for view in ("label_synonyms",):
        sources = {
            "pairwise_reranker": output_root
            / f"union_{view}_ranker_final_{args.run_id}"
            / "test_predictions_pre_gold.csv",
            "direct_xgboost": output_root
            / f"union_{view}_direct_xgb_{args.run_id}"
            / "test_predictions_pre_gold.csv",
        }
        for family, prediction_path in sources.items():
            if args.family != "auto" and family != args.family:
                continue
            if not prediction_path.is_file():
                print(f"INDISPONIBLE {family}/{view}: {prediction_path}")
                continue
            predictions = _read_predictions(prediction_path)
            touches_train = predictions.src_iri.isin(train_src) | predictions.tgt_iri.isin(train_tgt)
            if touches_train.any():
                raise ValueError(
                    f"{family}/{view}: {int(touches_train.sum())} candidats touchent train.tsv"
                )

            valid_mask = predictions.src_iri.isin(valid_src) | predictions.tgt_iri.isin(valid_tgt)
            valid_pool = predictions.loc[valid_mask].reset_index(drop=True)
            hidden_pool = predictions.loc[~valid_mask].reset_index(drop=True)
            valid_candidate_pairs = set(zip(valid_pool.src_iri, valid_pool.tgt_iri))
            valid_covered = len(valid_candidate_pairs & valid_gold)
            valid_candidate_recall = valid_covered / len(valid_gold) if valid_gold else 0.0

            mode, threshold, metrics = _best_one_to_one_protocol(
                valid_pool[["src_iri", "tgt_iri"]],
                valid_pool.score.to_numpy(dtype=float),
                valid_gold,
            )
            protocols.append(
                {
                    "family": family,
                    "view": view,
                    "prediction_path": str(prediction_path),
                    "valid_candidate_count": len(valid_pool),
                    "hidden_candidate_count": len(hidden_pool),
                    "valid_gold_count": len(valid_gold),
                    "valid_gold_covered": valid_covered,
                    "valid_candidate_recall": valid_candidate_recall,
                    "post_filter_mode": mode,
                    "threshold": float(threshold),
                    "valid_precision": float(metrics["precision"]),
                    "valid_recall": float(metrics["recall"]),
                    "valid_f1": float(metrics["f1"]),
                }
            )

    if not protocols:
        raise RuntimeError("Aucun fichier de prédictions terminé n'est disponible")
    eligible = [
        row for row in protocols
        if row["valid_candidate_recall"] >= args.min_valid_candidate_recall
    ]
    selection_pool = eligible or protocols
    recall_warning = None
    if not eligible:
        recall_warning = (
            "Aucun protocole n'atteint le recall candidat validation demandé "
            f"({args.min_valid_candidate_recall:.4f}); sélection du meilleur "
            "protocole disponible sans masquer cet échec."
        )
        print(f"WARNING: {recall_warning}")
    selected = max(
        selection_pool,
        key=lambda row: (
            row["valid_f1"],
            row["valid_recall"],
            row["valid_precision"],
            row["valid_candidate_recall"],
        ),
    )

    selected_predictions = _read_predictions(Path(selected["prediction_path"]))
    hidden_mask = (
        ~selected_predictions.src_iri.isin(train_src | valid_src)
        & ~selected_predictions.tgt_iri.isin(train_tgt | valid_tgt)
    )
    hidden = selected_predictions.loc[hidden_mask].reset_index(drop=True)
    hidden_matches = _matches(
        hidden,
        float(selected["threshold"]),
        str(selected["post_filter_mode"]),
    )
    _assert_one_to_one(hidden_matches)

    output_dir = output_root / f"best_public_valid_protocol_{args.run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    blind_path = output_dir / "alignments_hidden_pre_gold.tsv"
    hidden_matches.to_csv(blind_path, sep="\t", index=False)

    mode_tag = (
        "classes_properties"
        if args.entity_mode == "classes_and_properties"
        else "classes"
    )
    submission_dir = output_root / "submissions"
    submission_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.submission_suffix.strip('_')}" if args.submission_suffix else ""
    submission_path = submission_dir / f"{pair}_{mode_tag}{suffix}_submission.tsv"
    submission = hidden_matches[["SrcEntity", "TgtEntity"]].drop_duplicates().copy()
    submission["Relation"] = "="
    submission.to_csv(submission_path, sep="\t", index=False)

    comparison = pd.DataFrame(protocols).sort_values(
        ["valid_f1", "valid_recall", "valid_precision"], ascending=False
    )
    comparison.to_csv(output_dir / "public_valid_protocol_comparison.csv", index=False)
    payload = {
        "pair": pair,
        "entity_mode": args.entity_mode,
        "train_gold_count": len(train_gold),
        "valid_gold_count": len(valid_gold),
        "public_gold_count": len(train_gold | valid_gold),
        "train_valid_entity_overlap": overlap,
        "selected": selected,
        "min_valid_candidate_recall_requested": args.min_valid_candidate_recall,
        "valid_candidate_recall_floor_met": bool(eligible),
        "warning": recall_warning,
        "candidate_protocols": protocols,
        "hidden_candidate_count": len(hidden),
        "alignment_count": len(submission),
        "alignment_path": str(blind_path),
        "submission_path": str(submission_path),
        "elapsed_seconds": time.time() - started,
        "selection_data": "train-fitted predictions evaluated on public valid.tsv",
        "hidden_test_tsv_access": "never",
    }
    (output_dir / "selection_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(comparison.to_string(index=False))
    print(
        f"SÉLECTION VALID: {selected['family']} / {selected['view']} / "
        f"P={selected['valid_precision']:.4f} R={selected['valid_recall']:.4f} "
        f"F1={selected['valid_f1']:.4f}"
    )
    print(f"ENTITÉS VALID EXCLUES DU CACHÉ: src={len(valid_src)} tgt={len(valid_tgt)}")
    print(f"SOUMISSION: {submission_path} ({len(submission)} alignements)")
    print("GUARD: aucun test.tsv caché n'a été chargé")


if __name__ == "__main__":
    main()
