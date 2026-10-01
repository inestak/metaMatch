#!/usr/bin/env python3
"""Utilities for the blind BioML 2026 Union + XGBoost + 403 protocol.

The 403 model features are exactly:

* 79 canonical MetaMatch features;
* 6 ROUGE precision/recall features;
* 5 retrieval/provenance features;
* 313 ontology lexical/structural features (the 45 gold_support_* diagnostics
  from the raw 358-column export are deliberately excluded).

No command in this module accepts or opens a hidden ``test.tsv`` reference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.features.nlp_metrics import compute_nlp_features
from src.scripts.run_full117_deep_forest_train_only import (
    _load_deep,
    build_matches_with_postfilter,
)

PAIR = ["src_iri", "tgt_iri"]
ROUGE6 = (
    "nlp_rouge1_p", "nlp_rouge1_r",
    "nlp_rouge2_p", "nlp_rouge2_r",
    "nlp_rougeL_p", "nlp_rougeL_r",
)
RETRIEVAL5 = (
    "graph06_score", "graph06_cdf", "from_metamatch", "from_graph06", "from_both",
)


def _read_pairs(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",", dtype=str)
    frame = frame.rename(columns={"SrcEntity": "src_iri", "TgtEntity": "tgt_iri", "Score": "score"})
    missing = set(PAIR) - set(frame)
    if missing:
        raise ValueError(f"{path}: missing pair columns {sorted(missing)}")
    frame[PAIR] = frame[PAIR].astype(str)
    return frame


def _scores(path: Path) -> pd.DataFrame:
    frame = _read_pairs(path)
    score = next((c for c in ("score", "oof_score", "probability", "prediction") if c in frame), None)
    if score is None:
        raise ValueError(f"{path}: no score column")
    out = frame[PAIR + [score]].rename(columns={score: "graph06_score"})
    out["graph06_score"] = pd.to_numeric(out.graph06_score, errors="coerce").fillna(0.0)
    return out.groupby(PAIR, as_index=False).graph06_score.max()


def _cdf(values: pd.Series, reference: np.ndarray) -> np.ndarray:
    ref = np.sort(np.asarray(reference, dtype=np.float64))
    raw = pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(np.float64)
    return np.searchsorted(ref, raw, side="right") / max(len(ref), 1)


def build_union(args: argparse.Namespace) -> None:
    meta = Path(args.meta_dir)
    graph = Path(args.graph_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    graph_train = _scores(graph / "validation_predictions.csv")
    graph_unseen = _scores(graph / "test_predictions_pre_gold.csv")
    reference = graph_train.graph06_score.to_numpy(np.float64)
    counts = {}
    for split, meta_name, graph_frame, out_name in (
        ("train", "train_oof_predictions.csv", graph_train, "train_oof_predictions.csv"),
        ("unseen", "test_candidates_pre_gold.csv", graph_unseen, "test_candidates_pre_gold.csv"),
    ):
        meta_frame = _read_pairs(meta / meta_name)[PAIR].drop_duplicates()
        meta_frame["from_metamatch"] = 1
        merged = meta_frame.merge(graph_frame, on=PAIR, how="outer")
        merged["from_metamatch"] = merged.from_metamatch.fillna(0).astype(np.int8)
        merged["from_graph06"] = merged.graph06_score.notna().astype(np.int8)
        merged["from_both"] = (merged.from_metamatch & merged.from_graph06).astype(np.int8)
        merged["graph06_score"] = merged.graph06_score.fillna(0.0)
        merged["graph06_cdf"] = _cdf(merged.graph06_score, reference)
        merged.loc[merged.from_graph06.eq(0), "graph06_cdf"] = 0.0
        merged = merged.sort_values(PAIR).reset_index(drop=True)
        merged.to_csv(output / out_name, index=False)
        counts[split] = {
            "metamatch": int(len(meta_frame)), "graph06": int(len(graph_frame)),
            "common": int(merged.from_both.sum()),
            "metamatch_only": int(((merged.from_metamatch == 1) & (merged.from_graph06 == 0)).sum()),
            "graph06_only": int(((merged.from_metamatch == 0) & (merged.from_graph06 == 1)).sum()),
            "union": int(len(merged)),
        }
    payload = {"protocol": "2026 deduplicated LexSem-Bi + Graph06 union", "counts": counts,
               "retrieval_features": list(RETRIEVAL5), "hidden_test_tsv_access": "never"}
    (output / "union_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


def prepare_work(args: argparse.Namespace) -> None:
    union, shared, work = Path(args.union_dir), Path(args.shared_dir), Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    src_text = pd.read_pickle(shared / "src_texts.pkl").set_index("iri").text.astype(str)
    tgt_text = pd.read_pickle(shared / "tgt_texts.pkl").set_index("iri").text.astype(str)
    gold = _read_pairs(Path(args.train_gold))
    gold_set = set(zip(gold.src_iri, gold.tgt_iri))
    for split, name in (("train", "train_oof_predictions.csv"), ("unseen", "test_candidates_pre_gold.csv")):
        pairs = _read_pairs(union / name)
        pairs["label"] = [int(p in gold_set) if split == "train" else 0 for p in zip(pairs.src_iri, pairs.tgt_iri)]
        pairs["src_label"] = pairs.src_iri.map(src_text).fillna("")
        pairs["tgt_label"] = pairs.tgt_iri.map(tgt_text).fillna("")
        pairs.to_pickle(work / f"{split}_final_candidates.pkl")
    # parallel79_features.merge_features records these manifests.
    blind = {"protocol": "union403 fixed input", "test_tsv_access": "never"}
    (work / "selected_lexical_protocol.json").write_text(json.dumps(blind))
    (work / "selected_semantic_protocol.json").write_text(json.dumps(blind))


def shard_csv(args: argparse.Namespace) -> None:
    pairs = _read_pairs(Path(args.pairs))
    start = (len(pairs) * args.shard_index) // args.num_shards
    end = (len(pairs) * (args.shard_index + 1)) // args.num_shards
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pairs.iloc[start:end].to_csv(out, index=False)
    print(f"shard {args.shard_index + 1}/{args.num_shards}: {end-start} rows")


def merge_deep(args: argparse.Namespace) -> None:
    root, output = Path(args.shard_dir), Path(args.output)
    paths = [root / f"{args.split}_{i:04d}" / "candidate_deep_features.csv" for i in range(args.num_shards)]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"missing deep shards: {missing[:5]}")
    frames = [pd.read_csv(p) for p in paths]
    columns = list(frames[0].columns) if frames else []
    if any(list(f.columns) != columns for f in frames):
        raise ValueError("deep shard contracts differ")
    merged = pd.concat(frames, ignore_index=True)
    if merged.duplicated(PAIR).any():
        raise ValueError("duplicate pairs in merged deep features")
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)
    print(f"{len(merged)} rows -> {output}")


def add_rouge(args: argparse.Namespace) -> None:
    meta, union, shared, output = map(Path, (args.meta79_dir, args.union_dir, args.shared_dir, args.output_dir))
    output.mkdir(parents=True, exist_ok=True)
    src = pd.read_pickle(shared / "src_texts.pkl").set_index("iri").text.astype(str)
    tgt = pd.read_pickle(shared / "tgt_texts.pkl").set_index("iri").text.astype(str)
    for split, pair_name, feature_name in (
        ("train", "train_oof_predictions.csv", "train_features.csv"),
        ("test", "test_candidates_pre_gold.csv", "test_features.csv"),
    ):
        pairs = _read_pairs(union / pair_name)
        features = pd.read_csv(meta / feature_name)
        if len(pairs) != len(features) or len(features.columns) != 79:
            raise ValueError(f"{split}: expected aligned 79-feature matrix")
        rouge = pd.DataFrame([
            {k: compute_nlp_features(src.get(s, ""), tgt.get(t, ""))[k] for k in ROUGE6}
            for s, t in zip(pairs.src_iri, pairs.tgt_iri)
        ], columns=ROUGE6)
        for name in RETRIEVAL5:
            rouge[name] = pd.to_numeric(pairs[name], errors="coerce").fillna(0.0)
        matrix = pd.concat([features.reset_index(drop=True), rouge.reset_index(drop=True)], axis=1)
        if len(matrix.columns) != 90:
            raise RuntimeError(f"{split}: expected 90 pre-deep features, got {len(matrix.columns)}")
        matrix.to_csv(output / feature_name, index=False)
        pairs.to_csv(output / pair_name, index=False)
    (output / "manifest.json").write_text(json.dumps({
        "feature_count": 90, "breakdown": {"socle79": 79, "rouge6": 6, "retrieval5": 5},
        "hidden_test_tsv_access": "never",
    }, indent=2) + "\n")


def _gold(path: Path) -> set[tuple[str, str]]:
    f = _read_pairs(path)
    return set(zip(f.src_iri, f.tgt_iri))


def _matrix(pair_path: Path, meta_path: Path, deep_path: Path):
    pairs = _read_pairs(pair_path).reset_index(drop=True)
    meta = pd.read_csv(meta_path).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if len(meta) != len(pairs) or len(meta.columns) != 90:
        raise ValueError(f"invalid 90-feature matrix {meta_path}")
    deep = _load_deep(deep_path, pairs, str(deep_path))
    if len(deep.columns) != 313:
        raise ValueError(f"expected 313 leakage-free deep features, got {len(deep.columns)}")
    meta.columns = [f"meta__{c}" for c in meta.columns]
    x = pd.concat([meta.reset_index(drop=True), deep.reset_index(drop=True)], axis=1)
    if len(x.columns) != 403:
        raise RuntimeError(f"expected 403 features, got {len(x.columns)}")
    return pairs, x


def _model(y: np.ndarray, workers: int, seed: int):
    pos, neg = max(1, int(y.sum())), max(1, int(len(y) - y.sum()))
    return xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", n_estimators=400,
        max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=1, reg_lambda=1.0, scale_pos_weight=neg / pos,
        tree_method="hist", random_state=seed, n_jobs=workers,
    )


def _metrics(matches: pd.DataFrame, gold: set[tuple[str, str]]) -> dict:
    predicted = set(zip(matches.SrcEntity.astype(str), matches.TgtEntity.astype(str)))
    tp = len(predicted & gold); p = tp / len(predicted) if predicted else 0.0
    r = tp / len(gold) if gold else 0.0
    return {"precision": p, "recall": r, "f1": 2*p*r/(p+r) if p+r else 0.0,
            "tp": tp, "predicted": len(predicted), "reference": len(gold)}


def train_valid(args: argparse.Namespace) -> None:
    pairs, matrix = _matrix(Path(args.train_pairs), Path(args.train_features), Path(args.train_deep))
    unseen, unseen_x = _matrix(Path(args.unseen_pairs), Path(args.unseen_features), Path(args.unseen_deep))
    train_gold, valid_gold = _gold(Path(args.train_gold)), _gold(Path(args.valid_gold))
    y = np.fromiter((int(p in train_gold) for p in zip(pairs.src_iri, pairs.tgt_iri)), dtype=np.int8)
    model = _model(y, args.workers, 2026); model.fit(matrix.to_numpy(np.float32), y)
    scores = model.predict_proba(unseen_x.to_numpy(np.float32))[:, 1]
    scored = unseen.copy(); scored["score"] = scores
    valid_src = {s for s, _ in valid_gold}; valid_tgt = {t for _, t in valid_gold}
    valid = scored[scored.src_iri.isin(valid_src) | scored.tgt_iri.isin(valid_tgt)].reset_index(drop=True)
    candidates = []
    thresholds = np.unique(np.quantile(valid.score, np.linspace(0, 1, 401))) if len(valid) else np.array([0.5])
    for mode in ("mutual_best", "greedy_1to1"):
        for threshold in thresholds:
            matches = build_matches_with_postfilter(valid, float(threshold), mode)
            m = _metrics(matches, valid_gold)
            candidates.append({"post_filter_mode": mode, "threshold": float(threshold), **m})
    selected = max(candidates, key=lambda z: (z["f1"], z["recall"], z["precision"]))
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(candidates).sort_values(["f1", "recall", "precision"], ascending=False).to_csv(output / "public_valid_grid.csv", index=False)
    scored.to_csv(output / "unseen_predictions_pre_gold.csv", index=False)
    pd.DataFrame({"feature": matrix.columns, "importance": model.feature_importances_}).sort_values("importance", ascending=False).to_csv(output / "feature_importance.csv", index=False)
    payload = {"protocol": "Union + fixed XGBoost + 403 features; public-valid selection",
               "feature_count": 403, "train_candidate_count": len(pairs), "train_gold_in_candidates": int(y.sum()),
               "valid_candidate_count": len(valid), "selected": selected, "hidden_test_tsv_access": "never"}
    (output / "selection_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


def refit(args: argparse.Namespace) -> None:
    pairs, matrix = _matrix(Path(args.train_pairs), Path(args.train_features), Path(args.train_deep))
    unseen, unseen_x = _matrix(Path(args.unseen_pairs), Path(args.unseen_features), Path(args.unseen_deep))
    gold = _gold(Path(args.train_gold)); y = np.fromiter((int(p in gold) for p in zip(pairs.src_iri, pairs.tgt_iri)), dtype=np.int8)
    selected = json.loads(Path(args.selection_json).read_text())["selected"]
    model = _model(y, args.workers, 2027); model.fit(matrix.to_numpy(np.float32), y)
    scored = unseen.copy(); scored["score"] = model.predict_proba(unseen_x.to_numpy(np.float32))[:, 1]
    matches = build_matches_with_postfilter(scored, float(selected["threshold"]), str(selected["post_filter_mode"]))
    if matches.SrcEntity.duplicated().any() or matches.TgtEntity.duplicated().any():
        raise RuntimeError("1:1 post-filter contract violated")
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output / "hidden_predictions_pre_gold.csv", index=False)
    matches.to_csv(output / "matches_pre_gold.tsv", sep="\t", index=False)
    submission = matches[["SrcEntity", "TgtEntity"]].copy(); submission["Relation"] = "="
    submission.to_csv(output / "submission.tsv", sep="\t", index=False)
    payload = {"protocol": "Union + fixed XGBoost + 403 features; train+valid refit",
               "feature_count": 403, "train_candidate_count": len(pairs), "train_gold_in_candidates": int(y.sum()),
               "hidden_candidate_count": len(unseen), "alignment_count": len(matches), "selected_public_valid": selected,
               "submission_path": str(output / "submission.tsv"), "hidden_test_tsv_access": "never"}
    (output / "final_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("build-union"); q.add_argument("--meta-dir", required=True); q.add_argument("--graph-dir", required=True); q.add_argument("--output-dir", required=True); q.set_defaults(fn=build_union)
    q = sub.add_parser("prepare-work"); q.add_argument("--union-dir", required=True); q.add_argument("--shared-dir", required=True); q.add_argument("--train-gold", required=True); q.add_argument("--work-dir", required=True); q.set_defaults(fn=prepare_work)
    q = sub.add_parser("shard-csv"); q.add_argument("--pairs", required=True); q.add_argument("--num-shards", type=int, default=128); q.add_argument("--shard-index", type=int, required=True); q.add_argument("--output", required=True); q.set_defaults(fn=shard_csv)
    q = sub.add_parser("merge-deep"); q.add_argument("--shard-dir", required=True); q.add_argument("--split", required=True); q.add_argument("--num-shards", type=int, default=128); q.add_argument("--output", required=True); q.set_defaults(fn=merge_deep)
    q = sub.add_parser("add-rouge"); q.add_argument("--meta79-dir", required=True); q.add_argument("--union-dir", required=True); q.add_argument("--shared-dir", required=True); q.add_argument("--output-dir", required=True); q.set_defaults(fn=add_rouge)
    for name, fn in (("train-valid", train_valid), ("refit", refit)):
        q = sub.add_parser(name)
        for a in ("train-pairs", "unseen-pairs", "train-features", "unseen-features", "train-deep", "unseen-deep", "train-gold", "output-dir"):
            q.add_argument(f"--{a}", required=True)
        q.add_argument("--workers", type=int, default=2)
        if name == "train-valid": q.add_argument("--valid-gold", required=True)
        else: q.add_argument("--selection-json", required=True)
        q.set_defaults(fn=fn)
    args = p.parse_args(); args.fn(args)
    print("GUARD: hidden test.tsv was never opened", flush=True)


if __name__ == "__main__":
    main()