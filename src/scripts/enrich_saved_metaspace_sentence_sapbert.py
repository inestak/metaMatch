#!/usr/bin/env python3
"""Add configurable Sentence-SapBERT text-view features to a saved MetaSpace.

The candidate spaces are not changed.  Entity texts are encoded independently
of all gold files, and bidirectional ranks are computed inside each saved
candidate split.  test.tsv is never opened.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from rdflib import Literal, URIRef

from src.config import BIOML_DIR, OUTPUTS_DIR
from src.data.ontology_loader import OntologyLoader
from src.scripts.experiment_saved_metaspace import BASE79_PREFIXES
from src.scripts.run_frozen_train_only_protocol import SAPBERT_FEATURES
from src.scripts.run_sapbert_bidir_baseline import SapBertVectorEncoder
from src.scripts.run_token_overlap_pipeline import infer_src_tgt_files


FEATURES = (
    "sentence_sapbert_cosine",
    "sentence_sapbert_euclidean",
    "sentence_rank_src_tgt",
    "sentence_rank_tgt_src",
    "sentence_min_rank",
    "sentence_max_rank",
    "sentence_abs_rank_difference",
    "sentence_rrf_bidirectional",
    "sentence_mutual_top1",
    "sentence_mutual_top5",
    "sentence_mutual_top10",
    "sentence_gap_from_src_best",
    "sentence_gap_from_tgt_best",
)

TEXT_VIEWS = (
    "label",
    "label_synonyms",
    "label_synonyms_context",
    "label_synonyms_definitions",
    "label_synonyms_definitions_context",
)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def _needed_entities(source_dir: Path) -> Tuple[Set[str], Set[str]]:
    src: Set[str] = set()
    tgt: Set[str] = set()
    for filename in ("train_oof_predictions.csv", "test_candidates_pre_gold.csv"):
        pairs = pd.read_csv(source_dir / filename, usecols=["src_iri", "tgt_iri"])
        src.update(pairs["src_iri"].astype(str))
        tgt.update(pairs["tgt_iri"].astype(str))
    return src, tgt


def _unique(values, limit: int) -> List[str]:
    output: List[str] = []
    seen = set()
    for value in values:
        text = " ".join(str(value).split())
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            output.append(text)
        if limit > 0 and len(output) >= limit:
            break
    return output


def _definition_map(
    ontology: OntologyLoader,
    needed: Set[str],
    max_definitions: int,
) -> Dict[str, List[str]]:
    # For this CLI, zero explicitly disables definitions.  This is useful when
    # only one ontology exposes definition annotations, which would otherwise
    # create an asymmetric semantic representation.
    if max_definitions <= 0:
        return {}
    wanted_suffixes = tuple(
        value.casefold()
        for value in (
            "iao_0000115",
            "#definition",
            "/definition",
            "#description",
            "/description",
            "#scopeNote",
            "/scopeNote",
            # NCIt FULL_SYN/DEFINITION properties use compact P-codes.
            "#p97",
        )
    )
    definitions: Dict[str, List[str]] = {}
    for subject, predicate, value in ontology.graph:
        iri = str(subject)
        if iri not in needed or not isinstance(subject, URIRef) or not isinstance(value, Literal):
            continue
        if not str(predicate).casefold().endswith(wanted_suffixes):
            continue
        text = " ".join(str(value).split())
        if text and text not in definitions.setdefault(iri, []):
            definitions[iri].append(text)
    return {iri: values[:max_definitions] for iri, values in definitions.items()}


def _extra_synonym_map(
    ontology: OntologyLoader,
    needed: Set[str],
    max_synonyms: int,
) -> Dict[str, List[str]]:
    """Extract NCIt synonym/preferred-name annotations missed by the loader."""
    if max_synonyms <= 0:
        return {}
    wanted_suffixes = ("#p90", "#p108")
    synonyms: Dict[str, List[str]] = {}
    for subject, predicate, value in ontology.graph:
        iri = str(subject)
        if iri not in needed or not isinstance(subject, URIRef) or not isinstance(value, Literal):
            continue
        if not str(predicate).casefold().endswith(wanted_suffixes):
            continue
        text = " ".join(str(value).split())
        if text and text not in synonyms.setdefault(iri, []):
            synonyms[iri].append(text)
    return {iri: values[:max_synonyms] for iri, values in synonyms.items()}


def _concept_texts(
    ontology: OntologyLoader,
    iris: List[str],
    max_synonyms: int,
    max_definitions: int,
    max_parents: int,
    max_children: int,
) -> Tuple[List[str], int, int]:
    definitions = _definition_map(ontology, set(iris), max_definitions)
    extra_synonyms = _extra_synonym_map(ontology, set(iris), max_synonyms)
    texts: List[str] = []
    context_count = 0
    for iri in iris:
        labels = _unique(
            list(ontology.get_all_labels(iri)) + extra_synonyms.get(iri, []),
            1 + max_synonyms,
        )
        primary = labels[0] if labels else ontology.get_label(iri)
        pieces = ["Label: " + primary]
        if definitions.get(iri):
            pieces.append("Definition: " + " ".join(definitions[iri]))
        if len(labels) > 1:
            pieces.append("Synonyms: " + "; ".join(labels[1:]))
        parents = _unique(
            (
                ontology.get_label(str(parent))
                for parent in sorted(ontology.get_parents(iri))
            ),
            max_parents,
        ) if max_parents > 0 else []
        children = _unique(
            (
                ontology.get_label(str(child))
                for child in sorted(ontology.get_children(iri))
            ),
            max_children,
        ) if max_children > 0 else []
        if parents:
            pieces.append("Parents: " + "; ".join(parents))
        if children:
            pieces.append("Children: " + "; ".join(children))
        if parents or children:
            context_count += 1
        texts.append(" [SEP] ".join(pieces))
    return texts, len(definitions), context_count


def _load_or_encode(
    ontology: OntologyLoader,
    needed: Set[str],
    cache_path: Path,
    args: argparse.Namespace,
    side: str,
) -> pd.DataFrame:
    cached = pd.read_pickle(cache_path) if cache_path.exists() else pd.DataFrame()
    cached.index = cached.index.astype(str)
    missing = sorted((needed & set(ontology.classes)) - set(cached.index))
    if not missing:
        print(f"Chargement cache Sentence-SapBERT {side}: {cache_path.name}")
        return cached

    texts, definition_count, context_count = _concept_texts(
        ontology,
        missing,
        args.max_synonyms,
        args.max_definitions,
        args.max_parents,
        args.max_children,
    )
    print(
        f"Sentence-SapBERT {side}: {len(missing)} entités à encoder, "
        f"{definition_count} avec définition, {context_count} avec contexte"
    )
    # Keep this enrichment independent from the large experimental BioGITOM
    # pipeline.  The small SapBERT vector encoder is the only semantic encoder
    # required by the transferable protocol.
    encoder = SapBertVectorEncoder(model_name=args.model, device=args.device)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_size = max(args.checkpoint_size, args.batch_size)
    for start in range(0, len(missing), checkpoint_size):
        stop = min(start + checkpoint_size, len(missing))
        pooled = encoder.encode_texts(
            texts[start:stop],
            batch_size=args.batch_size,
            max_length=args.max_tokens,
        )
        new = pd.DataFrame(_normalize_rows(pooled), index=missing[start:stop])
        cached = pd.concat([cached, new]).loc[
            lambda frame: ~frame.index.duplicated(keep="last")
        ]
        cached.to_pickle(cache_path)
        print(
            f"Checkpoint Sentence-SapBERT {side}: {stop}/{len(missing)} "
            f"-> {cache_path.name}"
        )
    return cached


def _pair_features(
    pairs: pd.DataFrame,
    src_embeddings: pd.DataFrame,
    tgt_embeddings: pd.DataFrame,
    rrf_constant: float,
) -> pd.DataFrame:
    left = src_embeddings.reindex(pairs["src_iri"].astype(str)).fillna(0).to_numpy(np.float32)
    right = tgt_embeddings.reindex(pairs["tgt_iri"].astype(str)).fillna(0).to_numpy(np.float32)
    cosine = np.sum(_normalize_rows(left) * _normalize_rows(right), axis=1)
    frame = pd.DataFrame({"sentence_sapbert_cosine": cosine.astype(np.float64)})
    frame["sentence_sapbert_euclidean"] = np.sqrt(
        np.maximum(0.0, 2.0 - 2.0 * frame["sentence_sapbert_cosine"])
    )

    score = frame["sentence_sapbert_cosine"]
    rank_src = score.groupby(pairs["src_iri"].astype(str)).rank(
        method="min", ascending=False
    )
    rank_tgt = score.groupby(pairs["tgt_iri"].astype(str)).rank(
        method="min", ascending=False
    )
    frame["sentence_rank_src_tgt"] = rank_src.to_numpy(dtype=np.float64)
    frame["sentence_rank_tgt_src"] = rank_tgt.to_numpy(dtype=np.float64)
    frame["sentence_min_rank"] = np.minimum(rank_src, rank_tgt).to_numpy(dtype=np.float64)
    frame["sentence_max_rank"] = np.maximum(rank_src, rank_tgt).to_numpy(dtype=np.float64)
    frame["sentence_abs_rank_difference"] = np.abs(rank_src - rank_tgt).to_numpy(dtype=np.float64)
    frame["sentence_rrf_bidirectional"] = (
        1.0 / (rrf_constant + rank_src.to_numpy(dtype=np.float64))
        + 1.0 / (rrf_constant + rank_tgt.to_numpy(dtype=np.float64))
    )
    for k in (1, 5, 10):
        frame[f"sentence_mutual_top{k}"] = (
            (rank_src <= k) & (rank_tgt <= k)
        ).to_numpy(dtype=np.int8)

    src_best = score.groupby(pairs["src_iri"].astype(str)).transform("max")
    tgt_best = score.groupby(pairs["tgt_iri"].astype(str)).transform("max")
    frame["sentence_gap_from_src_best"] = (src_best - score).to_numpy(dtype=np.float64)
    frame["sentence_gap_from_tgt_best"] = (tgt_best - score).to_numpy(dtype=np.float64)
    return frame[list(FEATURES)]


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    source_dir = OUTPUTS_DIR / args.pair / args.input_subdir
    output_dir = OUTPUTS_DIR / args.pair / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((source_dir / "final_results.json").read_text(encoding="utf-8"))
    old_features = list(metadata["features"])

    pair_dir = BIOML_DIR / args.pair
    src_file, tgt_file = infer_src_tgt_files(pair_dir, args.pair)
    src_ontology = OntologyLoader(src_file).load()
    tgt_ontology = OntologyLoader(tgt_file).load()
    needed_src, needed_tgt = _needed_entities(source_dir)
    cache_dir = OUTPUTS_DIR / args.pair / args.cache_subdir
    tag = (
        f"{args.text_view}_syn{args.max_synonyms}_def{args.max_definitions}_"
        f"par{args.max_parents}_chi{args.max_children}_tok{args.max_tokens}"
    )
    src_embeddings = _load_or_encode(
        src_ontology,
        needed_src,
        cache_dir / f"src_sentence_sapbert_{tag}.pkl",
        args,
        "source",
    )
    tgt_embeddings = _load_or_encode(
        tgt_ontology,
        needed_tgt,
        cache_dir / f"tgt_sentence_sapbert_{tag}.pkl",
        args,
        "cible",
    )

    for split, pair_filename in (
        ("train", "train_oof_predictions.csv"),
        ("test", "test_candidates_pre_gold.csv"),
    ):
        pairs = pd.read_csv(source_dir / pair_filename)
        base = pd.read_csv(source_dir / f"{split}_features.csv")
        sentence = _pair_features(pairs, src_embeddings, tgt_embeddings, args.rrf_constant)
        base = base.drop(columns=list(FEATURES), errors="ignore")
        enriched = pd.concat([base.reset_index(drop=True), sentence], axis=1)
        enriched.to_csv(output_dir / f"{split}_features.csv", index=False)
        shutil.copy2(source_dir / pair_filename, output_dir / pair_filename)
        print(f"{split}: {len(pairs)} candidats, {len(enriched.columns)} features")

    final_features = [name for name in old_features if name not in FEATURES] + list(FEATURES)
    if args.expected_feature_count and len(final_features) != args.expected_feature_count:
        raise RuntimeError(
            f"Contrat de features invalide pour {args.text_view}: "
            f"{len(final_features)} au lieu de {args.expected_feature_count}"
        )
    if args.expected_feature_count == 117:
        family_counts = {
            "base79": sum(name.startswith(BASE79_PREFIXES) for name in final_features),
            "nlp": sum(name.startswith("nlp_") for name in final_features),
            "sapbert_bidir": sum(name in SAPBERT_FEATURES for name in final_features),
            "alg": sum(name.startswith("alg_") for name in final_features),
            "sentence_sapbert": sum(name in FEATURES for name in final_features),
        }
        expected = {
            "base79": 79,
            "nlp": 10,
            "sapbert_bidir": 10,
            "alg": 5,
            "sentence_sapbert": 13,
        }
        if family_counts != expected:
            raise RuntimeError(
                f"Décomposition MetaMatch-117 invalide: observe={family_counts}, "
                f"attendu={expected}"
            )
    metadata.update({
        "features": final_features,
        "feature_count": len(final_features),
        "feature_contract": (
            "MetaMatch-117 = base79 + nlp10 + sapbert_bidir10 + alg5 + sentence13"
            if args.expected_feature_count == 117 else metadata.get("feature_contract")
        ),
        "sentence_sapbert_enrichment": {
            "model": args.model,
            "text_view": args.text_view,
            "text": "label + optional synonyms + definitions + parents + children",
            "pooling": "mean",
            "max_synonyms": args.max_synonyms,
            "max_definitions": args.max_definitions,
            "max_parents": args.max_parents,
            "max_children": args.max_children,
            "max_tokens": args.max_tokens,
            "feature_names": list(FEATURES),
            "test_tsv_access": "never",
            "elapsed_seconds": time.time() - started,
        },
        "test_tsv_access": "never",
    })
    (output_dir / "final_results.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"MetaSpace Sentence-SapBERT sauvegardé: {output_dir}")
    print("GUARD: test.tsv n'a jamais été chargé")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", required=True)
    parser.add_argument("--input-subdir", default="frozen_protocol_final")
    parser.add_argument("--output-subdir", default="frozen_protocol_sentence_sapbert")
    parser.add_argument("--cache-subdir", default="sentence_sapbert_shared_cache")
    parser.add_argument("--model", default="sapbert")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-synonyms", type=int, default=4)
    parser.add_argument(
        "--max-definitions", type=int, default=2,
        help="Maximum definitions per entity; 0 disables definitions",
    )
    parser.add_argument("--max-parents", type=int, default=0)
    parser.add_argument("--max-children", type=int, default=0)
    parser.add_argument("--text-view", choices=TEXT_VIEWS, default="label_synonyms_definitions")
    parser.add_argument("--checkpoint-size", type=int, default=512)
    parser.add_argument("--rrf-constant", type=float, default=10.0)
    parser.add_argument(
        "--expected-feature-count",
        type=int,
        default=0,
        help="Fail unless the enriched MetaSpace has exactly this many features",
    )
    args = parser.parse_args()
    expected = {
        "label": (0, 0, 0, 0),
        "label_synonyms": (args.max_synonyms, 0, 0, 0),
        "label_synonyms_context": (
            args.max_synonyms, 0, args.max_parents, args.max_children
        ),
        "label_synonyms_definitions": (
            args.max_synonyms, args.max_definitions, 0, 0
        ),
        "label_synonyms_definitions_context": (
            args.max_synonyms,
            args.max_definitions,
            args.max_parents,
            args.max_children,
        ),
    }[args.text_view]
    (
        args.max_synonyms,
        args.max_definitions,
        args.max_parents,
        args.max_children,
    ) = expected
    run(args)


if __name__ == "__main__":
    main()
