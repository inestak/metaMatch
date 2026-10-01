#!/usr/bin/env python3
"""Deep test-aware lexical, annotation and hierarchy analysis for Bio-ML 2025.

The input is a deliberately high-recall candidate pool.  Every candidate is
labelled with the 2025 test reference, then described with lexical n-grams,
all literal annotation predicates, and hierarchy layers up to a configurable
depth.  Outputs are diagnostic/oracle artefacts, never blind predictions.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import AbstractSet, Dict, FrozenSet, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from rdflib import Literal, URIRef
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from src.config import BIOML_DIR
from src.data.ontology_loader import OntologyLoader
from src.scripts.run_token_overlap_pipeline import infer_src_tgt_files


Pair = Tuple[str, str]
TOKEN_RE = re.compile(r"[a-z0-9]+")


def _normalise(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return " ".join(TOKEN_RE.findall(text.casefold()))


def _tokens(value: str) -> FrozenSet[str]:
    return frozenset(TOKEN_RE.findall(value))


def _jaccard(left: AbstractSet[str], right: AbstractSet[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _dice(left: AbstractSet[str], right: AbstractSet[str]) -> float:
    total = len(left) + len(right)
    return 2 * len(left & right) / total if total else 0.0


def _overlap(left: AbstractSet[str], right: AbstractSet[str]) -> float:
    denominator = min(len(left), len(right))
    return len(left & right) / denominator if denominator else 0.0


def _word_ngrams(value: str, n: int) -> Set[str]:
    words = value.split()
    return {" ".join(words[index : index + n]) for index in range(len(words) - n + 1)}


def _char_ngrams(value: str, n: int) -> Set[str]:
    compact = value.replace(" ", "_")
    return {compact[index : index + n] for index in range(len(compact) - n + 1)}


def _acronym(value: str) -> str:
    words = value.split()
    return "".join(word[0] for word in words if word)


def _digits(value: str) -> FrozenSet[str]:
    return frozenset(re.findall(r"\d+", value))


def _local_name(iri: object) -> str:
    value = str(iri)
    local = value.rsplit("#", 1)[-1].rsplit("/", 1)[-1]
    return re.sub(r"[^a-zA-Z0-9]+", "_", local).strip("_").casefold() or "unknown"


def _tag_role(tag: str) -> str:
    value = tag.casefold()
    if any(token in value for token in ("definition", "description", "scope_note", "iao_0000115", "p97")):
        return "definition"
    if any(token in value for token in ("synonym", "altlabel", "full_syn", "p90")):
        return "synonym"
    if any(token in value for token in ("label", "pref", "p108")):
        return "label"
    if any(token in value for token in ("xref", "code", "identifier", "database", "hasdb")):
        return "xref"
    return "other"


@dataclass(frozen=True)
class TextSet:
    values: FrozenSet[str]
    tokens: FrozenSet[str]


EMPTY_TEXT = TextSet(frozenset(), frozenset())


def _text_set(values: Iterable[object], limit: int = 0) -> TextSet:
    normalised: List[str] = []
    seen: Set[str] = set()
    for raw in values:
        value = _normalise(raw)
        if value and value not in seen:
            seen.add(value)
            normalised.append(value)
            if limit > 0 and len(normalised) >= limit:
                break
    return TextSet(frozenset(normalised), frozenset().union(*(_tokens(x) for x in normalised))) if normalised else EMPTY_TEXT


def _text_comparison(left: TextSet, right: TextSet, prefix: str) -> Dict[str, float]:
    return {
        f"{prefix}_exact_count": float(len(left.values & right.values)),
        f"{prefix}_value_jaccard": _jaccard(left.values, right.values),
        f"{prefix}_token_jaccard": _jaccard(left.tokens, right.tokens),
        f"{prefix}_token_dice": _dice(left.tokens, right.tokens),
        f"{prefix}_token_overlap": _overlap(left.tokens, right.tokens),
    }


class Hierarchy:
    def __init__(self, ontology: OntologyLoader, depth: int):
        self.ontology = ontology
        self.depth = depth
        self._up: Dict[str, Tuple[FrozenSet[str], ...]] = {}
        self._down: Dict[str, Tuple[FrozenSet[str], ...]] = {}

    def layers(self, iri: str, direction: str) -> Tuple[FrozenSet[str], ...]:
        cache = self._up if direction == "up" else self._down
        if iri in cache:
            return cache[iri]
        getter = self.ontology.get_parents if direction == "up" else self.ontology.get_children
        visited = {iri}
        frontier = {iri}
        result: List[FrozenSet[str]] = []
        for _ in range(self.depth):
            next_nodes: Set[str] = set()
            for node in frontier:
                next_nodes.update(str(value) for value in getter(node))
            next_nodes -= visited
            result.append(frozenset(next_nodes))
            visited.update(next_nodes)
            frontier = next_nodes
            if not frontier:
                result.extend([frozenset()] * (self.depth - len(result)))
                break
        cache[iri] = tuple(result)
        return cache[iri]

    def siblings(self, iri: str) -> FrozenSet[str]:
        return frozenset(str(value) for value in self.ontology.get_siblings(iri))

    def cousins(self, iri: str) -> FrozenSet[str]:
        parents = set(self.layers(iri, "up")[0])
        parent_siblings: Set[str] = set()
        for parent in parents:
            parent_siblings.update(self.siblings(parent))
        cousins: Set[str] = set()
        for node in parent_siblings:
            cousins.update(str(value) for value in self.ontology.get_children(node))
        cousins.discard(iri)
        cousins -= set(self.siblings(iri))
        return frozenset(cousins)


@dataclass
class EntityProfile:
    iri: str
    label_raw: str
    label: str
    label_tokens: FrozenSet[str]
    labels: TextSet
    literals: TextSet
    role_texts: Dict[str, TextSet]
    tags: FrozenSet[str]
    values_by_tag: Dict[str, FrozenSet[str]]
    relation_nodes: Dict[str, FrozenSet[str]]
    relation_texts: Dict[str, TextSet]


def _extract_annotations(
    ontology: OntologyLoader,
    needed: Set[str],
    max_literals: int,
) -> Tuple[Dict[str, Dict[str, List[str]]], Dict[str, Set[str]], pd.DataFrame]:
    values: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    tags: Dict[str, Set[str]] = defaultdict(set)
    inventory: Counter[Tuple[str, str]] = Counter()
    for subject, predicate, obj in ontology.graph:
        iri = str(subject)
        if iri not in needed or not isinstance(subject, URIRef):
            continue
        tag = _local_name(predicate)
        kind = "literal" if isinstance(obj, Literal) else "uri" if isinstance(obj, URIRef) else "other"
        tags[iri].add(tag)
        inventory[(tag, kind)] += 1
        if isinstance(obj, Literal) and len(values[iri][tag]) < max_literals:
            text = " ".join(str(obj).split())
            if text and text not in values[iri][tag]:
                values[iri][tag].append(text)
    inventory_frame = pd.DataFrame(
        [{"tag": tag, "object_kind": kind, "triple_count": count} for (tag, kind), count in inventory.items()]
    ).sort_values("triple_count", ascending=False)
    return values, tags, inventory_frame


def _relation_groups(hierarchy: Hierarchy, iri: str) -> Dict[str, FrozenSet[str]]:
    up = hierarchy.layers(iri, "up")
    down = hierarchy.layers(iri, "down")
    groups: Dict[str, FrozenSet[str]] = {}
    for index, nodes in enumerate(up, 1):
        groups[f"ancestor_d{index}"] = nodes
    for index, nodes in enumerate(down, 1):
        groups[f"descendant_d{index}"] = nodes
    groups["ancestors_any"] = frozenset().union(*up)
    groups["descendants_any"] = frozenset().union(*down)
    groups["siblings"] = hierarchy.siblings(iri)
    groups["cousins"] = hierarchy.cousins(iri)
    groups["neighbors_any"] = groups["ancestors_any"] | groups["descendants_any"] | groups["siblings"] | groups["cousins"]
    return groups


def _build_profiles(
    ontology: OntologyLoader,
    needed: Set[str],
    depth: int,
    max_literals: int,
    max_related: int,
) -> Tuple[Dict[str, EntityProfile], pd.DataFrame]:
    annotation_values, entity_tags, inventory = _extract_annotations(ontology, needed, max_literals)
    hierarchy = Hierarchy(ontology, depth)
    profiles: Dict[str, EntityProfile] = {}
    for iri in tqdm(sorted(needed), desc=f"Profils {ontology.owl_path.stem}"):
        label_raw = ontology.get_label(iri)
        builtin_labels = list(ontology.get_all_labels(iri))
        by_tag = annotation_values.get(iri, {})
        literal_values = [value for values in by_tag.values() for value in values]
        role_values: Dict[str, List[str]] = defaultdict(list)
        role_values["label"].append(label_raw)
        role_values["synonym"].extend(builtin_labels)
        for tag, values in by_tag.items():
            role_values[_tag_role(tag)].extend(values)
        relation_nodes = _relation_groups(hierarchy, iri)
        relation_texts = {
            name: _text_set(
                (ontology.get_label(node) for node in sorted(nodes)),
                limit=max_related,
            )
            for name, nodes in relation_nodes.items()
        }
        values_by_tag = {
            tag: frozenset(filter(None, (_normalise(value) for value in values)))
            for tag, values in by_tag.items()
        }
        label = _normalise(label_raw)
        profiles[iri] = EntityProfile(
            iri=iri,
            label_raw=label_raw,
            label=label,
            label_tokens=_tokens(label),
            labels=_text_set(builtin_labels, limit=max_literals),
            literals=_text_set(literal_values, limit=max_literals),
            role_texts={role: _text_set(values, limit=max_literals) for role, values in role_values.items()},
            tags=frozenset(entity_tags.get(iri, set())),
            values_by_tag=values_by_tag,
            relation_nodes=relation_nodes,
            relation_texts=relation_texts,
        )
    return profiles, inventory


def _gold_support(
    src_nodes: FrozenSet[str],
    tgt_nodes: FrozenSet[str],
    gold_by_src: Mapping[str, Set[str]],
) -> int:
    return sum(len(gold_by_src.get(node, set()) & tgt_nodes) for node in src_nodes)


def _exact_tag_pairs(src: EntityProfile, tgt: EntityProfile) -> Set[Tuple[str, str]]:
    output: Set[Tuple[str, str]] = set()
    for src_tag, src_values in src.values_by_tag.items():
        if not src_values:
            continue
        for tgt_tag, tgt_values in tgt.values_by_tag.items():
            if src_values & tgt_values:
                output.add((src_tag, tgt_tag))
    return output


def _textset_word_ngrams(texts: TextSet, n: int) -> Set[str]:
    output: Set[str] = set()
    for value in texts.values:
        output.update(_word_ngrams(value, n))
    return output


def _textset_char_ngrams(texts: TextSet, n: int) -> Set[str]:
    output: Set[str] = set()
    for value in texts.values:
        output.update(_char_ngrams(value, n))
    return output


def _ngram_scopes(profile: EntityProfile) -> Mapping[str, TextSet]:
    return {
        "label": _text_set([profile.label]),
        "all_labels": profile.labels,
        "synonyms": profile.role_texts.get("synonym", EMPTY_TEXT),
        "definitions": profile.role_texts.get("definition", EMPTY_TEXT),
        "all_literals": profile.literals,
        "ancestors": profile.relation_texts["ancestors_any"],
        "descendants": profile.relation_texts["descendants_any"],
        "siblings": profile.relation_texts["siblings"],
        "cousins": profile.relation_texts["cousins"],
        "neighbors": profile.relation_texts["neighbors_any"],
    }


def _shared_ngram_signals(
    src: EntityProfile,
    tgt: EntityProfile,
) -> Iterable[Tuple[Tuple[str, str, int], Set[str]]]:
    src_scopes = _ngram_scopes(src)
    tgt_scopes = _ngram_scopes(tgt)
    for scope in src_scopes:
        left = src_scopes[scope]
        right = tgt_scopes[scope]
        for n in range(1, 4):
            yield (
                (scope, "word", n),
                _textset_word_ngrams(left, n) & _textset_word_ngrams(right, n),
            )
        for n in range(3, 7):
            yield (
                (scope, "char", n),
                _textset_char_ngrams(left, n) & _textset_char_ngrams(right, n),
            )


def _tag_presence_signals(src: EntityProfile, tgt: EntityProfile) -> Set[str]:
    signals = {f"source::{tag}" for tag in src.tags}
    signals.update(f"target::{tag}" for tag in tgt.tags)
    signals.update(f"shared::{tag}" for tag in src.tags & tgt.tags)
    return signals


def _pair_features(
    src: EntityProfile,
    tgt: EntityProfile,
    depth: int,
    test_gold_by_src: Mapping[str, Set[str]],
    train_gold_by_src: Mapping[str, Set[str]],
) -> Dict[str, object]:
    left, right = src.label, tgt.label
    left_tokens, right_tokens = src.label_tokens, tgt.label_tokens
    features: Dict[str, object] = {
        "src_label": src.label_raw,
        "tgt_label": tgt.label_raw,
        "label_exact_normalized": float(bool(left) and left == right),
        "label_sequence_ratio": SequenceMatcher(None, left, right).ratio(),
        "label_token_jaccard": _jaccard(left_tokens, right_tokens),
        "label_token_dice": _dice(left_tokens, right_tokens),
        "label_token_overlap": _overlap(left_tokens, right_tokens),
        "label_token_containment": float(bool(left_tokens) and (left_tokens <= right_tokens or right_tokens <= left_tokens)),
        "label_prefix": float(bool(left) and (left.startswith(right) or right.startswith(left))),
        "label_suffix": float(bool(left) and (left.endswith(right) or right.endswith(left))),
        "label_length_difference": float(abs(len(left) - len(right))),
        "label_token_count_difference": float(abs(len(left_tokens) - len(right_tokens))),
        "acronym_exact": float(bool(_acronym(left)) and _acronym(left) == _acronym(right)),
        "digit_exact": float(bool(_digits(left)) and _digits(left) == _digits(right)),
        "tag_count_src": float(len(src.tags)),
        "tag_count_tgt": float(len(tgt.tags)),
        "tag_count_difference": float(abs(len(src.tags) - len(tgt.tags))),
        "tag_common_count": float(len(src.tags & tgt.tags)),
        "tag_jaccard": _jaccard(src.tags, tgt.tags),
    }
    for n in range(2, 7):
        features[f"char_ngram_{n}_jaccard"] = _jaccard(_char_ngrams(left, n), _char_ngrams(right, n))
    for n in range(1, 4):
        features[f"word_ngram_{n}_jaccard"] = _jaccard(_word_ngrams(left, n), _word_ngrams(right, n))
    features.update(_text_comparison(src.labels, tgt.labels, "all_labels"))
    features.update(_text_comparison(src.literals, tgt.literals, "all_literals"))
    for role in ("label", "synonym", "definition", "xref", "other"):
        features.update(
            _text_comparison(
                src.role_texts.get(role, EMPTY_TEXT),
                tgt.role_texts.get(role, EMPTY_TEXT),
                f"role_{role}",
            )
        )

    relation_names = [
        *(f"ancestor_d{index}" for index in range(1, depth + 1)),
        *(f"descendant_d{index}" for index in range(1, depth + 1)),
        "ancestors_any",
        "descendants_any",
        "siblings",
        "cousins",
        "neighbors_any",
    ]
    for name in relation_names:
        src_nodes = src.relation_nodes[name]
        tgt_nodes = tgt.relation_nodes[name]
        features[f"{name}_count_src"] = float(len(src_nodes))
        features[f"{name}_count_tgt"] = float(len(tgt_nodes))
        features[f"{name}_count_difference"] = float(abs(len(src_nodes) - len(tgt_nodes)))
        features.update(_text_comparison(src.relation_texts[name], tgt.relation_texts[name], name))
        features[f"gold_support_{name}"] = float(
            _gold_support(src_nodes, tgt_nodes, test_gold_by_src)
        )
        features[f"train_support_{name}"] = float(
            _gold_support(src_nodes, tgt_nodes, train_gold_by_src)
        )

    # Cross-depth preservation reveals granularity shifts: a parent in one
    # ontology may correspond to a grandparent in the other.
    for src_depth in range(1, depth + 1):
        for tgt_depth in range(1, depth + 1):
            for direction in ("ancestor", "descendant"):
                src_name = f"{direction}_d{src_depth}"
                tgt_name = f"{direction}_d{tgt_depth}"
                prefix = f"cross_{direction}_d{src_depth}_d{tgt_depth}"
                features[prefix + "_token_jaccard"] = _jaccard(
                    src.relation_texts[src_name].tokens,
                    tgt.relation_texts[tgt_name].tokens,
                )
                features[prefix + "_exact_count"] = float(
                    len(src.relation_texts[src_name].values & tgt.relation_texts[tgt_name].values)
                )
                features["gold_support_" + prefix] = float(
                    _gold_support(
                        src.relation_nodes[src_name],
                        tgt.relation_nodes[tgt_name],
                        test_gold_by_src,
                    )
                )
                features["train_support_" + prefix] = float(
                    _gold_support(
                        src.relation_nodes[src_name],
                        tgt.relation_nodes[tgt_name],
                        train_gold_by_src,
                    )
                )

    # Concept-to-context similarities identify broader/narrower mappings.
    concept_src = _text_set([src.label])
    concept_tgt = _text_set([tgt.label])
    for relation in ("ancestors_any", "descendants_any", "siblings", "cousins"):
        features.update(_text_comparison(concept_src, tgt.relation_texts[relation], f"src_concept_to_tgt_{relation}"))
        features.update(_text_comparison(src.relation_texts[relation], concept_tgt, f"src_{relation}_to_tgt_concept"))
    return features


def _discrete_signal_rows(
    true_counter: Counter,
    false_counter: Counter,
    true_total: int,
    false_total: int,
    min_support: int,
    extra: Optional[Mapping[str, object]] = None,
) -> List[dict]:
    rows = []
    for signal in set(true_counter) | set(false_counter):
        positive = true_counter[signal]
        negative = false_counter[signal]
        if positive + negative < min_support:
            continue
        true_rate = positive / max(true_total, 1)
        false_rate = negative / max(false_total, 1)
        log_odds = math.log((positive + 0.5) / (true_total - positive + 0.5)) - math.log(
            (negative + 0.5) / (false_total - negative + 0.5)
        )
        row = {
            "signal": signal,
            "true_count": positive,
            "false_count": negative,
            "true_rate": true_rate,
            "false_rate": false_rate,
            "lift_true_vs_false": true_rate / max(false_rate, 1e-12),
            "log_odds_true": log_odds,
            "support": positive + negative,
        }
        if extra:
            row.update(extra)
        rows.append(row)
    return rows


def _best_prefix(values: np.ndarray, labels: np.ndarray, gold_total: int) -> dict:
    best = None
    for direction, oriented in ((">=", values), ("<=", -values)):
        order = np.argsort(-oriented, kind="mergesort")
        scores = oriented[order]
        truth = labels[order]
        cumulative = np.cumsum(truth)
        tie_end = np.r_[scores[:-1] != scores[1:], True]
        indices = np.flatnonzero(tie_end)
        tp = cumulative[indices]
        predicted = indices + 1
        precision = tp / predicted
        recall = tp / max(gold_total, 1)
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision, dtype=float), where=(precision + recall) > 0)
        idx = int(np.argmax(f1))
        candidate = {
            "direction": direction,
            "threshold": float(scores[indices[idx]] if direction == ">=" else -scores[indices[idx]]),
            "best_f1": float(f1[idx]),
            "best_precision": float(precision[idx]),
            "best_recall": float(recall[idx]),
            "best_count": int(predicted[idx]),
        }
        if best is None or candidate["best_f1"] > best["best_f1"]:
            best = candidate
    return best or {}


def _feature_statistics(frame: pd.DataFrame, gold_total: int) -> pd.DataFrame:
    labels = frame["label"].to_numpy(dtype=np.int8)
    excluded = {"label"}
    rows = []
    for column in frame.select_dtypes(include=[np.number]).columns:
        if column in excluded:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=float)
        true_values = values[labels == 1]
        false_values = values[labels == 0]
        if not len(true_values) or not len(false_values):
            continue
        auc = 0.5
        ap = float(labels.mean())
        if np.unique(values).size > 1:
            auc = float(roc_auc_score(labels, values))
            ap = float(average_precision_score(labels, values))
        pooled_std = float(np.std(values))
        row = {
            "feature": column,
            "true_mean": float(np.mean(true_values)),
            "false_mean": float(np.mean(false_values)),
            "true_median": float(np.median(true_values)),
            "false_median": float(np.median(false_values)),
            "true_nonzero_rate": float(np.mean(true_values != 0)),
            "false_nonzero_rate": float(np.mean(false_values != 0)),
            "standardized_mean_difference": float((np.mean(true_values) - np.mean(false_values)) / max(pooled_std, 1e-12)),
            "roc_auc": auc,
            "roc_auc_discrimination": max(auc, 1 - auc),
            "average_precision": ap,
        }
        row.update(_best_prefix(values, labels, gold_total))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["roc_auc_discrimination", "best_f1"], ascending=False
    )


def _feature_family(name: str) -> str:
    if name.startswith("gold_support_"):
        return "oracle_test_structure_support"
    if name.startswith("train_support_"):
        return "usable_train_structure_support"
    if name.startswith(("present__", "score__", "cdf__", "rank_src__", "rank_tgt__")) or name in {
        "support_count",
        "support_fraction",
        "max_channel_cdf",
        "mean_channel_cdf",
        "min_src_rank",
        "min_tgt_rank",
        "min_bidirectional_rank",
        "high_recall_score",
    }:
        return "retrieval_channels"
    if any(
        token in name
        for token in (
            "ancestor",
            "descendant",
            "siblings",
            "cousins",
            "neighbors",
        )
    ):
        return "hierarchy_lexical"
    if name.startswith(("tag_", "role_", "all_literals")):
        return "annotations_and_tags"
    if name.startswith(("label_", "char_ngram_", "word_ngram_", "all_labels", "acronym_", "digit_")):
        return "concept_lexical"
    return "other"


def _feature_family_summary(feature_statistics: pd.DataFrame) -> pd.DataFrame:
    tagged = feature_statistics.copy()
    tagged["family"] = tagged["feature"].map(_feature_family)
    rows = []
    for family, group in tagged.groupby("family", sort=True):
        best = group.sort_values(
            ["roc_auc_discrimination", "best_f1"], ascending=False
        ).iloc[0]
        rows.append(
            {
                "family": family,
                "feature_count": len(group),
                "best_feature": best["feature"],
                "best_roc_auc_discrimination": best["roc_auc_discrimination"],
                "median_roc_auc_discrimination": group["roc_auc_discrimination"].median(),
                "best_average_precision": group["average_precision"].max(),
                "best_univariate_f1": group["best_f1"].max(),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "best_roc_auc_discrimination", ascending=False
    )


def _oracle_cross_validation(
    frame: pd.DataFrame,
    gold_total: int,
    output_dir: Path,
    folds: int,
    workers: int,
) -> dict:
    numeric = list(frame.select_dtypes(include=[np.number]).columns)
    features = [
        column for column in numeric
        if column != "label" and not column.startswith("gold_support_")
    ]
    X = frame[features].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=np.float32)
    y = frame["label"].to_numpy(dtype=np.int8)
    groups = frame["src_iri"].astype(str).to_numpy()
    n_splits = min(folds, len(np.unique(groups)))
    splitter = GroupKFold(n_splits=n_splits)
    oof = np.zeros(len(frame), dtype=np.float32)
    importance = np.zeros(len(features), dtype=np.float64)
    for fold, (train_index, valid_index) in enumerate(splitter.split(X, y, groups), 1):
        model = ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42 + fold,
            n_jobs=workers,
        )
        model.fit(X[train_index], y[train_index])
        oof[valid_index] = model.predict_proba(X[valid_index])[:, 1]
        importance += model.feature_importances_ / n_splits
    decision = _best_prefix(oof.astype(float), y, gold_total)
    selected = (
        oof >= float(decision["threshold"])
        if decision["direction"] == ">="
        else oof <= float(decision["threshold"])
    )
    predictions = frame.loc[selected, ["src_iri", "tgt_iri"]].copy()
    predictions["Score"] = oof[selected]
    predictions.rename(columns={"src_iri": "SrcEntity", "tgt_iri": "TgtEntity"}).to_csv(
        output_dir / "oracle_cv_best_alignment.tsv", sep="\t", index=False
    )
    scored = frame[["src_iri", "tgt_iri", "label"]].copy()
    scored["oracle_oof_score"] = oof
    scored.to_csv(output_dir / "oracle_cv_predictions.csv", index=False)
    pd.DataFrame({"feature": features, "importance": importance}).sort_values(
        "importance", ascending=False
    ).to_csv(output_dir / "oracle_cv_feature_importance.csv", index=False)
    summary = {
        "warning": "Exploratory GroupKFold trained on labelled 2025 test candidates; not a blind score.",
        "folds": n_splits,
        "feature_count": len(features),
        "candidate_roc_auc": float(roc_auc_score(y, oof)),
        "candidate_average_precision": float(average_precision_score(y, oof)),
        **decision,
    }
    (output_dir / "oracle_cv_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", default="ncit-doid")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--test-gold", type=Path, default=None)
    parser.add_argument("--train-gold", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--max-literals", type=int, default=64)
    parser.add_argument("--max-related", type=int, default=64)
    parser.add_argument("--ngram-min-support", type=int, default=5)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="Échantillon déterministe pour un smoke test; 0 analyse tout le réservoir.",
    )
    parser.add_argument("--skip-missed-gold", action="store_true")
    parser.add_argument("--skip-oracle-cv", action="store_true")
    parser.add_argument(
        "--features-only",
        action="store_true",
        help="Build pair features without opening test gold or producing oracle diagnostics.",
    )
    args = parser.parse_args()

    if args.depth < 1:
        parser.error("--depth doit être >= 1")
    candidates = pd.read_csv(args.candidates)
    required = {"src_iri", "tgt_iri"}
    if not required.issubset(candidates.columns):
        parser.error(f"Candidats: colonnes absentes {required - set(candidates.columns)}")
    if args.max_candidates > 0 and len(candidates) > args.max_candidates:
        candidates = candidates.sample(n=args.max_candidates, random_state=42).reset_index(drop=True)
    if args.test_gold is None:
        gold = set()
    else:
        gold_frame = pd.read_csv(args.test_gold, sep="\t", dtype=str)
        gold = set(zip(gold_frame.SrcEntity.astype(str), gold_frame.TgtEntity.astype(str)))
    train_frame = pd.read_csv(args.train_gold, sep="\t", dtype=str)
    train_gold = set(zip(train_frame.SrcEntity.astype(str), train_frame.TgtEntity.astype(str)))
    pair_index = pd.MultiIndex.from_frame(candidates[["src_iri", "tgt_iri"]])
    if gold:
        gold_index = pd.MultiIndex.from_tuples(
            sorted(gold), names=["src_iri", "tgt_iri"]
        )
        candidates["label"] = pair_index.isin(gold_index).astype(np.int8)
    else:
        # Feature-only blind mode: no test reference is opened and labels are
        # intentionally absent.  The downstream train-only stage derives its
        # training labels from train.tsv and ignores this placeholder column.
        candidates["label"] = np.zeros(len(candidates), dtype=np.int8)
    gold_by_src: Dict[str, Set[str]] = defaultdict(set)
    for src, tgt in gold:
        gold_by_src[src].add(tgt)
    train_gold_by_src: Dict[str, Set[str]] = defaultdict(set)
    for src, tgt in train_gold:
        train_gold_by_src[src].add(tgt)

    pair_dir = BIOML_DIR / args.pair
    src_file, tgt_file = infer_src_tgt_files(pair_dir, args.pair)
    src_onto = OntologyLoader(src_file).load()
    tgt_onto = OntologyLoader(tgt_file).load()
    src_needed = set(candidates.src_iri.astype(str))
    tgt_needed = set(candidates.tgt_iri.astype(str))
    if not args.skip_missed_gold:
        src_needed.update(src for src, _ in gold)
        tgt_needed.update(tgt for _, tgt in gold)
    src_profiles, src_inventory = _build_profiles(
        src_onto, src_needed, args.depth, args.max_literals, args.max_related
    )
    tgt_profiles, tgt_inventory = _build_profiles(
        tgt_onto, tgt_needed, args.depth, args.max_literals, args.max_related
    )

    feature_rows: List[dict] = []
    ngram_true: Dict[Tuple[str, str, int], Counter] = defaultdict(Counter)
    ngram_false: Dict[Tuple[str, str, int], Counter] = defaultdict(Counter)
    tag_true: Counter = Counter()
    tag_false: Counter = Counter()
    tag_presence_true: Counter = Counter()
    tag_presence_false: Counter = Counter()
    for row in tqdm(candidates.itertuples(index=False), total=len(candidates), desc="Paires oracle"):
        src = src_profiles[str(row.src_iri)]
        tgt = tgt_profiles[str(row.tgt_iri)]
        label = int(row.label)
        features = {
            "src_iri": src.iri,
            "tgt_iri": tgt.iri,
            "label": label,
            **_pair_features(src, tgt, args.depth, gold_by_src, train_gold_by_src),
        }
        feature_rows.append(features)
        target_counters = ngram_true if label else ngram_false
        for signal_key, signals in _shared_ngram_signals(src, tgt):
            target_counters[signal_key].update(signals)
        (tag_true if label else tag_false).update(
            f"{left_tag} -> {right_tag}" for left_tag, right_tag in _exact_tag_pairs(src, tgt)
        )
        (tag_presence_true if label else tag_presence_false).update(
            _tag_presence_signals(src, tgt)
        )

    deep = pd.DataFrame(feature_rows)
    base_columns = [column for column in candidates.columns if column not in {"label"}]
    combined = candidates[base_columns + ["label"]].merge(
        deep, on=["src_iri", "tgt_iri", "label"], how="left", validate="one_to_one"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output_dir / "candidate_deep_features.csv", index=False)
    if args.features_only:
        manifest = {
            "pair": args.pair,
            "protocol": "fresh ontology pair features only",
            "candidate_count": len(combined),
            "feature_count": len(combined.columns),
            "hierarchy_depth": args.depth,
            "test_gold_access": "never" if args.test_gold is None else "explicit",
        }
        (args.output_dir / "analysis_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"Features fraîches sauvegardées: {args.output_dir / 'candidate_deep_features.csv'}")
        print(f"GUARD test.tsv: {manifest['test_gold_access']}")
        return
    feature_statistics = _feature_statistics(combined, len(gold))
    feature_statistics.to_csv(args.output_dir / "feature_discrimination.csv", index=False)
    family_summary = _feature_family_summary(feature_statistics)
    family_summary.to_csv(args.output_dir / "feature_family_summary.csv", index=False)

    n_true = int(combined.label.sum())
    n_false = len(combined) - n_true
    ngram_rows: List[dict] = []
    for scope_kind_n in sorted(set(ngram_true) | set(ngram_false)):
        scope, kind, n = scope_kind_n
        ngram_rows.extend(
            _discrete_signal_rows(
                ngram_true[scope_kind_n], ngram_false[scope_kind_n], n_true, n_false,
                args.ngram_min_support, {"scope": scope, "ngram_type": kind, "n": n},
            )
        )
    ngram_statistics = pd.DataFrame(ngram_rows)
    if not ngram_statistics.empty:
        ngram_statistics = ngram_statistics.sort_values(
            ["log_odds_true", "support"], ascending=[False, False]
        )
    ngram_statistics.to_csv(args.output_dir / "shared_ngram_statistics.csv", index=False)
    tag_rows = _discrete_signal_rows(
        tag_true, tag_false, n_true, n_false, args.ngram_min_support
    )
    tag_statistics = pd.DataFrame(tag_rows)
    if not tag_statistics.empty:
        tag_statistics = tag_statistics.sort_values(
            ["log_odds_true", "support"], ascending=[False, False]
        )
    tag_statistics.to_csv(args.output_dir / "exact_annotation_tag_pair_statistics.csv", index=False)
    tag_presence_rows = _discrete_signal_rows(
        tag_presence_true,
        tag_presence_false,
        n_true,
        n_false,
        args.ngram_min_support,
    )
    tag_presence_statistics = pd.DataFrame(tag_presence_rows)
    if not tag_presence_statistics.empty:
        tag_presence_statistics = tag_presence_statistics.sort_values(
            ["log_odds_true", "support"], ascending=[False, False]
        )
    tag_presence_statistics.to_csv(
        args.output_dir / "annotation_tag_presence_statistics.csv", index=False
    )
    src_inventory.assign(side="source").to_csv(args.output_dir / "source_annotation_tag_inventory.csv", index=False)
    tgt_inventory.assign(side="target").to_csv(args.output_dir / "target_annotation_tag_inventory.csv", index=False)

    observable_columns = [
        column for column in combined.select_dtypes(include=[np.number]).columns
        if column != "label" and not column.startswith("gold_support_")
    ]
    similarity_columns = [
        column for column in observable_columns
        if any(token in column for token in ("jaccard", "dice", "overlap", "exact", "sequence_ratio"))
    ]
    weak_signal = combined[similarity_columns].fillna(0).max(axis=1)
    hard_true = combined[combined.label == 1].assign(_weak=weak_signal[combined.label == 1]).sort_values("_weak")
    hard_true.head(500).drop(columns="_weak").to_csv(args.output_dir / "hard_true_matches.csv", index=False)
    deceptive_score = (
        combined.get("all_labels_token_jaccard", 0)
        + combined.get("all_literals_token_jaccard", 0)
        + combined.get("neighbors_any_token_jaccard", 0)
    )
    deceptive = combined[combined.label == 0].assign(_deceptive=deceptive_score[combined.label == 0]).sort_values("_deceptive", ascending=False)
    deceptive.head(500).drop(columns="_deceptive").to_csv(args.output_dir / "deceptive_false_candidates.csv", index=False)

    candidate_pairs = set(zip(candidates.src_iri.astype(str), candidates.tgt_iri.astype(str)))
    missed_rows = []
    if not args.skip_missed_gold:
        for src_iri, tgt_iri in sorted(gold - candidate_pairs):
            src = src_profiles[src_iri]
            tgt = tgt_profiles[tgt_iri]
            missed_rows.append(
                {
                    "SrcEntity": src_iri,
                    "TgtEntity": tgt_iri,
                    **_pair_features(
                        src,
                        tgt,
                        args.depth,
                        gold_by_src,
                        train_gold_by_src,
                    ),
                }
            )
    pd.DataFrame(missed_rows).to_csv(args.output_dir / "missed_gold_deep_analysis.csv", index=False)

    oracle = (
        _oracle_cross_validation(
            combined, len(gold), args.output_dir, args.cv_folds, args.workers
        )
        if not args.skip_oracle_cv
        else {
            "warning": "Oracle CV skipped by command-line option.",
            "candidate_roc_auc": float("nan"),
            "candidate_average_precision": float("nan"),
            "best_f1": float("nan"),
            "best_precision": float("nan"),
            "best_recall": float("nan"),
        }
    )
    manifest = {
        "pair": args.pair,
        "warning": "Explicit 2025 test-aware oracle analysis; not valid as a blind 2026 evaluation.",
        "candidate_count": len(combined),
        "candidate_true_count": n_true,
        "candidate_false_count": n_false,
        "gold_count": len(gold),
        "train_gold_count": len(train_gold),
        "candidate_recall_ceiling": n_true / max(len(gold), 1),
        "deep_feature_count": len(combined.columns),
        "hierarchy_depth": args.depth,
        "max_candidates_option": args.max_candidates,
        "oracle_cv": oracle,
    }
    (args.output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report = [
        "# Analyse oracle lexicale et structurelle — NCIT–DOID 2025",
        "",
        "> Attention : `test.tsv` est utilisé explicitement. Ces résultats servent à comprendre les signaux, pas à annoncer un score aveugle 2026.",
        "",
        f"- Candidats : {len(combined)}",
        f"- Vrais candidats : {n_true}/{len(gold)}",
        f"- Plafond de rappel : {n_true / max(len(gold), 1):.6f}",
        f"- Faux candidats : {n_false}",
        f"- Profondeur hiérarchique : {args.depth}",
        "",
        "## Pouvoir discriminant par famille de signaux",
        "",
        "```",
        family_summary.to_string(index=False),
        "```",
        "",
        "## Validation croisée oracle (sans les features `gold_support_*`)",
        "",
        f"- ROC-AUC candidat : {oracle['candidate_roc_auc']:.6f}",
        f"- Average precision : {oracle['candidate_average_precision']:.6f}",
        f"- Meilleur F1 OOF : {oracle['best_f1']:.6f}",
        f"- Précision/Rappel OOF : {oracle['best_precision']:.6f} / {oracle['best_recall']:.6f}",
        "",
        "## Meilleures features univariées",
        "",
        "```",
        feature_statistics.head(25)[
            ["feature", "true_mean", "false_mean", "roc_auc_discrimination", "best_precision", "best_recall", "best_f1"]
        ].to_string(index=False),
        "```",
    ]
    if not ngram_statistics.empty:
        report.extend(
            [
                "",
                "## N-grammes partagés les plus associés aux vrais matchs",
                "",
                "```",
                ngram_statistics.head(30)[
                    ["scope", "ngram_type", "n", "signal", "true_count", "false_count", "lift_true_vs_false", "log_odds_true"]
                ].to_string(index=False),
                "```",
            ]
        )
    if not tag_statistics.empty:
        report.extend(
            [
                "",
                "## Paires de balises avec valeurs littérales identiques",
                "",
                "```",
                tag_statistics.head(30)[
                    ["signal", "true_count", "false_count", "lift_true_vs_false", "log_odds_true"]
                ].to_string(index=False),
                "```",
            ]
        )
    (args.output_dir / "oracle_analysis_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print("\nANALYSE ORACLE TERMINÉE")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Résultats: {args.output_dir}")


if __name__ == "__main__":
    main()
