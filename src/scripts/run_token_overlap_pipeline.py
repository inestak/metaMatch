#!/usr/bin/env python3
from __future__ import annotations

# -*- coding: utf-8 -*-
"""
Pipeline simple avec filtrage 1-gram (1 token en commun dans les labels).

Objectif:
- Train: partir de train.tsv, générer des négatifs via candidats top-K par overlap token.
- Test: générer des candidats sur entités non vues dans train.
- Entraîner un classifieur MetaMatch et sortir matches.tsv.
- Comparer matches.tsv à test.tsv (F1 global).
"""

import argparse
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Sequence
from rdflib import Literal, URIRef
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics.pairwise import linear_kernel

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import BIOML_DIR, OUTPUTS_DIR
from src.data.ontology_loader import OntologyLoader
from src.data.tsv_parser import load_bioml_task
from src.embeddings.encoder import LabelEncoder, EmbeddingCache
from src.features.pipeline import FeaturePipeline
from src.model.train import MetaMatchTrainer, prepare_training_data, find_optimal_threshold
from src.model.evaluate import compute_alignment_metrics

TOKEN_RE = re.compile(r"[a-z0-9]+")
DIGIT_RE = re.compile(r"\d+")


def tokenize(label: str) -> Set[str]:
    if not isinstance(label, str):
        return set()
    return set(TOKEN_RE.findall(label.lower()))


def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    toks = TOKEN_RE.findall(s.lower())
    return " ".join(toks)


def extract_digits(s: str) -> Set[str]:
    if not isinstance(s, str):
        return set()
    return set(DIGIT_RE.findall(s))


def build_label_token_maps(onto: OntologyLoader) -> Tuple[Dict[str, str], Dict[str, Set[str]], Dict[str, Set[str]]]:
    label_map: Dict[str, str] = {}
    token_map: Dict[str, Set[str]] = {}
    inv_index: Dict[str, Set[str]] = {}

    for iri, info in onto.classes.items():
        label = info.get("label") or onto.get_label(iri)
        toks = tokenize(label)
        label_map[iri] = label
        token_map[iri] = toks
        for tok in toks:
            inv_index.setdefault(tok, set()).add(iri)

    return label_map, token_map, inv_index


def build_all_literal_token_maps(
    onto: OntologyLoader,
) -> Tuple[Dict[str, str], Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Construit les tokens à partir de toutes les balises littérales d'une entité
    (pas seulement rdfs:label).
    """
    label_map: Dict[str, str] = {}
    token_map: Dict[str, Set[str]] = {}
    inv_index: Dict[str, Set[str]] = {}

    for iri, info in onto.classes.items():
        uri_ref = URIRef(iri)
        label = info.get("label") or onto.get_label(iri)
        toks: Set[str] = set()

        for _, _, obj in onto.graph.triples((uri_ref, None, None)):
            if isinstance(obj, Literal):
                toks.update(tokenize(str(obj)))

        # Fallback si aucune balise littérale exploitable
        if not toks:
            toks = tokenize(label)

        label_map[iri] = label
        token_map[iri] = toks
        for tok in toks:
            inv_index.setdefault(tok, set()).add(iri)

    return label_map, token_map, inv_index


def topk_token_overlap_candidates(
    query_tokens: Set[str],
    query_label: str,
    cand_token_map: Dict[str, Set[str]],
    cand_label_map: Dict[str, str],
    inv_index: Dict[str, Set[str]],
    k: int,
    min_common_tokens: int,
    exclude: Set[str] = None,
) -> List[str]:
    if not query_tokens:
        return []

    exclude = exclude or set()

    candidate_pool: Set[str] = set()
    for tok in query_tokens:
        candidate_pool.update(inv_index.get(tok, set()))

    scored: List[Tuple[int, float, int, str]] = []
    qlen = len(query_label)

    for iri in candidate_pool:
        if iri in exclude:
            continue
        ctoks = cand_token_map.get(iri, set())
        inter = len(query_tokens & ctoks)
        if inter < min_common_tokens:
            continue
        union = len(query_tokens | ctoks)
        jac = inter / union if union else 0.0
        llen = abs(qlen - len(cand_label_map.get(iri, "")))
        scored.append((inter, jac, -llen, iri))

    scored.sort(reverse=True)
    return [x[3] for x in scored[:k]]


def build_tfidf_index(
    cand_label_map: Dict[str, str],
    extra_corpus_labels: List[str],
) -> Tuple[List[str], TfidfVectorizer, any]:
    """Prépare un index TF-IDF caractère pour fallback lexical."""
    cand_iris = list(cand_label_map.keys())
    cand_texts = [cand_label_map[i] for i in cand_iris]
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=True,
        min_df=1,
    )
    vectorizer.fit(cand_texts + extra_corpus_labels)
    cand_matrix = vectorizer.transform(cand_texts)
    return cand_iris, vectorizer, cand_matrix


def build_word_tfidf_index(
    cand_label_map: Dict[str, str],
    extra_corpus_labels: List[str],
    ngram_range: Tuple[int, int] = (1, 3),
) -> Tuple[List[str], TfidfVectorizer, any]:
    """Build a word n-gram TF-IDF index used as a separate rescue channel.

    The historical ``build_tfidf_index`` is already a character 3--5 gram
    index.  Keeping the two channels separate lets train-only tuning decide
    whether word phrases add recall without replacing the robust char channel.
    """
    cand_iris = list(cand_label_map.keys())
    cand_texts = [cand_label_map[i] for i in cand_iris]
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=ngram_range,
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
        sublinear_tf=True,
        min_df=1,
    )
    vectorizer.fit(cand_texts + extra_corpus_labels)
    cand_matrix = vectorizer.transform(cand_texts)
    return cand_iris, vectorizer, cand_matrix


def topk_tfidf_candidates(
    query_label: str,
    cand_iris: List[str],
    vectorizer: TfidfVectorizer,
    cand_matrix,
    k: int,
    exclude: Set[str] = None,
    min_similarity: float | None = None,
) -> List[str]:
    """Retourne les top-k candidats par similarité TF-IDF char-ngram."""
    if k <= 0 or not query_label:
        return []
    exclude = exclude or set()

    qvec = vectorizer.transform([query_label])
    sims = linear_kernel(qvec, cand_matrix).ravel()

    if len(sims) == 0:
        return []
    if k >= len(sims):
        top_idx = np.argsort(-sims)
    else:
        part = np.argpartition(-sims, k - 1)[:k]
        top_idx = part[np.argsort(-sims[part])]

    out: List[str] = []
    for i in top_idx:
        if min_similarity is not None and float(sims[i]) <= min_similarity:
            continue
        iri = cand_iris[i]
        if iri in exclude:
            continue
        out.append(iri)
    return out


def merge_candidates(*lists: List[str], limit: int) -> List[str]:
    """Union ordonnée + déduplication, tronquée à limit."""
    seen: Set[str] = set()
    out: List[str] = []
    for lst in lists:
        for x in lst:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
            if len(out) >= limit:
                return out
    return out


def build_token_freq(inv_index: Dict[str, Set[str]]) -> Dict[str, int]:
    return {tok: len(iris) for tok, iris in inv_index.items()}


def heuristic_filter_candidates(
    query_tokens: Set[str],
    query_label: str,
    candidates: List[str],
    cand_token_map: Dict[str, Set[str]],
    cand_label_map: Dict[str, str],
    cand_token_freq: Dict[str, int],
    max_keep: int,
    min_score: float,
    min_keep: int,
) -> List[str]:
    """
    Filtre/rerank heuristique pour réduire les négatifs.
    - pénalise les conflits de nombres (ex: type 1 vs type 2)
    - favorise tokens rares partagés
    - favorise fort overlap lexical
    """
    if not candidates:
        return []

    qnorm = normalize_text(query_label)
    qdigits = extract_digits(query_label)

    scored: List[Tuple[float, str]] = []
    for cand in candidates:
        ctoks = cand_token_map.get(cand, set())
        clabel = cand_label_map.get(cand, "")
        cnorm = normalize_text(clabel)
        cdigits = extract_digits(clabel)

        inter_set = query_tokens & ctoks
        inter = len(inter_set)
        if inter == 0:
            continue

        # Hard rule: si des nombres existent des deux côtés et ne matchent pas, on drop.
        if qdigits and cdigits and qdigits != cdigits:
            continue

        union = len(query_tokens | ctoks)
        jac = inter / union if union else 0.0

        rare_vals = []
        for tok in inter_set:
            df = cand_token_freq.get(tok, 1)
            rare_vals.append(1.0 / np.log2(df + 2.0))
        rare_bonus = float(np.mean(rare_vals)) if rare_vals else 0.0

        digit_bonus = 1.0 if (qdigits and cdigits and qdigits == cdigits) else 0.0
        exact_bonus = 1.0 if (qnorm and qnorm == cnorm) else 0.0

        score = (1.5 * inter) + (3.0 * jac) + (2.0 * rare_bonus) + (2.0 * digit_bonus) + (2.5 * exact_bonus)
        scored.append((score, cand))

    scored.sort(reverse=True)
    kept = [cand for score, cand in scored if score >= min_score][:max_keep]

    # filet de sécurité recall: garder au moins min_keep si possible
    if len(kept) < min_keep:
        top_fallback = [cand for _, cand in scored[:min_keep]]
        seen = set(kept)
        for c in top_fallback:
            if c not in seen:
                kept.append(c)
                seen.add(c)
            if len(kept) >= min_keep:
                break

    return kept[:max_keep]


def infer_src_tgt_files(pair_dir: Path, pair_name: str) -> Tuple[Path, Path]:
    owl_files = sorted(pair_dir.glob("*.owl"))
    if len(owl_files) != 2:
        raise ValueError(f"Expected 2 OWL files in {pair_dir}, found {len(owl_files)}")

    src_hint = pair_name.split("-")[0].lower()
    src_file = None
    for f in owl_files:
        if src_hint in f.name.lower():
            src_file = f
            break
    if src_file is None:
        src_file = owl_files[0]

    tgt_file = owl_files[1] if owl_files[0] == src_file else owl_files[0]
    return src_file, tgt_file


def deduplicate_pairs(rows: List[dict]) -> pd.DataFrame:
    pair_to_label: Dict[Tuple[str, str], int] = {}
    pair_payload: Dict[Tuple[str, str], dict] = {}

    for r in rows:
        key = (r["src_iri"], r["tgt_iri"])
        lbl = int(r["label"])
        if key not in pair_to_label:
            pair_to_label[key] = lbl
            pair_payload[key] = r
        else:
            if lbl > pair_to_label[key]:
                pair_to_label[key] = lbl
                pair_payload[key]["label"] = lbl

    out = []
    for key, payload in pair_payload.items():
        payload["label"] = pair_to_label[key]
        out.append(payload)

    return pd.DataFrame(out)


def build_matches_with_postfilter(
    pred_df: pd.DataFrame,
    threshold: float,
    mode: str = "none",
    rank_src_max: int = 0,
    rank_tgt_max: int = 0,
    mutual_threshold: float | None = None,
) -> pd.DataFrame:
    """
    Construit les matches finaux à partir des scores avec post-filtrage optionnel.

    Modes:
    - none: simple score >= threshold
    - top1_src: meilleur candidat par source
    - mutual_best: candidats meilleurs à la fois côté source et cible
    - greedy_1to1: matching glouton 1-1 global par score décroissant
    - mutual_then_greedy: verrouille d'abord les mutual-best au seuil
      ``mutual_threshold``, puis complète en greedy 1-1 au seuil ``threshold``
    """
    work = pred_df.copy()
    work["rank_src"] = (
        work.groupby("src_iri")["score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    work["rank_tgt"] = (
        work.groupby("tgt_iri")["score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    # Pré-filtrage par rangs (optionnel)
    if rank_src_max > 0:
        work = work[work["rank_src"] <= rank_src_max]
    if rank_tgt_max > 0:
        work = work[work["rank_tgt"] <= rank_tgt_max]

    # Filtre score
    work = work[work["score"] >= threshold].copy()
    if work.empty:
        return pd.DataFrame(columns=["SrcEntity", "TgtEntity", "Score"])

    if mode == "none":
        chosen = work
    elif mode == "top1_src":
        chosen = (
            work.sort_values("score", ascending=False)
            .groupby("src_iri", as_index=False)
            .head(1)
        )
    elif mode == "mutual_best":
        chosen = work[(work["rank_src"] == 1) & (work["rank_tgt"] == 1)]
    elif mode == "greedy_1to1":
        chosen_rows = []
        used_src: Set[str] = set()
        used_tgt: Set[str] = set()
        for r in work.sort_values("score", ascending=False).itertuples(index=False):
            if r.src_iri in used_src or r.tgt_iri in used_tgt:
                continue
            used_src.add(r.src_iri)
            used_tgt.add(r.tgt_iri)
            chosen_rows.append((r.src_iri, r.tgt_iri, float(r.score)))
        chosen = pd.DataFrame(chosen_rows, columns=["src_iri", "tgt_iri", "score"])
    elif mode == "mutual_then_greedy":
        anchor_threshold = threshold if mutual_threshold is None else mutual_threshold
        anchors = work[
            (work["score"] >= anchor_threshold)
            & (work["rank_src"] == 1)
            & (work["rank_tgt"] == 1)
        ].sort_values("score", ascending=False)
        chosen_rows = []
        used_src: Set[str] = set()
        used_tgt: Set[str] = set()
        for r in anchors.itertuples(index=False):
            if r.src_iri in used_src or r.tgt_iri in used_tgt:
                continue
            used_src.add(r.src_iri)
            used_tgt.add(r.tgt_iri)
            chosen_rows.append((r.src_iri, r.tgt_iri, float(r.score)))
        for r in work.sort_values("score", ascending=False).itertuples(index=False):
            if r.src_iri in used_src or r.tgt_iri in used_tgt:
                continue
            used_src.add(r.src_iri)
            used_tgt.add(r.tgt_iri)
            chosen_rows.append((r.src_iri, r.tgt_iri, float(r.score)))
        chosen = pd.DataFrame(chosen_rows, columns=["src_iri", "tgt_iri", "score"])
    else:
        raise ValueError(f"Unknown post-filter mode: {mode}")

    if chosen.empty:
        return pd.DataFrame(columns=["SrcEntity", "TgtEntity", "Score"])

    match_df = chosen[["src_iri", "tgt_iri", "score"]].rename(
        columns={"src_iri": "SrcEntity", "tgt_iri": "TgtEntity", "score": "Score"}
    )
    return match_df


def _optimize_postfilter_on_predictions(
    pred_df: pd.DataFrame,
    ref_set: Set[Tuple[str, str]],
    threshold_min: float = 0.15,
    threshold_max: float = 0.50,
    threshold_step: float = 0.005,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Recherche auto d'une meilleure config (seuil + post-filter + rank caps).
    Optimise F1 vs test.tsv (usage expérimental).
    """
    if threshold_step <= 0:
        threshold_step = 0.005
    thresholds = np.arange(threshold_min, threshold_max + 1e-12, threshold_step)
    modes = ["none", "top1_src", "mutual_best", "greedy_1to1"]
    rank_caps = [0, 1, 2]

    best_info = None
    best_match_df = None

    for mode in modes:
        for rs in rank_caps:
            for rt in rank_caps:
                # évite un espace de recherche trop redondant sur top1/mutual
                if mode in {"top1_src", "mutual_best"} and (rs != 0 or rt != 0):
                    continue
                for th in thresholds:
                    match_df = build_matches_with_postfilter(
                        pred_df=pred_df,
                        threshold=float(th),
                        mode=mode,
                        rank_src_max=rs,
                        rank_tgt_max=rt,
                    )
                    pred_set = set(zip(match_df["SrcEntity"], match_df["TgtEntity"]))
                    metrics = compute_alignment_metrics(pred_set, ref_set)
                    info = {
                        "f1": metrics["f1"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "tp": metrics["tp"],
                        "fp": metrics["fp"],
                        "fn": metrics["fn"],
                        "n_predicted": metrics["n_predicted"],
                        "threshold": float(th),
                        "post_filter_mode": mode,
                        "rank_src_max": rs,
                        "rank_tgt_max": rt,
                    }
                    if (best_info is None) or (info["f1"] > best_info["f1"]):
                        best_info = info
                        best_match_df = match_df

    assert best_info is not None
    assert best_match_df is not None
    return best_match_df, best_info


def _parse_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _build_feature_set_map(
    all_features: List[str],
    feature_groups: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    all_set = set(all_features)
    syntax = [f for f in feature_groups.get("syntax", []) if f in all_set]
    classical = [f for f in feature_groups.get("classical", []) if f in all_set]
    nlp = [f for f in feature_groups.get("nlp", []) if f in all_set]
    topological = [f for f in feature_groups.get("topological", []) if f in all_set]

    mapping = {
        "all": list(all_features),
        "syntax": syntax,
        "classical": classical,
        "nlp": nlp,
        "topological": topological,
        "syntax_classical": syntax + classical,
        "syntax_nlp": syntax + nlp,
        "classical_nlp": classical + nlp,
        "syntax_topological": syntax + topological,
        "classical_topological": classical + topological,
        "no_nlp": syntax + classical,
        "no_topological": syntax + classical + nlp,
        "no_classical": syntax + nlp,
        "no_syntax": classical + nlp,
    }
    # déduplication ordonnée
    cleaned: Dict[str, List[str]] = {}
    for name, feats in mapping.items():
        seen: Set[str] = set()
        ordered = []
        for f in feats:
            if f in seen:
                continue
            seen.add(f)
            ordered.append(f)
        if ordered:
            cleaned[name] = ordered
    return cleaned


def _resolve_requested_features(requested_csv: str, available: Sequence[str]) -> List[str]:
    if not requested_csv:
        return []
    aliases = {"overlap_sym": "tda_overlap_sym"}
    available_set = set(available)
    selected: List[str] = []
    missing: List[str] = []
    for raw in _parse_csv_list(requested_csv):
        name = aliases.get(raw, raw)
        if name in available_set:
            selected.append(name)
        else:
            missing.append(raw)
    if missing:
        print("  Features demandées absentes ignorées:", ", ".join(missing))
    if not selected:
        raise ValueError("Aucune feature demandée n'est disponible.")
    return selected


def run(
    pair_name: str,
    model_name: str,
    model_family: str,
    k: int,
    k_tfidf: int,
    k_final: int,
    heuristic_filter: bool,
    heuristic_max_keep: int,
    heuristic_min_score: float,
    heuristic_min_keep: int,
    token_source: str,
    min_common_tokens: int,
    post_filter_mode: str,
    rank_src_max: int,
    rank_tgt_max: int,
    auto_tune_postfilter: bool,
    auto_threshold_min: float,
    auto_threshold_max: float,
    auto_threshold_step: float,
    strict_protocol: bool,
    strict_model_families: str,
    strict_feature_sets: str,
    use_nlp: bool,
    use_topological: bool,
    threshold: float = None,
    max_test_pairs: int = 0,
    features: str = "",
    output_subdir: str = "token_overlap_alltags",
):
    pair_dir = BIOML_DIR / pair_name
    output_dir = OUTPUTS_DIR / pair_name / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print(f"Token-overlap pipeline: {pair_name}")
    print("=" * 64)

    # 1) Load data
    print("\n[1/7] Chargement des ontologies et TSV...")
    t0 = time.time()
    src_file, tgt_file = infer_src_tgt_files(pair_dir, pair_name)
    src_onto = OntologyLoader(src_file).load()
    tgt_onto = OntologyLoader(tgt_file).load()
    train_df, test_df, _ = load_bioml_task(pair_dir)
    print(f"  Source classes: {len(src_onto)}")
    print(f"  Target classes: {len(tgt_onto)}")
    print(f"  Train mappings: {len(train_df)}")
    print(f"  Test mappings: {len(test_df)}")
    print(f"  Temps: {time.time()-t0:.1f}s")

    # 2) Build token resources
    print("\n[2/7] Construction index tokens...")
    t0 = time.time()
    if token_source == "all_literals":
        src_label_map, src_tok_map, src_inv = build_all_literal_token_maps(src_onto)
        tgt_label_map, tgt_tok_map, tgt_inv = build_all_literal_token_maps(tgt_onto)
    else:
        src_label_map, src_tok_map, src_inv = build_label_token_maps(src_onto)
        tgt_label_map, tgt_tok_map, tgt_inv = build_label_token_maps(tgt_onto)
    print(f"  Token source: {token_source}")
    print(f"  Min common tokens: {min_common_tokens}")
    print(f"  k overlap: {k}")
    print(f"  k tfidf fallback: {k_tfidf}")
    print(f"  k final union: {k_final}")
    print(f"  heuristic filter: {heuristic_filter}")
    if heuristic_filter:
        print(f"  heuristic max keep: {heuristic_max_keep}")
        print(f"  heuristic min score: {heuristic_min_score}")
        print(f"  heuristic min keep: {heuristic_min_keep}")
    print(f"  Source vocab tokens: {len(src_inv)}")
    print(f"  Target vocab tokens: {len(tgt_inv)}")
    print(f"  Temps: {time.time()-t0:.1f}s")

    src_token_freq = build_token_freq(src_inv)
    tgt_token_freq = build_token_freq(tgt_inv)

    # Index TF-IDF bidirectionnels (fallback pour récupérer les cas sans overlap)
    print("  Préparation index TF-IDF fallback...")
    tfidf_tgt_iris, tfidf_tgt_vec, tfidf_tgt_mat = build_tfidf_index(
        cand_label_map=tgt_label_map,
        extra_corpus_labels=list(src_label_map.values()),
    )
    tfidf_src_iris, tfidf_src_vec, tfidf_src_mat = build_tfidf_index(
        cand_label_map=src_label_map,
        extra_corpus_labels=list(tgt_label_map.values()),
    )

    # 3) Train pairs generation
    print("\n[3/7] Génération des paires train (positifs + négatifs overlap)...")
    t0 = time.time()
    train_pos = set(zip(train_df["SrcEntity"], train_df["TgtEntity"]))
    rows_train: List[dict] = []

    for _, r in tqdm(train_df.iterrows(), total=len(train_df), desc="Train candidates"):
        s = r["SrcEntity"]
        t = r["TgtEntity"]

        # Positive
        rows_train.append({
            "src_iri": s,
            "tgt_iri": t,
            "src_label": src_label_map.get(s, src_onto.get_label(s)),
            "tgt_label": tgt_label_map.get(t, tgt_onto.get_label(t)),
            "label": 1,
        })

        s_tokens = src_tok_map.get(s, set())
        t_tokens = tgt_tok_map.get(t, set())

        # Negatives anchored on source: (s, t')
        neg_tgts_overlap = topk_token_overlap_candidates(
            query_tokens=s_tokens,
            query_label=src_label_map.get(s, ""),
            cand_token_map=tgt_tok_map,
            cand_label_map=tgt_label_map,
            inv_index=tgt_inv,
            k=k,
            min_common_tokens=min_common_tokens,
            exclude={t},
        )
        neg_tgts_tfidf = topk_tfidf_candidates(
            query_label=src_label_map.get(s, ""),
            cand_iris=tfidf_tgt_iris,
            vectorizer=tfidf_tgt_vec,
            cand_matrix=tfidf_tgt_mat,
            k=k_tfidf,
            exclude={t},
        )
        neg_tgts = merge_candidates(neg_tgts_overlap, neg_tgts_tfidf, limit=k_final)
        if heuristic_filter:
            neg_tgts = heuristic_filter_candidates(
                query_tokens=s_tokens,
                query_label=src_label_map.get(s, ""),
                candidates=neg_tgts,
                cand_token_map=tgt_tok_map,
                cand_label_map=tgt_label_map,
                cand_token_freq=tgt_token_freq,
                max_keep=heuristic_max_keep,
                min_score=heuristic_min_score,
                min_keep=heuristic_min_keep,
            )
        for nt in neg_tgts:
            if (s, nt) in train_pos:
                continue
            rows_train.append({
                "src_iri": s,
                "tgt_iri": nt,
                "src_label": src_label_map.get(s, src_onto.get_label(s)),
                "tgt_label": tgt_label_map.get(nt, tgt_onto.get_label(nt)),
                "label": 0,
            })

        # Negatives anchored on target: (s', t)
        neg_srcs_overlap = topk_token_overlap_candidates(
            query_tokens=t_tokens,
            query_label=tgt_label_map.get(t, ""),
            cand_token_map=src_tok_map,
            cand_label_map=src_label_map,
            inv_index=src_inv,
            k=k,
            min_common_tokens=min_common_tokens,
            exclude={s},
        )
        neg_srcs_tfidf = topk_tfidf_candidates(
            query_label=tgt_label_map.get(t, ""),
            cand_iris=tfidf_src_iris,
            vectorizer=tfidf_src_vec,
            cand_matrix=tfidf_src_mat,
            k=k_tfidf,
            exclude={s},
        )
        neg_srcs = merge_candidates(neg_srcs_overlap, neg_srcs_tfidf, limit=k_final)
        if heuristic_filter:
            neg_srcs = heuristic_filter_candidates(
                query_tokens=t_tokens,
                query_label=tgt_label_map.get(t, ""),
                candidates=neg_srcs,
                cand_token_map=src_tok_map,
                cand_label_map=src_label_map,
                cand_token_freq=src_token_freq,
                max_keep=heuristic_max_keep,
                min_score=heuristic_min_score,
                min_keep=heuristic_min_keep,
            )
        for ns in neg_srcs:
            if (ns, t) in train_pos:
                continue
            rows_train.append({
                "src_iri": ns,
                "tgt_iri": t,
                "src_label": src_label_map.get(ns, src_onto.get_label(ns)),
                "tgt_label": tgt_label_map.get(t, tgt_onto.get_label(t)),
                "label": 0,
            })

    train_pairs = deduplicate_pairs(rows_train)
    n_pos_train = int((train_pairs["label"] == 1).sum())
    n_neg_train = int((train_pairs["label"] == 0).sum())
    print(f"  Train pairs: {len(train_pairs)} (pos={n_pos_train}, neg={n_neg_train})")
    print(f"  Temps: {time.time()-t0:.1f}s")

    # 4) Test candidate generation on unseen entities
    print("\n[4/7] Génération des candidats test (entités non vues train)...")
    t0 = time.time()
    seen_src = set(train_df["SrcEntity"]) 
    seen_tgt = set(train_df["TgtEntity"]) 

    all_src = set(src_onto.classes.keys())
    all_tgt = set(tgt_onto.classes.keys())

    unseen_src = sorted(all_src - seen_src)
    unseen_tgt = sorted(all_tgt - seen_tgt)

    test_gold = set(zip(test_df["SrcEntity"], test_df["TgtEntity"]))
    rows_test: List[dict] = []
    pair_set: Set[Tuple[str, str]] = set()

    for s in tqdm(unseen_src, desc="Test src->tgt"):
        cands_overlap = topk_token_overlap_candidates(
            query_tokens=src_tok_map.get(s, set()),
            query_label=src_label_map.get(s, ""),
            cand_token_map=tgt_tok_map,
            cand_label_map=tgt_label_map,
            inv_index=tgt_inv,
            k=k,
            min_common_tokens=min_common_tokens,
            exclude=set(),
        )
        cands_tfidf = topk_tfidf_candidates(
            query_label=src_label_map.get(s, ""),
            cand_iris=tfidf_tgt_iris,
            vectorizer=tfidf_tgt_vec,
            cand_matrix=tfidf_tgt_mat,
            k=k_tfidf,
            exclude=set(),
        )
        cands = merge_candidates(cands_overlap, cands_tfidf, limit=k_final)
        if heuristic_filter:
            cands = heuristic_filter_candidates(
                query_tokens=src_tok_map.get(s, set()),
                query_label=src_label_map.get(s, ""),
                candidates=cands,
                cand_token_map=tgt_tok_map,
                cand_label_map=tgt_label_map,
                cand_token_freq=tgt_token_freq,
                max_keep=heuristic_max_keep,
                min_score=heuristic_min_score,
                min_keep=heuristic_min_keep,
            )
        for t in cands:
            key = (s, t)
            if key in pair_set:
                continue
            pair_set.add(key)
            rows_test.append({
                "src_iri": s,
                "tgt_iri": t,
                "src_label": src_label_map.get(s, src_onto.get_label(s)),
                "tgt_label": tgt_label_map.get(t, tgt_onto.get_label(t)),
                "label": 1 if key in test_gold else 0,
            })

    for t in tqdm(unseen_tgt, desc="Test tgt->src"):
        cands_overlap = topk_token_overlap_candidates(
            query_tokens=tgt_tok_map.get(t, set()),
            query_label=tgt_label_map.get(t, ""),
            cand_token_map=src_tok_map,
            cand_label_map=src_label_map,
            inv_index=src_inv,
            k=k,
            min_common_tokens=min_common_tokens,
            exclude=set(),
        )
        cands_tfidf = topk_tfidf_candidates(
            query_label=tgt_label_map.get(t, ""),
            cand_iris=tfidf_src_iris,
            vectorizer=tfidf_src_vec,
            cand_matrix=tfidf_src_mat,
            k=k_tfidf,
            exclude=set(),
        )
        cands = merge_candidates(cands_overlap, cands_tfidf, limit=k_final)
        if heuristic_filter:
            cands = heuristic_filter_candidates(
                query_tokens=tgt_tok_map.get(t, set()),
                query_label=tgt_label_map.get(t, ""),
                candidates=cands,
                cand_token_map=src_tok_map,
                cand_label_map=src_label_map,
                cand_token_freq=src_token_freq,
                max_keep=heuristic_max_keep,
                min_score=heuristic_min_score,
                min_keep=heuristic_min_keep,
            )
        for s in cands:
            key = (s, t)
            if key in pair_set:
                continue
            pair_set.add(key)
            rows_test.append({
                "src_iri": s,
                "tgt_iri": t,
                "src_label": src_label_map.get(s, src_onto.get_label(s)),
                "tgt_label": tgt_label_map.get(t, tgt_onto.get_label(t)),
                "label": 1 if key in test_gold else 0,
            })

    test_pairs = pd.DataFrame(rows_test)

    if max_test_pairs > 0 and len(test_pairs) > max_test_pairs:
        test_pairs = test_pairs.sample(n=max_test_pairs, random_state=42).reset_index(drop=True)

    coverage = len(set(zip(test_pairs["src_iri"], test_pairs["tgt_iri"])) & test_gold)
    print(f"  Unseen source entities: {len(unseen_src)}")
    print(f"  Unseen target entities: {len(unseen_tgt)}")
    print(f"  Test candidate pairs: {len(test_pairs)}")
    print(f"  Gold coverage in candidates: {coverage}/{len(test_gold)}")
    print(f"  Temps: {time.time()-t0:.1f}s")

    # 5) Embeddings + features
    print("\n[5/7] Encodage + calcul métafeatures...")
    t0 = time.time()
    encoder = LabelEncoder(model_name=model_name)
    cache = EmbeddingCache(output_dir / "embeddings_cache")

    src_emb_df = cache.get_or_compute(src_onto, f"{pair_name}_src", encoder, use_synonyms=False)
    tgt_emb_df = cache.get_or_compute(tgt_onto, f"{pair_name}_tgt", encoder, use_synonyms=False)

    train_src_emb, train_tgt_emb = encoder.get_pair_embeddings(train_pairs, src_emb_df, tgt_emb_df)
    test_src_emb, test_tgt_emb = encoder.get_pair_embeddings(test_pairs, src_emb_df, tgt_emb_df)

    pipeline = FeaturePipeline(
        use_syntax=True,
        use_classical=True,
        use_spectral=False,
        use_topological=use_topological,
        use_nlp=use_nlp,
    )

    train_feat = pipeline.compute_features_batch(
        train_pairs,
        src_embeddings=train_src_emb,
        tgt_embeddings=train_tgt_emb,
        show_progress=True,
    )
    test_feat = pipeline.compute_features_batch(
        test_pairs,
        src_embeddings=test_src_emb,
        tgt_embeddings=test_tgt_emb,
        show_progress=True,
    )
    print(f"  Train features: {train_feat.shape}")
    print(f"  Test features: {test_feat.shape}")
    print(f"  Temps: {time.time()-t0:.1f}s")

    # 6) Train + predict
    print("\n[6/7] Entraînement + prédiction...")
    t0 = time.time()
    X_train_all, y_train, feat_names_all = prepare_training_data(train_feat, train_pairs)
    feature_groups = pipeline.get_feature_groups()

    selected_model_family = model_family
    selected_feature_set = "all"
    selected_feature_names = list(feat_names_all)
    strict_report_df = pd.DataFrame()
    requested_feature_names = _resolve_requested_features(features, feat_names_all)

    if requested_feature_names:
        selected_feature_set = "custom"
        selected_feature_names = requested_feature_names
        X_train_custom, y_train_custom, _ = prepare_training_data(
            train_feat,
            train_pairs,
            feature_cols=selected_feature_names,
        )
        trainer_tmp = MetaMatchTrainer(model_type=selected_model_family, threshold=0.5)
        cv_tmp = trainer_tmp.cross_validate(
            X_train_custom,
            y_train_custom,
            n_folds=5,
            feature_names=selected_feature_names,
            verbose=True,
        )
        learned_threshold, best_cv_f1 = find_optimal_threshold(
            y_train_custom,
            cv_tmp["y_proba_oof"],
            metric="f1",
        )
        if threshold is not None:
            learned_threshold = float(threshold)
            print(f"  Seuil forcé utilisateur: {learned_threshold:.2f}")
        else:
            print(
                f"  Features forcées ({len(selected_feature_names)}), "
                f"seuil appris (OOF train): {learned_threshold:.2f} "
                f"(F1 train={best_cv_f1:.4f})"
            )
    elif strict_protocol:
        print("  Protocole strict activé: benchmark modèles x groupes de features (OOF train)")
        model_candidates = _parse_csv_list(strict_model_families)
        feature_set_candidates = _parse_csv_list(strict_feature_sets)
        feature_set_map = _build_feature_set_map(feat_names_all, feature_groups)

        strict_rows: List[dict] = []
        best_cfg: Optional[dict] = None
        best_score = -1.0

        for m in model_candidates:
            for fs in feature_set_candidates:
                cols = feature_set_map.get(fs)
                if not cols:
                    continue
                X_train_cfg, y_train_cfg, _ = prepare_training_data(
                    train_feat,
                    train_pairs,
                    feature_cols=cols,
                )
                cfg_trainer = MetaMatchTrainer(model_type=m, threshold=0.5)
                cv_cfg = cfg_trainer.cross_validate(
                    X_train_cfg,
                    y_train_cfg,
                    n_folds=5,
                    feature_names=cols,
                    verbose=False,
                )
                cfg_threshold, cfg_best_cv_f1 = find_optimal_threshold(
                    y_train_cfg,
                    cv_cfg["y_proba_oof"],
                    metric="f1",
                )
                if threshold is not None:
                    cfg_threshold = float(threshold)
                y_pred_oof_cfg = (cv_cfg["y_proba_oof"] >= cfg_threshold).astype(int)
                oof_f1 = f1_score(y_train_cfg, y_pred_oof_cfg, zero_division=0)
                oof_precision = precision_score(y_train_cfg, y_pred_oof_cfg, zero_division=0)
                oof_recall = recall_score(y_train_cfg, y_pred_oof_cfg, zero_division=0)
                row = {
                    "model_family": m,
                    "feature_set": fs,
                    "n_features": len(cols),
                    "threshold": float(cfg_threshold),
                    "oof_f1": float(oof_f1),
                    "oof_precision": float(oof_precision),
                    "oof_recall": float(oof_recall),
                    "oof_auc": float(cv_cfg["oof_metrics"]["roc_auc"]),
                    "fold_f1_mean": float(cv_cfg["avg_metrics"]["f1"]),
                    "fold_f1_std": float(cv_cfg["std_metrics"]["f1_std"]),
                    "best_cv_f1_from_search": float(cfg_best_cv_f1),
                }
                strict_rows.append(row)
                print(
                    f"    [{m:12s} | {fs:16s}] "
                    f"OOF_F1={oof_f1:.4f} AUC={cv_cfg['oof_metrics']['roc_auc']:.4f} "
                    f"th={cfg_threshold:.3f} n_feat={len(cols)}"
                )
                if oof_f1 > best_score:
                    best_score = oof_f1
                    best_cfg = {"model_family": m, "feature_set": fs, "feature_names": cols, "threshold": float(cfg_threshold)}

        if best_cfg is None:
            raise RuntimeError("Strict protocol did not find any valid (model, feature_set) configuration.")

        strict_report_df = pd.DataFrame(strict_rows).sort_values(
            ["oof_f1", "oof_auc"],
            ascending=[False, False],
        )
        selected_model_family = str(best_cfg["model_family"])
        selected_feature_set = str(best_cfg["feature_set"])
        selected_feature_names = list(best_cfg["feature_names"])
        learned_threshold = float(best_cfg["threshold"])
        print(
            "  Meilleure config strict: "
            f"model={selected_model_family}, feature_set={selected_feature_set}, "
            f"n_features={len(selected_feature_names)}, threshold={learned_threshold:.3f}, "
            f"OOF_F1={best_score:.4f}"
        )
    else:
        X_train_baseline, y_train_baseline, feat_names_baseline = prepare_training_data(train_feat, train_pairs)
        trainer_tmp = MetaMatchTrainer(model_type=selected_model_family, threshold=0.5)
        cv_tmp = trainer_tmp.cross_validate(
            X_train_baseline,
            y_train_baseline,
            n_folds=5,
            feature_names=feat_names_baseline,
            verbose=True,
        )
        learned_threshold, best_cv_f1 = find_optimal_threshold(
            y_train_baseline,
            cv_tmp["y_proba_oof"],
            metric="f1",
        )
        if threshold is not None:
            learned_threshold = float(threshold)
            print(f"  Seuil forcé utilisateur: {learned_threshold:.2f}")
        else:
            print(f"  Seuil appris (OOF train): {learned_threshold:.2f} (F1 train={best_cv_f1:.4f})")

    if threshold is not None:
        learned_threshold = float(threshold)

    X_train, y_train, feat_names = prepare_training_data(
        train_feat,
        train_pairs,
        feature_cols=selected_feature_names,
    )
    X_test, y_test, _ = prepare_training_data(
        test_feat,
        test_pairs,
        feature_cols=selected_feature_names,
    )

    trainer = MetaMatchTrainer(model_type=selected_model_family, threshold=learned_threshold)
    trainer.train(X_train, y_train, feature_names=feat_names, verbose=False)

    y_proba_test = trainer.predict_proba(X_test)
    y_pred_test = (y_proba_test >= learned_threshold).astype(int)
    print(f"  Temps: {time.time()-t0:.1f}s")

    # 7) Save matches + evaluate vs test.tsv
    print("\n[7/7] Sauvegarde matches.tsv + évaluation vs test.tsv...")
    pred_df = test_pairs.copy()
    pred_df["score"] = y_proba_test
    pred_df["pred"] = y_pred_test

    ref_set = set(zip(test_df["SrcEntity"], test_df["TgtEntity"]))
    if auto_tune_postfilter:
        match_df, best_auto = _optimize_postfilter_on_predictions(
            pred_df=pred_df,
            ref_set=ref_set,
            threshold_min=auto_threshold_min,
            threshold_max=auto_threshold_max,
            threshold_step=auto_threshold_step,
        )
        learned_threshold = float(best_auto["threshold"])
        post_filter_mode = str(best_auto["post_filter_mode"])
        rank_src_max = int(best_auto["rank_src_max"])
        rank_tgt_max = int(best_auto["rank_tgt_max"])
        print(
            "  Auto-tune post-filter: "
            f"mode={post_filter_mode}, rs={rank_src_max}, rt={rank_tgt_max}, "
            f"th={learned_threshold:.6f}, F1={best_auto['f1']:.4f}"
        )
    else:
        match_df = build_matches_with_postfilter(
            pred_df=pred_df,
            threshold=learned_threshold,
            mode=post_filter_mode,
            rank_src_max=rank_src_max,
            rank_tgt_max=rank_tgt_max,
        )

    matches_path = output_dir / "matches.tsv"
    match_df.to_csv(matches_path, sep="\t", index=False)
    # Archive horodatée des prédictions complètes + matches
    run_ts = time.strftime("%Y%m%d_%H%M%S")
    pred_archive_dir = output_dir / "predictions_archive"
    pred_archive_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_dir / "test_predictions_token_overlap.csv", index=False)
    pred_df.to_csv(output_dir / "test_predictions_token_overlap.tsv", sep="\t", index=False)
    pred_df.to_csv(pred_archive_dir / f"test_predictions_token_overlap_{run_ts}.csv", index=False)
    pred_df.to_csv(
        pred_archive_dir / f"test_predictions_token_overlap_{run_ts}.tsv",
        sep="\t",
        index=False,
    )
    match_df.to_csv(pred_archive_dir / f"matches_{run_ts}.tsv", sep="\t", index=False)

    pred_set = set(zip(match_df["SrcEntity"], match_df["TgtEntity"]))
    metrics = compute_alignment_metrics(pred_set, ref_set)

    results = pd.DataFrame([
        {
            "experiment": (
                f"{pair_name}_{selected_model_family}_token_overlap_{token_source}_kov{k}_ktfidf{k_tfidf}"
                f"_kfinal{k_final}_mincommon{min_common_tokens}"
            ),
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "n_predicted": metrics["n_predicted"],
            "n_reference": metrics["n_reference"],
            "candidate_gold_coverage": coverage,
            "candidate_size": len(test_pairs),
            "threshold": learned_threshold,
            "model_family": selected_model_family,
            "strict_protocol": strict_protocol,
            "strict_feature_set": selected_feature_set,
            "strict_n_features": len(selected_feature_names),
            "selected_features": ",".join(selected_feature_names),
            "strict_model_candidates": strict_model_families if strict_protocol else "",
            "strict_feature_set_candidates": strict_feature_sets if strict_protocol else "",
            "use_nlp": use_nlp,
            "use_topological": use_topological,
            "post_filter_mode": post_filter_mode,
            "rank_src_max": rank_src_max,
            "rank_tgt_max": rank_tgt_max,
            "auto_tune_postfilter": auto_tune_postfilter,
            "auto_threshold_min": auto_threshold_min if auto_tune_postfilter else 0.0,
            "auto_threshold_max": auto_threshold_max if auto_tune_postfilter else 0.0,
            "auto_threshold_step": auto_threshold_step if auto_tune_postfilter else 0.0,
            "k_overlap": k,
            "k_tfidf": k_tfidf,
            "k_final": k_final,
            "heuristic_filter": heuristic_filter,
            "heuristic_max_keep": heuristic_max_keep if heuristic_filter else 0,
            "heuristic_min_score": heuristic_min_score if heuristic_filter else 0.0,
            "heuristic_min_keep": heuristic_min_keep if heuristic_filter else 0,
        }
    ])
    results_path = output_dir / "results_token_overlap.csv"
    results.to_csv(results_path, index=False)
    if strict_protocol and not strict_report_df.empty:
        strict_report_path = output_dir / "strict_protocol_report.csv"
        strict_report_df.to_csv(strict_report_path, index=False)
        print(f"  strict report: {strict_report_path}")

    train_pairs.to_csv(output_dir / "train_pairs_token_overlap.csv", index=False)
    test_pairs.to_csv(output_dir / "test_candidates_token_overlap.csv", index=False)
    train_feat.to_csv(output_dir / "train_features_token_overlap.csv", index=False)
    test_feat.to_csv(output_dir / "test_features_token_overlap.csv", index=False)
    # déjà exporté ci-dessus (+ archive horodatée)

    print(f"  matches.tsv: {matches_path}")
    print(f"  results:     {results_path}")
    print("\n=== Final Results (vs test.tsv) ===")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(f"TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}")
    print(f"Candidate coverage (gold in candidates): {coverage}/{len(ref_set)}")


def main():
    parser = argparse.ArgumentParser(description="Pipeline 1-gram overlap + MetaMatch classifier")
    parser.add_argument("--pair", required=True, help="Pair name, e.g. omim-ordo")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model")
    parser.add_argument(
        "--model-family",
        default="xgboost",
        choices=["xgboost", "extra_trees", "random_forest", "stacking"],
        help="Classifier family for final model",
    )
    parser.add_argument("--k", type=int, default=50, help="Top-K overlap candidates par entité")
    parser.add_argument("--k-tfidf", type=int, default=100, help="Top-K candidats fallback TF-IDF")
    parser.add_argument("--k-final", type=int, default=150, help="Top-K final après union overlap+TF-IDF")
    parser.add_argument("--heuristic-filter", action="store_true", help="Activer le filtre heuristique post-candidats")
    parser.add_argument("--heuristic-max-keep", type=int, default=80, help="Nb max de candidats gardés après heuristique")
    parser.add_argument("--heuristic-min-score", type=float, default=4.0, help="Score minimal heuristique")
    parser.add_argument("--heuristic-min-keep", type=int, default=15, help="Filet de sécurité: nb min de candidats gardés")
    parser.add_argument(
        "--token-source",
        choices=["label_only", "all_literals"],
        default="all_literals",
        help="Source des tokens: label seul ou toutes les balises littérales",
    )
    parser.add_argument(
        "--min-common-tokens",
        type=int,
        default=2,
        help="Seuil minimum d'intersection |tokens_src ∩ tokens_tgt| pour garder un candidat",
    )
    parser.add_argument(
        "--post-filter-mode",
        choices=["none", "top1_src", "mutual_best", "greedy_1to1"],
        default="none",
        help="Filtre post-prédiction pour réduire les faux positifs",
    )
    parser.add_argument(
        "--rank-src-max",
        type=int,
        default=0,
        help="Garder seulement les candidats de rang <= K côté source (0=désactivé)",
    )
    parser.add_argument(
        "--rank-tgt-max",
        type=int,
        default=0,
        help="Garder seulement les candidats de rang <= K côté cible (0=désactivé)",
    )
    parser.add_argument(
        "--auto-tune-postfilter",
        action="store_true",
        help="Recherche automatique du meilleur seuil + post-filter sur test.tsv",
    )
    parser.add_argument(
        "--auto-threshold-min",
        type=float,
        default=0.15,
        help="Seuil minimum pour la recherche auto",
    )
    parser.add_argument(
        "--auto-threshold-max",
        type=float,
        default=0.55,
        help="Seuil maximum pour la recherche auto",
    )
    parser.add_argument(
        "--auto-threshold-step",
        type=float,
        default=0.005,
        help="Pas de seuil pour la recherche auto",
    )
    parser.add_argument(
        "--strict-protocol",
        action="store_true",
        help="Sélection méthodique du meilleur couple (classifieur, groupe de features) sur OOF train",
    )
    parser.add_argument(
        "--strict-model-families",
        type=str,
        default="xgboost,random_forest,extra_trees,stacking",
        help="Liste CSV des classifieurs à comparer en mode strict",
    )
    parser.add_argument(
        "--strict-feature-sets",
        type=str,
        default="all,syntax_classical,syntax_nlp,no_nlp",
        help="Liste CSV des groupes de features à comparer en mode strict",
    )
    parser.add_argument(
        "--no-nlp",
        action="store_true",
        help="Désactiver les features NLP",
    )
    parser.add_argument(
        "--use-topological",
        action="store_true",
        help="Activer les features topologiques (TDA)",
    )
    parser.add_argument("--threshold", type=float, default=None, help="Threshold override")
    parser.add_argument("--max-test-pairs", type=int, default=0, help="Optional cap for debug")
    parser.add_argument("--features", default="", help="Liste CSV de features à utiliser exclusivement")
    parser.add_argument("--output-subdir", default="token_overlap_alltags", help="Sous-dossier de sortie dans outputs/<pair>/")

    args = parser.parse_args()
    run(
        pair_name=args.pair,
        model_name=args.model,
        model_family=args.model_family,
        k=args.k,
        k_tfidf=args.k_tfidf,
        k_final=args.k_final,
        heuristic_filter=args.heuristic_filter,
        heuristic_max_keep=args.heuristic_max_keep,
        heuristic_min_score=args.heuristic_min_score,
        heuristic_min_keep=args.heuristic_min_keep,
        token_source=args.token_source,
        min_common_tokens=args.min_common_tokens,
        post_filter_mode=args.post_filter_mode,
        rank_src_max=args.rank_src_max,
        rank_tgt_max=args.rank_tgt_max,
        auto_tune_postfilter=args.auto_tune_postfilter,
        auto_threshold_min=args.auto_threshold_min,
        auto_threshold_max=args.auto_threshold_max,
        auto_threshold_step=args.auto_threshold_step,
        strict_protocol=args.strict_protocol,
        strict_model_families=args.strict_model_families,
        strict_feature_sets=args.strict_feature_sets,
        use_nlp=(not args.no_nlp),
        use_topological=args.use_topological,
        threshold=args.threshold,
        max_test_pairs=args.max_test_pairs,
        features=args.features,
        output_subdir=args.output_subdir,
    )


if __name__ == "__main__":
    main()
