#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Métriques NLP: BLEU et ROUGE (10 features).

Ajoutées suite à CR3.
"""

from typing import Dict, List
from collections import Counter
import math

# Import optionnel de rouge_score
try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False


def _get_ngrams(tokens: List[str], n: int) -> Counter:
    """Extrait les n-grammes d'une liste de tokens."""
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def bleu_score(
    reference: str,
    candidate: str,
    max_n: int = 4,
    smoothing: bool = True
) -> float:
    """
    Calcule le score BLEU entre un référence et un candidat.

    Version simplifiée sans corpus, adaptée pour la comparaison de labels.

    Args:
        reference: Texte de référence
        candidate: Texte candidat
        max_n: N-gramme maximum
        smoothing: Appliquer le smoothing pour éviter les zéros

    Returns:
        Score BLEU (0-1)
    """
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()

    if not ref_tokens or not cand_tokens:
        return 0.0

    # Brevity penalty
    ref_len = len(ref_tokens)
    cand_len = len(cand_tokens)

    if cand_len == 0:
        return 0.0

    if cand_len <= ref_len:
        bp = math.exp(1 - ref_len / cand_len)
    else:
        bp = 1.0

    # Précision pour chaque n-gramme
    precisions = []
    for n in range(1, min(max_n, len(cand_tokens)) + 1):
        ref_ngrams = _get_ngrams(ref_tokens, n)
        cand_ngrams = _get_ngrams(cand_tokens, n)

        if not cand_ngrams:
            if smoothing:
                precisions.append(1.0 / (len(cand_tokens) + 1))
            else:
                precisions.append(0.0)
            continue

        # Clipped counts
        clipped = 0
        for ngram, count in cand_ngrams.items():
            clipped += min(count, ref_ngrams.get(ngram, 0))

        total = sum(cand_ngrams.values())

        if smoothing and clipped == 0:
            precisions.append(1.0 / (total + 1))
        else:
            precisions.append(clipped / total if total > 0 else 0.0)

    if not precisions or all(p == 0 for p in precisions):
        return 0.0

    # Moyenne géométrique
    log_precisions = [math.log(p) if p > 0 else -float('inf') for p in precisions]
    avg_log = sum(log_precisions) / len(log_precisions)

    if avg_log == -float('inf'):
        return 0.0

    return bp * math.exp(avg_log)


def rouge_l_score(reference: str, candidate: str) -> Dict[str, float]:
    """
    Calcule le score ROUGE-L (Longest Common Subsequence).

    Args:
        reference: Texte de référence
        candidate: Texte candidat

    Returns:
        Dict avec precision, recall, fmeasure
    """
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()

    if not ref_tokens or not cand_tokens:
        return {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0}

    # LCS dynamique
    m, n = len(ref_tokens), len(cand_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i-1] == cand_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    lcs_len = dp[m][n]

    precision = lcs_len / n if n > 0 else 0.0
    recall = lcs_len / m if m > 0 else 0.0
    fmeasure = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "fmeasure": fmeasure}


def rouge_1_score(reference: str, candidate: str) -> Dict[str, float]:
    """
    Calcule le score ROUGE-1 (unigrams).

    Args:
        reference: Texte de référence
        candidate: Texte candidat

    Returns:
        Dict avec precision, recall, fmeasure
    """
    ref_tokens = set(reference.lower().split())
    cand_tokens = set(candidate.lower().split())

    if not ref_tokens or not cand_tokens:
        return {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0}

    overlap = len(ref_tokens & cand_tokens)

    precision = overlap / len(cand_tokens)
    recall = overlap / len(ref_tokens)
    fmeasure = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "fmeasure": fmeasure}


def rouge_2_score(reference: str, candidate: str) -> Dict[str, float]:
    """
    Calcule le score ROUGE-2 (bigrams).

    Args:
        reference: Texte de référence
        candidate: Texte candidat

    Returns:
        Dict avec precision, recall, fmeasure
    """
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()

    if len(ref_tokens) < 2 or len(cand_tokens) < 2:
        return {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0}

    ref_bigrams = set(zip(ref_tokens[:-1], ref_tokens[1:]))
    cand_bigrams = set(zip(cand_tokens[:-1], cand_tokens[1:]))

    overlap = len(ref_bigrams & cand_bigrams)

    precision = overlap / len(cand_bigrams) if cand_bigrams else 0.0
    recall = overlap / len(ref_bigrams) if ref_bigrams else 0.0
    fmeasure = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "fmeasure": fmeasure}


# =============================================================================
# Liste des features
# =============================================================================

NLP_FEATURES = [
    "nlp_bleu",
    "nlp_rouge1_p",
    "nlp_rouge1_r",
    "nlp_rouge1_f",
    "nlp_rouge2_p",
    "nlp_rouge2_r",
    "nlp_rouge2_f",
    "nlp_rougeL_p",
    "nlp_rougeL_r",
    "nlp_rougeL_f",
]


def compute_nlp_features(text1: str, text2: str) -> Dict[str, float]:
    """
    Calcule toutes les features NLP entre deux textes.

    Args:
        text1: Premier texte
        text2: Second texte

    Returns:
        Dictionnaire de features
    """
    rouge1 = rouge_1_score(text1, text2)
    rouge2 = rouge_2_score(text1, text2)
    rougeL = rouge_l_score(text1, text2)

    return {
        "nlp_bleu": bleu_score(text1, text2),
        "nlp_rouge1_p": rouge1["precision"],
        "nlp_rouge1_r": rouge1["recall"],
        "nlp_rouge1_f": rouge1["fmeasure"],
        "nlp_rouge2_p": rouge2["precision"],
        "nlp_rouge2_r": rouge2["recall"],
        "nlp_rouge2_f": rouge2["fmeasure"],
        "nlp_rougeL_p": rougeL["precision"],
        "nlp_rougeL_r": rougeL["recall"],
        "nlp_rougeL_f": rougeL["fmeasure"],
    }


# =============================================================================
# CLI pour tester
# =============================================================================

if __name__ == "__main__":
    examples = [
        ("diabetes mellitus", "diabetes mellitus type 2"),
        ("heart failure", "cardiac failure"),
        ("Alzheimer disease", "Alzheimer's disease"),
        ("acute myocardial infarction", "heart attack"),
    ]

    for ref, cand in examples:
        print(f"\n'{ref}' vs '{cand}':")
        features = compute_nlp_features(ref, cand)
        for k, v in features.items():
            print(f"  {k}: {v:.4f}")
