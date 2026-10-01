#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Features syntaxiques (22 features).

Levenshtein, Jaro-Winkler, Jaccard, etc.
Refactoré depuis compute_features.py de Nour.
"""

import math
import re
import unicodedata
from collections import Counter
from typing import Dict, List, Optional

import numpy as np


# =============================================================================
# Utilitaires de normalisation
# =============================================================================

_RX_NON_ALNUM = re.compile(r"[^0-9a-z]+", flags=re.IGNORECASE)


def strip_accents(s: str) -> str:
    """Retire les accents d'une chaîne."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )


def normalize_text(s: str) -> str:
    """Normalise un texte pour comparaison."""
    if s is None:
        return ""
    s = strip_accents(str(s)).lower().strip()
    s = _RX_NON_ALNUM.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize(s: str) -> List[str]:
    """Tokenise un texte normalisé."""
    s = normalize_text(s)
    return s.split() if s else []


def char_ngrams(s: str, n: int) -> List[str]:
    """Extrait les n-grammes de caractères."""
    s = normalize_text(s).replace(" ", "")
    if n <= 0 or len(s) < n:
        return []
    return [s[i:i+n] for i in range(len(s) - n + 1)]


# =============================================================================
# Similarités ensemblistes
# =============================================================================

def jaccard(a: List, b: List) -> float:
    """Similarité de Jaccard entre deux listes."""
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def dice(a: List, b: List) -> float:
    """Coefficient de Dice entre deux listes."""
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return (2 * len(A & B)) / (len(A) + len(B))


def cosine_counts(a: List, b: List) -> float:
    """Similarité cosinus basée sur les comptages."""
    A, B = Counter(a), Counter(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    dot = sum(A[k] * B.get(k, 0) for k in A)
    na = math.sqrt(sum(v*v for v in A.values()))
    nb = math.sqrt(sum(v*v for v in B.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


# =============================================================================
# Distances d'édition
# =============================================================================

def levenshtein(a: str, b: str) -> int:
    """Distance de Levenshtein (édition)."""
    a, b = a or "", b or ""
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la

    prev = list(range(lb + 1))
    curr = [0] * (lb + 1)

    for i in range(1, la + 1):
        curr[0] = i
        ai = a[i-1]
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j-1] else 1
            curr[j] = min(
                prev[j] + 1,      # deletion
                curr[j-1] + 1,    # insertion
                prev[j-1] + cost  # substitution
            )
        prev, curr = curr, prev

    return prev[lb]


def damerau_osa(a: str, b: str) -> int:
    """Distance de Damerau-Levenshtein (Optimal String Alignment)."""
    a, b = a or "", b or ""
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la

    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        d[i][0] = i
    for j in range(lb + 1):
        d[0][j] = j

    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            d[i][j] = min(
                d[i-1][j] + 1,
                d[i][j-1] + 1,
                d[i-1][j-1] + cost
            )
            if i > 1 and j > 1 and a[i-1] == b[j-2] and a[i-2] == b[j-1]:
                d[i][j] = min(d[i][j], d[i-2][j-2] + 1)

    return d[la][lb]


def lcs_length(a: str, b: str) -> int:
    """Longueur de la plus longue sous-séquence commune (LCS)."""
    a, b = a or "", b or ""
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0

    if la < lb:
        a, b = b, a
        la, lb = lb, la

    prev = [0] * (lb + 1)
    for i in range(1, la + 1):
        curr = [0] * (lb + 1)
        ai = a[i-1]
        for j in range(1, lb + 1):
            if ai == b[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(prev[j], curr[j-1])
        prev = curr

    return prev[lb]


# =============================================================================
# Préfixe/Suffixe communs
# =============================================================================

def common_prefix_len(a: str, b: str) -> int:
    """Longueur du préfixe commun."""
    a, b = a or "", b or ""
    m = min(len(a), len(b))
    i = 0
    while i < m and a[i] == b[i]:
        i += 1
    return i


def common_suffix_len(a: str, b: str) -> int:
    """Longueur du suffixe commun."""
    a, b = a or "", b or ""
    m = min(len(a), len(b))
    i = 0
    while i < m and a[-(i+1)] == b[-(i+1)]:
        i += 1
    return i


# =============================================================================
# Jaro & Jaro-Winkler
# =============================================================================

def jaro(a: str, b: str) -> float:
    """Similarité de Jaro."""
    a, b = a or "", b or ""
    la, lb = len(a), len(b)

    if la == 0 and lb == 0:
        return 1.0
    if la == 0 or lb == 0:
        return 0.0

    match_dist = max(0, (max(la, lb) // 2) - 1)
    a_flags = [False] * la
    b_flags = [False] * lb

    matches = 0
    for i in range(la):
        start = max(0, i - match_dist)
        end = min(i + match_dist + 1, lb)
        for j in range(start, end):
            if not b_flags[j] and a[i] == b[j]:
                a_flags[i] = b_flags[j] = True
                matches += 1
                break

    if matches == 0:
        return 0.0

    transpositions = 0
    k = 0
    for i in range(la):
        if a_flags[i]:
            while not b_flags[k]:
                k += 1
            if a[i] != b[k]:
                transpositions += 1
            k += 1
    transpositions //= 2

    return (matches/la + matches/lb + (matches - transpositions)/matches) / 3.0


def jaro_winkler(a: str, b: str, p: float = 0.1, max_l: int = 4) -> float:
    """Similarité de Jaro-Winkler."""
    ja = jaro(a, b)
    na, nb = a or "", b or ""
    mx = min(max_l, len(na), len(nb))
    l = 0
    while l < mx and na[l] == nb[l]:
        l += 1
    return ja + l * p * (1 - ja)


# =============================================================================
# Calcul de toutes les features syntaxiques
# =============================================================================

SYNTAX_FEATURES = [
    "syn_len_a",
    "syn_len_b",
    "syn_equal_exact",
    "syn_equal_casefold",
    "syn_contains_a_in_b",
    "syn_contains_b_in_a",
    "syn_levenshtein",
    "syn_levenshtein_sim",
    "syn_damerau_osa",
    "syn_damerau_osa_sim",
    "syn_lcs_len",
    "syn_lcs_ratio",
    "syn_common_prefix_ratio",
    "syn_common_suffix_ratio",
    "syn_jaccard_tokens",
    "syn_dice_tokens",
    "syn_jaccard_bigrams",
    "syn_jaccard_trigrams",
    "syn_cosine_bigrams",
    "syn_cosine_trigrams",
    "syn_jaro",
    "syn_jaro_winkler",
]


def compute_syntax_features(name1: str, name2: str) -> Dict[str, float]:
    """
    Calcule toutes les features syntaxiques entre deux labels.

    Args:
        name1: Premier label
        name2: Second label

    Returns:
        Dictionnaire {feature_name: value}
    """
    raw_a = str(name1 or "")
    raw_b = str(name2 or "")
    a = normalize_text(raw_a)
    b = normalize_text(raw_b)

    # Tokens et n-grammes
    tok_a, tok_b = tokenize(raw_a), tokenize(raw_b)
    big_a, big_b = char_ngrams(raw_a, 2), char_ngrams(raw_b, 2)
    tri_a, tri_b = char_ngrams(raw_a, 3), char_ngrams(raw_b, 3)

    maxlen = max(len(a), len(b), 1)

    # Distances
    lev = levenshtein(a, b)
    osa = damerau_osa(a, b)
    lcs = lcs_length(a, b)
    cp = common_prefix_len(a, b)
    cs = common_suffix_len(a, b)

    return {
        "syn_len_a": float(len(a)),
        "syn_len_b": float(len(b)),
        "syn_equal_exact": 1.0 if raw_a == raw_b else 0.0,
        "syn_equal_casefold": 1.0 if raw_a.casefold() == raw_b.casefold() else 0.0,
        "syn_contains_a_in_b": 1.0 if a and a in b else 0.0,
        "syn_contains_b_in_a": 1.0 if b and b in a else 0.0,
        "syn_levenshtein": float(lev),
        "syn_levenshtein_sim": 1.0 - (lev / maxlen),
        "syn_damerau_osa": float(osa),
        "syn_damerau_osa_sim": 1.0 - (osa / maxlen),
        "syn_lcs_len": float(lcs),
        "syn_lcs_ratio": lcs / maxlen,
        "syn_common_prefix_ratio": cp / maxlen,
        "syn_common_suffix_ratio": cs / maxlen,
        "syn_jaccard_tokens": jaccard(tok_a, tok_b),
        "syn_dice_tokens": dice(tok_a, tok_b),
        "syn_jaccard_bigrams": jaccard(big_a, big_b),
        "syn_jaccard_trigrams": jaccard(tri_a, tri_b),
        "syn_cosine_bigrams": cosine_counts(big_a, big_b),
        "syn_cosine_trigrams": cosine_counts(tri_a, tri_b),
        "syn_jaro": jaro(a, b),
        "syn_jaro_winkler": jaro_winkler(a, b),
    }


def compute_syntax_features_batch(
    pairs: List[tuple],
    show_progress: bool = True
) -> List[Dict[str, float]]:
    """
    Calcule les features syntaxiques pour un batch de paires.

    Args:
        pairs: Liste de tuples (label1, label2)
        show_progress: Afficher la progression

    Returns:
        Liste de dictionnaires de features
    """
    from tqdm import tqdm

    iterator = tqdm(pairs, desc="Syntax features") if show_progress else pairs
    return [compute_syntax_features(a, b) for a, b in iterator]


# =============================================================================
# CLI pour tester
# =============================================================================

if __name__ == "__main__":
    # Test rapide
    examples = [
        ("diabetes mellitus", "diabetes mellitus type 2"),
        ("heart failure", "cardiac failure"),
        ("Alzheimer disease", "Alzheimer's disease"),
        ("COVID-19", "coronavirus disease 2019"),
    ]

    for a, b in examples:
        print(f"\n'{a}' vs '{b}':")
        features = compute_syntax_features(a, b)
        for k, v in features.items():
            print(f"  {k}: {v:.4f}")
