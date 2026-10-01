#!/usr/bin/env python3
"""Create a non-destructive Bio-ML 2026 view with train+valid as train.tsv."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import pandas as pd


PAIRS = ("NCIT-DOID", "SNOMED-FMA", "SNOMED-NCIT")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _link_children(source: Path, destination: Path, excluded: set[str]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        if child.name in excluded:
            continue
        target = destination / child.name
        if target.exists() or target.is_symlink():
            continue
        try:
            target.symlink_to(child.resolve(), target_is_directory=child.is_dir())
        except FileExistsError:
            # Another independently submitted variant may prepare the same
            # public70 view concurrently. Both links have the same target.
            pass


def prepare_pair(source_root: Path, output_root: Path, pair: str) -> dict:
    source_pair = source_root / pair
    output_pair = output_root / pair
    refs_source = source_pair / "refs_equiv"
    refs_output = output_pair / "refs_equiv"
    train_path = refs_source / "train.tsv"
    valid_path = refs_source / "valid.tsv"
    if not train_path.is_file() or not valid_path.is_file():
        raise FileNotFoundError(f"train.tsv/valid.tsv absents pour {pair}")

    _link_children(source_pair, output_pair, {"refs_equiv"})
    refs_output.mkdir(parents=True, exist_ok=True)
    _link_children(refs_source, refs_output, {"train.tsv", "valid.tsv"})

    train = pd.read_csv(train_path, sep="\t")
    valid = pd.read_csv(valid_path, sep="\t")
    required = {"SrcEntity", "TgtEntity"}
    if not required.issubset(train.columns) or not required.issubset(valid.columns):
        raise ValueError(f"Colonnes SrcEntity/TgtEntity absentes pour {pair}")
    public = pd.concat([train, valid], ignore_index=True).drop_duplicates(
        ["SrcEntity", "TgtEntity"], keep="first"
    )
    output_train = refs_output / "train.tsv"
    temporary = refs_output / f"train.tsv.tmp.{os.getpid()}"
    public.to_csv(temporary, sep="\t", index=False)
    os.replace(temporary, output_train)
    shutil.copy2(valid_path, refs_output / "valid_public_selection_only.tsv")

    manifest = {
        "pair": pair,
        "source_root": str(source_root.resolve()),
        "output_root": str(output_root.resolve()),
        "train_rows": len(train),
        "valid_rows": len(valid),
        "public70_rows": len(public),
        "public70_train_path": str(output_train),
        "public70_train_sha256": _sha256(output_train),
        "hidden_test_access": "never",
    }
    (refs_output / "public70_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pair", default="all")
    args = parser.parse_args()
    requested = args.pair.upper().replace("_", "-")
    pairs = PAIRS if requested == "ALL" else (requested,)
    for pair in pairs:
        if pair not in PAIRS:
            raise ValueError(f"Paire inconnue: {pair}")
        manifest = prepare_pair(args.source_root, args.output_root, pair)
        print(
            f"{pair}: train={manifest['train_rows']} + valid={manifest['valid_rows']} "
            f"=> public70={manifest['public70_rows']}"
        )


if __name__ == "__main__":
    main()
