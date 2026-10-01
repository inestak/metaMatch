#!/usr/bin/env python3
"""Build the two-table Union403 artifact from a completed pipeline pair."""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--pipeline-project", required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(args.pipeline_project)))
    from src.scripts.union403_2026 import _matrix

    root = Path(args.work_root) / "outputs_bioml2026_union403" / args.pair / "classes"
    holdout = root / "holdout"
    refit = root / "refit"
    hp, hx = _matrix(
        holdout / "union/train_oof_predictions.csv",
        holdout / "meta90/train_features.csv",
        holdout / "deep/train_deep.csv",
    )
    hvp, hvx = _matrix(
        holdout / "union/test_candidates_pre_gold.csv",
        holdout / "meta90/test_features.csv",
        holdout / "deep/unseen_deep.csv",
    )
    rp, rx = _matrix(
        refit / "union/test_candidates_pre_gold.csv",
        refit / "meta90/test_features.csv",
        refit / "deep/unseen_deep.csv",
    )
    # _matrix returns the original union pair table as its first value. That
    # table also carries five retrieval/provenance columns which are already
    # present inside the canonical 403-feature matrix. Keep only pair IDs here
    # to avoid duplicating those five columns in the exported artifact.
    pair_columns = ["src_iri", "tgt_iri"]
    public = pd.concat([
        pd.concat([hp[pair_columns].reset_index(drop=True), hx.reset_index(drop=True)], axis=1),
        pd.concat([hvp[pair_columns].reset_index(drop=True), hvx.reset_index(drop=True)], axis=1),
    ], ignore_index=True).drop_duplicates(["src_iri", "tgt_iri"], keep="first")
    hidden = pd.concat([
        rp[pair_columns].reset_index(drop=True), rx.reset_index(drop=True)
    ], axis=1)
    if len([c for c in public if c not in {"src_iri", "tgt_iri"}]) != 403:
        raise RuntimeError("Public table does not contain exactly 403 features")
    if len([c for c in hidden if c not in {"src_iri", "tgt_iri"}]) != 403:
        raise RuntimeError("Hidden table does not contain exactly 403 features")

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    archive = artifact_dir / f"{args.pair.replace('-', '_')}_Union403_for_Ines.zip"
    temporary = Path(tempfile.mkdtemp(prefix="union403_artifact_", dir=artifact_dir))
    try:
        public_csv = temporary / "train_valid_metaspace_403.csv"
        hidden_csv = temporary / "test_hidden_metaspace_403.csv"
        public.to_csv(public_csv, index=False)
        hidden.to_csv(hidden_csv, index=False)
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as stream:
            stream.write(public_csv, public_csv.name)
            stream.write(hidden_csv, hidden_csv.name)
        with zipfile.ZipFile(archive) as stream:
            corrupt = stream.testzip()
            if corrupt:
                raise RuntimeError(f"Corrupt artifact member: {corrupt}")
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    print(f"Verified artifact: {archive}")


if __name__ == "__main__":
    main()
