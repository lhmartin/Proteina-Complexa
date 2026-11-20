"""Stage 7 - assemble the train / val / test metadata parquets.

`StructureDataModule` takes a parquet of metadata and a directory of structure
files it points at, and nothing else.  The columns that matter are `path` and
`example_id`, matching `path_column` / `id_column` in
`configs/dataset/unified/ted_dimers.yaml`.

Three files, two of which training consumes: `metadata_file` points at train
and `val_metadata_file` at val.  That path exists (`setup`,
`structure_data.py:810`) and is the only one that gives a deterministic split --
with `val_metadata_file` unset the module splits positionally on
`iloc[:n_train]`, so the split is whatever order the rows happen to be in.
`ted_dimers.yaml` relies on exactly that with `train_split: 0.99`, which is fine
for a million-row corpus and not fine for a few hundred antibody complexes.

There is no test dataloader: the module defines `train_dataloader` and
`val_dataloader` and nothing else, so `test.parquet` is not consumed by training
at all.  It is the input to the generation and evaluation path, which is why it
is written even though nothing in training reads it.

    python -m script_utils.abdata.s7_parquet
    python -m script_utils.abdata.s7_parquet --split-mode temporal
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from loguru import logger

from script_utils.abdata.config import (
    BuildConfig,
    add_common_args,
    config_from_args,
    record_stage,
)


def _random_split(df: pd.DataFrame, cfg: BuildConfig
                  ) -> dict[str, pd.DataFrame]:
    """Seeded random split, with the pinned entries forced into train.

    Pinned entries exist to connect the slice to the probe scripts, which means
    they have to be in the set the model actually trains on.
    """
    pinned = df[df["is_pinned"].astype(str).str.lower() == "true"]
    rest = df.drop(index=pinned.index)
    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(len(rest))
    rest = rest.iloc[order].reset_index(drop=True)

    n_val = min(cfg.n_val, len(rest))
    n_test = min(cfg.n_test, max(0, len(rest) - n_val))
    val = rest.iloc[:n_val]
    test = rest.iloc[n_val:n_val + n_test]
    train = pd.concat([pinned, rest.iloc[n_val + n_test:]], ignore_index=True)
    return {"train": train, "val": val, "test": test}


def _temporal_split(df: pd.DataFrame, cfg: BuildConfig
                    ) -> dict[str, pd.DataFrame]:
    """Split on RCSB deposition date, not release date.

    Deposition is the leak-relevant date: a structure deposited before the
    pretraining cut-off may have been released after it, and the pretrained
    weights may still have seen a related entry.
    """
    dep = pd.to_datetime(df["deposit_date"], errors="coerce")
    undated = int(dep.isna().sum())
    if undated:
        logger.warning(f"{undated} examples have no deposition date and go to train")
    val_from = pd.Timestamp(cfg.temporal_val_from)
    test_from = pd.Timestamp(cfg.temporal_test_from)
    return {
        "train": df[dep.isna() | (dep < val_from)],
        "val": df[(dep >= val_from) & (dep < test_from)],
        "test": df[dep >= test_from],
    }


def assemble(cfg: BuildConfig) -> dict[str, pd.DataFrame]:
    src = cfg.cache / "written.parquet"
    if not src.exists():
        raise FileNotFoundError(f"{src} missing; run s6_write first")
    df = pd.read_parquet(src)
    if df.empty:
        raise RuntimeError(f"{src} is empty; nothing was written in stage 6")

    # One row per example, ordered so the files are reproducible.
    df = df.drop_duplicates(subset=["example_id"]).sort_values(
        "example_id").reset_index(drop=True)

    splits = (_temporal_split(df, cfg) if cfg.split_mode == "temporal"
              else _random_split(df, cfg))

    cfg.mkdirs()
    summary = {}
    for name, part in splits.items():
        part = part.reset_index(drop=True)
        dest = cfg.parquet / f"{name}.parquet"
        part.to_parquet(dest, index=False)
        summary[name] = {
            "n": len(part),
            "path": str(dest),
            "ig_type": part["ig_type"].value_counts().to_dict() if len(part) else {},
            "deposit_date_range": [str(part["deposit_date"].min()),
                                   str(part["deposit_date"].max())] if len(part) else [],
            "residues": {
                "min": int(part["n_residues_total"].min()),
                "median": int(part["n_residues_total"].median()),
                "max": int(part["n_residues_total"].max()),
            } if len(part) else {},
            "framework_tokens": {
                "min": int(part["n_framework"].min()),
                "median": int(part["n_framework"].median()),
                "max": int(part["n_framework"].max()),
            } if len(part) else {},
        }
        logger.info(f"{name}: {len(part)} examples -> {dest}")

    # No entry may appear in two splits: the split is by example, but an entry
    # contributes exactly one example after stage 5, so this is a cheap
    # assertion that stage 5's one-per-entry selection actually held.
    seen: dict[str, str] = {}
    for name, part in splits.items():
        for pdb_id in part["pdb_id"]:
            if pdb_id in seen and seen[pdb_id] != name:
                raise RuntimeError(
                    f"entry {pdb_id} appears in both {seen[pdb_id]} and {name}")
            seen[pdb_id] = name

    logger.info(f"no entry leaks across splits ({len(seen)} entries)")
    record_stage(cfg, "s7_parquet", {
        "split_mode": cfg.split_mode,
        "seed": cfg.seed,
        "splits": summary,
        "n_entries": len(seen),
    })
    return splits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    assemble(config_from_args(parser.parse_args(argv)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
