"""Stage 8 - assemble the train / val / test metadata parquets.

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

The split is drawn over **antigen homology groups**, not examples: stage 7
merges examples whose antigen chains cluster together at 40% identity, and a
group is assigned to exactly one split.  That is AbData's cross-split homology
exclusion (Origin-1 supplement S7.1, "we removed all entries containing an
antigen with greater than 40% sequence similarity to any antigen in the
relevant reference set") in the form that applies to a drawn rather than a
dated split.  Without it a validation antigen can be a near-copy of a training
antigen and the validation loss measures memorisation.

    python -m script_utils.abdata.s8_parquet
    python -m script_utils.abdata.s8_parquet --split-mode temporal
    python -m script_utils.abdata.s8_parquet --split-mode random   # leaks
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


def _cluster_split(df: pd.DataFrame, cfg: BuildConfig
                   ) -> dict[str, pd.DataFrame]:
    """Draw whole antigen homology groups into val and test.

    Groups are drawn in a seeded random order, and a group is taken only when
    it fits the remaining budget.  Taking whatever came next instead lets one
    group swallow the whole split: the slice's largest antigen group holds 30
    of 163 examples, so a naive draw produced a 34-example validation set in
    which 30 shared a single antigen -- numerically the right size and useless
    as a validation set.  Skipping an oversized group costs nothing, because it
    simply stays in train.

    If nothing fits and the split is still empty, the smallest remaining group
    is taken anyway and the overshoot is reported: an empty validation set is
    worse than an oversized one.  Groups containing a pinned entry are never
    drawn.  A group is never split, which is the whole point of this mode.
    """
    pinned_groups = set(
        df.loc[df["is_pinned"].astype(str).str.lower() == "true", "antigen_group"])
    sizes = df.groupby("antigen_group").size()
    drawable = [g for g in sizes.index if g not in pinned_groups]

    rng = np.random.default_rng(cfg.seed)
    order = [drawable[i] for i in rng.permutation(len(drawable))]

    assignment: dict[str, str] = {g: "train" for g in sizes.index}
    budget = {"val": cfg.n_val, "test": cfg.n_test}
    for split in ("val", "test"):
        want = budget[split]
        if want <= 0:
            continue
        filled = 0
        remaining = []
        for g in order:
            size = int(sizes[g])
            if filled + size <= want:
                assignment[g] = split
                filled += size
            else:
                remaining.append(g)
        order = remaining
        if filled == 0 and order:
            g = min(order, key=lambda g: int(sizes[g]))
            order.remove(g)
            assignment[g] = split
            filled = int(sizes[g])
            logger.warning(
                f"{split}: no antigen group fits a budget of {want}; took "
                f"{g} with {filled} examples instead")
        if filled < want:
            logger.info(
                f"{split}: {filled} of {want} requested examples -- the "
                "remaining antigen groups are all too large to fit")

    assigned = df["antigen_group"].map(assignment)
    out = {name: df[assigned == name] for name in ("train", "val", "test")}
    for name, part in out.items():
        logger.info(f"{name}: {len(part)} examples over "
                    f"{part['antigen_group'].nunique()} antigen groups")
    return out


def _load(cfg: BuildConfig) -> tuple[pd.DataFrame, bool]:
    """Stage 7's table when it exists, else stage 6's.

    Clustering is required for split_mode="cluster" and for dedup; the other
    modes still run without it, which is what keeps a build reproducible when
    mmseqs is unavailable.
    """
    if cfg.clusters_parquet.exists():
        return pd.read_parquet(cfg.clusters_parquet), True
    src = cfg.cache / "written.parquet"
    if not src.exists():
        raise FileNotFoundError(f"{src} missing; run s6_write first")
    if cfg.split_mode == "cluster" or cfg.dedup_by_cluster:
        raise FileNotFoundError(
            f"{cfg.clusters_parquet} missing; run s7_cluster first, or pass "
            "--split-mode random --dedup-by-cluster false to skip clustering "
            "(which leaks homologous antigens across splits)")
    return pd.read_parquet(src), False


def assemble(cfg: BuildConfig) -> dict[str, pd.DataFrame]:
    df, clustered = _load(cfg)
    if df.empty:
        raise RuntimeError("nothing to assemble; stage 6 wrote no structures")

    # One row per example, ordered so the files are reproducible.
    df = df.drop_duplicates(subset=["example_id"]).sort_values(
        "example_id").reset_index(drop=True)

    n_before = len(df)
    if clustered and cfg.dedup_by_cluster:
        df = df[df["is_representative"]].reset_index(drop=True)
        logger.info(f"dedup by (antigen_group, fv_cluster): "
                    f"{n_before} -> {len(df)} examples")

    # The column the training sampler groups on.  Named `cluster` to match
    # `configs/dataset/unified/pdb_interfaces.yaml`, which already carries a
    # `cluster` column for the PDB interfaces corpus.
    if clustered:
        df["cluster"] = df["antigen_group"]

    if cfg.split_mode == "cluster":
        splits = _cluster_split(df, cfg)
    elif cfg.split_mode == "temporal":
        splits = _temporal_split(df, cfg)
    elif cfg.split_mode == "random":
        splits = _random_split(df, cfg)
    else:
        raise ValueError(f"unknown split_mode {cfg.split_mode!r}")

    # AbData applies its homology filter in one direction: filters defined on
    # the training set are applied to val and test.  For a dated split that
    # means dropping val/test examples homologous to a training example, which
    # is what this does.  The "cluster" mode cannot produce any.
    if clustered and cfg.split_mode != "cluster":
        train_groups = set(splits["train"]["antigen_group"])
        for name in ("val", "test"):
            part = splits[name]
            leaking = part["antigen_group"].isin(train_groups)
            if leaking.any():
                logger.warning(
                    f"{name}: dropping {int(leaking.sum())} of {len(part)} "
                    "examples whose antigen clusters with a training antigen")
                splits[name] = part[~leaking]
        val_groups = set(splits["val"]["antigen_group"])
        test = splits["test"]
        leaking = test["antigen_group"].isin(val_groups)
        if leaking.any():
            logger.warning(f"test: dropping {int(leaking.sum())} examples "
                           "homologous to a validation antigen")
            splits["test"] = test[~leaking]

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
        if clustered and len(part):
            summary[name]["n_antigen_groups"] = int(part["antigen_group"].nunique())
            summary[name]["n_fv_clusters"] = int(part["fv_cluster"].nunique())
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

    # The homology assertion: no antigen group may span two splits.  This is
    # the invariant the whole clustering stage exists to establish, so it is
    # checked rather than assumed, for every split mode.
    if clustered:
        group_split: dict[str, str] = {}
        for name, part in splits.items():
            for group in part["antigen_group"]:
                if group in group_split and group_split[group] != name:
                    raise RuntimeError(
                        f"antigen group {group} appears in both "
                        f"{group_split[group]} and {name}: the splits leak "
                        "homologous antigens")
                group_split[group] = name
        logger.info(f"no antigen group spans two splits "
                    f"({len(group_split)} groups)")

    record_stage(cfg, "s8_parquet", {
        "split_mode": cfg.split_mode,
        "seed": cfg.seed,
        "clustered": clustered,
        "dedup_by_cluster": bool(clustered and cfg.dedup_by_cluster),
        "n_examples_before_dedup": n_before,
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
