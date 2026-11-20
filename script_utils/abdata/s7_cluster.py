"""Stage 7 - cluster antigens and Fvs, then pick one example per cluster-pair.

This is AbData's redundancy recipe, as described in the Origin-1 supplement
S7.1: cluster the antigen sequences at 40% identity and the Fv sequences at
100%, and within each cluster-pair keep a single example chosen on resolution,
resolved fraction and interface contacts.  We keep resolution and interface
contacts; the resolved fraction needs the deposited entity length, which stage 3
does not fetch, so it is not used as a tie-break and the ranking falls back to
`example_id` for determinism.

Two outputs, both of which matter:

1.  **Deduplication.**  SAbDab is dominated by a handful of antigens -- lysozyme,
    HIV gp120, SARS-CoV-2 spike -- each with dozens of entries.  Without this,
    a few antigens supply most of the gradient.
2.  **The cross-split homology exclusion.**  Stage 8 splits on
    `antigen_group`, so no antigen cluster can span two splits.  AbData removes
    val/test entries whose antigen is >40% identical to any training antigen;
    splitting on the group achieves the same thing and is the only form that
    makes sense when the split is drawn rather than dated.

**`antigen_group` is a connected component, not a single mmseqs cluster.**  An
example may carry several antigen chains, so the unit mmseqs clusters (a chain)
is not the unit we split on (an example).  Two examples are treated as
homologous when *any* of their antigen chains land in the same 40% cluster, and
the groups are the connected components of that relation.  This is stricter than
clustering one nominated chain per example, and it is the only version that
cannot leak: an example sharing a chain cluster with a training example is in
the training group by construction.  AbData does not say how it handles
multi-chain antigens.

Fv identity is clustered at 100%, which separates genuinely identical antibodies
(the same mAb solved against the same antigen several times) from an antibody
family raised against one antigen.  Keeping the Fv axis means two *different*
antibodies against lysozyme are still two examples, which is the signal a
framework-conditioned model needs; it is only the same antibody against the same
antigen that is redundant.

    python -m script_utils.abdata.s7_cluster
    python -m script_utils.abdata.s7_cluster --antigen-cluster-identity 0.3
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
from loguru import logger
from scipy.spatial import cKDTree

from script_utils.abdata.config import (
    BuildConfig,
    add_common_args,
    config_from_args,
    record_stage,
)

FV_CHAINS = ("H", "L")


# --------------------------------------------------------------------------- #
# sequences and interface contacts, read off the structures stage 6 wrote
# --------------------------------------------------------------------------- #
def _sequences_and_contacts(path: Path, contact_distance: float) -> dict:
    """Fv sequence, antigen chain sequences, and Fv/antigen contact counts.

    Read from the *written* file rather than the deposited one, so that what is
    clustered is exactly what the model will see: the trimmed variable domains
    and the antigen chains as kept.
    """
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    st.remove_alternative_conformations()
    st.remove_hydrogens()
    if len(st) == 0:
        raise RuntimeError(f"{path} has no model")
    model = st[0]

    fv_seq_parts: list[str] = []
    antigen_seqs: dict[str, str] = {}
    fv_xyz: list[tuple[float, float, float]] = []
    ag_xyz: list[tuple[float, float, float]] = []
    ag_res_of_atom: list[int] = []
    ag_res_counter = 0

    for chain in model:
        seq = gemmi.one_letter_code([r.name for r in chain]).upper()
        is_fv = chain.name in FV_CHAINS
        if is_fv:
            fv_seq_parts.append(seq)
        else:
            antigen_seqs[chain.name] = seq
        for res in chain:
            if not is_fv:
                ag_res_counter += 1
            for atom in res:
                pos = (atom.pos.x, atom.pos.y, atom.pos.z)
                if is_fv:
                    fv_xyz.append(pos)
                else:
                    ag_xyz.append(pos)
                    ag_res_of_atom.append(ag_res_counter)

    n_contacts = 0
    epitope: set[int] = set()
    if fv_xyz and ag_xyz:
        fv_tree = cKDTree(np.asarray(fv_xyz, dtype=float))
        ag_arr = np.asarray(ag_xyz, dtype=float)
        neighbours = fv_tree.query_ball_point(ag_arr, r=contact_distance)
        res_of_atom = np.asarray(ag_res_of_atom)
        for atom_idx, hits in enumerate(neighbours):
            if hits:
                n_contacts += len(hits)
                epitope.add(int(res_of_atom[atom_idx]))

    return {
        # H and L are written in that order by stage 6; joining with ":" keeps
        # the two domains from forming a spurious junction peptide.
        "fv_sequence": ":".join(fv_seq_parts),
        "antigen_seqs": antigen_seqs,
        "n_interface_contacts": n_contacts,
        "n_epitope_residues": len(epitope),
    }


# --------------------------------------------------------------------------- #
# mmseqs
# --------------------------------------------------------------------------- #
def _write_fasta(path: Path, records: dict[str, str]) -> None:
    with path.open("w") as fh:
        for name, seq in records.items():
            fh.write(f">{name}\n{seq}\n")


def mmseqs_cluster(
    records: dict[str, str],
    work_dir: Path,
    tag: str,
    min_seq_id: float,
    coverage: float,
    cov_mode: int,
    mmseqs_exec: str,
) -> dict[str, str]:
    """Cluster `records` and return {record id -> cluster representative id}.

    `easy-cluster` writes `<prefix>_cluster.tsv` with one
    `representative<TAB>member` line per member.  Everything is passed as an
    absolute prefix inside `work_dir`: the repo's own `cluster_sequences`
    helper in `proteinfoundation.utils.cluster_utils` instead moves fixed
    filenames out of the current working directory, which makes two builds in
    the same checkout collide, so it is deliberately not reused here.
    """
    if not records:
        return {}
    work_dir.mkdir(parents=True, exist_ok=True)
    fasta = work_dir / f"{tag}.fasta"
    prefix = work_dir / f"{tag}_res"
    tmp = work_dir / f"{tag}_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    _write_fasta(fasta, records)

    cmd = [
        mmseqs_exec, "easy-cluster", str(fasta), str(prefix), str(tmp),
        "--min-seq-id", str(min_seq_id),
        "-c", str(coverage),
        "--cov-mode", str(cov_mode),
    ]
    logger.info(f"mmseqs {tag}: {len(records)} sequences at "
                f"min-seq-id={min_seq_id} -c={coverage} cov-mode={cov_mode}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"mmseqs easy-cluster failed for {tag} (exit {proc.returncode}).\n"
            f"command: {' '.join(cmd)}\n"
            f"stderr tail:\n{proc.stderr[-2000:]}")

    tsv = prefix.with_name(prefix.name + "_cluster.tsv")
    if not tsv.exists():
        raise RuntimeError(f"mmseqs produced no {tsv}")

    assignment: dict[str, str] = {}
    with tsv.open() as fh:
        for line in fh:
            rep, member = line.rstrip("\n").split("\t")
            assignment[member] = rep

    # Every input must be assigned.  mmseqs silently skips sequences it cannot
    # handle (empty, or shorter than the k-mer length), and an unassigned
    # example would otherwise sail through dedup as its own singleton cluster
    # and land in whichever split the shuffle put it in.
    missing = sorted(set(records) - set(assignment))
    if missing:
        raise RuntimeError(
            f"mmseqs assigned {len(assignment)} of {len(records)} {tag} "
            f"sequences; unassigned: {missing[:5]}")
    logger.info(f"mmseqs {tag}: {len(set(assignment.values()))} clusters")
    return assignment


# --------------------------------------------------------------------------- #
# union-find over examples that share an antigen cluster
# --------------------------------------------------------------------------- #
def _group_by_shared_cluster(example_clusters: dict[str, list[str]]) -> dict[str, str]:
    """example_id -> group id, merging examples that share any antigen cluster."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Each example is a node; each antigen cluster is a node too, so an example
    # joins every cluster it touches and clusters merge transitively through it.
    for example_id, clusters in example_clusters.items():
        find(example_id)
        for c in clusters:
            union(example_id, f"cluster::{c}")

    # Name each component after its lexicographically smallest example, so the
    # group ids are stable across runs regardless of row order.
    members: dict[str, list[str]] = {}
    for example_id in example_clusters:
        members.setdefault(find(example_id), []).append(example_id)
    out: dict[str, str] = {}
    for component in members.values():
        name = min(component)
        for example_id in component:
            out[example_id] = name
    return out


# --------------------------------------------------------------------------- #
# stage
# --------------------------------------------------------------------------- #
def cluster(cfg: BuildConfig) -> pd.DataFrame:
    src = cfg.cache / "written.parquet"
    if not src.exists():
        raise FileNotFoundError(f"{src} missing; run s6_write first")
    written = pd.read_parquet(src)
    if written.empty:
        raise RuntimeError(f"{src} is empty; nothing was written in stage 6")
    written = written.drop_duplicates(subset=["example_id"]).sort_values(
        "example_id").reset_index(drop=True)
    cfg.mkdirs()

    # --- sequences and contacts ------------------------------------------- #
    fv_records: dict[str, str] = {}
    antigen_records: dict[str, str] = {}      # "<example_id>__<chain>" -> seq
    per_example: dict[str, dict] = {}
    for n, row in enumerate(written.itertuples(index=False), 1):
        info = _sequences_and_contacts(
            Path(row.path), cfg.interface_contact_distance)
        per_example[row.example_id] = info
        fv_records[row.example_id] = info["fv_sequence"].replace(":", "")
        for chain, seq in info["antigen_seqs"].items():
            antigen_records[f"{row.example_id}__{chain}"] = seq
        if n % 50 == 0:
            logger.info(f"  read {n}/{len(written)} structures")
    logger.info(f"{len(fv_records)} Fvs, {len(antigen_records)} antigen chains")

    if not cfg.cluster_antigens:
        logger.warning("cluster_antigens=False: every example is its own group")
        antigen_assign: dict[str, str] = {k: k for k in antigen_records}
        fv_assign: dict[str, str] = {k: k for k in fv_records}
    else:
        mmseqs = cfg.resolve_mmseqs()
        if shutil.which(mmseqs) is None and not Path(mmseqs).exists():
            raise RuntimeError(
                f"mmseqs not found at {mmseqs!r}. Set MMSEQS_EXEC (it is "
                "defined in .env) or pass --mmseqs-exec.")
        antigen_assign = mmseqs_cluster(
            antigen_records, cfg.cluster_work, "antigen",
            cfg.antigen_cluster_identity, cfg.antigen_cluster_coverage,
            cfg.cluster_cov_mode, mmseqs)
        fv_assign = mmseqs_cluster(
            fv_records, cfg.cluster_work, "fv",
            cfg.fv_cluster_identity, cfg.fv_cluster_coverage,
            cfg.cluster_cov_mode, mmseqs)

    # --- antigen homology groups ------------------------------------------ #
    example_antigen_clusters: dict[str, list[str]] = {
        example_id: [] for example_id in per_example}
    for key, rep in antigen_assign.items():
        example_id = key.split("__", 1)[0]
        example_antigen_clusters[example_id].append(rep)
    groups = _group_by_shared_cluster(example_antigen_clusters)

    rows = []
    for row in written.itertuples(index=False):
        info = per_example[row.example_id]
        clusters = sorted(set(example_antigen_clusters[row.example_id]))
        rows.append({
            "example_id": row.example_id,
            "antigen_clusters": "|".join(clusters),
            "n_antigen_clusters": len(clusters),
            "antigen_group": groups[row.example_id],
            "fv_cluster": fv_assign[row.example_id],
            "fv_sequence": info["fv_sequence"],
            "n_interface_contacts": info["n_interface_contacts"],
            "n_epitope_residues": info["n_epitope_residues"],
        })
    out = written.merge(pd.DataFrame(rows), on="example_id", how="left")
    if out["antigen_group"].isna().any():
        raise RuntimeError("some examples got no antigen group")

    # --- one example per (antigen group, Fv cluster) ---------------------- #
    # AbData ranks on resolution, resolved fraction and interface contacts.
    # Lower resolution first, then more interface contacts, then example_id so
    # that the choice does not depend on row order.
    out["resolution_rank"] = pd.to_numeric(out["resolution"], errors="coerce")
    out = out.sort_values(
        ["antigen_group", "fv_cluster", "resolution_rank",
         "n_interface_contacts", "example_id"],
        ascending=[True, True, True, False, True]).reset_index(drop=True)
    first = ~out.duplicated(subset=["antigen_group", "fv_cluster"], keep="first")

    # A pinned entry must survive dedup: the probe scripts and the autoencoder
    # reconstruction test reference it by id.  Promote it to representative of
    # its own cluster-pair rather than dropping it.
    pinned = out["is_pinned"].astype(str).str.lower() == "true"
    keep = first | pinned
    n_pinned_rescued = int((pinned & ~first).sum())
    if n_pinned_rescued:
        logger.info(f"{n_pinned_rescued} pinned examples kept despite being "
                    "cluster-redundant")

    out["is_representative"] = keep
    out["is_redundant"] = ~keep
    out = out.drop(columns=["resolution_rank"])
    out.to_parquet(cfg.clusters_parquet, index=False)

    n_groups = out["antigen_group"].nunique()
    n_pairs = out.loc[keep].groupby(["antigen_group", "fv_cluster"]).ngroups
    biggest = Counter(out["antigen_group"]).most_common(5)
    logger.info(f"wrote {cfg.clusters_parquet} ({len(out)} rows)")
    logger.info(f"{len(out)} examples -> {n_groups} antigen groups, "
                f"{out['fv_cluster'].nunique()} Fv clusters, "
                f"{n_pairs} kept cluster-pairs "
                f"({int(keep.sum())} representatives, "
                f"{int((~keep).sum())} redundant)")
    logger.info("largest antigen groups: " + ", ".join(
        f"{g}={n}" for g, n in biggest))
    logger.info("epitope residues: min=%d median=%d max=%d" % (
        out["n_epitope_residues"].min(),
        int(out["n_epitope_residues"].median()),
        out["n_epitope_residues"].max()))

    record_stage(cfg, "s7_cluster", {
        "n_examples": len(out),
        "antigen_cluster_identity": cfg.antigen_cluster_identity,
        "fv_cluster_identity": cfg.fv_cluster_identity,
        "cluster_coverage": [cfg.antigen_cluster_coverage,
                             cfg.fv_cluster_coverage],
        "cov_mode": cfg.cluster_cov_mode,
        "n_antigen_chains": len(antigen_records),
        "n_antigen_chain_clusters": len(set(antigen_assign.values())),
        "n_antigen_groups": n_groups,
        "n_fv_clusters": out["fv_cluster"].nunique(),
        "n_representatives": int(keep.sum()),
        "n_redundant": int((~keep).sum()),
        "n_pinned_rescued": n_pinned_rescued,
        "largest_antigen_groups": dict(biggest),
        "interface_contact_distance": cfg.interface_contact_distance,
        "epitope_residues": {
            "min": int(out["n_epitope_residues"].min()),
            "median": int(out["n_epitope_residues"].median()),
            "max": int(out["n_epitope_residues"].max()),
        },
    })
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    cluster(config_from_args(parser.parse_args(argv)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
