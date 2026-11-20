"""Gate for cluster-balanced sampling and the clustering stage's invariants.

The sampler tests are pure and always run.  The dataset tests skip when the
build has not been run, in the same way as `tests/test_antibody_dataset.py`.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from proteinfoundation.datasets.cluster_sampling import MetadataClusterSampler

PARQUET_DIR = Path(os.environ.get("DATA_PATH", "data")) / "abdata" / "parquet"
CLUSTERS = Path(os.environ.get("DATA_PATH", "data")) / "abdata" / "cache" / "clusters.parquet"

requires_dataset = pytest.mark.skipif(
    not (PARQUET_DIR / "train.parquet").exists() or not CLUSTERS.exists(),
    reason="antibody dataset not built; run script_utils.abdata stages 1-8",
)

# Three clusters of very different sizes, which is the case the sampler exists
# for: "a" would otherwise supply 6 of 9 draws.
CLUSTER_IDS = ["a"] * 6 + ["b"] * 2 + ["c"]


# --------------------------------------------------------------------------- #
# the sampler
# --------------------------------------------------------------------------- #
def test_one_index_per_cluster_per_epoch():
    sampler = MetadataClusterSampler(CLUSTER_IDS, mode="cluster-random", seed=0)
    drawn = list(sampler)
    assert len(drawn) == 3 == len(sampler)
    # Exactly one index from each cluster.
    assert Counter(CLUSTER_IDS[i] for i in drawn) == {"a": 1, "b": 1, "c": 1}


def test_cluster_random_draws_different_members_across_epochs():
    sampler = MetadataClusterSampler(CLUSTER_IDS, mode="cluster-random", seed=0)
    seen: set[int] = set()
    for epoch in range(30):
        sampler.set_epoch(epoch)
        seen.update(i for i in sampler if CLUSTER_IDS[i] == "a")
    # All six members of cluster "a" must be reachable, or the large clusters
    # are permanently represented by one example and the rest is dead data.
    assert seen == {0, 1, 2, 3, 4, 5}, seen


def test_cluster_reps_is_deterministic_and_picks_the_first_member():
    sampler = MetadataClusterSampler(CLUSTER_IDS, mode="cluster-reps", shuffle=False)
    assert list(sampler) == [0, 6, 8]
    sampler.set_epoch(7)
    assert list(sampler) == [0, 6, 8]


def test_same_seed_and_epoch_reproduce_the_epoch():
    a = MetadataClusterSampler(CLUSTER_IDS, seed=5)
    b = MetadataClusterSampler(CLUSTER_IDS, seed=5)
    a.set_epoch(3)
    b.set_epoch(3)
    assert list(a) == list(b)


def test_different_epochs_change_the_cluster_order():
    sampler = MetadataClusterSampler(CLUSTER_IDS, seed=5)
    sampler.set_epoch(0)
    first = list(sampler)
    sampler.set_epoch(1)
    assert list(sampler) != first


def test_rejects_an_unknown_mode_and_empty_input():
    with pytest.raises(ValueError, match="mode must be one of"):
        MetadataClusterSampler(CLUSTER_IDS, mode="cluster-everything")
    with pytest.raises(ValueError, match="empty"):
        MetadataClusterSampler([])


def test_rejects_a_missing_cluster_id():
    # A null cluster would otherwise become its own singleton cluster and be
    # over-sampled by exactly the factor this sampler exists to remove.
    with pytest.raises(ValueError, match="no cluster id"):
        MetadataClusterSampler(["a", None, "b"])
    with pytest.raises(ValueError, match="no cluster id"):
        MetadataClusterSampler(["a", float("nan"), "b"])


def test_ddp_ranks_agree_on_the_partition(monkeypatch):
    """Every cluster must go to exactly one rank in an epoch.

    The legacy `ClusterSampler` shuffles with the unseeded global `random`
    module before slicing `indices[rank::world]`, so ranks disagree about which
    cluster each stride position holds.  This is the test that would fail for
    it.
    """
    cluster_ids = [f"c{i // 2}" for i in range(40)]  # 20 clusters

    def sampler_for(rank: int, world: int) -> MetadataClusterSampler:
        s = MetadataClusterSampler(cluster_ids, seed=11, drop_last=False)
        monkeypatch.setattr(s, "_rank_and_world", lambda r=rank, w=world: (r, w))
        s.set_epoch(2)
        return s

    world = 4
    per_rank = [list(sampler_for(r, world)) for r in range(world)]
    for drawn in per_rank:
        assert len(drawn) == 5
    clusters = [cluster_ids[i] for drawn in per_rank for i in drawn]
    assert sorted(clusters) == sorted(set(cluster_ids)), (
        "the union over ranks must cover every cluster exactly once")


# --------------------------------------------------------------------------- #
# the clustering stage
# --------------------------------------------------------------------------- #
@requires_dataset
def test_no_antigen_group_spans_two_splits():
    """The invariant the clustering stage exists to establish."""
    seen: dict[str, str] = {}
    for name in ("train", "val", "test"):
        path = PARQUET_DIR / f"{name}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if df.empty:
            continue
        assert "antigen_group" in df.columns, f"{name}.parquet has no antigen_group"
        for group in df["antigen_group"]:
            assert seen.get(group, name) == name, (
                f"antigen group {group} is in both {seen[group]} and {name}")
            seen[group] = name
    assert seen, "no splits carried any antigen group"


@requires_dataset
def test_one_example_per_cluster_pair_after_dedup():
    train = pd.read_parquet(PARQUET_DIR / "train.parquet")
    val = pd.read_parquet(PARQUET_DIR / "val.parquet")
    both = pd.concat([train, val], ignore_index=True)
    pinned = both["is_pinned"].astype(str).str.lower() == "true"
    counts = both[~pinned].groupby(["antigen_group", "fv_cluster"]).size()
    assert (counts == 1).all(), (
        f"{int((counts > 1).sum())} cluster-pairs kept more than one example:\n"
        f"{counts[counts > 1]}")


@requires_dataset
def test_cluster_column_is_present_and_usable_as_a_sampler_key():
    train = pd.read_parquet(PARQUET_DIR / "train.parquet")
    assert "cluster" in train.columns
    assert train["cluster"].notna().all()
    sampler = MetadataClusterSampler(train["cluster"].tolist(), seed=0)
    drawn = list(sampler)
    assert len(drawn) == train["cluster"].nunique()
    assert len(set(drawn)) == len(drawn), "an index was drawn twice in one epoch"


@requires_dataset
def test_dedup_actually_removed_redundancy():
    """The slice must contain real redundancy, or the stage proves nothing."""
    clusters = pd.read_parquet(CLUSTERS)
    assert clusters["is_redundant"].sum() > 0, (
        "no example was marked redundant; either the thresholds are wrong or "
        "the clustering silently assigned every sequence its own cluster")
    assert clusters["antigen_group"].nunique() < len(clusters)
    # Every example belongs to exactly one group, and groups are closed: two
    # examples sharing an antigen cluster must share a group.
    cluster_to_group: dict[str, str] = {}
    for row in clusters.itertuples(index=False):
        for antigen_cluster in str(row.antigen_clusters).split("|"):
            if not antigen_cluster:
                continue
            assert cluster_to_group.setdefault(
                antigen_cluster, row.antigen_group) == row.antigen_group, (
                f"antigen cluster {antigen_cluster} appears in two groups")


@requires_dataset
def test_interface_contacts_were_found_for_every_example():
    """A zero-contact example is an Fv that is not touching its antigen."""
    clusters = pd.read_parquet(CLUSTERS)
    bad = clusters[clusters["n_interface_contacts"] == 0]
    assert bad.empty, f"{len(bad)} examples have no Fv/antigen contact:\n{bad['example_id'].tolist()[:10]}"
    assert (clusters["n_epitope_residues"] >= 1).all()
