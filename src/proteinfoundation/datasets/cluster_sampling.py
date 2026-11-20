"""Cluster-balanced sampling over a metadata-backed structure dataset.

One epoch visits each cluster once rather than each example once, so a corpus
dominated by a handful of antigens stops being dominated by them.  In the
antibody slice the largest antigen homology group holds 30 of 163 examples:
under uniform sampling one antigen supplies 18% of the gradient, and under
cluster sampling it supplies 1/100th of it.

Two modes, matching `proteinfoundation.utils.cluster_utils.ClusterSampler`:

* ``cluster-random`` draws a different member of each cluster every epoch, so
  the model eventually sees every example while never over-weighting a large
  cluster.  This is the useful one.
* ``cluster-reps`` always yields the same member (the first in metadata order),
  which makes an epoch deterministic and is mostly useful for debugging.

Why not the existing `ClusterSampler`: it resolves dataset indices through
attributes that only the legacy datasets have (`dataset.database`,
`dataset.file_names`, `protein_to_idx`, `pinder_id_to_idx`), none of which
`StructureDataset` defines, and it is instantiated nowhere in the repo.  It
also shuffles with the global `random` module under DDP, so each rank draws a
*different* permutation before slicing it with `indices[rank::world_size]`;
ranks then disagree about which cluster each stride position holds, and an
epoch silently covers some clusters more than once and others not at all.  This
sampler seeds the permutation from (seed, epoch) so every rank computes the
same order, which is what makes the stride partition sound.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import torch.distributed as dist
from loguru import logger
from torch.utils.data import Sampler

MODES = ("cluster-random", "cluster-reps")


class MetadataClusterSampler(Sampler[int]):
    """Yield one dataset index per cluster per epoch.

    Args:
        cluster_ids: the cluster of each dataset row, in dataset order.  Index
            ``i`` of this sequence must correspond to ``dataset[i]``, which for
            `StructureDataset` means the ``cluster`` column of its
            ``metadata`` frame after the ``reset_index`` in its constructor.
        mode: ``"cluster-random"`` or ``"cluster-reps"``.
        shuffle: shuffle the cluster order each epoch.  Off, clusters are
            visited in first-appearance order.
        drop_last: under DDP, truncate to a whole number of per-rank batches
            instead of padding by repeating clusters.
        seed: base seed; the per-epoch seed is ``seed + epoch``.
    """

    def __init__(
        self,
        cluster_ids: Sequence,
        mode: str = "cluster-random",
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if len(cluster_ids) == 0:
            raise ValueError("cluster_ids is empty: nothing to sample")

        self.mode = mode
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        # Cluster -> dataset indices, in first-appearance order so that the
        # cluster list itself does not depend on hash ordering.
        members: dict[object, list[int]] = {}
        for idx, cluster in enumerate(cluster_ids):
            if cluster is None or (isinstance(cluster, float) and np.isnan(cluster)):
                raise ValueError(
                    f"row {idx} has no cluster id; filter the metadata (e.g. "
                    '"cluster.notnull()") or leave cluster_column unset')
            members.setdefault(cluster, []).append(idx)
        self.clusters: list[object] = list(members)
        self.members: list[list[int]] = [members[c] for c in self.clusters]

        logger.info(
            f"MetadataClusterSampler: {len(cluster_ids)} examples in "
            f"{len(self.clusters)} clusters, mode={mode}, "
            f"largest cluster={max(len(m) for m in self.members)}")

    # Lightning calls this per epoch when the attribute exists, which is what
    # makes "cluster-random" draw a different member each epoch rather than
    # the same one forever.
    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rank_and_world(self) -> tuple[int, int]:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        return 0, 1

    def __iter__(self) -> Iterator[int]:
        rank, world = self._rank_and_world()
        # Identical on every rank: same seed, same epoch, same permutation.
        rng = np.random.default_rng(self.seed + self.epoch)

        order = list(range(len(self.clusters)))
        if self.shuffle:
            order = [order[i] for i in rng.permutation(len(order))]

        if world > 1:
            per_rank = len(order) // world if self.drop_last else -(-len(order) // world)
            total = per_rank * world
            if self.drop_last:
                order = order[:total]
            elif total > len(order):
                order = order + order[: total - len(order)]
            order = order[rank:total:world]

        for cluster_idx in order:
            member_indices = self.members[cluster_idx]
            if self.mode == "cluster-reps":
                yield member_indices[0]
            else:
                # Draw from the same per-epoch stream, so the choice is
                # reproducible for a given (seed, epoch) and rank.
                yield int(member_indices[rng.integers(len(member_indices))])

    def __len__(self) -> int:
        rank, world = self._rank_and_world()
        if world == 1:
            return len(self.clusters)
        if self.drop_last:
            return len(self.clusters) // world
        return -(-len(self.clusters) // world)
