"""Transforms for antibody framework conditioning.

Three pieces, in the order they run:

1. ``FrameworkSidecarTransform`` resolves the framework sidecar against the
   loaded structure and writes per-residue arrays of full length ``n``.
2. ``AntibodyCroppingTransform`` keeps the whole Fv as the binder and crops the
   antigen around it.  ``CroppingTransform2`` cannot be used directly because
   it adopts a *single* chain as the binder (``asym_id != binder_chain_id``),
   and a paired Fv's binder is two chains.
3. ``ExtractFrameworkCoordinatesTransform`` compacts the framework positions
   onto a ``[k]`` axis and registers the ``"framework"`` token group.

Running the resolve *before* the crop is deliberate: ``_crop_graph``
(``transforms.py:780``) subsets every tensor whose leading dimension equals the
residue count, so the per-residue arrays are carried through the crop for free
and the key resolution happens once, against the whole structure.

The sidecar is keyed on ``(chain_id, res_id)`` rather than position because
both survive the atom37 conversion -- ``atomarray_to_atom37`` passes
``extra_annotations=["chain_id"]`` and sets ``residue_pdb_idx`` from
``token.res_id`` -- while a positional array would be silently invalidated by
any dropped or reordered residue.  Insertion codes do *not* survive, which is
why stage 6 of the build renumbers residues sequentially.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from loguru import logger


from proteinfoundation.datasets.transforms import (
    BaseTransform,
    CroppingTransform2,
    Data,
)

# Must match FrameworkChainTypeSeqFeat: VH = 0, VL = 1.
FRAMEWORK_CHAIN_VH = 0
FRAMEWORK_CHAIN_VL = 1

FRAMEWORK_FIELDS = [
    "x_framework",
    "framework_mask",
    "seq_framework",
    "seq_framework_mask",
    "framework_chain",
]

_SIDECAR_CACHE: dict[str, dict[str, pd.DataFrame]] = {}


def _load_sidecar(path: str) -> dict[str, pd.DataFrame]:
    """Load and index the sidecar once per process.

    Dataloader workers are separate processes, so this is per worker; the
    sidecar is a few hundred thousand rows at full scale and the alternative is
    a parquet read per example.
    """
    key = str(Path(path).resolve())
    if key not in _SIDECAR_CACHE:
        df = pd.read_parquet(key)
        required = {"example_id", "chain_id", "res_id", "is_framework",
                    "cdr_id", "framework_chain", "res_name", "aa1"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"framework sidecar {key} is missing columns {missing}")
        _SIDECAR_CACHE[key] = {
            eid: g for eid, g in df.groupby("example_id", sort=False)}
        logger.info(f"loaded framework sidecar {key}: {len(df)} rows over "
                    f"{len(_SIDECAR_CACHE[key])} examples")
    return _SIDECAR_CACHE[key]


class PerExampleSeedTransform(BaseTransform):
    """Seed the RNGs from a hash of ``example_id``, for a reproducible val set.

    ``setup`` builds the val dataset with ``is_inference=True``
    (``structure_data.py:874``), but that flag is only read when building an
    atomworks pipeline (line 254); in the simple ``atom37_transforms`` mode this
    dataset uses it does nothing at all.  Val batches would therefore go through
    the same random rotation and crop as training, and val loss would move
    between epochs for reasons unrelated to the model -- intolerable on a
    twenty-complex val set.

    Placing this first in the *val* transform chain makes each example's
    randomness a deterministic function of its identity, so the val set is
    identical every epoch and comparable across runs, while training stays
    stochastic.  It is deliberately not in the train chain.
    """

    def __init__(self, salt: int = 0):
        self.salt = salt

    def __call__(self, graph: Data) -> Data:
        import hashlib
        import random as _random

        example_id = str(getattr(graph, "id", ""))
        digest = hashlib.sha256(f"{self.salt}:{example_id}".encode()).digest()
        seed = int.from_bytes(digest[:8], "big") % (2**31 - 1)
        torch.manual_seed(seed)
        _random.seed(seed)
        return graph


class FrameworkSidecarTransform(BaseTransform):
    """Resolve the framework sidecar onto the structure, as ``[n]`` arrays.

    Writes ``framework_residue_mask``, ``framework_chain_full``,
    ``framework_cdr_id`` and ``framework_is_antibody``, all of length ``n``.

    Args:
        sidecar_path: parquet written by ``script_utils.abdata.s6_write``.
        verify_identity: compare each resolved residue's identity against the
            sidecar and raise on a mismatch.  This is the property that
            atomworks' ``get_masks`` has and a hand-rolled mask does not: a mask
            that disagrees with the structure is wrong in precisely the way
            nobody notices.
        require_match: raise if an example has no sidecar entry, or if the
            sidecar names residues that the structure does not contain.  Off
            would mean silently conditioning on nothing, which is
            indistinguishable from the scientific negative result this dataset
            exists to measure.
    """

    def __init__(
        self,
        sidecar_path: str,
        verify_identity: bool = True,
        require_match: bool = True,
        min_framework_residues: int = 1,
    ):
        self.sidecar_path = sidecar_path
        self.verify_identity = verify_identity
        self.require_match = require_match
        self.min_framework_residues = min_framework_residues

    def __call__(self, graph: Data) -> Data:
        index = _load_sidecar(self.sidecar_path)
        example_id = str(getattr(graph, "id", ""))
        rows = index.get(example_id)

        n = int(graph.coords.shape[0])
        device = graph.coords.device
        fw = torch.zeros(n, dtype=torch.bool, device=device)
        chain = torch.zeros(n, dtype=torch.long, device=device)
        cdr = torch.zeros(n, dtype=torch.long, device=device)
        is_ab = torch.zeros(n, dtype=torch.bool, device=device)

        if rows is None or rows.empty:
            if self.require_match:
                raise ValueError(
                    f"no framework sidecar rows for example_id={example_id!r}; "
                    f"sidecar={self.sidecar_path}. Conditioning on nothing "
                    "silently would be indistinguishable from the conditioning "
                    "not transferring.")
            logger.warning(f"no framework sidecar rows for {example_id}")
            graph.framework_residue_mask = fw
            graph.framework_chain_full = chain
            graph.framework_cdr_id = cdr
            graph.framework_is_antibody = is_ab
            return graph

        # (chain_id, res_id) -> token index, from what actually loaded.
        chain_ids = list(graph.chain_id)
        res_ids = graph.residue_pdb_idx.tolist()
        position = {}
        for i, (c, r) in enumerate(zip(chain_ids, res_ids)):
            position.setdefault((str(c), int(r)), i)

        unresolved: list[tuple[str, int]] = []
        mismatched: list[str] = []
        for row in rows.itertuples(index=False):
            key = (str(row.chain_id), int(row.res_id))
            i = position.get(key)
            if i is None:
                unresolved.append(key)
                continue
            if self.verify_identity:
                observed = str(graph.residues[i]).strip().upper()
                expected_3 = str(row.res_name).strip().upper()
                expected_1 = str(row.aa1).strip().upper()
                if observed not in (expected_3, expected_1):
                    mismatched.append(
                        f"{key}: structure has {observed}, sidecar says "
                        f"{expected_3}/{expected_1}")
                    continue
            is_ab[i] = True
            cdr[i] = int(row.cdr_id)
            chain[i] = int(row.framework_chain)
            if bool(row.is_framework):
                fw[i] = True

        if mismatched:
            raise ValueError(
                f"{example_id}: {len(mismatched)} framework residues disagree "
                f"with the structure, e.g. {mismatched[:3]}. The mask and the "
                "coordinates describe different residues.")

        if unresolved and self.require_match:
            raise ValueError(
                f"{example_id}: {len(unresolved)} sidecar residues absent from "
                f"the loaded structure, e.g. {unresolved[:5]}. Either the "
                "structure file and the sidecar are out of step, or a "
                "transform dropped residues before this one ran.")

        n_fw = int(fw.sum())
        if n_fw < self.min_framework_residues:
            raise ValueError(
                f"{example_id}: resolved only {n_fw} framework residues "
                f"(minimum {self.min_framework_residues}).")

        graph.framework_residue_mask = fw
        graph.framework_chain_full = chain
        graph.framework_cdr_id = cdr
        graph.framework_is_antibody = is_ab
        return graph


class ExtractFrameworkCoordinatesTransform(BaseTransform):
    """Compact the framework positions onto a ``[k]`` axis and register them.

    Mirrors ``ExtractMotifCoordinatesTransform`` in compact mode, with one
    deliberate difference: the token group is registered **unconditionally**,
    including when no framework residue survives.  ``_discover_groups``
    (``structure_data.py:482``) raises when samples in a batch disagree about
    which groups exist, so registering only on the non-empty path turns batch
    composition into a crash.  A group with zero tokens is also exempt from the
    ``pad_group_priority`` check (``_allocate_budget``, line 569), so an
    all-empty batch stays valid.

    Args:
        use_coords: emit ``x_framework`` and ``framework_mask`` as well.  The
            sequence-only arm (22 dims) is the default because supplying
            coordinates supplies a committed pose.
    """

    def __init__(self, use_coords: bool = False):
        self.use_coords = use_coords

    def __call__(self, graph: Data) -> Data:
        if not hasattr(graph, "framework_residue_mask"):
            raise ValueError(
                "framework_residue_mask not found; apply "
                "FrameworkSidecarTransform before this transform.")

        mask = graph.framework_residue_mask.bool()
        device = graph.coords.device
        k = int(mask.sum())

        if k > 0:
            graph.seq_framework = graph.residue_type[mask]              # [k]
            graph.seq_framework_mask = torch.ones(
                k, dtype=torch.bool, device=device)                     # [k]
            graph.framework_chain = graph.framework_chain_full[mask]    # [k]
        else:
            graph.seq_framework = torch.zeros(
                0, dtype=graph.residue_type.dtype, device=device)
            graph.seq_framework_mask = torch.zeros(0, dtype=torch.bool, device=device)
            graph.framework_chain = torch.zeros(0, dtype=torch.long, device=device)

        if self.use_coords:
            coords = getattr(graph, "coords_nm", None)
            if coords is None:
                raise ValueError(
                    "use_coords=True needs coords_nm; apply "
                    "CoordsToNanometers before this transform.")
            graph.x_framework = (coords[mask] if k > 0
                                 else torch.zeros(0, 37, 3, device=device))
            graph.framework_mask = (graph.coord_mask.bool()[mask] if k > 0
                                    else torch.zeros(0, 37, dtype=torch.bool,
                                                     device=device))
            fields = FRAMEWORK_FIELDS
        else:
            fields = ["seq_framework", "seq_framework_mask", "framework_chain"]

        graph.register_token_group("framework", fields)
        return graph


class AntibodyCroppingTransform(CroppingTransform2):
    """Crop with the whole Fv as the binder and the antigen as the target.

    ``CroppingTransform2`` picks a random interface residue and adopts *its
    chain* as the binder, so for a paired Fv it would keep VH or VL but not
    both, and would additionally crop a contiguous window out of it.  Both are
    wrong here: the framework conditioning supplies the whole Fv, so the Fv must
    arrive whole.

    The antibody chains are identified by the convention stage 6 writes --
    ``H`` and ``L`` -- resolved through ``chain_id`` strings rather than the
    integer ``chains`` encoding, whose values depend on chain order.

    Note that for most antibody complexes this is a no-op beyond setting
    ``target_mask``: the parent returns early when the structure has fewer
    residues than ``crop_size`` (``transforms.py:588``), and a trimmed Fv plus
    one antigen chain is usually well under 500 residues.
    """

    def __init__(
        self,
        crop_size: int = 500,
        interface_threshold: float = 5.0,
        dist_mode: str = "bb_ca",
        data_mode: str = "all-atom",
        antibody_chain_ids: tuple[str, ...] = ("H", "L"),
        target_spatial_crop_threshold: float = 15.0,
        require_antibody_chains: bool = True,
    ):
        # The parent's binder_* knobs are unused here: the binder is the whole
        # Fv, never a sampled window.  binder_min_length is passed only to
        # satisfy the parent's assertion.
        super().__init__(
            crop_size=crop_size,
            interface_threshold=interface_threshold,
            dist_mode=dist_mode,
            data_mode=data_mode,
            binder_min_length=21,
            binder_max_length=crop_size,
            binder_padding_length=10,
            target_spatial_crop_threshold=target_spatial_crop_threshold,
        )
        self.antibody_chain_ids = tuple(antibody_chain_ids)
        self.require_antibody_chains = require_antibody_chains

    def __call__(self, graph: Data) -> Data:
        chain_ids = [str(c) for c in graph.chain_id]
        binder_mask = torch.tensor(
            [c in self.antibody_chain_ids for c in chain_ids],
            dtype=torch.bool, device=graph.coords.device)

        if not bool(binder_mask.any()):
            if self.require_antibody_chains:
                raise ValueError(
                    f"no chain named {self.antibody_chain_ids} in "
                    f"{getattr(graph, 'id', '?')} (chains present: "
                    f"{sorted(set(chain_ids))}). Stage 6 writes the antibody as "
                    "H and L; load_any returns label_asym_id, so this is also "
                    "what fails if the written file's label and auth chain ids "
                    "are not identical.")
            logger.warning(f"no antibody chains in {getattr(graph, 'id', '?')}")
            return graph

        target_mask = ~binder_mask
        if not bool(target_mask.any()):
            if self.data_mode == "all-atom":
                graph.target_mask = target_mask[:, None] * graph.coord_mask.bool()
            else:
                graph.target_mask = target_mask
            return graph

        if self.data_mode == "all-atom":
            graph.target_mask = target_mask[:, None] * graph.coord_mask.bool()
        else:
            graph.target_mask = target_mask

        n_res = len(chain_ids)
        if n_res <= self.crop_size:
            return graph

        # Over budget: keep the whole Fv and take the antigen residues nearest
        # to it, which is an epitope-centred spatial crop by construction.
        _, min_dist_per_res = self._get_interface_residues(graph)
        residue_idxs = torch.arange(n_res, device=graph.coords.device)
        binder_idxs = residue_idxs[binder_mask]
        budget = self.crop_size - int(binder_mask.sum())

        if budget <= 0:
            raise ValueError(
                f"{getattr(graph, 'id', '?')}: the Fv alone is "
                f"{int(binder_mask.sum())} residues, at or over crop_size="
                f"{self.crop_size}; raise crop_size or tighten the trim.")

        to_binder = min_dist_per_res[target_mask][:, binder_idxs].min(dim=-1).values
        order = torch.argsort(to_binder)[:budget]
        keep_target = residue_idxs[target_mask][order]

        selected = torch.zeros(n_res, dtype=torch.bool, device=graph.coords.device)
        selected[binder_idxs] = True
        selected[keep_target] = True
        # Sorted so that chain order and contiguity are preserved.
        return self._crop_graph(graph, residue_idxs[selected])
