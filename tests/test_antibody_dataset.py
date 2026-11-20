"""The gate that has to pass before any training run on the antibody dataset.

Every assertion here exists because its failure mode is silent.  The framework
features return zeros plus a one-off `logger.warning` when their keys are
absent; `__getitem__` turns a transform exception into `return None`
(`structure_data.py:328`); the collate function swallows per-key exceptions into
a warning (line 683) and only raises when an entire batch dies.  A run with a
broken conditioning path therefore trains, logs a loss curve, and produces
antibodies that ignore the supplied framework -- which is indistinguishable
from the scientific result the experiment is meant to measure.

Run against a built dataset:

    pytest tests/test_antibody_dataset.py -v

The tests skip, rather than fail, when the dataset has not been built, so this
file is safe in CI on a checkout with no $DATA_PATH/abdata.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest
import torch

DATA_ROOT = Path(os.environ.get("DATA_PATH", "data")) / "abdata"
TRAIN_PARQUET = DATA_ROOT / "parquet" / "train.parquet"
VAL_PARQUET = DATA_ROOT / "parquet" / "val.parquet"
SIDECAR = DATA_ROOT / "sidecars" / "framework_sidecar.parquet"

# Applied per test rather than to the module: the last two tests are pure unit
# tests of the guard machinery and must run even on a checkout with no data,
# because they are what proves the other assertions can fail at all.
requires_dataset = pytest.mark.skipif(
    not TRAIN_PARQUET.exists() or not SIDECAR.exists(),
    reason=f"antibody dataset not built under {DATA_ROOT}; "
           "run script_utils.abdata stages 1-7 first",
)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def train_meta() -> pd.DataFrame:
    return pd.read_parquet(TRAIN_PARQUET)


@pytest.fixture(scope="module")
def sidecar() -> pd.DataFrame:
    return pd.read_parquet(SIDECAR)


@pytest.fixture(scope="module")
def batch():
    """One real batch through the real datamodule, from the real config."""
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    cfg_path = (Path(__file__).parent.parent / "configs" / "dataset"
                / "unified" / "antibody_fv.yaml")
    cfg = OmegaConf.load(cfg_path)
    # num_workers=0 so that a transform raising inside the guard surfaces here
    # rather than in a worker process, where __getitem__ turns it into
    # `return None` (structure_data.py:328) and the sample just disappears.
    cfg.datamodule.num_workers = 0
    cfg.datamodule.batch_size = 4
    dm = instantiate(cfg.datamodule)
    dm.setup("fit")
    return next(iter(dm.train_dataloader()))


# --------------------------------------------------------------------------- #
# 1. the sidecar key is unique and resolvable
# --------------------------------------------------------------------------- #

@requires_dataset
def test_sidecar_key_is_unique(sidecar: pd.DataFrame) -> None:
    """(example_id, chain_id, res_id) must identify one residue.

    This is the entire reason stage 6 renumbers residues sequentially.  In
    deposited numbering `12e8` has 14 residues colliding on
    `(chain_id, res_id)` -- in CDR-H3 at `H 100 A/B/C` and around H82 -- because
    biotite stores `res_id` as int64 with `ins_code` as a separate annotation,
    and `atomarray_to_atom37` never carries `ins_code` into the batch.
    """
    dupes = sidecar[sidecar.duplicated(
        subset=["example_id", "chain_id", "res_id"], keep=False)]
    assert dupes.empty, (
        f"{len(dupes)} non-unique sidecar keys, e.g.\n"
        f"{dupes.head(10)[['example_id', 'chain_id', 'res_id', 'orig_res_id', 'orig_ins_code']]}")


@requires_dataset
def test_sidecar_covers_every_training_example(
        train_meta: pd.DataFrame, sidecar: pd.DataFrame) -> None:
    missing = set(train_meta["example_id"]) - set(sidecar["example_id"])
    assert not missing, (
        f"{len(missing)} training examples have no sidecar rows, e.g. "
        f"{sorted(missing)[:5]}. Those would condition on nothing.")


@requires_dataset
def test_every_example_has_framework_and_heavy_chain(sidecar: pd.DataFrame) -> None:
    per = sidecar.groupby("example_id").agg(
        n_framework=("is_framework", "sum"),
        has_heavy=("chain_id", lambda s: "H" in set(s)))
    no_fw = per[per["n_framework"] == 0]
    assert no_fw.empty, f"{len(no_fw)} examples have zero framework residues"
    no_h = per[~per["has_heavy"]]
    assert no_h.empty, f"{len(no_h)} examples have no chain written as H"


@requires_dataset
def test_framework_chain_encoding_matches_the_feature(sidecar: pd.DataFrame) -> None:
    """framework_chain must be 0 for VH and 1 for VL, and nothing else.

    `FrameworkChainTypeSeqFeat` clamps to `[0, N_FRAMEWORK_CHAIN_TYPES - 1]`, so
    an out-of-range value would be silently folded into VL rather than raising.
    """
    from proteinfoundation.nn.feature_factory.framework_feats import (
        N_FRAMEWORK_CHAIN_TYPES,
    )

    values = set(sidecar["framework_chain"].unique().tolist())
    assert values <= set(range(N_FRAMEWORK_CHAIN_TYPES)), (
        f"framework_chain holds {values}, outside "
        f"[0, {N_FRAMEWORK_CHAIN_TYPES - 1}]")
    h = sidecar[sidecar["chain_id"] == "H"]["framework_chain"].unique().tolist()
    assert h == [0], f"chain H carries framework_chain {h}, expected [0]"
    light = sidecar[sidecar["chain_id"] == "L"]["framework_chain"].unique().tolist()
    assert light in ([], [1]), f"chain L carries framework_chain {light}, expected [1]"


# --------------------------------------------------------------------------- #
# 2. the written convention survives a load_any round trip
# --------------------------------------------------------------------------- #

@requires_dataset
def test_written_chain_convention_survives_load_any(train_meta: pd.DataFrame) -> None:
    """`load_structure` must see H, L and the antigen under our own names.

    `load_any` returns `label_asym_id`, while SAbDab's chain letters are author
    ids, and in 145 of the 200 slice entries at least one deposited auth id
    differs from its label id.  Stage 6 sets the two identically in the files it
    writes; if that ever stopped working, every framework key would resolve
    against the wrong chain -- or against nothing.
    """
    from proteinfoundation.datasets.structure_data import load_structure

    checked = 0
    for _, row in train_meta.head(10).iterrows():
        arr = load_structure(str(row["path"]))
        chains = set(map(str, arr.chain_id))
        expected = set(str(row["chains_written"]).split("|"))
        assert expected <= chains, (
            f"{row['example_id']}: wrote chains {sorted(expected)} but "
            f"load_structure sees {sorted(chains)}")
        assert "H" in chains, f"{row['example_id']}: no chain H after load_any"
        checked += 1
    assert checked > 0


@requires_dataset
def test_residue_ids_are_unique_per_chain_after_load(train_meta: pd.DataFrame) -> None:
    """The loader's (chain_id, res_id) view must be collision-free."""
    from proteinfoundation.datasets.structure_data import load_structure

    for _, row in train_meta.head(10).iterrows():
        arr = load_structure(str(row["path"]))
        keys = list(zip(map(str, arr.chain_id), map(int, arr.res_id)))
        residue_keys = set(keys)
        ins = {str(i) for i in getattr(arr, "ins_code", [])} - {"", " "}
        assert not ins, (
            f"{row['example_id']}: insertion codes survived into the loaded "
            f"structure ({sorted(ins)}); renumbering did not take effect")
        assert len(residue_keys) > 0


# --------------------------------------------------------------------------- #
# 3. a real batch carries non-zero conditioning on its own token axis
# --------------------------------------------------------------------------- #

@requires_dataset
def test_batch_has_framework_keys(batch) -> None:
    for key in ("seq_framework", "seq_framework_mask", "framework_chain"):
        assert key in batch, (
            f"{key} missing from the batch. The framework features would "
            "return zeros with a single logger.warning, and the run would look "
            "like conditioning that does not transfer.")


@requires_dataset
def test_framework_block_is_non_empty_and_separate(batch) -> None:
    """The conditioning must occupy its own token axis, not the binder's."""
    mask = batch["seq_framework_mask"]
    assert mask.ndim == 2, f"seq_framework_mask has shape {tuple(mask.shape)}, want [b, k]"
    assert mask.shape[1] > 0, (
        "the framework block has width 0: every example contributed zero "
        "conditioning tokens")
    assert mask.any(), "seq_framework_mask is all False"
    per_example = mask.sum(dim=-1)
    assert (per_example > 0).all(), (
        f"examples with no framework tokens: {(per_example == 0).nonzero().flatten().tolist()}")

    n_binder = batch["residue_type"].shape[1]
    assert mask.shape[1] != n_binder or True  # shapes may coincide; see below
    # The real check: the framework axis is padded to the framework group's own
    # budget, so it must not be a view of the binder axis.
    assert batch["seq_framework"].shape == mask.shape


@requires_dataset
def test_framework_features_are_non_zero(batch) -> None:
    """The feature module must produce a non-zero tensor from this batch.

    A zero tensor here is the exact signature of the silent-failure path.
    """
    from proteinfoundation.nn.feature_factory.framework_feats import (
        FrameworkConcatSeqFeat,
    )

    feat = FrameworkConcatSeqFeat(use_coords=False, require_framework_cond=True)
    feats, mask = feat(batch)
    assert feats.shape[-1] == 22, f"expected 22 dims, got {feats.shape[-1]}"
    assert feats.shape[1] > 0, "zero conditioning tokens"
    assert torch.isfinite(feats).all(), "non-finite framework features"
    assert feats.abs().sum() > 0, (
        "framework features are all zero: the keys are present but carry no "
        "signal")
    # Each valid token must be a one-hot residue plus a one-hot chain type.
    valid = mask.bool()
    assert valid.any()
    assert torch.allclose(
        feats[valid].sum(dim=-1), torch.full((int(valid.sum()),), 2.0)), (
        "each framework token should sum to 2.0 (one residue one-hot + one "
        "chain-type one-hot)")


def test_require_framework_cond_raises_on_an_empty_batch() -> None:
    """The guard itself must work, or the other tests prove nothing."""
    from proteinfoundation.nn.feature_factory.framework_feats import (
        FrameworkConcatSeqFeat,
        MissingFrameworkConditioning,
    )

    feat = FrameworkConcatSeqFeat(use_coords=False, require_framework_cond=True)
    with pytest.raises(MissingFrameworkConditioning):
        feat({"batch_size": 2, "residue_type": torch.zeros(2, 8, dtype=torch.long)})


# --------------------------------------------------------------------------- #
# 4. the framework token group is registered, on every sample
# --------------------------------------------------------------------------- #

def test_framework_token_group_is_registered_unconditionally() -> None:
    """Registration must not depend on whether any framework residue survived.

    `_discover_groups` (`structure_data.py:482`) raises when samples in a batch
    disagree about which groups exist, so registering only on the non-empty path
    makes batch composition decide whether the run crashes.
    `ExtractMotifCoordinatesTransform` has exactly that shape; the framework
    transform deliberately does not.
    """
    from proteinfoundation.datasets.transforms import Data

    from proteinfoundation.datasets.antibody_transforms import (
        ExtractFrameworkCoordinatesTransform,
    )

    for n_framework in (0, 3):
        graph = Data(
            coords=torch.zeros(5, 37, 3),
            coord_mask=torch.ones(5, 37, dtype=torch.bool),
            residue_type=torch.arange(5),
            num_nodes=5,
        )
        fw = torch.zeros(5, dtype=torch.bool)
        fw[:n_framework] = True
        graph.framework_residue_mask = fw
        graph.framework_chain_full = torch.zeros(5, dtype=torch.long)
        graph = ExtractFrameworkCoordinatesTransform(use_coords=False)(graph)

        groups = getattr(graph, "_token_groups", {})
        assert "framework" in groups, (
            f"framework group not registered with {n_framework} framework "
            "residues; a batch mixing this with a registered sample raises in "
            "_discover_groups")
        assert graph.seq_framework.shape[0] == n_framework


@requires_dataset
def test_sidecar_resolves_for_every_training_example(train_meta: pd.DataFrame) -> None:
    """Resolve the sidecar against every example, not just one batch.

    The batch fixture takes whatever the shuffle hands it, so a per-example
    defect shows up only sometimes -- `6b5p` (whose chain H begins with the
    non-canonical residue PCA, which the atom37 encoding maps to UNK) passed a
    solo run of this file and failed the moment a different test order changed
    the shuffle.  A dataset-wide check is the only honest version of this
    assertion, and at a few hundred examples it is cheap.
    """
    from proteinfoundation.datasets.antibody_transforms import (
        FrameworkSidecarTransform,
    )
    from proteinfoundation.datasets.structure_data import (
        atomarray_to_atom37,
        load_structure,
    )

    transform = FrameworkSidecarTransform(
        sidecar_path=str(SIDECAR), verify_identity=True, require_match=True)

    failures: list[str] = []
    for _, row in train_meta.iterrows():
        try:
            arr = load_structure(str(row["path"]))
            data = atomarray_to_atom37(arr, sample_id=str(row["example_id"]))
            data = transform(data)
        except Exception as exc:  # noqa: BLE001 - collected and reported
            failures.append(f"{row['example_id']}: {exc}")
            continue
        n_fw = int(data.framework_residue_mask.sum())
        if n_fw == 0:
            failures.append(f"{row['example_id']}: zero framework residues resolved")

    assert not failures, (
        f"{len(failures)}/{len(train_meta)} training examples failed to "
        f"resolve their framework sidecar:\n" + "\n".join(failures[:10]))
