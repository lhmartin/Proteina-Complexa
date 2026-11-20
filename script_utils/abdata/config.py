"""Build parameters and on-disk layout for the antibody dataset.

Every curation threshold lives here rather than inline in a stage, so that the
manifest can record the exact configuration a build ran under and a re-run with
a changed filter is a config edit rather than a code edit.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

SABDAB_API = "https://sabdab.opig.stats.ox.ac.uk/api"
RCSB_GRAPHQL = "https://data.rcsb.org/graphql"
RCSB_FILES = "https://files.rcsb.org/download"

# A generic client name: never identify the user to an external service.
USER_AGENT = "complexa-abdata/0.1"

# SAbDab `type` -> our ig_type.  SD-H is the nanobody/VHH class; VNAR (shark
# IgNAR) and SD-L (single light domain) are separate single-domain families and
# are kept distinguishable rather than folded into vhh.
IG_TYPE_FROM_SABDAB = {
    "SD-H": "vhh",
    "VNAR": "vnar",
    "SD-L": "sd_l",
    "FV": "paired_fv",
    "FAB": "paired_fv",
    "FAB+FC": "paired_fv",
}

# Antigen types we accept, applied per pipe-delimited element.  Haptens, sugars,
# ions and nucleic acids belong on the ligand path.
ANTIGEN_TYPES_KEPT = ("PROTEIN", "PEPTIDE")

# `chainsharing_construct` values that are dropped.  SCFV is deliberately NOT
# here: one chain carrying two variable domains is exactly the scFv case, which
# is admitted with the linker discarded.  Diabodies and the bispecific formats
# put two *different* specificities on shared chains, which the VH=0/VL=1
# encoding of `framework_chain` cannot express.
CHAIN_SHARING_DROPPED = ("DIABODY", "CODV", "SCDB", "DVD", "OTHER")

# Entries pinned into the slice so it connects to the probe scripts and
# autoencoder reconstruction tests that already use them.
#
# 12e8 is deliberately absent despite being one of those five: both its
# antibody instances have an empty antigen_chain, so it is an unbound Fab and
# cannot appear in a bound-complex dataset.  It remains the reference case for
# the insertion-code collision (14 residues colliding on (chain_id, res_id),
# in CDR-H3 and H82) that motivates sequential renumbering in stage 6.
PINNED_PDB_IDS = ("1a2y", "1bj1", "1n8z", "4hkx")

# IMGT variable-domain span used to trim constant domains off the antibody.
# Mirrors atomworks.ml.utils.antibody.VARIABLE_DOMAIN_IMGT_RANGE.
VARIABLE_DOMAIN_IMGT_RANGE = (1, 128)


@dataclass
class BuildConfig:
    """One antibody dataset build."""

    # ---- layout -------------------------------------------------------------
    root: Path = field(default_factory=lambda: Path(
        os.environ.get("DATA_PATH", "data")) / "abdata")

    # ---- entry-level curation ----------------------------------------------
    max_resolution: float = 3.5
    require_resolution: bool = True        # drop entries with no resolution
    require_heavy_chain: bool = True
    require_antigen: bool = True           # bound, not apo
    antigen_types_kept: tuple[str, ...] = ANTIGEN_TYPES_KEPT
    ig_types_kept: tuple[str, ...] = ("vhh", "paired_fv")
    allow_scfv: bool = True
    methods_kept: tuple[str, ...] = ("XRAY", "ELECTRON_MICROSCOPY")

    # ---- slice -------------------------------------------------------------
    # The first build is a proving slice, not the corpus.  slice_size=None
    # takes everything that survives curation.
    slice_size: int | None = 200
    pinned_pdb_ids: tuple[str, ...] = PINNED_PDB_IDS
    seed: int = 20261120

    # ---- splits ------------------------------------------------------------
    # Random for the slice; the second build replaces this with temporal
    # windows plus a deposition-date leak check.
    split_mode: str = "random"             # "random" | "temporal"
    n_val: int = 20
    n_test: int = 0
    temporal_val_from: str = "2022-01-01"  # split_mode="temporal" only
    temporal_test_from: str = "2023-01-01"

    # ---- structure handling -------------------------------------------------
    trim_to_variable_domain: bool = True
    variable_domain_imgt_range: tuple[int, int] = VARIABLE_DOMAIN_IMGT_RANGE
    keep_antigen_whole: bool = True        # no epitope crop at build time
    require_fully_resolved_cdrs: bool = True
    # anarcii "accuracy" or "speed".  Accuracy is a torch model on CPU and is
    # the build's bottleneck: ~600 chains of the 200-entry slice take tens of
    # minutes, so the full corpus at ~24,000 chains would be impractical.
    # Speed mode is defensible because numbering is independently guarded --
    # the IMGT-conserved residues must carry their defined identity and the
    # North CDR spans must contain SAbDab's own loop sequences, so a bad
    # numbering is dropped rather than trusted.
    anarcii_mode: str = "accuracy"
    renumber_sequentially: bool = True     # (chain_id, res_id) must be unique

    # ---- model-side defaults recorded alongside the data -------------------
    crop_size: int = 500
    binder_max_length: int = 250
    batch_size: int = 4
    num_workers: int = 8

    # ---- runtime -----------------------------------------------------------
    n_workers_fetch: int = 8
    force: bool = False

    # -- derived paths --------------------------------------------------------
    @property
    def cache(self) -> Path:
        return self.root / "cache"

    @property
    def cif_cache(self) -> Path:
        return self.cache / "cif"

    @property
    def structures(self) -> Path:
        return self.root / "structures"

    @property
    def sidecars(self) -> Path:
        return self.root / "sidecars"

    @property
    def parquet(self) -> Path:
        return self.root / "parquet"

    @property
    def summary_csv(self) -> Path:
        return self.cache / "sabdab_summary.csv"

    @property
    def candidates_csv(self) -> Path:
        return self.cache / "candidates.csv"

    @property
    def rcsb_csv(self) -> Path:
        return self.cache / "rcsb_metadata.csv"

    @property
    def resolved_csv(self) -> Path:
        """Stage 3 output.  Kept separate from candidates.csv so that a stage
        can always be re-run without having to re-run the stage before it."""
        return self.cache / "resolved.csv"

    @property
    def annotations_parquet(self) -> Path:
        return self.cache / "annotations.parquet"

    @property
    def manifest_json(self) -> Path:
        return self.root / "build_manifest.json"

    def mkdirs(self) -> None:
        for p in (self.cache, self.cif_cache, self.structures,
                  self.sidecars, self.parquet):
            p.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["root"] = str(self.root)
        return d


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Expose every BuildConfig field as a flag, so defaults are overridable."""
    defaults = BuildConfig()
    for f in dataclasses.fields(defaults):
        value = getattr(defaults, f.name)
        flag = "--" + f.name.replace("_", "-")
        if f.name == "force":
            parser.add_argument(flag, action="store_true",
                                help="redo work even if the cache has it")
        elif isinstance(value, tuple):
            parser.add_argument(flag, nargs="*", default=None,
                                help=f"default: {list(value)}")
        elif isinstance(value, bool):
            parser.add_argument(flag, type=_boolish, default=None,
                                help=f"default: {value}")
        elif isinstance(value, Path):
            parser.add_argument(flag, type=Path, default=None,
                                help=f"default: {value}")
        elif value is None or isinstance(value, int) and not isinstance(value, bool):
            parser.add_argument(flag, type=_int_or_none, default=None,
                                help=f"default: {value}")
        elif isinstance(value, float):
            parser.add_argument(flag, type=float, default=None,
                                help=f"default: {value}")
        else:
            parser.add_argument(flag, default=None, help=f"default: {value}")
    return parser


def _boolish(s: str) -> bool:
    if s.lower() in ("1", "true", "yes", "y", "on"):
        return True
    if s.lower() in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"not a boolean: {s}")


def _int_or_none(s: str) -> int | None:
    if s.lower() in ("none", "null", ""):
        return None
    return int(s)


def config_from_args(args: argparse.Namespace) -> BuildConfig:
    """Build a config from parsed args, leaving unspecified flags at default."""
    kwargs = {}
    for f in dataclasses.fields(BuildConfig):
        v = getattr(args, f.name, None)
        if v is None:
            continue
        if isinstance(getattr(BuildConfig(), f.name), tuple):
            v = tuple(v)
        kwargs[f.name] = v
    return BuildConfig(**kwargs)


def record_stage(cfg: BuildConfig, stage: str, payload: dict) -> None:
    """Append one stage's counts to the build manifest.

    Per-filter drop counts live here instead of in a log line, because the
    question "how many complexes did we lose, and to which filter" is the one
    that gets asked after the build, not during it.
    """
    cfg.root.mkdir(parents=True, exist_ok=True)
    manifest: dict = {}
    if cfg.manifest_json.exists():
        manifest = json.loads(cfg.manifest_json.read_text())
    manifest.setdefault("config", cfg.to_dict())
    manifest.setdefault("stages", {})[stage] = payload
    cfg.manifest_json.write_text(json.dumps(manifest, indent=2, default=str))
