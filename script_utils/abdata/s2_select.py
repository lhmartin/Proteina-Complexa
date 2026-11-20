"""Stage 2 - curation filters and slice selection, from the cached summary.

Produces one row per surviving antibody instance.  Choosing *one* instance per
PDB entry is deliberately not done here: the tie-break is resolved-residue
count, which needs the structure, so it belongs to stage 5.

Every filter reports how many rows and entries it removed, into the build
manifest.  A yield number is an output of this stage, not an input to it.

Two things about the SAbDab2 summary that the filters here depend on, both
established by inspecting the file rather than assumed:

  * `antigen_type` is a *set* of the polymer types present among the antigen
    chains, not one element per chain.  `antigen_chain="C|D"` with
    `antigen_type="PROTEIN"` is two protein chains; `"A|B|J"` with
    `"PROTEIN|SUGAR"` says nothing about which of the three is the sugar.  So
    this stage can only require that *some* protein or peptide is present; the
    per-chain decision needs the structure and is made in stage 3.
  * For chain-sharing constructs the chain id carries a domain suffix: an scFv
    on auth chain N is reported as `Hchain="N1"`, `Lchain="N2"`.  That suffix
    must only be stripped for chain-sharing rows, because `1h`, `2h` and `Ah`
    are genuine multi-character auth chain ids elsewhere in the file.

    python -m script_utils.abdata.s2_select
    python -m script_utils.abdata.s2_select --slice-size none --max-resolution 3.0
"""

from __future__ import annotations

import argparse
import re
import sys

import numpy as np
import pandas as pd
from loguru import logger

from script_utils.abdata.config import (
    CHAIN_SHARING_DROPPED,
    IG_TYPE_FROM_SABDAB,
    BuildConfig,
    add_common_args,
    config_from_args,
    record_stage,
)

CDR_H = ["CDR-H1", "CDR-H2", "CDR-H3"]
CDR_L = ["CDR-L1", "CDR-L2", "CDR-L3"]

_DOMAIN_SUFFIX = re.compile(r"^(?P<base>.+?)(?P<domain>\d)$")


def pdb_id_from_extended(extended: str) -> str:
    """`pdb_00001a2y` -> `1a2y`.  SAbDab2 reports the 12-char extended id."""
    s = str(extended).strip().lower()
    return s[-4:] if s.startswith("pdb_") else s


def parse_sabdab_chain(chain: object, is_chain_sharing: bool) -> tuple[str, int | None]:
    """Split a SAbDab chain id into (auth chain, domain index).

    The domain suffix is only a suffix on chain-sharing rows.  Applying the
    split unconditionally would turn the real auth chain id `B1` of an ordinary
    Fab into chain `B`, which is a different chain in the same entry.
    """
    c = str(chain or "").strip()
    if not c or not is_chain_sharing:
        return c, None
    m = _DOMAIN_SUFFIX.match(c)
    if not m:
        return c, None
    return m.group("base"), int(m.group("domain"))


def _split_pipe(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    s = str(value).strip()
    if not s or s.upper() == "NA":
        return []
    return [p.strip() for p in s.split("|") if p.strip()]


def _parse_resolution(value: object) -> float | None:
    try:
        r = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return r if np.isfinite(r) and r > 0 else None


def assign_ig_type(row: pd.Series, allow_scfv: bool) -> str | None:
    """Our format label, which is not quite SAbDab's `type` column.

    `chainsharing_construct == "SCFV"` takes precedence over `type`, because
    every scFv is reported as FV: the construct column is the only place the
    single-polypeptide fact is recorded.
    """
    construct = str(row.get("chainsharing_construct", "") or "").strip().upper()
    if construct == "SCFV":
        return "scfv" if allow_scfv else None
    return IG_TYPE_FROM_SABDAB.get(str(row.get("type", "") or "").strip().upper())


def select(cfg: BuildConfig) -> pd.DataFrame:
    if not cfg.summary_csv.exists():
        raise FileNotFoundError(
            f"{cfg.summary_csv} missing; run s1_fetch_summary first")

    df = pd.read_csv(cfg.summary_csv, dtype=str, keep_default_na=False)
    df["pdb_id"] = df["PDB"].map(pdb_id_from_extended)
    n0, e0 = len(df), df["pdb_id"].nunique()
    logger.info(f"summary: {n0} instance rows over {e0} entries")

    drops: list[dict] = []

    def apply(name: str, keep_fn, note: str = "") -> None:
        """Apply one filter.

        `keep_fn` is a callable rather than a Series so that it is evaluated
        against the *current* frame.  Precomputing masks and applying them
        after an intervening filter silently reindexes them.
        """
        nonlocal df
        keep = pd.Series(keep_fn(df), index=df.index).fillna(False).astype(bool)
        before_rows, before_entries = len(df), df["pdb_id"].nunique()
        df = df[keep].copy()
        drops.append({
            "filter": name,
            "rows_dropped": before_rows - len(df),
            "entries_dropped": before_entries - df["pdb_id"].nunique(),
            "rows_remaining": len(df),
            "entries_remaining": int(df["pdb_id"].nunique()),
            "note": note,
        })
        logger.info(f"{name}: -{before_rows - len(df)} rows "
                    f"-> {len(df)} rows / {df['pdb_id'].nunique()} entries")

    # --- experimental method and resolution ---------------------------------
    apply("method_kept", lambda d: d["method"].str.upper().isin(cfg.methods_kept),
          f"kept {list(cfg.methods_kept)}")

    df["resolution_value"] = df["resolution"].map(_parse_resolution)
    if cfg.require_resolution:
        apply("resolution_present", lambda d: d["resolution_value"].notna())
    apply("resolution_le_max",
          lambda d: d["resolution_value"].isna()
          | (d["resolution_value"] <= cfg.max_resolution),
          f"<= {cfg.max_resolution} A")

    # --- construct format ----------------------------------------------------
    apply("chain_sharing_construct",
          lambda d: ~d["chainsharing_construct"].str.strip().str.upper()
          .isin(CHAIN_SHARING_DROPPED),
          f"dropped {list(CHAIN_SHARING_DROPPED)}; SCFV kept as its own ig_type. "
          "Diabody-class constructs pair domains across different base chains "
          "(A1 with C2), which framework_chain's VH=0/VL=1 cannot express")

    df["ig_type"] = df.apply(assign_ig_type, axis=1, allow_scfv=cfg.allow_scfv)
    apply("ig_type_known", lambda d: d["ig_type"].notna(),
          "unmapped SAbDab `type` values")
    kept_ig = tuple(cfg.ig_types_kept) + (("scfv",) if cfg.allow_scfv else ())
    apply("ig_type_kept", lambda d: d["ig_type"].isin(kept_ig),
          f"kept {list(kept_ig)}")

    # --- chain roles ---------------------------------------------------------
    is_sharing = df["chainsharing_construct"].str.strip().ne("")
    h_parsed = [parse_sabdab_chain(c, s) for c, s in zip(df["Hchain"], is_sharing)]
    l_parsed = [parse_sabdab_chain(c, s) for c, s in zip(df["Lchain"], is_sharing)]
    df["h_chain"] = [p[0] for p in h_parsed]
    df["h_domain"] = [p[1] for p in h_parsed]
    df["l_chain"] = [p[0] for p in l_parsed]
    df["l_domain"] = [p[1] for p in l_parsed]

    if cfg.require_heavy_chain:
        apply("heavy_chain_present", lambda d: d["h_chain"].ne(""))

    apply("light_chain_consistent_with_format",
          lambda d: ~d["ig_type"].isin(["paired_fv", "scfv"]) | d["l_chain"].ne(""),
          "paired_fv / scfv require a light chain")

    # An scFv is one polypeptide, so its two domains must sit on the same auth
    # chain.  7 SCFV-labelled rows in the file point at two genuinely different
    # chains (22jx, 22jz, 6s2i); whatever those are, they are not a single-chain
    # Fv, and reconciling them silently would put a mislabelled construct in.
    apply("scfv_domains_share_one_chain",
          lambda d: d["ig_type"].ne("scfv") | (d["h_chain"] == d["l_chain"]),
          "scfv requires base(Hchain) == base(Lchain)")
    apply("paired_fv_chains_distinct",
          lambda d: d["ig_type"].ne("paired_fv") | (d["h_chain"] != d["l_chain"]),
          "paired_fv requires two different chains")

    # --- CDR loop sequences present -----------------------------------------
    def _cdrs_ok(d: pd.DataFrame) -> pd.Series:
        h = pd.concat([d[c].str.strip().ne("") for c in CDR_H], axis=1).all(axis=1)
        lt = pd.concat([d[c].str.strip().ne("") for c in CDR_L], axis=1).all(axis=1)
        needs_light = d["ig_type"].isin(["paired_fv", "scfv"])
        return h & (~needs_light | lt)

    apply("cdr_sequences_present", _cdrs_ok,
          "SAbDab CDR loop sequences are the stage-5 cross-check reference")

    # --- antigen -------------------------------------------------------------
    df["antigen_chains"] = ["|".join(dict.fromkeys(_split_pipe(v)))
                            for v in df["antigen_chain"]]
    df["antigen_types_set"] = ["|".join(dict.fromkeys(
        t.upper() for t in _split_pipe(v))) for v in df["antigen_type"]]

    if cfg.require_antigen:
        apply("antigen_present",
              lambda d: d["antigen_chains"].ne("") & d["antigen_types_set"].ne(""),
              "bound, not apo")

    kept_ag = {t.upper() for t in cfg.antigen_types_kept}
    apply("antigen_type_includes_protein_or_peptide",
          lambda d: [bool(set(_split_pipe(v)) & kept_ag) for v in d["antigen_types_set"]],
          f"set semantics: at least one of {sorted(kept_ag)} among the antigen "
          "chains. Which chain is which is decided from the structure in stage 3, "
          "because antigen_type carries no per-chain mapping")

    # A chain cannot be both antibody and antigen.
    apply("antigen_distinct_from_antibody",
          lambda d: [not (set(_split_pipe(ac)) & ({h, lc} - {""}))
                     for ac, h, lc in zip(d["antigen_chains"], d["h_chain"],
                                          d["l_chain"])],
          "overlap means the role assignment is unresolvable")

    curated = df.copy()

    # --- slice ---------------------------------------------------------------
    pinned = [p.lower() for p in cfg.pinned_pdb_ids]
    surviving = set(curated["pdb_id"])
    pinned_present = [p for p in pinned if p in surviving]
    pinned_missing = [p for p in pinned if p not in surviving]
    if pinned_missing:
        logger.warning(
            f"pinned entries did not survive curation: {pinned_missing}. "
            "They are NOT force-included: bypassing a curation filter would "
            "hand stage 6 an entry it cannot write.")

    if cfg.slice_size is None:
        chosen_entries = sorted(surviving)
    else:
        rest = sorted(surviving - set(pinned_present))
        n_extra = max(0, cfg.slice_size - len(pinned_present))
        picked: list[str] = []
        if rest and n_extra:
            rng = np.random.default_rng(cfg.seed)
            picked = [str(x) for x in rng.choice(
                rest, size=min(n_extra, len(rest)), replace=False)]
        chosen_entries = sorted(set(pinned_present) | set(picked))

    out = curated[curated["pdb_id"].isin(chosen_entries)].copy()
    out["is_pinned"] = out["pdb_id"].isin(pinned_present)
    out = out.sort_values(["pdb_id", "h_chain", "l_chain"]).reset_index(drop=True)

    logger.info(f"slice: {len(out)} instance rows over "
                f"{out['pdb_id'].nunique()} entries "
                f"({len(pinned_present)} pinned)")
    logger.info("ig_type: " + ", ".join(
        f"{k}={v}" for k, v in out["ig_type"].value_counts().items()))

    cfg.mkdirs()
    out.to_csv(cfg.candidates_csv, index=False)
    logger.info(f"wrote {cfg.candidates_csv}")

    record_stage(cfg, "s2_select", {
        "n_rows_in": n0,
        "n_entries_in": e0,
        "filters": drops,
        "n_rows_curated": len(curated),
        "n_entries_curated": int(curated["pdb_id"].nunique()),
        "n_rows_slice": len(out),
        "n_entries_slice": int(out["pdb_id"].nunique()),
        "ig_type_counts_curated": curated["ig_type"].value_counts().to_dict(),
        "ig_type_counts_slice": out["ig_type"].value_counts().to_dict(),
        "pinned_present": pinned_present,
        "pinned_missing": pinned_missing,
        "seed": cfg.seed,
    })
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    select(config_from_args(parser.parse_args(argv)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
