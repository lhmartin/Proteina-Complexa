"""Stage 5 - number the variable domains, cross-check them, pick one instance.

Produces the two artifacts stage 6 writes from:

  * `annotations.parquet` - long format, one row per antibody residue, carrying
    the original `(chain_id, res_id, ins_code)` key, the IMGT number, the North
    CDR id and the framework flag.  This is the sidecar in its pre-renumbering
    form.
  * `instances.parquet` - one row per surviving antibody instance, with the
    cross-check results, the resolved-residue counts and the one-per-entry
    selection already applied.

Only the chains SAbDab names are numbered.  Numbering every chain of every
entry would spend most of its time rejecting histones and DNA: on `6e0c` the
ten non-antibody chains took the great majority of a two-minute run and all
correctly returned nothing.

Three checks decide whether an instance survives, and all three are drops
rather than reconciliations:

  * **Role agreement.**  A chain SAbDab calls heavy must number as `H`; a light
    chain must number as `K` or `L`.
  * **CDR agreement.**  Each of SAbDab's CDR loop sequences must be contained
    in our North CDR span for that loop.
  * **Fully resolved CDRs.**  Every CDR residue must carry N, CA, C and O, and
    consecutive CDR residues must be consecutive in `label_seq_id`.  A CDR with
    missing density means the residues we are asking the model to generate were
    never observed.

    python -m script_utils.abdata.s5_annotate
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from loguru import logger

from script_utils.abdata import structure_io
from script_utils.abdata.config import (
    BuildConfig,
    add_common_args,
    config_from_args,
    record_stage,
)
from script_utils.abdata.ig_numbering import (
    CDR_KEYS,
    Domain,
    cdr_crosscheck,
    number_chains,
)
from script_utils.abdata.s4_fetch_structures import cif_path

LIGHT_TYPES = ("K", "L")

# The 20 residues FrameworkSequenceSeqFeat can express: it is a 20-dim one-hot
# over ResidueTypeSeqFeat.  A modified residue cannot be a conditioning token
# because the atom37 encoding maps it to UNK, so it is left out of the mask
# while keeping its coordinates in the written structure.
#
# The test has to be on the THREE-letter name, not the one-letter code.
# `find_tabulated_residue` resolves modified residues to their parent letter --
# MSE -> M, PCA -> E, TYS -> Y -- which is what we want for numbering (anarcii
# aligns them correctly) but means a one-letter test sees nothing wrong.  The
# encoding keys on the residue name, so PCA at 6b5p/9c7f/5kvf H1, TYS at 6w4m
# H104-105 and MSE through 2p43 all arrive in the batch as UNK.
CANONICAL_AA3 = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}


def example_id(row: pd.Series) -> str:
    parts = [row["pdb_id"], str(row["h_chain"])]
    if str(row["l_chain"]):
        parts.append(str(row["l_chain"]))
    return "_".join(parts)


def cdrs_from_sabdab(row: pd.Series, heavy: bool) -> dict[int, str]:
    prefix = "CDR-H" if heavy else "CDR-L"
    return {i: str(row.get(f"{prefix}{i}", "") or "").strip() for i in (1, 2, 3)}


def cdr_fully_resolved(residues: list[structure_io.Residue], domain: Domain
                       ) -> tuple[bool, str]:
    """Every CDR residue observed with a complete backbone, and no gaps."""
    for which in (1, 2, 3):
        idx = np.where(domain.cdr_id == which)[0]
        if len(idx) == 0:
            return False, f"cdr{which}_empty"
        for i in idx:
            if not residues[int(i)].has_backbone:
                return False, f"cdr{which}_incomplete_backbone"
        labels = [residues[int(i)].label_seq for i in idx]
        if all(v > 0 for v in labels):
            if any(b - a != 1 for a, b in zip(labels, labels[1:])):
                return False, f"cdr{which}_unresolved_gap"
    return True, ""


def _pick_domains(doms: list[Domain], ig_type: str
                  ) -> tuple[Domain | None, Domain | None, str]:
    """Which numbered domains play the heavy and light roles."""
    heavy = next((d for d in doms if d.chain_type == "H"), None)
    if ig_type == "vhh":
        return heavy, None, "" if heavy else "no_heavy_domain"
    light = next((d for d in doms
                  if d.chain_type in LIGHT_TYPES
                  and (heavy is None or d.start > heavy.end)), None)
    if heavy is None:
        return None, light, "no_heavy_domain"
    if light is None:
        return heavy, None, "no_light_domain"
    return heavy, light, ""


def annotate(cfg: BuildConfig) -> pd.DataFrame:
    src = cfg.resolved_csv if cfg.resolved_csv.exists() else cfg.candidates_csv
    if not src.exists():
        raise FileNotFoundError(f"{src} missing; run the earlier stages first")
    rows = pd.read_csv(src, dtype=str, keep_default_na=False)
    logger.info(f"{len(rows)} instances over {rows['pdb_id'].nunique()} entries")

    # --- read every structure once, keeping only the named chains -----------
    needed: dict[str, set[str]] = defaultdict(set)
    for _, r in rows.iterrows():
        for c in (r["h_chain"], r["l_chain"]):
            if c:
                needed[r["pdb_id"]].add(c)

    residues: dict[tuple[str, str], list[structure_io.Residue]] = {}
    auth_label: dict[str, dict[str, list[str]]] = {}
    missing_file: list[str] = []
    missing_chain: list[str] = []
    for n, (pdb_id, chains) in enumerate(sorted(needed.items()), 1):
        path = cif_path(cfg, pdb_id)
        if not path.exists():
            missing_file.append(pdb_id)
            continue
        try:
            by_chain = structure_io.read_chains(path)
            auth_label[pdb_id] = structure_io.auth_label_map(path)
        except Exception as exc:  # noqa: BLE001 - counted, not fatal
            logger.warning(f"{pdb_id}: unreadable ({exc})")
            missing_file.append(pdb_id)
            continue
        for c in chains:
            if c not in by_chain:
                missing_chain.append(f"{pdb_id}:{c}")
                continue
            residues[(pdb_id, c)] = by_chain[c]
        if n % 50 == 0:
            logger.info(f"  read {n}/{len(needed)} structures")

    logger.info(f"{len(residues)} named chains read; "
                f"{len(missing_file)} unreadable entries, "
                f"{len(missing_chain)} chains absent from their file")

    # --- number them, batched across the whole slice ------------------------
    sequences = {f"{p}|{c}": structure_io.sequence_of(rs)
                 for (p, c), rs in residues.items()}
    logger.info(f"numbering {len(sequences)} chains (anarcii mode={cfg.anarcii_mode})")
    domains = number_chains(sequences, mode=cfg.anarcii_mode)
    n_with = sum(1 for v in domains.values() if v)
    logger.info(f"{n_with}/{len(sequences)} chains yielded at least one domain")

    # --- per-instance assembly and checks -----------------------------------
    drops: dict[str, int] = defaultdict(int)
    kept: list[dict] = []
    long_rows: list[dict] = []

    for _, row in rows.iterrows():
        eid = example_id(row)
        pdb_id, ig_type = row["pdb_id"], row["ig_type"]
        h_key = f"{pdb_id}|{row['h_chain']}"
        l_key = f"{pdb_id}|{row['l_chain']}" if row["l_chain"] else None

        if h_key not in sequences or (l_key and l_key not in sequences):
            drops["chain_missing_from_structure"] += 1
            continue

        if ig_type == "scfv":
            # Both domains sit on one chain, so both come from the heavy key.
            h_dom, l_dom, why = _pick_domains(domains[h_key], "paired_fv")
            l_res_key = h_key
        else:
            h_dom, _, why = _pick_domains(domains[h_key], "vhh")
            l_dom, l_res_key = None, None
            if ig_type == "paired_fv":
                l_dom = next((d for d in domains[l_key]
                              if d.chain_type in LIGHT_TYPES), None)
                l_res_key = l_key
                if h_dom is not None and l_dom is None:
                    why = "no_light_domain"
        if why:
            drops[why] += 1
            continue

        h_seq = sequences[h_key]
        h_res = residues[(pdb_id, row["h_chain"])]

        # Role agreement: SAbDab's heavy must number as H, light as K or L.
        if h_dom.chain_type != "H":
            drops["heavy_not_numbered_as_H"] += 1
            continue
        if l_dom is not None and l_dom.chain_type not in LIGHT_TYPES:
            drops["light_not_numbered_as_K_or_L"] += 1
            continue

        # Scheme-conserved residues must carry their defined identity.
        if not h_dom.conserved_ok() or (l_dom is not None and not l_dom.conserved_ok()):
            drops["imgt_conserved_residue_mismatch"] += 1
            continue

        # CDR agreement with SAbDab.
        h_ok, h_detail = cdr_crosscheck(h_seq, h_dom, cdrs_from_sabdab(row, True))
        if not h_ok:
            drops["heavy_cdr_disagreement"] += 1
            continue
        l_detail = {}
        if l_dom is not None:
            l_seq = sequences[l_res_key]
            l_ok, l_detail = cdr_crosscheck(l_seq, l_dom, cdrs_from_sabdab(row, False))
            if not l_ok:
                drops["light_cdr_disagreement"] += 1
                continue

        # Fully resolved CDRs.
        if cfg.require_fully_resolved_cdrs:
            ok, why2 = cdr_fully_resolved(h_res, h_dom)
            if not ok:
                drops[f"heavy_{why2}"] += 1
                continue
            if l_dom is not None:
                l_res = residues[(pdb_id, row["l_chain"] if ig_type != "scfv"
                                  else row["h_chain"])]
                ok, why2 = cdr_fully_resolved(l_res, l_dom)
                if not ok:
                    drops[f"light_{why2}"] += 1
                    continue

        # --- emit the per-residue sidecar rows ------------------------------
        n_fw = n_cdr = 0
        for role, dom, res_list, chain_id in (
                ("H", h_dom, h_res, row["h_chain"]),
                ("L", l_dom,
                 residues[(pdb_id, row["l_chain"])] if (
                     l_dom is not None and ig_type != "scfv") else h_res,
                 row["l_chain"] if ig_type != "scfv" else row["h_chain"])):
            if dom is None:
                continue
            inside = np.where(dom.is_framework | (dom.cdr_id > 0))[0]
            for i in inside:
                r = res_list[int(i)]
                if r.res_name.strip().upper() not in CANONICAL_AA3:
                    drops[f"noncanonical_residue_{r.res_name}"] += 1
                    continue
                long_rows.append({
                    "example_id": eid,
                    "role": role,
                    "chain_id": chain_id,
                    "res_id": r.res_id,
                    "ins_code": r.ins_code,
                    "label_seq": r.label_seq,
                    "res_name": r.res_name,
                    "aa1": r.aa1,
                    "imgt": int(dom.imgt[i]),
                    "cdr_id": int(dom.cdr_id[i]),
                    "is_framework": bool(dom.is_framework[i]),
                    "chain_type": dom.chain_type,
                })
            n_fw += int(dom.is_framework.sum())
            n_cdr += int((dom.cdr_id > 0).sum())

        kept.append({
            "example_id": eid,
            "pdb_id": pdb_id,
            "ig_type": ig_type,
            "h_chain": row["h_chain"],
            "l_chain": row["l_chain"],
            "h_chain_type": h_dom.chain_type,
            "l_chain_type": l_dom.chain_type if l_dom is not None else "",
            "antigen_chains": row["antigen_chains_resolved"]
            if "antigen_chains_resolved" in row else row["antigen_chains"],
            "n_framework": n_fw,
            "n_cdr": n_cdr,
            "n_resolved_antibody_residues": len(h_res) + (
                0 if (l_dom is None or ig_type == "scfv")
                else len(residues[(pdb_id, row["l_chain"])])),
            "h_score": h_dom.score,
            "l_score": l_dom.score if l_dom is not None else 0.0,
            "resolution": row.get("rcsb_resolution", "") or row.get("resolution", ""),
            "deposit_date": row.get("deposit_date", ""),
            "release_date": row.get("release_date", ""),
            "is_pinned": row.get("is_pinned", "False"),
            "cdr_crosscheck_heavy": str(h_detail),
            "cdr_crosscheck_light": str(l_detail),
        })

    inst = pd.DataFrame(kept)
    logger.info(f"{len(inst)} instances passed all checks "
                f"({inst['pdb_id'].nunique() if len(inst) else 0} entries)")
    for k, v in sorted(drops.items(), key=lambda kv: -kv[1]):
        logger.info(f"  dropped {v:>4}  {k}")

    # --- one instance per entry, deterministically --------------------------
    n_before = len(inst)
    if len(inst):
        inst["_antigen_n"] = inst["antigen_chains"].str.count(r"\|") + 1
        inst = (inst
                .sort_values(["pdb_id", "n_resolved_antibody_residues",
                              "_antigen_n", "h_chain", "l_chain"],
                             ascending=[True, False, False, True, True])
                .groupby("pdb_id", as_index=False, sort=True)
                .head(1)
                .drop(columns=["_antigen_n"])
                .reset_index(drop=True))
        long_df = pd.DataFrame(long_rows)
        long_df = long_df[long_df["example_id"].isin(set(inst["example_id"]))]
    else:
        long_df = pd.DataFrame(long_rows)

    logger.info(f"one per entry: {n_before} -> {len(inst)} instances")
    if len(inst):
        logger.info("ig_type: " + ", ".join(
            f"{k}={v}" for k, v in inst["ig_type"].value_counts().items()))
        logger.info(f"framework tokens per example: "
                    f"min={inst['n_framework'].min()} "
                    f"median={int(inst['n_framework'].median())} "
                    f"max={inst['n_framework'].max()}")

    cfg.mkdirs()
    long_df.to_parquet(cfg.annotations_parquet, index=False)
    inst.to_parquet(cfg.cache / "instances.parquet", index=False)
    logger.info(f"wrote {cfg.annotations_parquet} ({len(long_df)} residue rows)")
    logger.info(f"wrote {cfg.cache / 'instances.parquet'} ({len(inst)} instances)")

    record_stage(cfg, "s5_annotate", {
        "n_instances_in": len(rows),
        "n_chains_numbered": len(sequences),
        "n_chains_with_domain": n_with,
        "unreadable_entries": missing_file,
        "chains_absent_from_file": missing_chain[:50],
        "drops": dict(drops),
        "n_instances_passing": n_before,
        "n_instances_after_one_per_entry": len(inst),
        "n_entries": int(inst["pdb_id"].nunique()) if len(inst) else 0,
        "n_residue_rows": len(long_df),
        "ig_type_counts": inst["ig_type"].value_counts().to_dict() if len(inst) else {},
        "framework_tokens": {
            "min": int(inst["n_framework"].min()),
            "median": int(inst["n_framework"].median()),
            "max": int(inst["n_framework"].max()),
        } if len(inst) else {},
        "cdr_definition": f"North under IMGT, {list(CDR_KEYS)}; cdr4 is framework",
        "anarcii_mode": cfg.anarcii_mode,
    })
    return inst


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    annotate(config_from_args(parser.parse_args(argv)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
