"""Stage 3 - resolve per-entry metadata and per-chain identity from RCSB.

Three things SAbDab cannot settle:

  * **Deposition and release dates.**  SAbDab's `date` is the release date and
    its `SABDABdepo_date` is the date SAbDab itself recorded the entry (every
    row in the current file reads as the download date), so neither supports a
    temporal split or a deposition-date leak check.
  * **Which antigen chain is actually a protein.**  `antigen_type` is a set
    over the antigen chains with no per-chain mapping, so the polymer entity
    table is the only authority.  In `1a2y` the chains are `C|D` against
    `PROTEIN|ION`, and `D` is not an auth chain at all: it is the *label* id of
    a phosphate ion that sits on auth chain `C`.  Intersecting the antigen
    chains with the real protein-polymer auth chains removes the ion and the
    phantom id in one step.
  * **The auth/label chain-id mapping.**  `load_any` returns `label_asym_id`
    while SAbDab's letters are author ids, and the two genuinely disagree: in
    `6e0c` the scFv is auth `M`,`N` but label `K`,`L`.  The mapping is cached
    here so stage 6 can resolve roles against auth ids and write the file with
    both namespaces set identically.

Results are cached per entry, so re-running after a curation change costs
nothing for entries already resolved.

    python -m script_utils.abdata.s3_resolve
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import pandas as pd
import requests
from loguru import logger

from script_utils.abdata.config import (
    RCSB_GRAPHQL,
    USER_AGENT,
    BuildConfig,
    add_common_args,
    config_from_args,
    record_stage,
)

BATCH_SIZE = 50
MAX_RETRIES = 4

# `asym_ids` is label_asym_id and `auth_asym_ids` is auth_asym_id; the two
# lists are positionally parallel within one entity (unlike SAbDab's antigen
# fields, this parallelism is part of the RCSB schema).
QUERY = """
query Entries($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    rcsb_accession_info { deposit_date initial_release_date revision_date }
    rcsb_entry_info {
      resolution_combined
      experimental_method
      deposited_polymer_monomer_count
      polymer_entity_count_protein
    }
    polymer_entities {
      entity_poly { rcsb_entity_polymer_type rcsb_sample_sequence_length }
      rcsb_polymer_entity { pdbx_description }
      rcsb_polymer_entity_container_identifiers { auth_asym_ids asym_ids }
    }
    nonpolymer_entities {
      rcsb_nonpolymer_entity_container_identifiers { auth_asym_ids asym_ids }
      nonpolymer_comp { chem_comp { id name } }
    }
  }
}
"""


def _post(ids: list[str]) -> list[dict]:
    payload = {"query": QUERY, "variables": {"ids": ids}}
    last: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                RCSB_GRAPHQL, json=payload, timeout=120,
                headers={"User-Agent": USER_AGENT,
                         "Content-Type": "application/json"})
            resp.raise_for_status()
            body = resp.json()
            if body.get("errors"):
                raise RuntimeError(f"RCSB GraphQL errors: {body['errors']}")
            return [e for e in (body.get("data", {}).get("entries") or []) if e]
        except Exception as exc:  # noqa: BLE001 - retried, then raised
            last = exc
            wait = 2 ** attempt
            logger.warning(f"RCSB batch failed ({exc}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"RCSB GraphQL failed after {MAX_RETRIES} attempts") from last


def entry_record(entry: dict) -> dict:
    """Flatten one entry into the per-chain tables the later stages need."""
    acc = entry.get("rcsb_accession_info") or {}
    info = entry.get("rcsb_entry_info") or {}
    res = info.get("resolution_combined") or []

    auth_to_label: dict[str, list[str]] = {}
    chain_polymer_type: dict[str, str] = {}
    chain_length: dict[str, int] = {}
    chain_description: dict[str, str] = {}

    for pe in entry.get("polymer_entities") or []:
        ci = pe.get("rcsb_polymer_entity_container_identifiers") or {}
        auth = ci.get("auth_asym_ids") or []
        label = ci.get("asym_ids") or []
        poly = (pe.get("entity_poly") or {})
        ptype = poly.get("rcsb_entity_polymer_type") or ""
        plen = poly.get("rcsb_sample_sequence_length")
        desc = ((pe.get("rcsb_polymer_entity") or {}).get("pdbx_description") or "")
        for i, a in enumerate(auth):
            chain_polymer_type[a] = ptype
            if plen is not None:
                chain_length[a] = int(plen)
            chain_description[a] = desc
            # Positionally parallel within an entity; fall back to the whole
            # list if the lengths ever disagree rather than mispair silently.
            auth_to_label.setdefault(a, []).append(
                label[i] if i < len(label) else "")

    nonpolymer_auth: dict[str, list[str]] = {}
    for ne in entry.get("nonpolymer_entities") or []:
        ci = ne.get("rcsb_nonpolymer_entity_container_identifiers") or {}
        comp = ((ne.get("nonpolymer_comp") or {}).get("chem_comp") or {})
        for a in ci.get("auth_asym_ids") or []:
            nonpolymer_auth.setdefault(a, []).append(comp.get("id") or "?")

    return {
        "pdb_id": str(entry["rcsb_id"]).lower(),
        "deposit_date": (acc.get("deposit_date") or "")[:10],
        "release_date": (acc.get("initial_release_date") or "")[:10],
        "revision_date": (acc.get("revision_date") or "")[:10],
        "rcsb_resolution": float(res[0]) if res else None,
        "rcsb_method": info.get("experimental_method") or "",
        "deposited_polymer_monomer_count":
            info.get("deposited_polymer_monomer_count"),
        "protein_entity_count": info.get("polymer_entity_count_protein"),
        "auth_to_label": json.dumps(auth_to_label, sort_keys=True),
        "chain_polymer_type": json.dumps(chain_polymer_type, sort_keys=True),
        "chain_length": json.dumps(chain_length, sort_keys=True),
        "chain_description": json.dumps(chain_description, sort_keys=True),
        "nonpolymer_chains": json.dumps(nonpolymer_auth, sort_keys=True),
        "protein_auth_chains": "|".join(sorted(
            a for a, t in chain_polymer_type.items() if t == "Protein")),
    }


def resolve(cfg: BuildConfig) -> pd.DataFrame:
    if not cfg.candidates_csv.exists():
        raise FileNotFoundError(
            f"{cfg.candidates_csv} missing; run s2_select first")

    cand = pd.read_csv(cfg.candidates_csv, dtype=str, keep_default_na=False)
    wanted = sorted(set(cand["pdb_id"]))

    have: dict[str, dict] = {}
    if cfg.rcsb_csv.exists() and not cfg.force:
        cached = pd.read_csv(cfg.rcsb_csv, dtype=str, keep_default_na=False)
        have = {r["pdb_id"]: dict(r) for _, r in cached.iterrows()}
        logger.info(f"{len(have)} entries already cached in {cfg.rcsb_csv.name}")

    todo = [p for p in wanted if p not in have]
    logger.info(f"resolving {len(todo)} of {len(wanted)} entries from RCSB")

    for i in range(0, len(todo), BATCH_SIZE):
        batch = todo[i:i + BATCH_SIZE]
        for entry in _post(batch):
            rec = entry_record(entry)
            have[rec["pdb_id"]] = rec
        logger.info(f"  {min(i + BATCH_SIZE, len(todo))}/{len(todo)}")

    missing = [p for p in wanted if p not in have]
    if missing:
        logger.warning(f"RCSB returned nothing for {len(missing)} entries: "
                       f"{missing[:10]}")

    meta = pd.DataFrame([have[p] for p in wanted if p in have])
    cfg.mkdirs()
    meta.to_csv(cfg.rcsb_csv, index=False)
    logger.info(f"wrote {cfg.rcsb_csv} ({len(meta)} entries)")

    # --- narrow the antigen chains to real protein polymers -----------------
    protein_by_entry = {
        r["pdb_id"]: set(str(r["protein_auth_chains"]).split("|")) - {""}
        for _, r in meta.iterrows()}
    dep_by_entry = {r["pdb_id"]: r["deposit_date"] for _, r in meta.iterrows()}
    rel_by_entry = {r["pdb_id"]: r["release_date"] for _, r in meta.iterrows()}
    res_by_entry = {r["pdb_id"]: r["rcsb_resolution"] for _, r in meta.iterrows()}

    def _narrow(row: pd.Series) -> str:
        prot = protein_by_entry.get(row["pdb_id"], set())
        claimed = [c for c in str(row["antigen_chains"]).split("|") if c]
        return "|".join(c for c in claimed if c in prot)

    cand["antigen_chains_resolved"] = cand.apply(_narrow, axis=1)
    cand["antigen_chains_dropped"] = cand.apply(
        lambda r: "|".join(
            c for c in str(r["antigen_chains"]).split("|")
            if c and c not in protein_by_entry.get(r["pdb_id"], set())), axis=1)
    cand["deposit_date"] = cand["pdb_id"].map(dep_by_entry).fillna("")
    cand["release_date"] = cand["pdb_id"].map(rel_by_entry).fillna("")
    cand["rcsb_resolution"] = cand["pdb_id"].map(res_by_entry)

    # The antibody chains must also exist as protein auth chains, or the role
    # assignment cannot be carried out against the structure.
    def _ab_chains_present(row: pd.Series) -> bool:
        prot = protein_by_entry.get(row["pdb_id"], set())
        needed = {row["h_chain"], row["l_chain"]} - {""}
        return bool(needed) and needed <= prot

    before = len(cand)
    cand["antibody_chains_present"] = cand.apply(_ab_chains_present, axis=1)
    n_ab_missing = int((~cand["antibody_chains_present"]).sum())
    cand = cand[cand["antibody_chains_present"]].copy()

    n_no_antigen = int(cand["antigen_chains_resolved"].eq("").sum())
    cand = cand[cand["antigen_chains_resolved"].ne("")].copy()

    # Resolution from RCSB supersedes SAbDab's, which can disagree.
    n_res = 0
    if cfg.require_resolution:
        bad_res = cand["rcsb_resolution"].isna() | (
            cand["rcsb_resolution"].astype(float) > cfg.max_resolution)
        n_res = int(bad_res.sum())
        cand = cand[~bad_res].copy()

    logger.info(f"resolved: {before} -> {len(cand)} rows "
                f"({cand['pdb_id'].nunique()} entries); dropped "
                f"{n_ab_missing} antibody-chain-missing, "
                f"{n_no_antigen} no-protein-antigen, "
                f"{n_res} rcsb-resolution")

    n_narrowed = int(cand["antigen_chains_dropped"].ne("").sum())
    logger.info(f"{n_narrowed} rows had non-protein or phantom antigen chains "
                f"removed (ions, sugars, nucleic acids, label-id artefacts)")

    cand.to_csv(cfg.resolved_csv, index=False)
    logger.info(f"wrote {cfg.resolved_csv}")

    record_stage(cfg, "s3_resolve", {
        "n_entries_resolved": len(meta),
        "n_entries_missing_from_rcsb": len(missing),
        "entries_missing_from_rcsb": missing[:50],
        "rows_in": before,
        "rows_out": len(cand),
        "entries_out": int(cand["pdb_id"].nunique()),
        "dropped_antibody_chain_not_a_protein_polymer": n_ab_missing,
        "dropped_no_protein_antigen_chain": n_no_antigen,
        "dropped_rcsb_resolution": n_res,
        "rows_with_antigen_chains_narrowed": n_narrowed,
        "ig_type_counts": cand["ig_type"].value_counts().to_dict(),
        "deposit_date_range": [cand["deposit_date"].min(),
                               cand["deposit_date"].max()],
    })
    return cand


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    resolve(config_from_args(parser.parse_args(argv)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
