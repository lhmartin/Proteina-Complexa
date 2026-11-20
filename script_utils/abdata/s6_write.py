"""Stage 6 - write the trimmed complex and its framework sidecar.

One mmCIF per surviving instance, holding the trimmed Fv and the whole antigen,
under our own chain convention: `H`, then `L`, then the antigen chains from `A`
onwards (skipping H and L).  Because we write the file, `label_asym_id` and
`auth_asym_id` are set identically, which is what makes the convention survive
`load_any` -- that returns `label_asym_id`, and in 145 of the 200 slice entries
at least one deposited auth id differs from its label id.

Trimming cuts the antibody to the IMGT variable-domain range, discarding CH1,
CL and whole-IgG bulk; constant domains otherwise double the antibody token
count and are a conditioning signal the framework conditioner cannot represent.
Antigen chains are kept whole: cropping them here would sidestep the crop path,
and the crop code has to change regardless.

**Residues are renumbered sequentially per chain.**  `(chain_id, res_id)` is
the sidecar key and is not unique in deposited numbering: biotite splits `100A`
into `res_id=100` plus a separate `ins_code` annotation, and
`atomarray_to_atom37` passes only `extra_annotations=["chain_id"]` before
casting `token.res_id[0]` to long, so insertion codes never reach the batch.
Renumbering makes the key unique by construction.  The original
`(res_id, ins_code)` is kept in the sidecar as provenance, so nothing is
unrecoverable; what is given up is the gap structure of deposited numbering,
which `SequenceSeparationPairFeat` could otherwise read.

An scFv's two domains are written as chains `H` and `L` like any other paired
Fv, rather than as one chain with an internal break.  The linker is discarded
either way, so the two domains are not covalently continuous in the written
file; a uniform convention means the role resolution and the `framework_chain`
VH=0/VL=1 encoding have one code path instead of two.

    python -m script_utils.abdata.s6_write
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gemmi
import pandas as pd
from loguru import logger

from script_utils.abdata.config import (
    BuildConfig,
    add_common_args,
    config_from_args,
    record_stage,
)
from script_utils.abdata.s4_fetch_structures import cif_path

ANTIGEN_CHAIN_LETTERS = [c for c in "ABCDEFGIJKMNOPQRSTUVWXYZ" if c not in ("H", "L")]


def _copy_residue(res: gemmi.Residue, new_seqid: int, subchain: str) -> gemmi.Residue:
    out = gemmi.Residue()
    out.name = res.name
    out.seqid = gemmi.SeqId(new_seqid, " ")
    out.het_flag = "A"
    out.subchain = subchain
    for atom in res:
        if atom.element == gemmi.Element("H"):
            continue
        out.add_atom(atom)
    return out


def write_one(cfg: BuildConfig, inst: pd.Series, side: pd.DataFrame
              ) -> tuple[Path, pd.DataFrame] | tuple[None, str]:
    """Write one trimmed complex, returning its path and its sidecar rows."""
    src = cif_path(cfg, inst["pdb_id"])
    if not src.exists():
        return None, "cif_missing"

    st = gemmi.read_structure(str(src))
    st.setup_entities()
    st.remove_alternative_conformations()
    st.remove_hydrogens()
    st.remove_ligands_and_waters()
    if len(st) == 0:
        return None, "no_model"
    model = st[0]

    out_st = gemmi.Structure()
    out_st.name = inst["example_id"]
    out_st.spacegroup_hm = st.spacegroup_hm
    out_st.cell = st.cell
    out_model = gemmi.Model("1")

    sidecar_rows: list[dict] = []

    # --- the antibody chains, trimmed to the variable domain ----------------
    for role in ("H", "L"):
        want = side[side["role"] == role]
        if want.empty:
            continue
        src_chain_name = str(want["chain_id"].iloc[0])
        src_chain = model.find_chain(src_chain_name)
        if src_chain is None:
            return None, f"chain_{src_chain_name}_absent"

        # (res_id, ins_code) -> the annotation row for it
        by_key = {(int(r["res_id"]), str(r["ins_code"] or "").strip()): r
                  for _, r in want.iterrows()}

        out_chain = gemmi.Chain(role)
        new_seqid = 0
        for res in src_chain:
            key = (int(res.seqid.num), (res.seqid.icode or "").strip())
            ann = by_key.get(key)
            if ann is None:
                continue          # outside the variable domain, or the linker
            new_seqid += 1
            out_chain.add_residue(_copy_residue(res, new_seqid, role))
            sidecar_rows.append({
                "example_id": inst["example_id"],
                "chain_id": role,
                "res_id": new_seqid,            # the key the loader will see
                "orig_res_id": int(res.seqid.num),
                "orig_ins_code": (res.seqid.icode or "").strip(),
                "orig_chain_id": src_chain_name,
                "res_name": res.name,
                "aa1": str(ann["aa1"]),
                "imgt": int(ann["imgt"]),
                "cdr_id": int(ann["cdr_id"]),
                "is_framework": bool(ann["is_framework"]),
                "chain_type": str(ann["chain_type"]),
                "framework_chain": 0 if role == "H" else 1,
            })
        if len(out_chain) == 0:
            return None, f"chain_{role}_empty_after_trim"
        out_model.add_chain(out_chain)

    if not any(r["chain_id"] == "H" for r in sidecar_rows):
        return None, "no_heavy_chain_written"

    # --- the antigen chains, whole ------------------------------------------
    antigen = [c for c in str(inst["antigen_chains"]).split("|") if c]
    letters = iter(ANTIGEN_CHAIN_LETTERS)
    n_antigen_res = 0
    for src_name in antigen:
        src_chain = model.find_chain(src_name)
        if src_chain is None:
            return None, f"antigen_chain_{src_name}_absent"
        try:
            name = next(letters)
        except StopIteration:
            return None, "too_many_antigen_chains"
        out_chain = gemmi.Chain(name)
        new_seqid = 0
        for res in src_chain:
            info = gemmi.find_tabulated_residue(res.name)
            if info is None or not info.is_amino_acid():
                continue
            new_seqid += 1
            out_chain.add_residue(_copy_residue(res, new_seqid, name))
        if len(out_chain) == 0:
            continue
        n_antigen_res += len(out_chain)
        out_model.add_chain(out_chain)

    if n_antigen_res == 0:
        return None, "antigen_empty"

    out_st.add_model(out_model)
    out_st.setup_entities()
    # setup_entities reassigns subchains; force label == auth so that the
    # convention survives load_any, which returns label_asym_id.  One subchain
    # per chain also keeps each written chain a single entity.
    for chain in out_st[0]:
        for res in chain:
            res.subchain = chain.name
    # Rebuild entities against those subchains and then assign label_seq_id.
    # Overwriting subchain by hand leaves label_seq_id unset, and atomworks'
    # load_any rejects a polymer whose label_seq_id is missing
    # ("CIF polymer label_asym_id 'H' has missing label_seq_id values"), so
    # without this the written files are simply unloadable.
    out_st.ensure_entities()
    out_st.deduplicate_entities()
    out_st.assign_label_seq_id()
    for chain in out_st[0]:
        for i, res in enumerate(chain, start=1):
            if res.label_seq is None or int(res.label_seq) < 1:
                res.label_seq = i

    dest = cfg.structures / f"{inst['example_id']}.cif"
    doc = out_st.make_mmcif_document()
    doc.write_file(str(dest))
    return dest, pd.DataFrame(sidecar_rows)


def write(cfg: BuildConfig) -> pd.DataFrame:
    inst_path = cfg.cache / "instances.parquet"
    if not inst_path.exists() or not cfg.annotations_parquet.exists():
        raise FileNotFoundError("run s5_annotate first")
    instances = pd.read_parquet(inst_path)
    annotations = pd.read_parquet(cfg.annotations_parquet)
    cfg.mkdirs()
    logger.info(f"writing {len(instances)} complexes to {cfg.structures}")

    by_example = dict(tuple(annotations.groupby("example_id")))
    rows: list[dict] = []
    sidecars: list[pd.DataFrame] = []
    failures: dict[str, int] = {}

    for n, (_, inst) in enumerate(instances.iterrows(), 1):
        side = by_example.get(inst["example_id"])
        if side is None or side.empty:
            failures["no_annotation_rows"] = failures.get("no_annotation_rows", 0) + 1
            continue
        try:
            dest, payload = write_one(cfg, inst, side)
        except Exception as exc:  # noqa: BLE001 - counted, not fatal
            logger.warning(f"{inst['example_id']}: {exc}")
            failures["exception"] = failures.get("exception", 0) + 1
            continue
        if dest is None:
            failures[payload] = failures.get(payload, 0) + 1
            continue

        sidecars.append(payload)
        st = gemmi.read_structure(str(dest))
        n_res = sum(len(c) for c in st[0])
        rows.append({
            **{k: inst[k] for k in (
                "example_id", "pdb_id", "ig_type", "resolution",
                "deposit_date", "release_date", "is_pinned")},
            "path": str(dest.resolve()),
            "n_residues_total": n_res,
            "n_framework": int(payload["is_framework"].sum()),
            "n_cdr": int((payload["cdr_id"] > 0).sum()),
            "n_antibody_residues": len(payload),
            "n_antigen_residues": n_res - len(payload),
            "chains_written": "|".join(c.name for c in st[0]),
        })
        if n % 50 == 0:
            logger.info(f"  {n}/{len(instances)}")

    written = pd.DataFrame(rows)
    sidecar = (pd.concat(sidecars, ignore_index=True) if sidecars
               else pd.DataFrame())

    if failures:
        logger.warning("failures: " + ", ".join(
            f"{k}={v}" for k, v in sorted(failures.items(), key=lambda kv: -kv[1])))

    sidecar_path = cfg.sidecars / "framework_sidecar.parquet"
    sidecar.to_parquet(sidecar_path, index=False)
    written.to_parquet(cfg.cache / "written.parquet", index=False)
    logger.info(f"wrote {len(written)} structures and {len(sidecar)} sidecar rows")
    if len(written):
        logger.info(f"residues per complex: min={written['n_residues_total'].min()} "
                    f"median={int(written['n_residues_total'].median())} "
                    f"max={written['n_residues_total'].max()}")
        logger.info(f"framework tokens: min={written['n_framework'].min()} "
                    f"median={int(written['n_framework'].median())} "
                    f"max={written['n_framework'].max()}")
        over = int((written["n_residues_total"] > cfg.crop_size).sum())
        logger.info(f"{over}/{len(written)} complexes exceed crop_size="
                    f"{cfg.crop_size}; the rest bypass the crop entirely "
                    f"(transforms.py:588 returns early when n < crop_size)")

    # The sidecar key must be unique, which is the whole point of renumbering.
    if len(sidecar):
        dupes = sidecar.duplicated(subset=["example_id", "chain_id", "res_id"]).sum()
        if dupes:
            raise RuntimeError(
                f"{dupes} duplicate (example_id, chain_id, res_id) sidecar keys; "
                "renumbering failed and the mask would resolve ambiguously")
        logger.info("sidecar key (example_id, chain_id, res_id) is unique")

    record_stage(cfg, "s6_write", {
        "n_instances_in": len(instances),
        "n_written": len(written),
        "failures": failures,
        "n_sidecar_rows": len(sidecar),
        "sidecar_path": str(sidecar_path),
        "residues_per_complex": {
            "min": int(written["n_residues_total"].min()),
            "median": int(written["n_residues_total"].median()),
            "max": int(written["n_residues_total"].max()),
        } if len(written) else {},
        "framework_tokens": {
            "min": int(written["n_framework"].min()),
            "median": int(written["n_framework"].median()),
            "max": int(written["n_framework"].max()),
        } if len(written) else {},
        "n_exceeding_crop_size": int(
            (written["n_residues_total"] > cfg.crop_size).sum()) if len(written) else 0,
    })
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    write(config_from_args(parser.parse_args(argv)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
