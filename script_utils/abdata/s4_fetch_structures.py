"""Stage 4 - download the mmCIF for each surviving entry, cached on disk.

mmCIF rather than PDB because RCSB refuses to generate PDB format for oversized
entries (`6z6n.pdb` and `4v6x.pdb` both 404 while their mmCIFs are ~29 MB), and
a 404 on a format that usually works is a silent data-loss path.  mmCIF also has
no chain-id or atom-count ceiling.

Files land in their own directory and nothing else is written there, so the
later stages read them as data by path.  A download is skipped if the cached
file is present and gunzips cleanly; a truncated cache entry is re-fetched
rather than carried forward.

    python -m script_utils.abdata.s4_fetch_structures
    python -m script_utils.abdata.s4_fetch_structures --n-workers-fetch 16
"""

from __future__ import annotations

import argparse
import gzip
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from loguru import logger

from script_utils.abdata.config import (
    RCSB_FILES,
    USER_AGENT,
    BuildConfig,
    add_common_args,
    config_from_args,
    record_stage,
)


def cif_path(cfg: BuildConfig, pdb_id: str) -> Path:
    return cfg.cif_cache / f"{pdb_id}.cif.gz"


def _is_complete(path: Path) -> bool:
    """A cached file counts only if it gunzips and ends like an mmCIF.

    An interrupted download leaves a readable prefix, which would otherwise be
    parsed as a structure with silently missing chains.
    """
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with gzip.open(path, "rt", errors="replace") as fh:
            text = fh.read()
    except (OSError, EOFError, gzip.BadGzipFile):
        return False
    # gzip raises on a truncated member, so reaching here means the stream is
    # whole; these two then confirm it is an mmCIF with coordinates in it.
    return "_atom_site." in text and text.rstrip().endswith("#")


def fetch_one(cfg: BuildConfig, pdb_id: str) -> tuple[str, str]:
    dest = cif_path(cfg, pdb_id)
    if _is_complete(dest) and not cfg.force:
        return pdb_id, "cached"
    url = f"{RCSB_FILES}/{pdb_id}.cif.gz"
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=180)
        resp.raise_for_status()
    except requests.RequestException as exc:
        return pdb_id, f"error: {exc}"
    tmp = dest.with_suffix(".gz.part")
    tmp.write_bytes(resp.content)
    if not _is_complete(tmp):
        tmp.unlink(missing_ok=True)
        return pdb_id, "error: incomplete mmCIF"
    tmp.replace(dest)
    return pdb_id, "downloaded"


def fetch(cfg: BuildConfig) -> dict[str, int]:
    src = cfg.resolved_csv if cfg.resolved_csv.exists() else cfg.candidates_csv
    if not src.exists():
        raise FileNotFoundError(f"{src} missing; run the earlier stages first")
    ids = sorted(set(pd.read_csv(src, dtype=str, keep_default_na=False)["pdb_id"]))
    cfg.mkdirs()
    logger.info(f"{len(ids)} entries to ensure in {cfg.cif_cache}")

    counts = {"cached": 0, "downloaded": 0, "error": 0}
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=cfg.n_workers_fetch) as pool:
        futures = {pool.submit(fetch_one, cfg, p): p for p in ids}
        for n, fut in enumerate(as_completed(futures), 1):
            pdb_id, status = fut.result()
            if status.startswith("error"):
                counts["error"] += 1
                errors.append(f"{pdb_id}: {status}")
                logger.warning(f"{pdb_id} {status}")
            else:
                counts[status] += 1
            if n % 50 == 0:
                logger.info(f"  {n}/{len(ids)}")

    total_bytes = sum(p.stat().st_size for p in cfg.cif_cache.glob("*.cif.gz"))
    logger.info(f"{counts} | {total_bytes / 1e6:.1f} MB in cache "
                f"({total_bytes / max(1, len(ids)) / 1e3:.0f} KB/entry)")
    if errors:
        logger.warning(f"{len(errors)} entries failed and are NOT in the cache; "
                       "later stages will drop them and count them")

    record_stage(cfg, "s4_fetch_structures", {
        "n_entries": len(ids),
        **counts,
        "errors": errors[:50],
        "cache_bytes": total_bytes,
    })
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    fetch(config_from_args(parser.parse_args(argv)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
