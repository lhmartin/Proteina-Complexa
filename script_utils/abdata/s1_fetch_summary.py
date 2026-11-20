"""Stage 1 - fetch the SAbDab2 summary CSV.

One request, cached on disk.  Everything downstream reads the cache, so a
curation change never re-downloads it.

    python -m script_utils.abdata.s1_fetch_summary [--force]
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone

import requests
from loguru import logger

from script_utils.abdata.config import (
    SABDAB_API,
    USER_AGENT,
    BuildConfig,
    add_common_args,
    config_from_args,
    record_stage,
)

ALL_SUMMARY = f"{SABDAB_API}/download/all-summary"

# Columns the later stages actually read.  Checked on arrival so that an
# upstream schema change is a loud failure here rather than a KeyError in
# stage 5 after three thousand structures have been downloaded.
REQUIRED_COLUMNS = (
    "INSTANCE", "PDB", "Hchain", "Lchain", "VH", "VL",
    "CDR-H1", "CDR-H2", "CDR-H3", "CDR-L1", "CDR-L2", "CDR-L3",
    "antigen_chain", "antigen_type", "method", "resolution", "type",
    "chainsharing_construct", "non_chainsharing_construct", "date",
    "light_ctype", "compound", "antigen_name",
)


def fetch_summary(cfg: BuildConfig) -> int:
    cfg.mkdirs()
    if cfg.summary_csv.exists() and not cfg.force:
        n = sum(1 for _ in cfg.summary_csv.open()) - 1
        logger.info(f"summary already cached: {cfg.summary_csv} ({n} rows)")
        return n

    logger.info(f"GET {ALL_SUMMARY}")
    resp = requests.get(ALL_SUMMARY, headers={"User-Agent": USER_AGENT},
                        timeout=300)
    resp.raise_for_status()

    tmp = cfg.summary_csv.with_suffix(".csv.part")
    tmp.write_bytes(resp.content)

    with tmp.open(newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            tmp.unlink()
            raise RuntimeError(
                f"SAbDab summary is missing expected columns {missing}; "
                f"got {reader.fieldnames}")
        rows = sum(1 for _ in reader)

    tmp.replace(cfg.summary_csv)
    logger.info(f"wrote {cfg.summary_csv} ({rows} antibody-instance rows)")
    record_stage(cfg, "s1_fetch_summary", {
        "url": ALL_SUMMARY,
        "fetched_utc": datetime.now(timezone.utc).isoformat(),
        "n_instance_rows": rows,
        "bytes": len(resp.content),
    })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    cfg = config_from_args(parser.parse_args(argv))
    fetch_summary(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
