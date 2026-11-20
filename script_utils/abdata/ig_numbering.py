"""IMGT numbering of immunoglobulin variable domains, North CDR boundaries.

Differs from `script_utils/framework_cond/ab_numbering.py` in three ways that
the dataset build needs:

  * **Multiple domains per chain.**  anarcii numbers exactly one domain per
    sequence: given the 229 observed residues of the scFv on chain M of `6e0c`
    it returns the VH alone, `query_start=0, query_end=119`.  Numbering is
    therefore iterative — renumber the unconsumed tail until no domain is
    found — which recovers the VL at `[122, 228]` as chain type K.  The same
    loop covers a VHH (one domain) and a paired Fv (one domain per chain)
    without special-casing either.
  * **North CDR boundaries under IMGT**, from
    `atomworks.ml.utils.antibody.NORTH_CDR_RANGES_IMGT`, rather than the plain
    IMGT boundaries.  CDR4 (the HV4 loop, IMGT 80-87) is treated as framework:
    masking a loop that rarely contacts antigen enlarges the generation task
    for nothing.
  * **Batched across chains.**  Numbering runs in rounds, so one forward pass
    covers every chain still having an unnumbered tail, instead of one call per
    chain per domain.

A non-antibody chain comes back as chain type `F` with
`error="Less than 50 non insertion residues numbered"`, which is the rejection
signal the SAbDab cross-check relies on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from loguru import logger

# Imported for their values only; this module does no atomworks IO.
from atomworks.ml.utils.antibody import (
    ANTIBODY_CHAIN_TYPES,
    IG_CHAIN_TYPE_NAMES,
    NORTH_CDR_RANGES_IMGT,
    VARIABLE_DOMAIN_IMGT_RANGE,
)

# CDR4 is deliberately absent: it is folded into framework.
CDR_KEYS = ("cdr1", "cdr2", "cdr3")

# Plain-IMGT conserved positions, used as a numbering sanity check.  These are
# conserved *by definition* of the scheme, so a mismatch means the numbering is
# not trustworthy rather than that the antibody is unusual.
IMGT_CONSERVED = {23: "C", 41: "W", 104: "C"}

MAX_DOMAINS_PER_CHAIN = 4
_MODEL = None
_MODEL_MODE: str | None = None


def _model(mode: str = "accuracy"):
    """Load the anarcii model once; it is a torch model and startup is not free."""
    global _MODEL, _MODEL_MODE
    if _MODEL is None or _MODEL_MODE != mode:
        from anarcii import Anarcii

        _MODEL = Anarcii(seq_type="antibody", mode=mode, batch_size=64,
                         cpu=True, verbose=False)
        _MODEL_MODE = mode
    return _MODEL


@dataclass
class Domain:
    """One numbered variable domain, in coordinates of the chain it sits on."""

    chain_type: str                 # H, K, L (antibody) or A/B/G/D (TCR)
    start: int                      # first sequence index, inclusive
    end: int                        # last sequence index, inclusive
    score: float
    imgt: np.ndarray                # IMGT number per chain position, -1 outside
    cdr_id: np.ndarray              # 1/2/3 inside a North CDR, 0 elsewhere
    is_framework: np.ndarray        # bool, inside the variable domain, not CDR
    conserved: dict = field(default_factory=dict)  # imgt -> (observed, ok)

    @property
    def group(self) -> str:
        return IG_CHAIN_TYPE_NAMES.get(self.chain_type, "heavy")

    @property
    def is_antibody(self) -> bool:
        return self.chain_type in ANTIBODY_CHAIN_TYPES

    @property
    def is_heavy(self) -> bool:
        return self.chain_type == "H"

    def cdr_sequence(self, seq: str, which: int) -> str:
        idx = np.where(self.cdr_id == which)[0]
        return "".join(seq[i] for i in idx)

    def conserved_ok(self) -> bool:
        return all(ok for _, ok in self.conserved.values())


def _assign(seq_len: int, numbering, query_start: int, group: str
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Map one anarcii record onto per-position arrays over the whole chain."""
    imgt = np.full(seq_len, -1, dtype=np.int32)
    pos = int(query_start or 0)
    first = last = -1
    for (num, _ins), aa in numbering:
        if aa == "-":            # scheme gap, absent from the sequence
            continue
        if pos < seq_len:
            imgt[pos] = num
            if first < 0:
                first = pos
            last = pos
        pos += 1

    lo, hi = VARIABLE_DOMAIN_IMGT_RANGE
    in_domain = (imgt >= lo) & (imgt <= hi)

    cdr_id = np.zeros(seq_len, dtype=np.int8)
    ranges = NORTH_CDR_RANGES_IMGT[group]
    for k, key in enumerate(CDR_KEYS, start=1):
        c_lo, c_hi = ranges[key]
        cdr_id[(imgt >= c_lo) & (imgt <= c_hi)] = k

    is_framework = in_domain & (cdr_id == 0)
    return imgt, cdr_id, is_framework, first, last


def _in_input_order(recs: dict, n: int) -> list[dict | None]:
    """Put anarcii's records back into input order.

    anarcii keys its results "Sequence <i>" with the index **zero-padded to the
    width of the input count**: 12 sequences give "Sequence 01", 150 give
    "Sequence 001".  Looking records up with an unpadded f-string therefore
    finds nothing for indices 1-99 as soon as the batch reaches 100 sequences,
    and the chains silently come back with no domain -- 99 of 612 named chains
    in the first run of the build, every one of them an ordinary VH.

    So the index is parsed out of the key rather than reconstructed, which is
    immune to the padding width, and the result is cross-checked against the
    input count instead of trusted.
    """
    out: list[dict | None] = [None] * n
    unparsed: list[str] = []
    for key, rec in recs.items():
        digits = "".join(ch for ch in str(key) if ch.isdigit())
        if not digits:
            unparsed.append(str(key))
            continue
        i = int(digits) - 1
        if 0 <= i < n:
            out[i] = rec
        else:
            unparsed.append(str(key))

    if unparsed:
        raise RuntimeError(
            f"could not place anarcii records {unparsed[:5]} in input order "
            f"(batch of {n}); the result key format has changed")
    n_placed = sum(r is not None for r in out)
    if n_placed != len(recs):
        raise RuntimeError(
            f"placed {n_placed} of {len(recs)} anarcii records for a batch of "
            f"{n}; duplicate indices in the result keys")
    return out


def number_chains(sequences: dict[str, str], mode: str = "accuracy"
                  ) -> dict[str, list[Domain]]:
    """Number every variable domain of every sequence, batched across chains.

    `sequences` maps a chain key to its one-letter sequence; the return maps the
    same keys to their domains in N-to-C order.  A chain with no numberable
    domain maps to an empty list rather than being absent.
    """
    model = _model(mode)
    out: dict[str, list[Domain]] = {k: [] for k in sequences}
    # offset[k] is how much of sequence k has already been consumed.
    offset = {k: 0 for k in sequences}
    active = [k for k, s in sequences.items() if len(s) >= 50]

    for round_idx in range(MAX_DOMAINS_PER_CHAIN):
        if not active:
            break
        tails = [sequences[k][offset[k]:] for k in active]
        recs = model.number(list(tails))
        recs = model.to_scheme("imgt")
        ordered = _in_input_order(recs, len(tails))

        still: list[str] = []
        for key, tail, rec in zip(active, tails, ordered):
            if not rec or rec.get("error") or not rec.get("numbering"):
                continue
            qs = int(rec.get("query_start") or 0)
            qe = int(rec.get("query_end") or 0)
            if qe < qs:
                continue
            chain_type = str(rec.get("chain_type") or "")
            group = IG_CHAIN_TYPE_NAMES.get(chain_type)
            if group is None:
                # chain type F and friends: not an Ig variable domain.
                continue

            base = offset[key]
            full = sequences[key]
            imgt, cdr_id, is_fw, first, last = _assign(
                len(full), rec["numbering"], base + qs, group)
            if first < 0:
                continue

            conserved = {}
            for num, expect in IMGT_CONSERVED.items():
                idx = np.where(imgt == num)[0]
                got = full[int(idx[0])] if len(idx) else "-"
                conserved[num] = (got, got == expect)

            out[key].append(Domain(
                chain_type=chain_type, start=first, end=last,
                score=float(rec.get("score") or 0.0),
                imgt=imgt, cdr_id=cdr_id, is_framework=is_fw,
                conserved=conserved))

            offset[key] = base + qe + 1
            if len(full) - offset[key] >= 50:
                still.append(key)
        active = still
        if round_idx == MAX_DOMAINS_PER_CHAIN - 1 and still:
            logger.warning(f"{len(still)} chains still had numberable tail after "
                           f"{MAX_DOMAINS_PER_CHAIN} rounds; tail ignored")

    for doms in out.values():
        doms.sort(key=lambda d: d.start)
    return out


def cdr_crosscheck(seq: str, domain: Domain, sabdab_cdrs: dict[int, str]
                   ) -> tuple[bool, dict]:
    """Is each SAbDab CDR loop contained in our North CDR span for that loop?

    Containment rather than equality: SAbDab's loop definition is narrower than
    North's, so the North span should strictly contain it.  A disagreement means
    the two numberings put the loop in different places, and the right response
    is to drop the instance rather than pick a winner.
    """
    detail = {}
    ok = True
    for which, expected in sabdab_cdrs.items():
        got = domain.cdr_sequence(seq, which)
        contained = bool(expected) and expected in got
        detail[f"cdr{which}"] = {"sabdab": expected, "north": got,
                                 "contained": contained}
        ok = ok and contained
    return ok, detail
