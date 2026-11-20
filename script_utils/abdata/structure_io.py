"""Reading observed polymer residues out of a cached mmCIF, by author chain id.

Roles are resolved against `auth_asym_id` because that is the namespace SAbDab
reports in.  `label_seq_id` is carried alongside because it is the position in
the deposited entity sequence, which is what makes an unresolved residue
detectable: two observed residues whose `label_seq_id` differ by more than one
have missing density between them, whatever their author numbering says.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gemmi

BACKBONE = ("N", "CA", "C", "O")


@dataclass(slots=True)
class Residue:
    res_id: int               # auth_seq_id
    ins_code: str             # pdbx_PDB_ins_code, "" when absent
    label_seq: int            # label_seq_id, -1 if unset
    res_name: str
    aa1: str
    has_backbone: bool
    n_atoms: int

    @property
    def key(self) -> tuple[int, str]:
        return (self.res_id, self.ins_code)


def read_chains(path: Path) -> dict[str, list[Residue]]:
    """Observed amino-acid residues per author chain, in file order.

    Only the first model is read, matching `load_structure`'s `model=1`.
    """
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    st.remove_alternative_conformations()
    st.remove_hydrogens()
    if len(st) == 0:
        return {}

    out: dict[str, list[Residue]] = {}
    for chain in st[0]:
        residues: list[Residue] = []
        for res in chain:
            info = gemmi.find_tabulated_residue(res.name)
            if info is None or not info.is_amino_acid():
                continue
            names = {a.name for a in res}
            residues.append(Residue(
                res_id=int(res.seqid.num),
                ins_code=(res.seqid.icode or "").strip(),
                label_seq=int(res.label_seq) if res.label_seq is not None else -1,
                res_name=res.name,
                aa1=gemmi.find_tabulated_residue(res.name).one_letter_code.upper(),
                has_backbone=all(b in names for b in BACKBONE),
                n_atoms=len(res),
            ))
        if residues:
            out[chain.name] = residues
    return out


def sequence_of(residues: list[Residue]) -> str:
    """One-letter sequence of observed residues, X for anything untabulated."""
    return "".join(r.aa1 if r.aa1.isalpha() else "X" for r in residues)


def auth_label_map(path: Path) -> dict[str, list[str]]:
    """auth_asym_id -> the label_asym_ids seen under it, for the gate's round trip."""
    doc = gemmi.cif.read(str(path))
    block = doc.sole_block()
    pairs = block.find("_atom_site.", ["auth_asym_id", "label_asym_id"])
    seen: dict[str, list[str]] = {}
    for row in pairs:
        auth, label = row[0], row[1]
        bucket = seen.setdefault(auth, [])
        if label not in bucket:
            bucket.append(label)
    return seen
