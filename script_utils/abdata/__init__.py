"""Antibody dataset build for framework-conditioned LoRA fine-tuning.

Numbered stages, each reading the previous stage's cached artifact and skipping
work already done:

    s1_fetch_summary    SAbDab2 summary CSV            -> cache/sabdab_summary.csv
    s2_select           curation filters + slice       -> cache/candidates.csv
    s3_resolve          RCSB dates + chain identity   -> cache/resolved.csv
    s4_fetch_structures RCSB mmCIF                     -> cache/cif/
    s5_annotate         anarcii numbering + crosscheck -> cache/annotations.parquet
    s6_write            trimmed mmCIF + sidecar        -> structures/, sidecars/
    s7_parquet          train / val / test metadata    -> parquet/

Revising a curation filter must not re-download structures or re-run numbering,
which is why the expensive stages cache per entry rather than per build.
"""
