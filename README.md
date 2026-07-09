# PBPF-FTIR-GMFAN-MOBO

Minimal public transparency package for the manuscript:

**Spectroscopy-Guided Thermomechanical Trajectory Learning for Inverse Design of PBPF Bioplastics**

This repository provides manuscript-aligned metadata, minimal representative sample data, and a lightweight consistency-check script for the FT-IR-guided GMFAN/MOBO workflow described in the 260626 manuscript version.

## Scope

This is a **minimal public package**, not a full raw-data release and not a full training-code archive. The representative sample files demonstrate the public data format only. They are not intended to reproduce the full curve-prediction performance reported in the manuscript.

## Contents

```text
metadata/      Manuscript-aligned formulation, split, uncertainty, and MOBO summary CSV files
data_sample/   Small representative sample files for format demonstration only
scripts/       Lightweight repository consistency checker
docs/          Notes on GMFAN workflow scope and reproducibility boundaries
```

Figure source assets, full raw data, model checkpoints, internal debug outputs, and large intermediate results are intentionally excluded.

## Manuscript-aligned key facts

- PBPF-CNC formulation states: 24
- Integrated paired measurements: 459
- Fixed independent test subset: 108 paired measurements
- Split policy: formulation-disjoint splitting
- Model family: gated multimodal fusion attention network (GMFAN)
- Computationally prioritized MOBO candidates: `PBPF-91A-0.1` and `PBPF-91P-0.1`

## Quick check

```bash
python3 scripts/verify_public_package.py
```

## Data availability

See [`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md).
