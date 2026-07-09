# PBPF-FTIR-GMFAN-MOBO

Public repository for the manuscript:

**Spectroscopy-Guided Thermomechanical Trajectory Learning for Inverse Design of PBPF Bioplastics**

This repository provides manuscript-aligned resources for inspecting the FT-IR-guided GMFAN/MOBO workflow described in the associated manuscript. The materials are organized to make the formulation space, split design, representative FT-IR input format, uncertainty summary, and computationally prioritized MOBO candidates easy to review.

## Workflow overview

The manuscript describes a gated multimodal fusion attention network (GMFAN) that combines FT-IR spectral descriptors with encoded PBPF formulation descriptors to infer DSC heating, DSC cooling, and UTM stress-strain response curves. The same spectroscopy-conditioned response representation is then used for uncertainty-aware evaluation and multi-objective Bayesian optimization (MOBO) of candidate PBPF-CNC formulations.

## Repository contents

```text
metadata/      Manuscript-aligned formulation, split, input/target schema, uncertainty, metric, and MOBO summary CSV files
data_sample/   Representative sample files showing the public FT-IR input and formulation metadata format
scripts/       Lightweight consistency and metadata-inspection utilities
```

The representative sample files are provided to demonstrate data format and repository workflow. Full experimental datasets used for model training and evaluation are available from the corresponding authors upon reasonable request.

## What this repository enables

This repository enables readers to:

1. inspect the PBPF formulation space used in the manuscript;
2. verify the formulation-disjoint split summary and fixed-test composition list;
3. examine representative FT-IR input formatting and formulation metadata;
4. review model input, response-target, and evaluation-metric definitions;
5. review the reported curve-uncertainty summary;
6. review the computationally prioritized MOBO candidate formulations; and
7. run lightweight consistency and metadata-inspection utilities.

## Manuscript-aligned key facts

- PBPF-CNC formulation states: 24
- Integrated paired measurements: 459
- Fixed independent test subset: 108 paired measurements
- Split policy: formulation-disjoint splitting
- Response targets: DSC heating curves, DSC cooling curves, and UTM stress-strain curves
- Model family: gated multimodal fusion attention network (GMFAN)
- Curve uncertainty PICP values: 0.805 for DSC heating, 0.818 for DSC cooling, and 0.800 for UTM
- Computationally prioritized MOBO candidates: `PBPF-91A-0.1` and `PBPF-91P-0.1`

## Quick check

```bash
python3 scripts/verify_public_package.py
python3 scripts/inspect_public_metadata.py
```

The consistency check verifies the manuscript-level formulation counts, paired-row counts, fixed-test split summary, representative sample-file alignment, prioritized MOBO candidate identifiers, and absence of private/local path markers.

## Data availability

See [`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md).
