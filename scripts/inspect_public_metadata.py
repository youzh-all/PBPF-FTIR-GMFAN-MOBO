#!/usr/bin/env python3
"""Print a readable summary of the public PBPF-FTIR-GMFAN-MOBO metadata package."""
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]


def read_csv(rel):
    with (ROOT / rel).open(newline='') as f:
        return list(csv.DictReader(f))


def main():
    formulation = read_csv('metadata/formulation_space.csv')
    fixed_test = [r for r in formulation if r['split_role'] == 'test']
    cv_rows = [r for r in formulation if r['split_role'] == 'cv']
    samples = read_csv('data_sample/sample_ftir_msc_spectra_downsampled.csv')
    uncertainty = read_csv('metadata/curve_uncertainty_metrics.csv')
    candidates = read_csv('metadata/mobo_prioritized_candidates.csv')
    input_schema = read_csv('metadata/model_input_schema.csv')
    target_schema = read_csv('metadata/response_target_schema.csv')

    print('# PBPF-FTIR-GMFAN-MOBO public metadata summary')
    print()
    print('## Formulation and split summary')
    print(f"Formulation states: {len(formulation)}")
    print(f"Integrated paired measurements: {sum(int(r['paired_rows']) for r in formulation)}")
    print(f"Fixed-test formulations: {len(fixed_test)}")
    print(f"Fixed-test paired measurements: {sum(int(r['paired_rows']) for r in fixed_test)}")
    print(f"Cross-validation formulations: {len(cv_rows)}")
    print(f"Cross-validation paired measurements: {sum(int(r['paired_rows']) for r in cv_rows)}")
    print('Fixed-test composition IDs:', ', '.join(r['composition_id'] for r in fixed_test))
    print()

    print('## Public input schema')
    for row in input_schema:
        print(f"- {row['component']}: {row['representation']}")
    print()

    print('## Response targets')
    for row in target_schema:
        print(f"- {row['target']}: {row['response_representation']}")
    print()

    print('## Representative FT-IR sample file')
    spectral_columns = [c for c in samples[0].keys() if c.startswith('wn_')] if samples else []
    print(f"Sample rows: {len(samples)}")
    print(f"Downsampled spectral columns: {len(spectral_columns)}")
    if samples:
        print('Sample composition IDs:', ', '.join(r['composition_id'] for r in samples))
    print()

    print('## Curve uncertainty summary')
    for row in uncertainty:
        print(f"- {row['response_type']}: PICP={row['picp']}, IW={row['standardized_interval_width']}, IS={row['standardized_interval_score']}")
    print()

    print('## MOBO-prioritized candidates')
    for row in candidates:
        print(
            f"- {row['composition_id']}: Tm={row['Tm_C']} C, Tc={row['Tc_C']} C, "
            f"tensile strength={row['tensile_strength_MPa']} MPa, "
            f"elongation at break={row['elongation_at_break_percent']}%, "
            f"note={row['tradeoff_note']}"
        )


if __name__ == '__main__':
    main()
