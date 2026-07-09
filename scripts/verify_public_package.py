#!/usr/bin/env python3
"""Verify the minimal public package against manuscript-level consistency checks."""
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_MARKERS = ['/' + 'mnt' + '/mydisk', '/Users/' + 'jeon-yujin', '.' + 'hermes', 'ext_' + 'yujin', 'GITHUB' + '_TOKEN', 'OPENAI' + '_API_KEY', 'ANTHROPIC' + '_API_KEY']


def read_csv(rel):
    with (ROOT / rel).open(newline='') as f:
        return list(csv.DictReader(f))


def require(condition, message):
    if not condition:
        raise SystemExit(f'FAIL: {message}')


def main():
    formulation = read_csv('metadata/formulation_space.csv')
    require(len(formulation) == 24, f'expected 24 formulations, found {len(formulation)}')
    paired = sum(int(r['paired_rows']) for r in formulation)
    require(paired == 459, f'expected 459 paired rows, found {paired}')
    fixed_test = [r for r in formulation if r['split_role'] == 'test']
    require(len(fixed_test) == 6, f'expected 6 fixed-test formulations, found {len(fixed_test)}')
    fixed_rows = sum(int(r['paired_rows']) for r in fixed_test)
    require(fixed_rows == 108, f'expected 108 fixed-test paired rows, found {fixed_rows}')

    candidates = {r['composition_id'] for r in read_csv('metadata/mobo_prioritized_candidates.csv')}
    require(candidates == {'PBPF-91A-0.1', 'PBPF-91P-0.1'}, f'unexpected MOBO candidates: {sorted(candidates)}')

    sample_meta = read_csv('data_sample/sample_formulation_metadata.csv')
    sample_ftir = read_csv('data_sample/sample_ftir_msc_spectra_downsampled.csv')
    require(len(sample_meta) == len(sample_ftir) == 8, 'expected 8 sample metadata and FT-IR rows')
    require({r['composition_id'] for r in sample_meta} == {r['composition_id'] for r in sample_ftir}, 'sample metadata and FT-IR IDs differ')

    bad = []
    for path in ROOT.rglob('*'):
        if path.is_file() and '.git' not in path.parts:
            try:
                text = path.read_text(errors='ignore')
            except Exception:
                continue
            for marker in PRIVATE_MARKERS:
                if marker in text:
                    bad.append((str(path.relative_to(ROOT)), marker))
    require(not bad, f'private markers found: {bad[:10]}')

    figure_paths = [p for p in ROOT.rglob('*') if any(part.lower() in {'figure', 'figures'} for part in p.parts)]
    require(not figure_paths, f'figure assets should not be included: {[str(p.relative_to(ROOT)) for p in figure_paths[:10]]}')
    print('OK: public package consistency checks passed')


if __name__ == '__main__':
    main()
