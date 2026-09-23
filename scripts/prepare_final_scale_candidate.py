"""Apply the single scale already reported in the supplied ZIP's STATUS.md.

Scientific CSV processing; no fitting against public scores or hidden labels.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True,
                        help='Original ensemble CSV (before applying scale 0.97).')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('artifacts/final_candidate'))
    args = parser.parse_args()
    source = args.input
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    expected = '777633a4dee553192ba54c982d8691ba89d70ecbd105bb50561c2dff22c4ef7a'
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    assert source_hash == expected
    original = pd.read_csv(source)
    assert list(original.columns) == ['id', 'tp_mm_day']
    assert len(original) == original.id.nunique() == 1885464
    assert original.id.notna().all()
    values = original.tp_mm_day.to_numpy(dtype=np.float64)
    assert np.isfinite(values).all() and (values >= 0).all()
    candidate = original.copy()
    candidate['tp_mm_day'] = values * 0.97
    destination = out / 'submission_candidate_scale097.csv'
    assert not destination.exists(), 'Preserve any existing output; choose a new destination.'
    candidate.to_csv(destination, index=False, float_format='%.10f')
    saved = pd.read_csv(destination)
    assert saved.id.equals(original.id)
    assert len(saved) == saved.id.nunique() == 1885464
    assert np.isfinite(saved.tp_mm_day).all() and (saved.tp_mm_day >= 0).all()
    np.testing.assert_allclose(saved.tp_mm_day, values * 0.97, rtol=0, atol=5.1e-11)
    audit = {
        'source': str(source), 'source_sha256': source_hash,
        'output': str(destination),
        'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
        'rows': len(saved), 'unique_ids': saved.id.nunique(),
        'id_order_matches_source': True, 'finite': True, 'nonnegative': True,
        'scale': 0.97, 'min': float(saved.tp_mm_day.min()),
        'max': float(saved.tp_mm_day.max()), 'mean': float(saved.tp_mm_day.mean()),
        'basis': 'Supplied ZIP STATUS.md reports historical scale=0.97 experiment.',
        'reported_validation_not_recomputed': {
            '2015_2016': {'base': 1.78041, 'scale097': 1.77962},
            '2021_2022': {'base': 1.78996, 'scale097': 1.78735}},
        'limitations': 'No independent holdout for scale selection; year-2-only scale metrics unavailable. Private performance unknown.',
        'historical_public_score_reported_by_user': 1.66882,
        'submitted': False,
    }
    (out / 'scale097_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
