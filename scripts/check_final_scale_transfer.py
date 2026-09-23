"""Check one historical scale fit against a later block, without leaderboard tuning.

Run in the friend's repository:
python check_final_scale_transfer.py --results lucasteste1/results
No training, CSV generation, or Kaggle submission occurs.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load_pair(root, fold):
    stem = f'E00_fold{fold}_s8_l224_r700_lr0.025_years30_atm_three_ocean'
    items = []
    for suffix in ('', '_nmme_geoss2s'):
        path = root / f'{stem}{suffix}_predictions.npz'
        with np.load(path, allow_pickle=False) as archive:
            items.append({k: archive[k].copy() for k in
                          ('prediction', 'truth', 'time', 'lat', 'lon')})
    a, b = items
    for key in ('truth', 'time', 'lat', 'lon'):
        if not np.array_equal(a[key], b[key]):
            raise ValueError(f'{fold}: incompatible {key}')
    dates = a['time'].astype('datetime64[M]')
    start = '2015-01' if fold == '2015full' else '2021-01'
    expected_dates = np.datetime64(start, 'M') + np.arange(24)
    if not np.array_equal(dates, expected_dates):
        raise ValueError(f'{fold}: expected 24 consecutive months starting {start}')
    npoints = len(a['lat']) * len(a['lon'])
    shape = (24 * npoints,)
    if any(x[k].shape != shape or not np.isfinite(x[k]).all()
           for x in items for k in ('prediction', 'truth')):
        raise ValueError(f'{fold}: invalid prediction/truth shape or values')
    pred = np.maximum(0, 0.5 * a['prediction'].astype('float64')
                      + 0.5 * b['prediction'].astype('float64'))
    return pred, a['truth'].astype('float64'), np.repeat(np.arange(24) >= 12, npoints)


def score(pred, truth, second, factor):
    error = (factor * pred - truth) ** 2
    return {'rmse_24': float(np.sqrt(error.mean())),
            'rmse_year1': float(np.sqrt(error[~second].mean())),
            'rmse_year2': float(np.sqrt(error[second].mean()))}


def evaluate(early, late):
    pred, truth, second = early
    denominator = float(pred @ pred)
    if denominator <= 0:
        raise ValueError('Cannot fit a scale to zero predictions')
    # One parameter; fit exclusively on 2015-16, keep fixed for 2021-22.
    unconstrained = float((pred @ truth) / denominator)
    factor = float(np.clip(unconstrained, 0.90, 1.05))
    rows = {}
    for name, data in [('fit_2015_2016', early), ('later_2021_2022', late)]:
        rows[name] = {label: score(*data, scale) for label, scale in
                      [('original', 1.0), ('current_097', 0.97), ('candidate', factor)]}
    validation = rows['later_2021_2022']
    gains = {k: validation['current_097'][k] - validation['candidate'][k]
             for k in ('rmse_24', 'rmse_year1', 'rmse_year2')}
    # A tiny numeric improvement does not justify changing the final candidate.
    accepted = (gains['rmse_24'] >= 0.001 and gains['rmse_year2'] >= 0.001
                and gains['rmse_year1'] >= 0)
    return {'factor_fit_only_2015_2016': factor, 'unconstrained_factor': unconstrained,
            'metrics': rows, 'later_block_gain_vs_097': gains,
            'eligible_for_review': bool(accepted),
            'notes': ['2022 single-year fold excluded to avoid overlapping evaluation.',
                      'Later block already used in previous model development; not an untouched test.',
                      'No public score used to fit factor; no evidence yet about private 2024.',
                      'If rejected, retain current 0.97. Do not search more factors on the later block.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('lucasteste1/results'))
    args = parser.parse_args()
    report = evaluate(load_pair(args.results, '2015full'), load_pair(args.results, '2021'))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
