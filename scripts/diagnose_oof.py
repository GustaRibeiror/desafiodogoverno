"""Auditoria de previsões históricas, sem treino e sem envio ao Kaggle.

Uso: python scripts/diagnose_oof.py --oof experiments/*_oof.npz --output report.json
Forneça exatamente um arquivo por fold, todos da mesma configuração/amostra.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from worcap_pipeline import apply_lead_calibration, fit_lead_calibration, sha256_file


def metrics(pred, truth, lead):
    error = (np.asarray(pred, dtype='float64') - truth) ** 2
    return {name: float(np.sqrt(error[mask].mean())) for name, mask in (
        ('rmse_24', lead > 0), ('rmse_year1', lead <= 12), ('rmse_year2', lead > 12)
    )}


def load(path):
    with np.load(path, allow_pickle=False) as source:
        data = {key: source[key].copy() for key in ('prediction', 'truth', 'climatology', 'lead', 'point_flat')}
    n = len(data['point_flat'])
    if n == 0 or len(np.unique(data['point_flat'])) != n:
        raise ValueError(f'Grade vazia/duplicada: {path}')
    for key in ('prediction', 'truth', 'climatology', 'lead'):
        if data[key].shape != (24 * n,) or not np.isfinite(data[key]).all():
            raise ValueError(f'OOF inválido: {path}, {key}')
    if not np.array_equal(data['lead'], np.repeat(np.arange(1, 25), n)):
        raise ValueError(f'Ordem dos horizontes inválida: {path}')
    return data


def diagnose(paths):
    paths = [Path(p) for p in paths]
    if len(paths) < 3 or len(set(p.resolve() for p in paths)) != len(paths):
        raise ValueError('Informe pelo menos três OOFs distintos, um por fold')
    data = [load(p) for p in paths]
    if any(not np.array_equal(d['point_flat'], data[0]['point_flat']) for d in data):
        raise ValueError('Grades diferentes: não misture smoke tests e validação completa')
    rows, calibrated = [], []
    for i, (path, d) in enumerate(zip(paths, data)):
        other = {key: np.concatenate([v[key] for j, v in enumerate(data) if j != i])
                 for key in ('prediction', 'truth', 'climatology', 'lead')}
        weights = fit_lead_calibration(other['climatology'], other['prediction'], other['truth'], other['lead'])
        pred = apply_lead_calibration(d['climatology'], d['prediction'], d['lead'], weights)
        calibrated.append(pred)
        row = {'file': str(path), 'sha256': sha256_file(path), 'n_points': len(d['point_flat']),
               'raw': metrics(d['prediction'], d['truth'], d['lead']),
               'climatology': metrics(d['climatology'], d['truth'], d['lead']),
               'calibrated_held_out': metrics(pred, d['truth'], d['lead']), 'weights_from_other_folds': weights}
        row['monthly'] = []
        for month in range(1, 25):
            take = d['lead'] == month
            row['monthly'].append({'lead': month, **{
                name: float(np.sqrt(np.mean((values[take].astype('float64') - d['truth'][take]) ** 2)))
                for name, values in [('raw', d['prediction']), ('climatology', d['climatology']), ('calibrated', pred)]}})
        rows.append(row)
    joined = {key: np.concatenate([d[key] for d in data]) for key in ('prediction', 'truth', 'climatology', 'lead')}
    aggregate = {name: metrics(values, joined['truth'], joined['lead']) for name, values in (
        ('raw', joined['prediction']), ('climatology', joined['climatology']), ('calibrated_held_out', np.concatenate(calibrated)))}
    raw, cal = aggregate['raw'], aggregate['calibrated_held_out']
    accept = (cal['rmse_year2'] < raw['rmse_year2'] and cal['rmse_24'] <= raw['rmse_24']
              and all(r['calibrated_held_out']['rmse_year2'] <= r['raw']['rmse_year2'] + .02 for r in rows)
              and sum(r['calibrated_held_out']['rmse_year2'] < r['raw']['rmse_year2'] for r in rows) >= 2)
    weights = fit_lead_calibration(joined['climatology'], joined['prediction'], joined['truth'], joined['lead'])
    return {'folds': rows, 'aggregate_pooled_rmse': aggregate,
            'accept_calibration_vs_raw': accept, 'candidate_final_weights': weights,
            'deployment_weights': weights if accept else [1., 1., 1., 1.],
            'notes': ['RMSE agregado = raiz da média de todos os erros quadráticos, não média dos RMSEs.',
                      'Leave-one-fold-out avalia transferência entre regimes; não é validação cronológica do calibrador.',
                      'Os folds foram usados para decisões de desenvolvimento; não são um teste final intocado.',
                      'Esta comparação não prova superioridade à entrega 1,69874. Falta a referência no mesmo backtest.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oof', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = diagnose(args.oof)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'folds'}, indent=2, ensure_ascii=False))
    for row in report['folds']:
        print(json.dumps({k: v for k, v in row.items() if k != 'monthly'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
