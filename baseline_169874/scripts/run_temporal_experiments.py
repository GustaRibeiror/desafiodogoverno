"""Ablations causais, artefatos persistentes e validação temporal independente.

Nunca submete ao Kaggle. Exemplos:
python "lucasteste 1/run_temporal_experiments.py" --fold 2015
python "lucasteste 1/run_temporal_experiments.py" --fold 1997 --experiments E00 E01
"""
from pathlib import Path
import argparse
import gc
import json
import time
import hashlib

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from common import DATA_DIR
from backtest_residual_lgbm import make_frame
from train_residual_lightgbm import FEATURE_FILES

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'
FOLDS = {'1997': ('1997-05-01', '1998-04-01'),
         '2015': ('2015-01-01', '2016-05-01'),
         '2021': ('2021-01-01', '2022-12-01'),
         '2022': ('2022-01-01', '2022-12-01')}
CONFIGS = {
    'E00': ('LightGBM', 'baseline'),
    'E01': ('LightGBM', 'clim_stats'),
    'E02': ('LightGBM', 'anchor_history'),
    'E03': ('Ridge', 'anchor_history'),
    'E04': ('HistGradientBoosting', 'anchor_history'),
}
PARAMS = dict(objective='regression', metric='rmse', learning_rate=0.03,
              num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, max_bin=127,
              verbosity=-1, num_threads=8)


def score(pred, truth):
    return float(np.sqrt(np.mean((np.asarray(pred, dtype='float64') - truth) ** 2)))


def load_data(stride):
    # O ZIP original assumia que este diretório já existia. Criá-lo aqui torna
    # a primeira execução realmente reprodutível, inclusive em um Kaggle novo.
    RESULTS.mkdir(parents=True, exist_ok=True)
    cache = RESULTS / f'cache_stride{stride}.npz'
    if cache.exists():
        with np.load(cache) as z:
            arrays = {k: z[k] for k in (*FEATURE_FILES, 'target')}
            return arrays, z['lat'], z['lon'], pd.DatetimeIndex(z['time'])
    arrays = {}
    for name, filename in {**FEATURE_FILES, 'target': 'treino_tp_alvo.nc'}.items():
        print('Carregando', name, flush=True)
        with xr.open_dataset(DATA_DIR / filename) as ds:
            v = ds[list(ds.data_vars)[0]].transpose('time', 'lat', 'lon')
            sub = v.isel(time=slice(None, -1), lat=slice(None, None, stride),
                         lon=slice(None, None, stride))
            arrays[name] = sub.values.astype('float32')
            lat, lon, times = sub.lat.values, sub.lon.values, sub.time.values
    if not all(np.isfinite(x).all() for x in arrays.values()):
        raise ValueError('Dados não finitos: investigar antes de treinar.')
    np.savez(cache, **arrays, lat=lat, lon=lon, time=times)
    return arrays, lat, lon, pd.DatetimeIndex(times)


def augment(frame, arrays, times, idx, anchors, train_idx, train_months, group):
    if group == 'baseline':
        return frame
    target_month = (times[idx].month.to_numpy() % 12) + 1
    ytrain = arrays['target'][train_idx]
    # Todos os valores de validação usam exclusivamente estatísticas de TRAIN.
    median = np.stack([np.median(ytrain[train_months == m], axis=0) for m in range(1, 13)])
    std = np.stack([ytrain[train_months == m].std(axis=0) for m in range(1, 13)])
    frame['clim_median'] = median[target_month - 1].ravel()
    frame['clim_std'] = std[target_month - 1].ravel()
    if group == 'anchor_history':
        rain = arrays['tp']
        # Histórico relativo à ÚLTIMA observação conhecida, não ao mês-alvo.
        # A âncora fica congelada durante cada bloco, tal como no Kaggle.
        for lag in (1, 2, 5, 11):
            source = anchors - lag
            values = rain[np.maximum(source, 0)].copy()
            values[source < 0] = np.nan
            frame[f'tp_anchor_minus_{lag}'] = values.ravel()
        for width in (3, 6, 12):
            means, stds = [], []
            for anchor in anchors:
                window = rain[max(0, anchor - width + 1):anchor + 1]
                means.append(window.mean(axis=0))
                stds.append(window.std(axis=0))
            frame[f'tp_anchor_mean_{width}'] = np.asarray(means).ravel()
            if width != 12:
                frame[f'tp_anchor_std_{width}'] = np.asarray(stds).ravel()
    return frame


def diagnose(pred, truth, dates, lat, lon, prefix):
    p = pred.reshape(len(dates), len(lat), len(lon))
    y = truth.reshape(p.shape)
    err = p.astype('float64') - y
    se = err ** 2
    pd.DataFrame({'target_month': dates.astype(str), 'rmse': np.sqrt(se.mean(axis=(1, 2))),
                  'sse': se.sum(axis=(1, 2))}).to_csv(str(prefix) + '_months.csv', index=False)
    pd.DataFrame({'lat': lat, 'rmse': np.sqrt(se.mean(axis=(0, 2)))}).to_csv(str(prefix) + '_lat.csv', index=False)
    pd.DataFrame({'lon': lon, 'rmse': np.sqrt(se.mean(axis=(0, 1)))}).to_csv(str(prefix) + '_lon.csv', index=False)
    la, lo = np.meshgrid(lat, lon, indexing='ij')
    pd.DataFrame({'lat': la.ravel(), 'lon': lo.ravel(), 'rmse': np.sqrt(se.mean(axis=0)).ravel(),
                  'sse': se.sum(axis=0).ravel()}).sort_values('sse', ascending=False).to_csv(str(prefix) + '_cells.csv', index=False)
    top = np.argsort(se.ravel())[-30:][::-1]
    ti, li, lj = np.unravel_index(top, p.shape)
    pd.DataFrame({'month': dates[ti].astype(str), 'lat': lat[li], 'lon': lon[lj],
                  'observed': y.ravel()[top], 'predicted': p.ravel()[top],
                  'squared_error': se.ravel()[top]}).to_csv(str(prefix) + '_largest_errors.csv', index=False)
    return {'bias': float(err.mean()), 'error_quantiles': np.quantile(err, [0,.01,.1,.5,.9,.99,1]).tolist(),
            'top_1pct_sse_share': float(np.sort(se.ravel())[-max(1,se.size//100):].sum()/se.sum())}


def summarize():
    rows = [json.loads(p.read_text(encoding='utf8')) for p in RESULTS.glob('E*_fold*_s*.json')]
    summary = []
    for (exp, stride), group in pd.DataFrame(rows).groupby(['experiment', 'stride']):
        row = {'experiment': exp, 'model': group.iloc[0]['model'], 'features': group.iloc[0]['feature_group'],
               'stride': stride, 'validation_strategy': 'expanding window; tp frozen; atmosphere t-1',
               'training_time': group.training_time.sum(), 'notes': 'Partial spatial grid. No Kaggle submission.'}
        for i, fold in enumerate(FOLDS, 1):
            g = group[group.fold == fold]
            row[f'rmse_fold{i}'] = float(g.rmse.iloc[0]) if len(g) else np.nan
        row['rmse_mean'] = float(group.rmse.mean())
        row['rmse_std'] = float(group.rmse.std(ddof=0))
        row['rmse_pooled'] = float(np.sqrt(group.sse.sum()/group.n.sum()))
        row['folds_completed'] = len(group)
        summary.append(row)
    pd.DataFrame(summary).sort_values('rmse_mean').to_csv(RESULTS/'experiments.csv', index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', choices=FOLDS, default='2015')
    parser.add_argument('--experiments', nargs='+', choices=CONFIGS, default=list(CONFIGS))
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--leaves', type=int, default=63)
    parser.add_argument('--rounds', type=int, default=560)
    parser.add_argument('--lr', type=float, default=0.03)
    parser.add_argument('--multi-lag', action='store_true')
    parser.add_argument('--regional', action='store_true')
    parser.add_argument('--ocean-indices', action='store_true')
    parser.add_argument('--ocean-interactions', action='store_true')
    parser.add_argument('--nmme-ccsm4', action='store_true', help='Previsão NMME CCSM4 inicializada no mês de origem; requer stride=8')
    parser.add_argument('--nmme-geoss2s', action='store_true', help='Previsão NMME GEOSS2S inicializada no mês de origem; requer stride=8')
    parser.add_argument('--nmme-geoss2s-tos', action='store_true', help='TSM prevista pelo GEOSS2S no mês-alvo; requer stride=8')
    parser.add_argument('--atmosphere-history', choices=['none', 'three', 'trajectory'], default='none')
    parser.add_argument('--train-years', type=int, default=0, help='Janela anterior ao bloco; 0 usa todo o historico')
    args = parser.parse_args()
    PARAMS['num_leaves'] = args.leaves
    PARAMS['learning_rate'] = args.lr
    RESULTS.mkdir(exist_ok=True)
    arrays, lat, lon, times = load_data(args.stride)
    target_dates = times + pd.DateOffset(months=1)
    first, last = map(pd.Timestamp, FOLDS[args.fold])
    tr = np.where(target_dates < first)[0]
    if args.train_years < 0:
        parser.error('--train-years deve ser >= 0')
    if args.train_years:
        tr = tr[target_dates[tr] >= first - pd.DateOffset(years=args.train_years)]
    va = np.where((target_dates >= first) & (target_dates <= last))[0]
    months = target_dates[tr].month.to_numpy()
    clim = {m: arrays['target'][tr][months == m].mean(axis=0) for m in range(1,13)}
    lags = np.random.default_rng(2026).integers(1, np.minimum(24, tr+1)+1).astype('float32')
    anchors = tr - lags.astype('int16') + 1
    val_anchors = np.full(len(va), va[0])
    assert np.all(anchors <= tr) and np.all(val_anchors <= va)
    assert target_dates[tr].max() < target_dates[va].min()
    assert np.all(times[va] < target_dates[va])
    print(f'Fold {args.fold}; treino {len(tr)} meses; validação {len(va)}; grade {len(lat)}x{len(lon)}', flush=True)
    truth = arrays['target'][va].ravel()
    for exp in args.experiments:
        tag = '' if (args.leaves == 63 and args.rounds == 560 and args.lr == 0.03 and not args.multi_lag and not args.regional) else f'_l{args.leaves}_r{args.rounds}_lr{args.lr:g}' + ('_ml' if args.multi_lag else '') + ('_rg' if args.regional else '')
        prefix = RESULTS/f'{exp}_fold{args.fold}_s{args.stride}{tag}'
        if args.train_years:
            prefix = prefix.with_name(prefix.name + f'_years{args.train_years}')
        if args.atmosphere_history != 'none':
            prefix = prefix.with_name(prefix.name + '_atm_' + args.atmosphere_history)
        if args.ocean_indices:
            prefix = prefix.with_name(prefix.name + '_ocean')
        if args.ocean_interactions:
            prefix = prefix.with_name(prefix.name + '_interactions')
        if args.nmme_ccsm4:
            prefix = prefix.with_name(prefix.name + '_nmme_ccsm4')
        if args.nmme_geoss2s:
            prefix = prefix.with_name(prefix.name + '_nmme_geoss2s')
        if args.nmme_geoss2s_tos:
            prefix = prefix.with_name(prefix.name + '_nmme_geoss2s_tos')
        if prefix.with_name(prefix.name + '.json').exists():
            saved = json.loads(prefix.with_name(prefix.name + '.json').read_text(encoding='utf8'))
            print(f'{exp} resultado salvo: RMSE={saved["rmse"]:.6f}; climatologia={saved["climatology_rmse"]:.6f}', flush=True)
            continue
        model_name, group = CONFIGS[exp]
        started = time.perf_counter()
        train = make_frame(arrays,lat,lon,times,tr,clim,lag_values=lags,circulation=True,multi_lag=args.multi_lag,regional=args.regional)
        val = make_frame(arrays,lat,lon,times,va,clim,frozen_tp_idx=va[0],
                         lag_values=np.arange(1,len(va)+1),circulation=True,multi_lag=args.multi_lag,regional=args.regional)
        if args.atmosphere_history != 'none':
            del train, val
            train = make_frame(arrays,lat,lon,times,tr,clim,lag_values=lags,circulation=True,atmosphere_history=args.atmosphere_history)
            val = make_frame(arrays,lat,lon,times,va,clim,frozen_tp_idx=va[0],lag_values=np.arange(1,len(va)+1),circulation=True,atmosphere_history=args.atmosphere_history)
        train = augment(train,arrays,times,tr,anchors,tr,months,group)
        val = augment(val,arrays,times,va,val_anchors,tr,months,group)
        if args.ocean_indices:
            from ocean_indices import read_indices, add_indices
            series = read_indices()
            train = add_indices(train, target_dates[tr], len(lat)*len(lon), series, args.ocean_interactions)
            val = add_indices(val, target_dates[va], len(lat)*len(lon), series, args.ocean_interactions)
        if args.nmme_ccsm4:
            if args.stride != 8:
                parser.error('--nmme-ccsm4 requer --stride 8 com o cache atual')
            from nmme_features import add_ccsm4_lead1
            train = add_ccsm4_lead1(train, times[tr], lat, lon)
            val = add_ccsm4_lead1(val, times[va], lat, lon)
        if args.nmme_geoss2s:
            if args.stride != 8:
                parser.error('--nmme-geoss2s requer --stride 8 com o cache atual')
            from nmme_features import add_geoss2s_lead1
            train = add_geoss2s_lead1(train, times[tr], lat, lon)
            val = add_geoss2s_lead1(val, times[va], lat, lon)
        if args.nmme_geoss2s_tos:
            if args.stride != 8:
                parser.error('--nmme-geoss2s-tos requer --stride 8 com o cache atual')
            from nmme_features import add_geoss2s_tos_lead1
            train = add_geoss2s_tos_lead1(train, times[tr], lat, lon)
            val = add_geoss2s_tos_lead1(val, times[va], lat, lon)
        features = [c for c in train if c != 'target']
        baseline = val.climatology.to_numpy().copy()
        prep_time = time.perf_counter()-started
        print(f'{exp} {model_name}: {len(train):,} linhas, {len(features)} features; treino iniciado',flush=True)
        begin_fit = time.perf_counter()
        with threadpool_limits(limits=8):
            if model_name == 'LightGBM':
                model = lgb.train(PARAMS,lgb.Dataset(train[features],label=train.target),num_boost_round=args.rounds)
                model.save_model(str(prefix)+'.txt')
            elif model_name == 'Ridge':
                model = make_pipeline(SimpleImputer(),StandardScaler(),Ridge(alpha=1000.0))
                model.fit(train[features],train.target)
                joblib.dump(model,str(prefix)+'.joblib')
            else:
                # Mesmo treino temporal; nenhuma separação aleatória para early stopping.
                model = HistGradientBoostingRegressor(max_iter=200,learning_rate=.08,
                    max_leaf_nodes=31,min_samples_leaf=100,l2_regularization=10,
                    early_stopping=False,random_state=2026)
                model.fit(train[features],train.target)
                joblib.dump(model,str(prefix)+'.joblib')
            fit_time = time.perf_counter()-begin_fit
            pred = np.maximum(0,baseline+model.predict(val[features])).astype('float32')
        assert np.isfinite(pred).all()
        diagnostics = diagnose(pred,truth,target_dates[va],lat,lon,prefix)
        np.savez_compressed(str(prefix)+'_predictions.npz',prediction=pred,truth=truth,
                            climatology=baseline,time=target_dates[va].values,lat=lat,lon=lon)
        result = dict(experiment=exp,model=model_name,feature_group=group,features=features,
            fold=args.fold,stride=args.stride,train_years=args.train_years,atmosphere_history=args.atmosphere_history,rounds=args.rounds,rmse=score(pred,truth),climatology_rmse=score(baseline,truth),
            sse=float(np.sum((pred.astype('float64')-truth)**2)),n=len(truth),
            training_time=fit_time,preparation_time=prep_time,train_first=str(target_dates[tr][0]),
            train_last=str(target_dates[tr][-1]),validation_first=str(first),validation_last=str(last),
            params=PARAMS if model_name=='LightGBM' else str(model.get_params()),
            diagnostics=diagnostics,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        prefix.with_name(prefix.name + '.json').write_text(json.dumps(result,indent=2),encoding='utf8')
        summarize()
        print(f'{exp} CONCLUÍDO RMSE={result["rmse"]:.6f}; climatologia={result["climatology_rmse"]:.6f}; treino={fit_time:.1f}s',flush=True)
        del train,val,model
        gc.collect()


if __name__ == '__main__':
    main()
