"""Refit selected configuration and export predictions; never upload."""
from pathlib import Path
import gc
import json
import numpy as np
import pandas as pd
import xarray as xr
import lightgbm as lgb
from run_temporal_experiments import load_data, PARAMS
from backtest_residual_lgbm import make_frame
from train_residual_lightgbm import FEATURE_FILES, _clim_context
from ocean_indices import read_indices, add_indices
from common import DATA_DIR

OUT = Path(__file__).resolve().parents[1] / 'output'
OUT.mkdir(exist_ok=True)
arrays, lat, lon, times = load_data(4)
dates = times + pd.DateOffset(months=1)
idx = np.where((dates >= '1993-01-01') & (dates < '2023-01-01'))[0]
months = dates[idx].month.to_numpy()
clim = {m: arrays['target'][idx][months == m].mean(axis=0) for m in range(1,13)}
lags = np.random.default_rng(2026).integers(1, np.minimum(24,idx+1)+1).astype('float32')
series = read_indices()
frame = make_frame(arrays,lat,lon,times,idx,clim,lag_values=lags,circulation=True,atmosphere_history='three')
frame = add_indices(frame, dates[idx], len(lat)*len(lon), series)
features = [c for c in frame if c != 'target']
params = dict(PARAMS, num_leaves=224, learning_rate=.025)
print(f'Treino final: {dates[idx][0]} a {dates[idx][-1]}, {len(frame)} linhas',flush=True)
model = lgb.train(params,lgb.Dataset(frame[features],label=frame.target),num_boost_round=700)
model.save_model(str(OUT/'model_ocean_30years.txt'))
del frame,arrays
gc.collect()
with xr.open_dataset(DATA_DIR/'treino_tp_alvo.nc') as ds:
    fullclim = {m: ds.tp_alvo.isel(time=idx[months==m]).mean('time').transpose('lat','lon').values.astype('float32') for m in range(1,13)}
with xr.open_dataset(DATA_DIR/'teste_features.nc') as ds:
    test = ds.load()
testdates = pd.DatetimeIndex(test.time.values)
origins = pd.DatetimeIndex(test.time_origem.values)
assert (origins == testdates - pd.DateOffset(months=1)).all()
la,lo = test.lat.values,test.lon.values
context = {}
for key,value in _clim_context(np.stack([clim[m] for m in range(1,13)])).items():
    context[key] = xr.DataArray(value,dims=('month','lat','lon'),coords={'month':range(1,13),'lat':lat,'lon':lon}).interp(lat=la,lon=lo).values
history = {}
for name,file in FEATURE_FILES.items():
    if name == 'tp': continue
    with xr.open_dataset(DATA_DIR/file) as ds:
        v=ds[list(ds.data_vars)[0]].transpose('time','lat','lon')
        history[name] = v.sel(time=slice('2022-10-01','2022-11-01')).values.astype('float32')
    assert len(history[name]) == 2
parts=[]
for i,date in enumerate(testdates):
    a={}
    for name in history:
        a[name]=np.concatenate([history[name],test[name].isel(time=slice(0,i+1)).values.astype('float32')],axis=0)[-3:]
    rain=test.tp_ultima_obs.isel(time=i).values.astype('float32')
    a['tp']=np.repeat(rain[None],3,axis=0)
    a['target']=np.zeros_like(a['tp'])
    inputdates=pd.date_range(end=origins[i],periods=3,freq='MS')
    f=make_frame(a,la,lo,inputdates,np.array([2]),fullclim,frozen_tp_idx=2,lag_values=np.array([i+1]),circulation=True,atmosphere_history='three')
    for key,value in context.items(): f[key]=value[date.month-1].ravel()
    f=add_indices(f,pd.DatetimeIndex([date]),len(la)*len(lo),series)
    assert np.isfinite(f[features].to_numpy()).all()
    prediction=np.maximum(0,f.climatology.to_numpy()+model.predict(f[features]))
    ids=[f'{date.year}_{date.month:02d}_{x:.2f}_{y:.2f}' for x in la for y in lo]
    parts.append(pd.DataFrame({'id':ids,'tp_mm_day':prediction}))
    print(f'Previsoes {date:%Y-%m} concluidas',flush=True)
allpred=pd.concat(parts,ignore_index=True).set_index('id')
sample=pd.read_csv(DATA_DIR/'sample_submission.csv')
assert sample.id.is_unique and allpred.index.is_unique
assert set(sample.id)==set(allpred.index)
result=allpred.loc[sample.id].reset_index()
assert np.isfinite(result.tp_mm_day).all() and (result.tp_mm_day>=0).all()
result.to_csv(OUT/'submission_ocean_2023_2024.csv',index=False)
result[result.id.str.startswith('2023_')].to_csv(OUT/'predictions_ocean_2023.csv',index=False)
(OUT/'submission_ocean_metadata.json').write_text(json.dumps({'rows':len(result),'validation_2022_rmse':1.767188,'validation_2015_rmse':1.859466,'features':features,'train_end':str(dates[idx][-1]),'params':params,'rounds':700,'warning':'Retrospective monthly indices; release vintages not verified. Not a single December-issued annual forecast.'},indent=2))
print(f'CSV validado: {len(result)} linhas em {OUT}',flush=True)
