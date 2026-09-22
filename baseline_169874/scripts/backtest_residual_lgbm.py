"""Backtest temporal que imita o mecanismo de teste do Kaggle.

Treina somente antes de 2015 e valida janeiro/2015--maio/2016, mantendo a
precipitação de dezembro/2014 congelada em todo o bloco de validação. O RMSE
é calculado sobre todos os pixels do bloco, como no leaderboard.
"""

from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, rmse
from train_residual_lightgbm import FEATURE_FILES, STRIDE, _clim_context


FIRST_TARGET = pd.Timestamp("2015-01-01")
LAST_TARGET = pd.Timestamp("2016-05-01")


def load_arrays():
    with xr.open_dataset(DATA_DIR / "treino_tp.nc") as ds:
        lat, lon = ds["lat"].values, ds["lon"].values
        times = pd.to_datetime(ds["time"].values)
    ii, jj = np.arange(0, len(lat), STRIDE), np.arange(0, len(lon), STRIDE)
    arrays = {}
    for name, file in FEATURE_FILES.items():
        print(f"Carregando {name}...", flush=True)
        with xr.open_dataset(DATA_DIR / file) as ds:
            var = list(ds.data_vars)[0]
            arrays[name] = ds[var].values[:-1, ii, :][:, :, jj].astype("float32")
    with xr.open_dataset(DATA_DIR / "treino_tp_alvo.nc") as ds:
        arrays["target"] = ds["tp_alvo"].values[:-1, ii, :][:, :, jj].astype("float32")
    return arrays, lat[ii], lon[jj], times[:-1]


def make_frame(arrays, lat, lon, times, idx, clim, frozen_tp_idx=None,
               lag_values=None, circulation=False, multi_lag=False,
               regional=False, atmosphere_history='none'):
    target_dates = times[idx] + pd.DateOffset(months=1)
    months = target_dates.month.to_numpy().astype("int16")
    n_points = len(lat) * len(lon)
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    la, lo = lat_grid.ravel().astype("float32"), lon_grid.ravel().astype("float32")
    clim_stack = np.stack([clim[int(m)] for m in months])
    if lag_values is None:
        lag_values = np.ones(len(idx), dtype='float32')
    lag_values = np.asarray(lag_values, dtype='float32')
    context = _clim_context(np.stack([clim[m] for m in range(1, 13)]))
    out = {
        "lat": np.tile(la, len(idx)), "lon": np.tile(lo, len(idx)),
        "month_sin": np.repeat(np.sin(2 * np.pi * months / 12).astype("float32"), n_points),
        "month_cos": np.repeat(np.cos(2 * np.pi * months / 12).astype("float32"), n_points),
        "climatology": clim_stack.ravel(),
        "lat2": np.tile((la / 30) ** 2, len(idx)),
        "lon2": np.tile((lo / 60) ** 2, len(idx)),
        "latlon": np.tile((la / 30) * (lo / 60), len(idx)),
        "lag_meses": np.repeat(lag_values, n_points),
    }
    for key, val in context.items():
        out[key] = np.concatenate([val[int(m) - 1].ravel() for m in months])
    for name, arr in arrays.items():
        if name == "target":
            continue
        if name == 'tp' and frozen_tp_idx is not None:
            values = np.repeat(arr[frozen_tp_idx][None, ...], len(idx), axis=0)
        elif name == 'tp':
            values = arr[np.maximum(0, idx - (lag_values.astype('int16') - 1))]
        else:
            values = arr[idx]
        out[name] = values.ravel()
        if name != 'tp' and atmosphere_history != 'none':
            earlier = []
            for offset in (1, 2):
                source = idx - offset
                value = arr[np.maximum(source, 0)].copy()
                value[source < 0] = np.nan
                out[f'{name}_lag{offset + 1}'] = value.ravel()
                earlier.append(value)
            if atmosphere_history == 'trajectory':
                out[f'{name}_delta_recent'] = (values - earlier[0]).ravel()
                out[f'{name}_delta_previous'] = (earlier[0] - earlier[1]).ravel()
        if regional and name != 'tp':
            out[f'{name}_regional_mean'] = np.repeat(values.mean(axis=(1, 2)), n_points)
            out[f'{name}_regional_std'] = np.repeat(values.std(axis=(1, 2)), n_points)
        if name == 'tp' and multi_lag:
            for lag in (1, 2, 3, 6, 12):
                source = np.maximum(0, idx - (lag - 1))
                lagged = np.repeat(arr[frozen_tp_idx][None, ...], len(idx), axis=0) if frozen_tp_idx is not None else arr[source]
                out[f'tp_lag{lag}'] = lagged.ravel()
    if circulation:
        out['moisture_flux_u'] = out['u_850'] * out['shum_850']
        out['moisture_flux_v'] = out['v_850'] * out['shum_850']
        out['wind_speed_850'] = np.hypot(out['u_850'], out['v_850'])
    out["target"] = (arrays["target"][idx] - clim_stack).ravel()
    return pd.DataFrame(out)


def main():
    arrays, lat, lon, times = load_arrays()
    target_dates = times + pd.DateOffset(months=1)
    train_idx = np.where(target_dates < FIRST_TARGET)[0]
    val_idx = np.where((target_dates >= FIRST_TARGET) & (target_dates <= LAST_TARGET))[0]
    month_train = target_dates[train_idx].month.to_numpy()
    clim = {m: arrays["target"][train_idx][month_train == m].mean(axis=0) for m in range(1, 13)}
    print(f"Treino até {target_dates[train_idx][-1].date()} ({len(train_idx)} meses); validação {FIRST_TARGET.date()}--{LAST_TARGET.date()} ({len(val_idx)} meses)", flush=True)
    train = make_frame(arrays, lat, lon, times, train_idx, clim)
    features = [c for c in train.columns if c != "target"]
    model = lgb.train(
        dict(objective="regression", metric="rmse", learning_rate=0.03, num_leaves=63,
             min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
             bagging_freq=1, max_bin=127, verbosity=-1, num_threads=8),
        lgb.Dataset(train[features], label=train["target"]), num_boost_round=560,
    )
    frozen_tp_idx = val_idx[0]
    val = make_frame(arrays, lat, lon, times, val_idx, clim, frozen_tp_idx=frozen_tp_idx)
    raw_pred = val["climatology"].to_numpy() + model.predict(val[features])
    truth = arrays["target"][val_idx].ravel()
    baseline = np.stack([clim[int(m)] for m in target_dates[val_idx].month]).ravel()
    pred = np.clip(raw_pred, 0, None)
    weights = np.linspace(0.0, 1.0, 101)
    scores = np.array([
        rmse(np.clip(baseline + weight * (raw_pred - baseline), 0, None), truth)
        for weight in weights
    ])
    best = int(scores.argmin())
    print(f"RMSE climatologia: {rmse(baseline, truth):.4f}")
    print(f"RMSE residual LGBM (tp congelada): {rmse(pred, truth):.4f}")
    print(f"Melhor blend: {weights[best]:.2f} modelo + {1 - weights[best]:.2f} climatologia; RMSE={scores[best]:.4f}")


if __name__ == "__main__":
    main()
