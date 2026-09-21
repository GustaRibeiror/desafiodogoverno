"""Treina um modelo residual sobre a climatologia mensal.

O alvo é ``tp_alvo - climatologia(ponto, mês)``. A previsão final soma o
resíduo estimado à climatologia do mês-alvo. Isso reduz o trabalho do modelo:
ele aprende anomalias meteorológicas e conserva a estrutura espacial/sazonal
que já é forte no baseline.

Uso:
    python experimentos/residual_lgbm/train_residual_lightgbm.py
"""

from pathlib import Path
import gc
import argparse
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR

OUT_DIR = ROOT / "outputs"
STRIDE = 4

FEATURE_FILES = {
    "tp": "treino_tp.nc",
    "t2": "treino_t2.nc",
    "cloud_cover": "treino_cloud_cover.nc",
    "surface_pressure": "treino_surface_pressure.nc",
    "shum_850": "treino_shum_850.nc",
    "rel_hum_850": "treino_rel_hum_850.nc",
    "temperature_850": "treino_temperature_850.nc",
    "geopotential_850": "treino_geopotential_850.nc",
    "u_850": "treino_u_850.nc",
    "v_850": "treino_v_850.nc",
}


def _load(name: str, time_slice, lat_idx, lon_idx) -> np.ndarray:
    ds = xr.open_dataset(DATA_DIR / FEATURE_FILES[name])
    var = list(ds.data_vars)[0]
    arr = ds[var].values[time_slice, :, :][:, lat_idx, :][:, :, lon_idx].astype("float32")
    ds.close()
    return arr


def _clim_context(field: np.ndarray) -> dict[str, np.ndarray]:
    """Features estáticas de vizinhança para representar relevo/costa/regime local."""
    p = np.pad(field, ((0, 0), (1, 1), (1, 1)), mode="edge")
    neigh = np.stack([
        p[:, 0:-2, 0:-2], p[:, 0:-2, 1:-1], p[:, 0:-2, 2:],
        p[:, 1:-1, 0:-2], p[:, 1:-1, 2:],
        p[:, 2:, 0:-2], p[:, 2:, 1:-1], p[:, 2:, 2:],
    ], axis=0)
    return {
        "clim_local_mean": neigh.mean(axis=0).astype("float32"),
        "clim_local_std": neigh.std(axis=0).astype("float32"),
        "clim_grad_lat": np.gradient(field, axis=1).astype("float32"),
        "clim_grad_lon": np.gradient(field, axis=2).astype("float32"),
    }


def build_train_table():
    tp_ds = xr.open_dataset(DATA_DIR / "treino_tp.nc")
    lat = tp_ds["lat"].values
    lon = tp_ds["lon"].values
    times = pd.to_datetime(tp_ds["time"].values)
    tp_ds.close()

    lat_idx = np.arange(0, len(lat), STRIDE)
    lon_idx = np.arange(0, len(lon), STRIDE)
    lat_s, lon_s = lat[lat_idx], lon[lon_idx]
    n_months = len(times) - 1
    target_month = ((times[:n_months].month % 12) + 1).astype("int16")

    arrays = {}
    for name in FEATURE_FILES:
        print(f"Carregando {name}...", flush=True)
        arrays[name] = _load(name, slice(0, n_months), lat_idx, lon_idx)
    target_ds = xr.open_dataset(DATA_DIR / "treino_tp_alvo.nc")
    print("Carregando target...", flush=True)
    target = target_ds["tp_alvo"].values[:n_months, :, :][:, lat_idx, :][:, :, lon_idx].astype("float32")
    target_ds.close()

    # Climatologia do alvo, por ponto e mês previsto.
    clim = {m: target[target_month == m].mean(axis=0) for m in range(1, 13)}
    clim_stack = np.stack([clim[int(m)] for m in target_month])
    residual = target - clim_stack
    context = _clim_context(np.stack([clim[m] for m in range(1, 13)]))

    n_points = len(lat_s) * len(lon_s)
    lat_grid, lon_grid = np.meshgrid(lat_s, lon_s, indexing="ij")
    lat_flat = lat_grid.reshape(-1).astype("float32")
    lon_flat = lon_grid.reshape(-1).astype("float32")
    month_sin = np.sin(2 * np.pi * target_month / 12).astype("float32")
    month_cos = np.cos(2 * np.pi * target_month / 12).astype("float32")

    cols = {
        "lat": np.tile(lat_flat, n_months),
        "lon": np.tile(lon_flat, n_months),
        "month_sin": np.repeat(month_sin, n_points),
        "month_cos": np.repeat(month_cos, n_points),
        "climatology": clim_stack.reshape(-1),
        "lat2": np.tile((lat_flat / 30.0) ** 2, n_months),
        "lon2": np.tile((lon_flat / 60.0) ** 2, n_months),
        "latlon": np.tile((lat_flat / 30.0) * (lon_flat / 60.0), n_months),
    }
    for key, arr in context.items():
        cols[key] = np.concatenate([arr[int(m) - 1].reshape(-1) for m in target_month])
    for name, arr in arrays.items():
        cols[name] = arr.reshape(-1)
    cols["target"] = residual.reshape(-1)
    df = pd.DataFrame(cols).replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    print(f"Tabela pronta: {len(df):,} linhas; iniciando treinamento...", flush=True)
    return df, clim, lat_s, lon_s


def build_test_table(clim, lat_s, lon_s):
    ds = xr.open_dataset(DATA_DIR / "teste_features.nc")
    lat = ds["lat"].values
    lon = ds["lon"].values
    times = pd.to_datetime(ds["time"].values)
    n_months, n_lat, n_lon = len(times), len(lat), len(lon)
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    n_points = n_lat * n_lon
    target_month = times.month.to_numpy().astype("int16")
    # Calcular a base na grade completa; manter a escala espacial das
    # features de contexto usada pelo modelo salvo (stride=4).
    with xr.open_dataset(DATA_DIR / "treino_tp_alvo.nc") as target_ds:
        source = target_ds["tp_alvo"].isel(time=slice(0, -1))
        months = (pd.to_datetime(source.time.values).month % 12) + 1
        full = np.stack([source.isel(time=np.where(months == m)[0]).mean("time").transpose("lat", "lon").values for m in range(1, 13)])
    clim_full = full[target_month - 1]
    coarse = full[:, ::STRIDE, ::STRIDE]
    context = {}
    for key, values in _clim_context(coarse).items():
        field = xr.DataArray(values, dims=("month", "lat", "lon"),
                             coords={"month": np.arange(1, 13), "lat": lat[::STRIDE], "lon": lon[::STRIDE]})
        context[key] = field.interp(lat=lat, lon=lon).values.astype("float32")
    lat_flat = lat_grid.reshape(-1).astype("float32")
    lon_flat = lon_grid.reshape(-1).astype("float32")

    cols = {
        "lat": np.tile(lat_flat, n_months),
        "lon": np.tile(lon_flat, n_months),
        "month_sin": np.repeat(np.sin(2 * np.pi * target_month / 12).astype("float32"), n_points),
        "month_cos": np.repeat(np.cos(2 * np.pi * target_month / 12).astype("float32"), n_points),
        "climatology": clim_full.reshape(-1),
        "tp": ds["tp_ultima_obs"].values.reshape(-1).astype("float32"),
        "lat2": np.tile((lat_flat / 30.0) ** 2, n_months),
        "lon2": np.tile((lon_flat / 60.0) ** 2, n_months),
        "latlon": np.tile((lat_flat / 30.0) * (lon_flat / 60.0), n_months),
    }
    for key, arr in context.items():
        cols[key] = np.concatenate([arr[int(m) - 1].reshape(-1) for m in target_month])
    for name in FEATURE_FILES:
        if name == "tp":
            continue
        cols[name] = ds[name].values.reshape(-1).astype("float32")
    ds.close()
    lengths = {name: len(values) for name, values in cols.items()}
    if any(length != n_months * n_points for length in lengths.values()):
        raise ValueError(f"Tamanhos incorretos nas features: {lengths}")
    df = pd.DataFrame(cols).replace([np.inf, -np.inf], np.nan).fillna(0)
    df["year"] = np.repeat(times.year.astype("int32"), n_points)
    df["month"] = np.repeat(target_month.astype("int16"), n_points)
    df["lat_r"] = df["lat"].round(2)
    df["lon_r"] = df["lon"].round(2)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict-only", action="store_true", help="Reutiliza o modelo salvo sem treinar novamente")
    args = parser.parse_args()
    if args.predict_only:
        model = lgb.Booster(model_file=str(OUT_DIR / "lgb_residual_model.txt"))
        print("Modelo salvo carregado; gerando CSV sem novo treinamento...", flush=True)
        write_submission(model, model.feature_name(), model.current_iteration())
        return
    df, clim, lat_s, lon_s = build_train_table()
    features = [c for c in df.columns if c != "target"]
    # O último bloco cronológico serve apenas para escolher uma iteração segura.
    # O ajuste final abaixo usa todos os meses e o número de árvores selecionado.
    n = len(df)
    split = int(n * 0.9)
    train_set = lgb.Dataset(df.iloc[:split][features], label=df.iloc[:split]["target"])
    valid_set = lgb.Dataset(df.iloc[split:][features], label=df.iloc[split:]["target"], reference=train_set)
    params = dict(objective="regression", metric="rmse", learning_rate=0.03,
                  num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, max_bin=127,
                  verbosity=-1, num_threads=8)
    model = lgb.train(params, train_set, num_boost_round=3000, valid_sets=[valid_set],
                      callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
    best = model.best_iteration or 1500

    # Reajusta no histórico completo com a iteração escolhida.
    full_set = lgb.Dataset(df[features], label=df["target"])
    model = lgb.train(params, full_set, num_boost_round=best)
    model.save_model(str(OUT_DIR / "lgb_residual_model.txt"))
    print("Modelo final salvo; montando previsões de teste...", flush=True)
    del df, train_set, valid_set, full_set
    gc.collect()

    write_submission(model, features, best)


def write_submission(model, features, best):
    test = build_test_table(None, None, None)
    missing = [c for c in features if c not in test.columns]
    if missing:
        raise ValueError(f"Features ausentes no teste: {missing}")
    pred_res = model.predict(test[features], num_iteration=best)
    pred = np.clip(test["climatology"].to_numpy() + pred_res, 0, None)
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    parts = sample["id"].str.split("_", expand=True)
    sample_keys = pd.DataFrame({
        "year": parts[0].astype("int32"),
        "month": parts[1].astype("int16"),
        "lat_r": parts[2].astype(float).round(2),
        "lon_r": parts[3].astype(float).round(2),
    })
    pred_table = test[["year", "month", "lat_r", "lon_r"]].copy()
    pred_table["tp_mm_day"] = pred.astype("float32")
    sample_keys["id"] = sample["id"].values
    sample = sample_keys.merge(
        pred_table, on=["year", "month", "lat_r", "lon_r"], how="left", sort=False, validate="one_to_one"
    )
    if sample["tp_mm_day"].isna().any() or len(sample) != len(pred_table):
        raise ValueError("IDs da submissão não foram alinhados à grade de teste")
    sample = sample[["id", "tp_mm_day"]]
    if not np.isfinite(sample["tp_mm_day"].to_numpy()).all():
        raise ValueError("Previsões não finitas")
    out = OUT_DIR / "submission_residual_lgbm.csv"
    sample.to_csv(out, index=False)
    print(f"Submissão escrita em {out} ({len(sample):,} linhas), best_iteration={best}")


if __name__ == "__main__":
    main()
