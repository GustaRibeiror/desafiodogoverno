#!/usr/bin/env python3
"""Pipeline canônico e causal para o desafio WORCAP de precipitação.

O módulo reúne treino, backtest, calibração, inferência, blend e auditoria. Ele
NUNCA envia arquivos ao Kaggle. A precipitação permanece congelada na última
observação anterior a cada bloco de 24 meses; campos atmosféricos de um alvo T
são limitados a T-1.

Exemplos são documentados em ``docs/ENTREGA_MODELO.md``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import numpy as np
    import pandas as pd
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Dependência ausente: {exc.name}. Instale requirements-worcap.txt."
    ) from exc

try:
    import lightgbm as lgb
except ModuleNotFoundError:
    lgb = None  # type: ignore[assignment]

try:
    import xarray as xr
except ModuleNotFoundError:
    xr = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = ROOT / "experiments"
ARTIFACTS_DIR = ROOT / "artifacts"
REGISTRY_PATH = EXPERIMENTS_DIR / "registry.csv"
BASELINE_SHA256 = "2b5084b0bb00131948e1cba3720b7e96774b0f5b10d96119646c0508fdd8782b"
COMPETITION_SLUG = "previsao-climatica-de-precipitacao-sobre-a-america-do-sul"

ATMOS_FILES = {
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
LOCAL_VARS = ("cloud_cover", "shum_850", "rel_hum_850", "u_850", "v_850")
OCEAN_FILES = {
    "soi": "soi.txt",
    "nino12": "nina1.anom.data.txt",
    "nino3": "nina3.anom.data.txt",
    "nino4": "nina4.anom.data.txt",
}


@dataclass(frozen=True)
class Fold:
    name: str
    first_target: str
    last_target: str


FOLDS: dict[str, Fold] = {
    "1997": Fold("1997", "1997-01-01", "1998-12-01"),
    "2015": Fold("2015", "2015-01-01", "2016-12-01"),
    "2021": Fold("2021", "2021-01-01", "2022-12-01"),
}


@dataclass(frozen=True)
class ModelConfig:
    name: str
    num_leaves: int
    learning_rate: float
    rounds: int
    min_data_in_leaf: int
    feature_fraction: float
    bagging_fraction: float
    lambda_l1: float
    lambda_l2: float
    max_bin: int = 127
    seed: int = 2026

    def lightgbm_params(self) -> dict[str, Any]:
        return {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_data_in_leaf,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": 1,
            "lambda_l1": self.lambda_l1,
            "lambda_l2": self.lambda_l2,
            "max_bin": self.max_bin,
            "seed": self.seed,
            "feature_fraction_seed": self.seed,
            "bagging_seed": self.seed,
            "data_random_seed": self.seed,
            "deterministic": True,
            "force_col_wise": True,
            "num_threads": int(os.environ.get("WORCAP_NUM_THREADS", "8")),
            "verbosity": -1,
        }


LGB_CONFIGS: dict[str, ModelConfig] = {
    "wide": ModelConfig("wide", 224, 0.025, 700, 100, 0.80, 0.80, 0.0, 1.0),
    "regularized": ModelConfig("regularized", 128, 0.030, 600, 160, 0.82, 0.85, 0.1, 5.0),
    "shallow": ModelConfig("shallow", 96, 0.025, 800, 220, 0.90, 0.90, 0.0, 10.0),
}


@dataclass
class SampledBundle:
    times: pd.DatetimeIndex
    lat: np.ndarray
    lon: np.ndarray
    point_flat: np.ndarray
    series: dict[str, np.ndarray]
    local_mean: dict[str, np.ndarray]
    local_std: dict[str, np.ndarray]
    flux_convergence: np.ndarray
    target: np.ndarray
    target_clim_mean: np.ndarray
    target_clim_std: np.ndarray
    target_clim_context: dict[str, np.ndarray]


def _require_runtime(*packages: str) -> None:
    missing = [name for name in packages if (name == "lightgbm" and lgb is None) or (name == "xarray" and xr is None)]
    if missing:
        raise ModuleNotFoundError(
            f"Dependências necessárias para esta operação: {', '.join(missing)}. "
            "Instale requirements-worcap.txt."
        )


def sha256_file(path: str | Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def code_sha256() -> str:
    return sha256_file(Path(__file__))


def resolve_data_dir(explicit: str | Path | None = None) -> Path:
    """Resolve a pasta da competição em ambiente local ou Kaggle."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("WORCAP_DATA_DIR"):
        candidates.append(Path(os.environ["WORCAP_DATA_DIR"]))
    candidates.extend(
        [
            Path("/kaggle/input") / COMPETITION_SLUG,
            ROOT / "data" / COMPETITION_SLUG,
            ROOT / "data",
        ]
    )
    for candidate in candidates:
        if (candidate / "treino_tp.nc").exists() and (candidate / "sample_submission.csv").exists():
            return candidate.resolve()
    tried = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Dados da competição não encontrados. Caminhos testados:\n  - {tried}")


def _data_var(ds: xr.Dataset) -> xr.DataArray:
    if len(ds.data_vars) != 1:
        # teste_features possui várias variáveis; esta função não é usada nele.
        raise ValueError(f"Esperava uma variável, encontrei {list(ds.data_vars)}")
    return ds[next(iter(ds.data_vars))].transpose("time", "lat", "lon")


def _read_grid(data_dir: Path) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    _require_runtime("xarray")
    with xr.open_dataset(data_dir / "treino_tp.nc") as ds:
        field = _data_var(ds)
        return (
            pd.DatetimeIndex(field.time.values),
            field.lat.values.astype("float32"),
            field.lon.values.astype("float32"),
        )


def target_dates_from_origins(times: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return times + pd.DateOffset(months=1)


def fold_indices(times: pd.DatetimeIndex, fold: Fold, train_years: int = 30) -> tuple[np.ndarray, np.ndarray]:
    targets = target_dates_from_origins(times)
    first, last = pd.Timestamp(fold.first_target), pd.Timestamp(fold.last_target)
    train_start = first - pd.DateOffset(years=train_years)
    train = np.flatnonzero((targets >= train_start) & (targets < first))
    valid = np.flatnonzero((targets >= first) & (targets <= last))
    if len(valid) != 24:
        raise AssertionError(f"Fold {fold.name} deveria ter 24 meses, recebeu {len(valid)}")
    if not len(train) or targets[train].max() >= targets[valid].min():
        raise AssertionError("Separação temporal inválida")
    # O primeiro campo de validação é o mês anterior ao primeiro alvo.
    if times[valid[0]] != first - pd.DateOffset(months=1):
        raise AssertionError("Origem atmosférica não corresponde a alvo T-1")
    return train, valid


def _month_stats(values: np.ndarray, dates: pd.DatetimeIndex, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    months = dates[idx].month.to_numpy()
    means, stds = [], []
    for month in range(1, 13):
        take = idx[months == month]
        if not len(take):
            raise ValueError(f"Sem amostras para o mês {month}")
        means.append(np.nanmean(values[take], axis=0, dtype="float64").astype("float32"))
        stds.append(np.nanstd(values[take], axis=0, dtype="float64").astype("float32"))
    return np.stack(means), np.stack(stds)


def _local_stats_full(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Média/desvio 3x3 para array (..., lat, lon), bordas replicadas."""
    pad_width = [(0, 0)] * field.ndim
    pad_width[-2] = (1, 1)
    pad_width[-1] = (1, 1)
    padded = np.pad(field, pad_width, mode="edge")
    height, width = field.shape[-2:]
    total = np.zeros_like(field, dtype="float32")
    total2 = np.zeros_like(field, dtype="float32")
    for di in range(3):
        for dj in range(3):
            value = padded[..., di : di + height, dj : dj + width].astype("float32", copy=False)
            total += value
            total2 += value * value
    mean = total / 9.0
    variance = np.maximum(total2 / 9.0 - mean * mean, 0.0)
    return mean, np.sqrt(variance).astype("float32")


def _local_stats_points(field: np.ndarray, ii: np.ndarray, jj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Média/desvio 3x3 sem materializar nove cópias da grade completa."""
    total = np.zeros((field.shape[0], len(ii)), dtype="float32")
    total2 = np.zeros_like(total)
    for di in (-1, 0, 1):
        iii = np.clip(ii + di, 0, field.shape[1] - 1)
        for dj in (-1, 0, 1):
            jjj = np.clip(jj + dj, 0, field.shape[2] - 1)
            value = field[:, iii, jjj].astype("float32", copy=False)
            total += value
            total2 += value * value
    mean = total / 9.0
    return mean, np.sqrt(np.maximum(total2 / 9.0 - mean * mean, 0.0)).astype("float32")


def _clim_context_full(clim: np.ndarray) -> dict[str, np.ndarray]:
    mean, std = _local_stats_full(clim)
    return {
        "clim_local_mean": mean,
        "clim_local_std": std,
        "clim_grad_lat": np.gradient(clim, axis=1).astype("float32"),
        "clim_grad_lon": np.gradient(clim, axis=2).astype("float32"),
    }


def stratified_point_pair(
    rain_mean: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    n_each: int = 8000,
    seed: int = 2026,
) -> tuple[np.ndarray, np.ndarray]:
    """Duas amostras complementares por latitude, longitude e regime de chuva."""
    total = rain_mean.size
    if 2 * n_each > total:
        raise ValueError("A grade não contém pontos suficientes para duas amostras disjuntas")
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    rain = rain_mean.ravel()
    finite = np.isfinite(rain)
    q = np.quantile(rain[finite], [0.25, 0.50, 0.75])
    rain_bin = np.digitize(rain, q)
    lat_bin = np.clip(np.digitize(la.ravel(), np.quantile(lat, np.linspace(0, 1, 7)[1:-1])), 0, 5)
    lon_bin = np.clip(np.digitize(lo.ravel(), np.quantile(lon, np.linspace(0, 1, 7)[1:-1])), 0, 5)
    strata = lat_bin * 24 + lon_bin * 4 + rain_bin
    rng = np.random.default_rng(seed)
    wanted = 2 * n_each
    selected: list[int] = []
    for label in np.unique(strata[finite]):
        members = np.flatnonzero(finite & (strata == label))
        rng.shuffle(members)
        quota = max(1, int(round(wanted * len(members) / finite.sum())))
        selected.extend(members[: min(quota, len(members))].tolist())
    selected_array = np.asarray(selected, dtype="int64")
    selected_array = np.unique(selected_array)
    rng.shuffle(selected_array)
    if len(selected_array) < wanted:
        remaining = np.setdiff1d(np.flatnonzero(finite), selected_array, assume_unique=False)
        rng.shuffle(remaining)
        selected_array = np.concatenate([selected_array, remaining[: wanted - len(selected_array)]])
    selected_array = selected_array[:wanted]
    # Distribuição alternada evita favorecer uma amostra com um único estrato.
    return np.sort(selected_array[0::2][:n_each]), np.sort(selected_array[1::2][:n_each])


def canonical_validation_points(nlat: int, nlon: int, stride: int = 4) -> np.ndarray:
    ii, jj = np.meshgrid(np.arange(0, nlat, stride), np.arange(0, nlon, stride), indexing="ij")
    return np.ravel_multi_index((ii.ravel(), jj.ravel()), (nlat, nlon))


def _load_target_and_points(
    data_dir: Path,
    times: pd.DatetimeIndex,
    lat: np.ndarray,
    lon: np.ndarray,
    train_idx: np.ndarray,
    sample: str,
    n_points: int,
    explicit_points: np.ndarray | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    with xr.open_dataset(data_dir / "treino_tp_alvo.nc") as ds:
        full = _data_var(ds).values[:-1].astype("float32")
    target_times = target_dates_from_origins(times)
    clim_mean, clim_std = _month_stats(full, target_times, train_idx)
    if explicit_points is None:
        pair = stratified_point_pair(np.nanmean(full[train_idx], axis=0), lat, lon, n_points, seed)
        point_flat = pair[0 if sample == "a" else 1]
    else:
        point_flat = np.asarray(explicit_points, dtype="int64")
    ii, jj = np.unravel_index(point_flat, (len(lat), len(lon)))
    target = full[:, ii, jj].astype("float32")
    context = _clim_context_full(clim_mean)
    sampled_context = {name: value[:, ii, jj].astype("float32") for name, value in context.items()}
    return (
        point_flat,
        target,
        clim_mean[:, ii, jj].astype("float32"),
        clim_std[:, ii, jj].astype("float32"),
        sampled_context,
    )


def _sample_flux_convergence(
    data_dir: Path, ii: np.ndarray, jj: np.ndarray, ntime: int, chunk_size: int = 24
) -> np.ndarray:
    output = np.empty((ntime, len(ii)), dtype="float32")
    paths = [data_dir / ATMOS_FILES[name] for name in ("shum_850", "u_850", "v_850")]
    with xr.open_dataset(paths[0]) as qds, xr.open_dataset(paths[1]) as uds, xr.open_dataset(paths[2]) as vds:
        qvar, uvar, vvar = _data_var(qds), _data_var(uds), _data_var(vds)
        for start in range(0, ntime, chunk_size):
            stop = min(start + chunk_size, ntime)
            q = qvar.isel(time=slice(start, stop)).values.astype("float32")
            u = uvar.isel(time=slice(start, stop)).values.astype("float32")
            v = vvar.isel(time=slice(start, stop)).values.astype("float32")
            # Aproximação em coordenadas da grade; escala absoluta é aprendida pelo modelo.
            divergence = np.gradient(u * q, axis=2) + np.gradient(v * q, axis=1)
            output[start:stop] = (-divergence[:, ii, jj]).astype("float32")
    return output


def load_sampled_bundle(
    data_dir: str | Path,
    train_idx: np.ndarray,
    sample: str = "a",
    n_points: int = 8000,
    explicit_points: np.ndarray | None = None,
    seed: int = 2026,
) -> SampledBundle:
    """Carrega apenas pontos necessários, preservando estatísticas causais."""
    _require_runtime("xarray")
    data_dir = Path(data_dir)
    all_times, lat, lon = _read_grid(data_dir)
    times = all_times[:-1]
    point_flat, target, clim_mean, clim_std, clim_context = _load_target_and_points(
        data_dir, times, lat, lon, train_idx, sample, n_points, explicit_points, seed
    )
    ii, jj = np.unravel_index(point_flat, (len(lat), len(lon)))
    series: dict[str, np.ndarray] = {}
    local_mean: dict[str, np.ndarray] = {}
    local_std: dict[str, np.ndarray] = {}
    with xr.open_dataset(data_dir / "treino_tp.nc") as ds:
        series["tp_anchor"] = _data_var(ds).values[:-1, ii, jj].astype("float32")
    for name, filename in ATMOS_FILES.items():
        print(f"Carregando {name} ({len(point_flat):,} pontos)...", flush=True)
        with xr.open_dataset(data_dir / filename) as ds:
            full = _data_var(ds).values[:-1].astype("float32")
        series[name] = full[:, ii, jj].astype("float32")
        if name in LOCAL_VARS:
            local_mean[name], local_std[name] = _local_stats_points(full, ii, jj)
    flux_convergence = _sample_flux_convergence(data_dir, ii, jj, len(times))
    arrays = [target, clim_mean, clim_std, flux_convergence, *series.values()]
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("Há valores não finitos nas features históricas")
    return SampledBundle(
        times=times,
        lat=lat,
        lon=lon,
        point_flat=point_flat,
        series=series,
        local_mean=local_mean,
        local_std=local_std,
        flux_convergence=flux_convergence,
        target=target,
        target_clim_mean=clim_mean,
        target_clim_std=clim_std,
        target_clim_context=clim_context,
    )


def read_ocean_indices(indices_dir: str | Path | None = None) -> dict[str, pd.Series]:
    directory = Path(indices_dir) if indices_dir else ROOT / "baseline_169874" / "scripts" / "data" / "indices"
    output: dict[str, pd.Series] = {}
    for name, filename in OCEAN_FILES.items():
        text = (directory / filename).read_text(encoding="utf-8")
        if name == "soi":
            text = text.split("STANDARDIZED    DATA", 1)[1]
        values: dict[pd.Timestamp, float] = {}
        for line in text.splitlines():
            tokens = re.findall(r"-?\d+(?:\.\d+)?", line)
            if len(tokens) != 13 or not re.match(r"^\s*\d{4}", line):
                continue
            year = int(tokens[0])
            for month, raw in enumerate(tokens[1:], 1):
                value = float(raw)
                values[pd.Timestamp(year, month, 1)] = value if value > -90 else np.nan
        series = pd.Series(values, dtype="float32").sort_index()
        if series.empty:
            raise ValueError(f"Índice vazio: {filename}")
        output[name] = series
    return output


def _add_ocean_columns(
    columns: dict[str, np.ndarray], target_dates: pd.DatetimeIndex, npoints: int, series: Mapping[str, pd.Series]
) -> None:
    for name, values in series.items():
        for lag in (1, 2, 3):
            source = target_dates - pd.DateOffset(months=lag)
            if not (source < target_dates).all():
                raise AssertionError("Índice oceânico não está defasado")
            selected = values.reindex(source).to_numpy(dtype="float32")
            if not np.isfinite(selected).all():
                missing = source[~np.isfinite(selected)]
                raise ValueError(f"Índice {name} lag {lag} ausente em {list(missing)}")
            columns[f"{name}_lag{lag}"] = np.repeat(selected, npoints)


def build_table(
    bundle: SampledBundle,
    idx: np.ndarray,
    train_idx_for_stats: np.ndarray,
    mode: str,
    seed: int = 2026,
    ocean_series: Mapping[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """Monta tabela residual. ``mode`` é train ou validation."""
    if mode not in {"train", "validation"}:
        raise ValueError("mode deve ser train ou validation")
    idx = np.asarray(idx, dtype="int64")
    if np.any(idx < 2):
        raise ValueError("São necessários dois meses atmosféricos anteriores")
    target_dates = target_dates_from_origins(bundle.times[idx])
    target_months = target_dates.month.to_numpy(dtype="int16")
    origin_months_all = bundle.times.month.to_numpy(dtype="int16")
    npoints = len(bundle.point_flat)
    ii, jj = np.unravel_index(bundle.point_flat, (len(bundle.lat), len(bundle.lon)))
    lat_points, lon_points = bundle.lat[ii], bundle.lon[jj]
    climatology = bundle.target_clim_mean[target_months - 1]
    if mode == "train":
        rng = np.random.default_rng(seed)
        max_lead = np.minimum(24, idx + 1)
        lead = np.asarray([rng.integers(1, upper + 1) for upper in max_lead], dtype="int16")
        anchor_idx = idx - (lead - 1)
    else:
        lead = np.arange(1, len(idx) + 1, dtype="int16")
        anchor_idx = np.full(len(idx), idx[0], dtype="int64")
    if np.any(anchor_idx > idx):
        raise AssertionError("Âncora de precipitação no futuro")
    columns: dict[str, np.ndarray] = {
        "lat": np.tile(lat_points, len(idx)).astype("float32"),
        "lon": np.tile(lon_points, len(idx)).astype("float32"),
        "month_sin": np.repeat(np.sin(2 * np.pi * target_months / 12), npoints).astype("float32"),
        "month_cos": np.repeat(np.cos(2 * np.pi * target_months / 12), npoints).astype("float32"),
        "climatology": climatology.reshape(-1),
        "clim_std": bundle.target_clim_std[target_months - 1].reshape(-1),
        "lat2": np.tile((lat_points / 30.0) ** 2, len(idx)).astype("float32"),
        "lon2": np.tile((lon_points / 60.0) ** 2, len(idx)).astype("float32"),
        "latlon": np.tile((lat_points / 30.0) * (lon_points / 60.0), len(idx)).astype("float32"),
        "lag_meses": np.repeat(lead, npoints).astype("float32"),
        "tp_anchor": bundle.series["tp_anchor"][anchor_idx].reshape(-1),
    }
    for name, value in bundle.target_clim_context.items():
        columns[name] = value[target_months - 1].reshape(-1)
    for name in ATMOS_FILES:
        values = bundle.series[name]
        predictor_mean, _ = _month_stats(values, bundle.times, train_idx_for_stats)
        columns[name] = values[idx].reshape(-1)
        columns[f"{name}_lag2"] = values[idx - 1].reshape(-1)
        columns[f"{name}_lag3"] = values[idx - 2].reshape(-1)
        columns[f"{name}_anom"] = (
            values[idx] - predictor_mean[origin_months_all[idx] - 1]
        ).reshape(-1)
        if name in LOCAL_VARS:
            columns[f"{name}_local_mean"] = bundle.local_mean[name][idx].reshape(-1)
            columns[f"{name}_local_std"] = bundle.local_std[name][idx].reshape(-1)
    columns["wind_speed_850"] = np.hypot(columns["u_850"], columns["v_850"]).astype("float32")
    columns["moisture_flux_u"] = (columns["u_850"] * columns["shum_850"]).astype("float32")
    columns["moisture_flux_v"] = (columns["v_850"] * columns["shum_850"]).astype("float32")
    columns["moisture_flux_convergence"] = bundle.flux_convergence[idx].reshape(-1)
    _add_ocean_columns(columns, target_dates, npoints, ocean_series or read_ocean_indices())
    columns["target"] = (bundle.target[idx] - climatology).reshape(-1)
    frame = pd.DataFrame(columns, dtype="float32")
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Tabela contém NaN ou infinito")
    return frame


def rmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    error = np.asarray(prediction, dtype="float64") - np.asarray(truth, dtype="float64")
    return float(np.sqrt(np.mean(error * error)))


def score_24_months(prediction: np.ndarray, truth: np.ndarray, npoints: int) -> dict[str, float]:
    if len(prediction) != 24 * npoints:
        raise ValueError("A avaliação causal requer exatamente 24 meses")
    pred = np.asarray(prediction).reshape(24, npoints)
    obs = np.asarray(truth).reshape(24, npoints)
    return {
        "rmse_24": rmse(pred, obs),
        "rmse_year1": rmse(pred[:12], obs[:12]),
        "rmse_year2": rmse(pred[12:], obs[12:]),
    }


def lead_bucket(lead: np.ndarray) -> np.ndarray:
    lead = np.asarray(lead)
    if np.any((lead < 1) | (lead > 24)):
        raise ValueError("Horizonte fora de 1..24")
    return ((lead - 1) // 6).astype("int8")


def fit_lead_calibration(
    climatology: np.ndarray,
    raw_prediction: np.ndarray,
    truth: np.ndarray,
    lead: np.ndarray,
    lower: float = 0.0,
    upper: float = 1.5,
) -> list[float]:
    """Escala a anomalia prevista em quatro faixas de seis meses."""
    base = np.asarray(climatology, dtype="float64")
    anomaly = np.asarray(raw_prediction, dtype="float64") - base
    target_anomaly = np.asarray(truth, dtype="float64") - base
    buckets = lead_bucket(lead)
    weights: list[float] = []
    for bucket in range(4):
        take = buckets == bucket
        denominator = float(np.dot(anomaly[take], anomaly[take]))
        value = 0.0 if denominator == 0 else float(np.dot(anomaly[take], target_anomaly[take]) / denominator)
        weights.append(float(np.clip(value, lower, upper)))
    return weights


def apply_lead_calibration(
    climatology: np.ndarray, raw_prediction: np.ndarray, lead: np.ndarray, weights: Sequence[float]
) -> np.ndarray:
    if len(weights) != 4:
        raise ValueError("Calibração deve conter quatro pesos")
    base = np.asarray(climatology, dtype="float64")
    raw = np.asarray(raw_prediction, dtype="float64")
    scale = np.asarray(weights, dtype="float64")[lead_bucket(lead)]
    return np.maximum(0.0, base + scale * (raw - base)).astype("float32")


def _registry_header() -> list[str]:
    return [
        "run_id", "created_utc", "code_sha256", "config", "spatial_sample", "seed", "fold",
        "features", "train_first", "train_last", "validation_first", "validation_last",
        "rmse_24", "rmse_year1", "rmse_year2", "climatology_rmse_24", "seconds", "artifacts",
    ]


def append_registry(row: Mapping[str, Any], path: Path = REGISTRY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    normalized = {key: row.get(key, "") for key in _registry_header()}
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_registry_header())
        if write_header:
            writer.writeheader()
        writer.writerow(normalized)


def validate_model(
    config: ModelConfig,
    folds: Sequence[Fold],
    data_dir: str | Path | None = None,
    spatial_sample: str = "a",
    n_points: int = 8000,
    validation_stride: int = 4,
    output_dir: str | Path = EXPERIMENTS_DIR,
) -> pd.DataFrame:
    """Executa folds causais para uma configuração."""
    return validate_suite(
        [config], folds, data_dir, spatial_sample, n_points, validation_stride, output_dir
    )


def validate_suite(
    configs: Sequence[ModelConfig],
    folds: Sequence[Fold],
    data_dir: str | Path | None = None,
    spatial_sample: str = "a",
    n_points: int = 8000,
    validation_stride: int = 4,
    output_dir: str | Path = EXPERIMENTS_DIR,
) -> pd.DataFrame:
    """Valida várias configurações reutilizando a preparação cara de cada fold."""
    _require_runtime("xarray", "lightgbm")
    if not configs:
        raise ValueError("Informe ao menos uma configuração")
    if len({config.seed for config in configs}) != 1:
        raise ValueError("A suíte reutiliza a amostra; todas as configurações devem usar a mesma seed")
    data_path = resolve_data_dir(data_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ocean = read_ocean_indices()
    times_all, lat, lon = _read_grid(data_path)
    times = times_all[:-1]
    rows: list[dict[str, Any]] = []
    for fold in folds:
        prep_started = time.perf_counter()
        train_idx, valid_idx = fold_indices(times, fold, train_years=30)
        eval_points = canonical_validation_points(len(lat), len(lon), validation_stride)
        train_bundle = load_sampled_bundle(data_path, train_idx, spatial_sample, n_points, seed=configs[0].seed)
        valid_bundle = load_sampled_bundle(
            data_path, train_idx, spatial_sample, n_points, explicit_points=eval_points, seed=configs[0].seed
        )
        train = build_table(train_bundle, train_idx, train_idx, "train", configs[0].seed, ocean)
        valid = build_table(valid_bundle, valid_idx, train_idx, "validation", configs[0].seed, ocean)
        features = [column for column in train.columns if column != "target"]
        if features != [column for column in valid.columns if column != "target"]:
            raise AssertionError("Ordem de features divergiu entre treino e validação")
        climatology = valid["climatology"].to_numpy(dtype="float32")
        truth = valid_bundle.target[valid_idx].reshape(-1).astype("float32")
        baseline_score = rmse(climatology, truth)
        lead = np.repeat(np.arange(1, 25, dtype="int16"), len(eval_points))
        prep_seconds = time.perf_counter() - prep_started
        for config in configs:
            fit_started = time.perf_counter()
            booster = lgb.train(
                config.lightgbm_params(),
                lgb.Dataset(train[features], label=train["target"], feature_name=features),
                num_boost_round=config.rounds,
            )
            raw = np.maximum(0.0, climatology + booster.predict(valid[features])).astype("float32")
            metrics = score_24_months(raw, truth, len(eval_points))
            prefix = output / f"{config.name}_{spatial_sample}_fold{fold.name}"
            booster.save_model(str(prefix) + ".txt")
            np.savez_compressed(
                str(prefix) + "_oof.npz", prediction=raw, truth=truth,
                climatology=climatology, lead=lead, point_flat=eval_points,
            )
            metadata = {
                "config": asdict(config), "fold": asdict(fold), "spatial_sample": spatial_sample,
                "features": features, "metrics": metrics, "climatology_rmse_24": baseline_score,
                "code_sha256": code_sha256(), "data_dir": str(data_path),
                "preparation_seconds_shared": prep_seconds,
                "temporal_contract": "target T uses atmosphere <= T-1; precipitation frozen before block",
            }
            Path(str(prefix) + ".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            row = {
                "run_id": prefix.name, "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "code_sha256": metadata["code_sha256"], "config": config.name,
                "spatial_sample": spatial_sample, "seed": config.seed, "fold": fold.name,
                "features": len(features),
                "train_first": str(target_dates_from_origins(times[train_idx])[0].date()),
                "train_last": str(target_dates_from_origins(times[train_idx])[-1].date()),
                "validation_first": fold.first_target, "validation_last": fold.last_target,
                **metrics, "climatology_rmse_24": baseline_score,
                "seconds": round(prep_seconds + time.perf_counter() - fit_started, 3),
                "artifacts": str(prefix),
            }
            append_registry(row)
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)
    return pd.DataFrame(rows)


def _load_oof(paths: Sequence[str | Path]) -> dict[str, np.ndarray]:
    pieces: dict[str, list[np.ndarray]] = {key: [] for key in ("prediction", "truth", "climatology", "lead")}
    for path in paths:
        with np.load(path) as loaded:
            for key in pieces:
                pieces[key].append(loaded[key])
    return {key: np.concatenate(values) for key, values in pieces.items()}


def evaluate_spatial_average(
    component_a: Sequence[str | Path], component_b: Sequence[str | Path]
) -> dict[str, Any]:
    """Exige que a média a/b melhore o melhor componente em todos os folds."""
    if len(component_a) != len(component_b) or not component_a:
        raise ValueError("Componentes devem ter a mesma quantidade de folds")
    folds = []
    for path_a, path_b in zip(component_a, component_b):
        a, b = _load_oof([path_a]), _load_oof([path_b])
        for key in ("truth", "climatology", "lead"):
            if not np.allclose(a[key], b[key]):
                raise ValueError(f"OOFs desalinhados em {key}: {path_a} vs {path_b}")
        npoints = len(a["truth"]) // 24
        score_a = score_24_months(a["prediction"], a["truth"], npoints)
        score_b = score_24_months(b["prediction"], b["truth"], npoints)
        average = 0.5 * (a["prediction"] + b["prediction"])
        score_average = score_24_months(average, a["truth"], npoints)
        improved = score_average["rmse_year2"] < min(score_a["rmse_year2"], score_b["rmse_year2"])
        folds.append({
            "fold": Path(path_a).stem, "component_a": score_a, "component_b": score_b,
            "average": score_average, "year2_improved": improved,
        })
    return {"folds": folds, "accept_average": all(row["year2_improved"] for row in folds)}


def optimize_two_model_blend(
    component_a: Sequence[str | Path],
    component_b: Sequence[str | Path],
    step: float = 0.05,
    minimum_gain: float = 0.01,
) -> dict[str, Any]:
    """Escolhe peso por RMSE OOF do segundo ano e aplica gates por fold."""
    if len(component_a) != len(component_b) or not component_a:
        raise ValueError("Componentes devem ter a mesma quantidade de folds")
    if step <= 0 or step > 1 or not np.isclose(round(1 / step) * step, 1):
        raise ValueError("step deve dividir o intervalo 0..1")
    loaded = []
    for path_a, path_b in zip(component_a, component_b):
        a, b = _load_oof([path_a]), _load_oof([path_b])
        if not np.allclose(a["truth"], b["truth"]):
            raise ValueError("Truths OOF não estão alinhados")
        loaded.append((Path(path_a).stem, a, b))
    candidates = np.arange(0.0, 1.0 + step / 2, step)
    scores = []
    for weight_a in candidates:
        squared_error = 0.0
        count = 0
        for _, a, b in loaded:
            npoints = len(a["truth"]) // 24
            prediction = weight_a * a["prediction"] + (1 - weight_a) * b["prediction"]
            error = prediction.reshape(24, npoints)[12:] - a["truth"].reshape(24, npoints)[12:]
            squared_error += float(np.sum(error.astype("float64") ** 2))
            count += error.size
        scores.append(math.sqrt(squared_error / count))
    best_index = int(np.argmin(scores))
    best_weight = float(candidates[best_index])
    fold_rows = []
    for name, a, b in loaded:
        npoints = len(a["truth"]) // 24
        prediction = best_weight * a["prediction"] + (1 - best_weight) * b["prediction"]
        blend_metrics = score_24_months(prediction, a["truth"], npoints)
        a_metrics = score_24_months(a["prediction"], a["truth"], npoints)
        b_metrics = score_24_months(b["prediction"], b["truth"], npoints)
        best_component = min(a_metrics["rmse_year2"], b_metrics["rmse_year2"])
        fold_rows.append({
            "fold": name, "blend": blend_metrics, "component_a": a_metrics, "component_b": b_metrics,
            "year2_gain_vs_best": best_component - blend_metrics["rmse_year2"],
        })
    aggregate_best = min(
        np.mean([row["component_a"]["rmse_year2"] for row in fold_rows]),
        np.mean([row["component_b"]["rmse_year2"] for row in fold_rows]),
    )
    aggregate_blend = float(np.mean([row["blend"]["rmse_year2"] for row in fold_rows]))
    no_regression = all(row["year2_gain_vs_best"] >= -0.02 for row in fold_rows)
    return {
        "weight_a": best_weight, "weight_b": 1.0 - best_weight,
        "aggregate_year2_rmse": aggregate_blend,
        "aggregate_gain_vs_best_component": aggregate_best - aggregate_blend,
        "folds": fold_rows,
        "accept_blend": aggregate_best - aggregate_blend >= minimum_gain and no_regression,
    }


def cross_validated_calibration(oof_paths: Sequence[str | Path]) -> dict[str, Any]:
    """Calibra cada fold nos demais e reporta a estimativa sem reutilizar o alvo."""
    paths = [Path(path) for path in oof_paths]
    fold_scores = []
    for held_out in paths:
        train = _load_oof([path for path in paths if path != held_out])
        weights = fit_lead_calibration(**{
            "climatology": train["climatology"], "raw_prediction": train["prediction"],
            "truth": train["truth"], "lead": train["lead"],
        })
        val = _load_oof([held_out])
        calibrated = apply_lead_calibration(val["climatology"], val["prediction"], val["lead"], weights)
        npoints = len(val["prediction"]) // 24
        fold_scores.append({"fold": held_out.stem, "weights": weights, **score_24_months(calibrated, val["truth"], npoints)})
    all_oof = _load_oof(paths)
    final_weights = fit_lead_calibration(
        all_oof["climatology"], all_oof["prediction"], all_oof["truth"], all_oof["lead"]
    )
    return {"final_weights": final_weights, "leave_one_fold_out": fold_scores}


def _final_train_indices(times: pd.DatetimeIndex) -> np.ndarray:
    targets = target_dates_from_origins(times)
    idx = np.flatnonzero((targets >= pd.Timestamp("1993-01-01")) & (targets < pd.Timestamp("2023-01-01")))
    if len(idx) != 360 or targets[idx][-1] != pd.Timestamp("2022-12-01"):
        raise AssertionError("Janela final deve conter exatamente 1993-01 a 2022-12")
    return idx


def fit_final(
    config: ModelConfig,
    data_dir: str | Path | None = None,
    spatial_sample: str = "a",
    n_points: int = 8000,
    output_dir: str | Path = ARTIFACTS_DIR,
    calibration: Sequence[float] | None = None,
) -> Path:
    """Treina em 1993–2022 e salva modelo + contrato completo de reprodução."""
    _require_runtime("xarray", "lightgbm")
    data_path = resolve_data_dir(data_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    times = _read_grid(data_path)[0][:-1]
    train_idx = _final_train_indices(times)
    bundle = load_sampled_bundle(data_path, train_idx, spatial_sample, n_points, seed=config.seed)
    frame = build_table(bundle, train_idx, train_idx, "train", config.seed, read_ocean_indices())
    features = [column for column in frame if column != "target"]
    booster = lgb.train(
        config.lightgbm_params(),
        lgb.Dataset(frame[features], label=frame["target"], feature_name=features),
        num_boost_round=config.rounds,
    )
    stem = f"lgb_{config.name}_{spatial_sample}_final"
    model_path = output / f"{stem}.txt"
    booster.save_model(str(model_path))
    metadata = {
        "model_path": model_path.name,
        "config": asdict(config),
        "spatial_sample": spatial_sample,
        "n_spatial_points": n_points,
        "features": features,
        "train_targets": ["1993-01-01", "2022-12-01"],
        "calibration": list(calibration or [1.0, 1.0, 1.0, 1.0]),
        "seed": config.seed,
        "code_sha256": code_sha256(),
        "ocean_indices_sha256": {
            filename: sha256_file(ROOT / "baseline_169874" / "scripts" / "data" / "indices" / filename)
            for filename in OCEAN_FILES.values()
        },
        "environment": {"python": sys.version, "platform": platform.platform(), "lightgbm": lgb.__version__},
        "temporal_contract": "no precipitation after 2022-12; target T atmosphere <= T-1",
    }
    metadata_path = output / f"{stem}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Modelo final: {model_path}\nMetadata: {metadata_path}", flush=True)
    return metadata_path


def _full_month_stats_from_file(path: Path, idx: np.ndarray, dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    with xr.open_dataset(path) as ds:
        values = _data_var(ds).values[:-1].astype("float32")
    return _month_stats(values, dates, idx)


def _prepare_full_inference(data_dir: Path) -> dict[str, Any]:
    _require_runtime("xarray")
    times_all, lat, lon = _read_grid(data_dir)
    train_times = times_all[:-1]
    train_idx = _final_train_indices(train_times)
    train_targets = target_dates_from_origins(train_times)
    with xr.open_dataset(data_dir / "treino_tp_alvo.nc") as ds:
        target_full = _data_var(ds).values[:-1].astype("float32")
    target_mean, target_std = _month_stats(target_full, train_targets, train_idx)
    context = _clim_context_full(target_mean)
    predictor_mean: dict[str, np.ndarray] = {}
    history: dict[str, np.ndarray] = {}
    for name, filename in ATMOS_FILES.items():
        with xr.open_dataset(data_dir / filename) as ds:
            values = _data_var(ds).values.astype("float32")
            predictor_mean[name], _ = _month_stats(values[:-1], train_times, train_idx)
            history[name] = values[-3:-1].astype("float32")
    with xr.open_dataset(data_dir / "teste_features.nc") as ds:
        test = ds.load()
    dates = pd.DatetimeIndex(test.time.values)
    origins = pd.DatetimeIndex(test.time_origem.values)
    if len(dates) != 24 or not (origins == dates - pd.DateOffset(months=1)).all():
        raise AssertionError("Contrato time/time_origem do teste foi violado")
    if "tp_alvo" in test and not np.isnan(test["tp_alvo"].values).all():
        raise AssertionError("tp_alvo de teste deveria estar completamente oculto")
    frozen = test["tp_ultima_obs"].values.astype("float32")
    if not np.allclose(frozen, frozen[0:1], equal_nan=False):
        raise AssertionError("tp_ultima_obs deve permanecer congelada em dezembro/2022")
    if not np.array_equal(test.lat.values, lat) or not np.array_equal(test.lon.values, lon):
        raise AssertionError("Grade de treino e teste divergiu")
    test_values = {name: test[name].values.astype("float32") for name in ATMOS_FILES}
    flux = -(np.gradient(test_values["u_850"] * test_values["shum_850"], axis=2)
             + np.gradient(test_values["v_850"] * test_values["shum_850"], axis=1)).astype("float32")
    return {
        "lat": lat, "lon": lon, "dates": dates, "origins": origins, "test": test_values,
        "history": history, "predictor_mean": predictor_mean, "target_mean": target_mean,
        "target_std": target_std, "context": context, "frozen_tp": frozen,
        "flux_convergence": flux,
    }


def _test_month_frame(prepared: Mapping[str, Any], month_index: int, ocean: Mapping[str, pd.Series]) -> pd.DataFrame:
    lat, lon = prepared["lat"], prepared["lon"]
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    date = prepared["dates"][month_index]
    origin = prepared["origins"][month_index]
    npoints = la.size
    target_month = date.month
    lead = month_index + 1
    columns: dict[str, np.ndarray] = {
        "lat": la.ravel().astype("float32"), "lon": lo.ravel().astype("float32"),
        "month_sin": np.full(npoints, np.sin(2 * np.pi * target_month / 12), dtype="float32"),
        "month_cos": np.full(npoints, np.cos(2 * np.pi * target_month / 12), dtype="float32"),
        "climatology": prepared["target_mean"][target_month - 1].ravel(),
        "clim_std": prepared["target_std"][target_month - 1].ravel(),
        "lat2": ((la.ravel() / 30.0) ** 2).astype("float32"),
        "lon2": ((lo.ravel() / 60.0) ** 2).astype("float32"),
        "latlon": ((la.ravel() / 30.0) * (lo.ravel() / 60.0)).astype("float32"),
        "lag_meses": np.full(npoints, lead, dtype="float32"),
        "tp_anchor": prepared["frozen_tp"][month_index].ravel(),
    }
    for name, values in prepared["context"].items():
        columns[name] = values[target_month - 1].ravel()
    for name in ATMOS_FILES:
        available = np.concatenate([prepared["history"][name], prepared["test"][name][: month_index + 1]], axis=0)
        current, lag2, lag3 = available[-1], available[-2], available[-3]
        if origin.month != ((date.month - 2) % 12) + 1:
            raise AssertionError("Mês de origem atmosférica incorreto")
        columns[name] = current.ravel()
        columns[f"{name}_lag2"] = lag2.ravel()
        columns[f"{name}_lag3"] = lag3.ravel()
        columns[f"{name}_anom"] = (current - prepared["predictor_mean"][name][origin.month - 1]).ravel()
        if name in LOCAL_VARS:
            local_mean, local_std = _local_stats_full(current[None])
            columns[f"{name}_local_mean"] = local_mean[0].ravel()
            columns[f"{name}_local_std"] = local_std[0].ravel()
    columns["wind_speed_850"] = np.hypot(columns["u_850"], columns["v_850"]).astype("float32")
    columns["moisture_flux_u"] = (columns["u_850"] * columns["shum_850"]).astype("float32")
    columns["moisture_flux_v"] = (columns["v_850"] * columns["shum_850"]).astype("float32")
    columns["moisture_flux_convergence"] = prepared["flux_convergence"][month_index].ravel()
    _add_ocean_columns(columns, pd.DatetimeIndex([date]), npoints, ocean)
    frame = pd.DataFrame(columns, dtype="float32")
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f"Features não finitas em {date:%Y-%m}")
    return frame


def _load_model(metadata_path: str | Path) -> tuple[Any, dict[str, Any]]:
    _require_runtime("lightgbm")
    metadata_path = Path(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_path = metadata_path.parent / metadata["model_path"]
    return lgb.Booster(model_file=str(model_path)), metadata


def _assert_sample_grid_order(
    sample: pd.DataFrame, dates: pd.DatetimeIndex, lat: np.ndarray, lon: np.ndarray
) -> None:
    """Confirma a ordem sem reconstruir os IDs usados no arquivo de saída."""
    block = len(lat) * len(lon)
    if len(sample) != len(dates) * block:
        raise AssertionError("Tamanho do sample incompatível com tempo × grade")
    actual = sample["id"].to_numpy()
    for month_index, date in enumerate(dates):
        expected = np.asarray(
            [f"{date.year}_{date.month:02d}_{latitude:.2f}_{longitude:.2f}" for latitude in lat for longitude in lon],
            dtype=object,
        )
        start = month_index * block
        if not np.array_equal(actual[start : start + block], expected):
            raise AssertionError(f"Ordem do sample diverge da grade em {date:%Y-%m}")


def predict_submission(
    model: str | Path | Sequence[str | Path],
    data_dir: str | Path | None = None,
    output_path: str | Path = ARTIFACTS_DIR / "submission_candidate.csv",
    weights: Sequence[float] | None = None,
) -> Path:
    """Gera CSV oficial a partir de um ou mais metadados de modelo."""
    model_paths = [model] if isinstance(model, (str, Path)) else list(model)
    loaded = [_load_model(path) for path in model_paths]
    if weights is None:
        weights = [1.0 / len(loaded)] * len(loaded)
    weights_array = np.asarray(weights, dtype="float64")
    if len(weights_array) != len(loaded) or np.any(weights_array < 0) or not np.isclose(weights_array.sum(), 1):
        raise ValueError("Pesos devem ser não negativos, somar 1 e corresponder aos modelos")
    data_path = resolve_data_dir(data_dir)
    prepared = _prepare_full_inference(data_path)
    ocean = read_ocean_indices()
    predictions: list[np.ndarray] = []
    for month_index in range(24):
        frame = _test_month_frame(prepared, month_index, ocean)
        climatology = frame["climatology"].to_numpy(dtype="float32")
        month_models = []
        lead = np.full(len(frame), month_index + 1, dtype="int16")
        for booster, metadata in loaded:
            features = metadata["features"]
            if features != booster.feature_name():
                raise AssertionError("Features do metadata não correspondem ao modelo")
            raw = np.maximum(0.0, climatology + booster.predict(frame[features])).astype("float32")
            calibrated = apply_lead_calibration(climatology, raw, lead, metadata["calibration"])
            month_models.append(calibrated)
        blended = np.average(np.stack(month_models), axis=0, weights=weights_array).astype("float32")
        predictions.append(np.maximum(blended, 0.0))
        print(f"Inferência concluída: {prepared['dates'][month_index]:%Y-%m}", flush=True)
    sample = pd.read_csv(data_path / "sample_submission.csv")
    _assert_sample_grid_order(sample, prepared["dates"], prepared["lat"], prepared["lon"])
    values = np.concatenate(predictions)
    if len(values) != len(sample):
        raise AssertionError("Número de previsões não corresponde ao sample")
    result = sample[["id"]].copy()
    result["tp_mm_day"] = values
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    audit_submission(output, data_path / "sample_submission.csv")
    metadata_out = {
        "submission": output.name,
        "submission_sha256": sha256_file(output),
        "models": [str(Path(path)) for path in model_paths],
        "weights": weights_array.tolist(),
        "code_sha256": code_sha256(),
        "rows": len(result),
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata_out, indent=2), encoding="utf-8")
    return output


def audit_submission(submission_path: str | Path, sample_path: str | Path) -> dict[str, Any]:
    submission = pd.read_csv(submission_path)
    sample = pd.read_csv(sample_path)
    expected_columns = ["id", "tp_mm_day"]
    if list(submission.columns) != expected_columns:
        raise ValueError(f"Colunas esperadas: {expected_columns}")
    if len(submission) != 1_885_464 or len(submission) != len(sample):
        raise ValueError(f"Quantidade incorreta de linhas: {len(submission):,}")
    if submission["id"].duplicated().any() or not submission["id"].equals(sample["id"]):
        raise ValueError("IDs duplicados ou fora da ordem oficial")
    values = submission["tp_mm_day"].to_numpy(dtype="float64")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Previsões devem ser finitas e não negativas")
    report = {
        "rows": len(submission), "unique_ids": int(submission.id.nunique()),
        "min": float(values.min()), "max": float(values.max()), "mean": float(values.mean()),
        "sha256": sha256_file(submission_path),
    }
    print(json.dumps(report, indent=2), flush=True)
    return report


def blend_submissions(
    inputs: Sequence[str | Path],
    weights: Sequence[float],
    output_path: str | Path,
    sample_path: str | Path,
) -> Path:
    weights_array = np.asarray(weights, dtype="float64")
    if len(inputs) != len(weights_array) or np.any(weights_array < 0) or not np.isclose(weights_array.sum(), 1):
        raise ValueError("Pesos inválidos")
    frames = [pd.read_csv(path) for path in inputs]
    ids = frames[0]["id"]
    if any(not frame["id"].equals(ids) for frame in frames[1:]):
        raise ValueError("IDs dos componentes não são idênticos e ordenados")
    matrix = np.stack([frame["tp_mm_day"].to_numpy(dtype="float64") for frame in frames])
    result = pd.DataFrame({"id": ids, "tp_mm_day": np.maximum(0.0, weights_array @ matrix)})
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    audit_submission(output, sample_path)
    return output


def acceptance_report(candidate_rows: pd.DataFrame, reference_rows: pd.DataFrame) -> dict[str, Any]:
    """Aplica os guardrails definidos antes de olhar o leaderboard público."""
    candidate = candidate_rows.set_index("fold")
    reference = reference_rows.set_index("fold")
    common = sorted(set(candidate.index) & set(reference.index))
    if len(common) != 3:
        raise ValueError("São necessários os três folds para promoção")
    year2_gain = reference.loc[common, "rmse_year2"] - candidate.loc[common, "rmse_year2"]
    full_gain = reference.loc[common, "rmse_24"] - candidate.loc[common, "rmse_24"]
    report = {
        "folds": common,
        "year2_aggregate_improved": float(candidate.loc[common, "rmse_year2"].mean())
        < float(reference.loc[common, "rmse_year2"].mean()),
        "full24_maintained": float(candidate.loc[common, "rmse_24"].mean())
        <= float(reference.loc[common, "rmse_24"].mean()),
        "no_year2_regression_gt_0_02": bool((year2_gain >= -0.02).all()),
        "improved_multiple_folds": int((year2_gain > 0).sum()) >= 2,
        "year2_gain_by_fold": year2_gain.to_dict(),
        "full24_gain_by_fold": full_gain.to_dict(),
    }
    report["promote"] = all(
        report[key]
        for key in ("year2_aggregate_improved", "full24_maintained", "no_year2_regression_gt_0_02", "improved_multiple_folds")
    )
    return report


def _parse_folds(names: Sequence[str]) -> list[Fold]:
    return [FOLDS[name] for name in names]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", help="Pasta contendo os 13 arquivos da competição")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="Executa backtests causais")
    validate.add_argument("--config", choices=LGB_CONFIGS, required=True)
    validate.add_argument("--folds", nargs="+", choices=FOLDS, default=list(FOLDS))
    validate.add_argument("--spatial-sample", choices=("a", "b"), default="a")
    validate.add_argument("--n-points", type=int, default=8000)
    validate.add_argument("--validation-stride", type=int, default=4)

    suite = sub.add_parser("validate-suite", help="Valida as três configurações reutilizando as features")
    suite.add_argument("--configs", nargs="+", choices=LGB_CONFIGS, default=list(LGB_CONFIGS))
    suite.add_argument("--folds", nargs="+", choices=FOLDS, default=list(FOLDS))
    suite.add_argument("--spatial-sample", choices=("a", "b"), default="a")
    suite.add_argument("--n-points", type=int, default=8000)
    suite.add_argument("--validation-stride", type=int, default=4)

    calibrate = sub.add_parser("calibrate", help="Calibra anomalias por horizonte com OOF")
    calibrate.add_argument("oof", nargs="+")
    calibrate.add_argument("--output", required=True)

    fit = sub.add_parser("fit-final", help="Treina artefato final em 1993–2022")
    fit.add_argument("--config", choices=LGB_CONFIGS, required=True)
    fit.add_argument("--spatial-sample", choices=("a", "b"), required=True)
    fit.add_argument("--n-points", type=int, default=8000)
    fit.add_argument("--calibration-json")

    predict = sub.add_parser("predict", help="Gera submissão sem enviar")
    predict.add_argument("--model", action="append", required=True, help="Metadata JSON; pode repetir")
    predict.add_argument("--weights", nargs="+", type=float)
    predict.add_argument("--output", required=True)

    blend = sub.add_parser("blend", help="Combina CSVs já aprovados em OOF")
    blend.add_argument("--input", action="append", required=True)
    blend.add_argument("--weights", nargs="+", type=float, required=True)
    blend.add_argument("--sample", required=True)
    blend.add_argument("--output", required=True)

    evaluate_blend = sub.add_parser("evaluate-blend", help="Seleciona e aplica gates a dois componentes OOF")
    evaluate_blend.add_argument("--component-a", nargs="+", required=True)
    evaluate_blend.add_argument("--component-b", nargs="+", required=True)
    evaluate_blend.add_argument("--step", type=float, default=0.05)
    evaluate_blend.add_argument("--minimum-gain", type=float, default=0.01)
    evaluate_blend.add_argument("--spatial-average", action="store_true")
    evaluate_blend.add_argument("--output", required=True)

    audit = sub.add_parser("audit-submission", help="Valida contrato do CSV")
    audit.add_argument("submission")
    audit.add_argument("--sample")

    baseline = sub.add_parser("audit-baseline", help="Confirma hash da entrega 1,69874")
    baseline.add_argument("csv")

    args = parser.parse_args(argv)
    if args.command == "validate":
        validate_model(
            LGB_CONFIGS[args.config], _parse_folds(args.folds), args.data_dir,
            args.spatial_sample, args.n_points, args.validation_stride,
        )
    elif args.command == "validate-suite":
        validate_suite(
            [LGB_CONFIGS[name] for name in args.configs], _parse_folds(args.folds), args.data_dir,
            args.spatial_sample, args.n_points, args.validation_stride,
        )
    elif args.command == "calibrate":
        report = cross_validated_calibration(args.oof)
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    elif args.command == "fit-final":
        calibration = None
        if args.calibration_json:
            calibration = json.loads(Path(args.calibration_json).read_text(encoding="utf-8"))["final_weights"]
        fit_final(LGB_CONFIGS[args.config], args.data_dir, args.spatial_sample, args.n_points, calibration=calibration)
    elif args.command == "predict":
        predict_submission(args.model, args.data_dir, args.output, args.weights)
    elif args.command == "blend":
        blend_submissions(args.input, args.weights, args.output, args.sample)
    elif args.command == "evaluate-blend":
        if args.spatial_average:
            report = evaluate_spatial_average(args.component_a, args.component_b)
        else:
            report = optimize_two_model_blend(
                args.component_a, args.component_b, args.step, args.minimum_gain
            )
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    elif args.command == "audit-submission":
        sample = args.sample or resolve_data_dir(args.data_dir) / "sample_submission.csv"
        audit_submission(args.submission, sample)
    elif args.command == "audit-baseline":
        actual = sha256_file(args.csv)
        print(json.dumps({"expected": BASELINE_SHA256, "actual": actual, "matches": actual == BASELINE_SHA256}, indent=2))
        if actual != BASELINE_SHA256:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
