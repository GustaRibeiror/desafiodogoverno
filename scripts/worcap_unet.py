#!/usr/bin/env python3
"""U-Net espacial experimental, causal e limitada para o desafio WORCAP.

Este trilho só deve chegar à submissão se passar os gates OOF descritos em
``docs/ENTREGA_MODELO.md``. O script não envia nada ao Kaggle.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
from torch.utils.data import DataLoader, Dataset

from worcap_pipeline import (
    ARTIFACTS_DIR,
    ATMOS_FILES,
    EXPERIMENTS_DIR,
    FOLDS,
    Fold,
    _add_ocean_columns,
    _data_var,
    _final_train_indices,
    _month_stats,
    _prepare_full_inference,
    _read_grid,
    append_registry,
    audit_submission,
    code_sha256,
    fold_indices,
    read_ocean_indices,
    resolve_data_dir,
    score_24_months,
    sha256_file,
    target_dates_from_origins,
)


PAD_BOTTOM = 3
PAD_RIGHT = 3


def _channels() -> list[str]:
    names: list[str] = []
    for variable in ATMOS_FILES:
        names.extend([variable, f"{variable}_lag2", f"{variable}_lag3", f"{variable}_anom"])
    names.extend(["climatology", "clim_std", "tp_anchor", "latitude", "longitude", "month_sin", "month_cos", "lead"])
    for ocean in ("soi", "nino12", "nino3", "nino4"):
        names.extend([f"{ocean}_lag1", f"{ocean}_lag2", f"{ocean}_lag3"])
    return names


CHANNEL_NAMES = _channels()


class DoubleConv(nn.Module):
    def __init__(self, inputs: int, outputs: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, outputs), outputs),
            nn.SiLU(inplace=True),
            nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, outputs), outputs),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class SmallUNet(nn.Module):
    """U-Net base 16, três níveis, saída residual normalizada."""

    def __init__(self, input_channels: int, base: int = 16) -> None:
        super().__init__()
        self.enc1 = DoubleConv(input_channels, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.bridge = DoubleConv(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)
        self.output = nn.Conv2d(base, 1, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        original_h, original_w = inputs.shape[-2:]
        x = F.pad(inputs, (0, PAD_RIGHT, 0, PAD_BOTTOM), mode="replicate")
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        bridge = self.bridge(F.max_pool2d(e3, 2))
        d3 = self.dec3(torch.cat([self.up3(bridge), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.output(d1)[..., :original_h, :original_w]


def _global_mean_std(values: np.ndarray, idx: np.ndarray) -> tuple[float, float]:
    selected = values[idx].astype("float64", copy=False)
    mean = float(np.mean(selected))
    std = max(float(np.std(selected)), 1e-6)
    return mean, std


def load_full_maps(data_dir: Path, train_idx: np.ndarray) -> dict[str, Any]:
    times_all, lat, lon = _read_grid(data_dir)
    times = times_all[:-1]
    target_dates = target_dates_from_origins(times)
    arrays: dict[str, np.ndarray] = {}
    scalar_stats: dict[str, list[float]] = {}
    predictor_mean: dict[str, np.ndarray] = {}
    for name, filename in ATMOS_FILES.items():
        print(f"U-Net: carregando {name}...", flush=True)
        with xr.open_dataset(data_dir / filename) as ds:
            values = _data_var(ds).values[:-1].astype("float32")
        arrays[name] = values
        scalar_stats[name] = list(_global_mean_std(values, train_idx))
        predictor_mean[name], _ = _month_stats(values, times, train_idx)
    with xr.open_dataset(data_dir / "treino_tp.nc") as ds:
        tp = _data_var(ds).values[:-1].astype("float32")
    with xr.open_dataset(data_dir / "treino_tp_alvo.nc") as ds:
        target = _data_var(ds).values[:-1].astype("float32")
    target_mean, target_std = _month_stats(target, target_dates, train_idx)
    rain_mean, rain_scale = _global_mean_std(target, train_idx)
    return {
        "times": times, "lat": lat, "lon": lon, "arrays": arrays, "tp": tp, "target": target,
        "predictor_mean": predictor_mean, "target_mean": target_mean, "target_std": target_std,
        "scalar_stats": scalar_stats, "rain_mean": rain_mean, "rain_scale": rain_scale,
    }


class ClimateMapDataset(Dataset):
    def __init__(
        self,
        maps: Mapping[str, Any],
        indices: np.ndarray,
        mode: str,
        seed: int = 2026,
        ocean: Mapping[str, pd.Series] | None = None,
    ) -> None:
        self.maps = maps
        self.indices = np.asarray(indices, dtype="int64")
        self.mode = mode
        self.seed = seed
        self.ocean = ocean or read_ocean_indices()
        if mode not in {"train", "validation"}:
            raise ValueError("mode inválido")
        if np.any(self.indices < 2):
            raise ValueError("Histórico atmosférico insuficiente")
        lat, lon = maps["lat"], maps["lon"]
        la, lo = np.meshgrid(lat, lon, indexing="ij")
        self.latitude = (la / max(abs(float(lat.min())), abs(float(lat.max())))).astype("float32")
        self.longitude = ((lo - float(lon.mean())) / max(float(np.ptp(lon)) / 2, 1)).astype("float32")
        rng = np.random.default_rng(seed)
        if mode == "train":
            self.leads = np.asarray([rng.integers(1, min(24, int(idx) + 1) + 1) for idx in self.indices], dtype="int16")
            self.anchors = self.indices - (self.leads - 1)
        else:
            self.leads = np.arange(1, len(self.indices) + 1, dtype="int16")
            self.anchors = np.full(len(self.indices), self.indices[0], dtype="int64")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        idx = int(self.indices[item])
        lead = int(self.leads[item])
        date = self.maps["times"][idx] + pd.DateOffset(months=1)
        origin_month = int(self.maps["times"][idx].month)
        target_month = int(date.month)
        channels: list[np.ndarray] = []
        for name in ATMOS_FILES:
            mean, std = self.maps["scalar_stats"][name]
            values = self.maps["arrays"][name]
            channels.extend([(values[idx] - mean) / std, (values[idx - 1] - mean) / std, (values[idx - 2] - mean) / std])
            channels.append((values[idx] - self.maps["predictor_mean"][name][origin_month - 1]) / std)
        rain_mean, rain_scale = self.maps["rain_mean"], self.maps["rain_scale"]
        climatology = self.maps["target_mean"][target_month - 1]
        channels.extend([
            (climatology - rain_mean) / rain_scale,
            self.maps["target_std"][target_month - 1] / rain_scale,
            (self.maps["tp"][self.anchors[item]] - rain_mean) / rain_scale,
            self.latitude,
            self.longitude,
            np.full_like(climatology, np.sin(2 * np.pi * target_month / 12)),
            np.full_like(climatology, np.cos(2 * np.pi * target_month / 12)),
            np.full_like(climatology, lead / 24.0),
        ])
        scalar_columns: dict[str, np.ndarray] = {}
        _add_ocean_columns(scalar_columns, pd.DatetimeIndex([date]), 1, self.ocean)
        for ocean in ("soi", "nino12", "nino3", "nino4"):
            for lag in (1, 2, 3):
                channels.append(np.full_like(climatology, scalar_columns[f"{ocean}_lag{lag}"][0] / 3.0))
        inputs = np.stack(channels).astype("float32")
        if inputs.shape[0] != len(CHANNEL_NAMES):
            raise AssertionError("Quantidade de canais divergente")
        residual = ((self.maps["target"][idx] - climatology) / rain_scale).astype("float32")
        return {
            "inputs": torch.from_numpy(inputs),
            "target": torch.from_numpy(residual[None]),
            "climatology": torch.from_numpy(climatology[None].astype("float32")),
            "truth": torch.from_numpy(self.maps["target"][idx][None].astype("float32")),
            "lead": torch.tensor(lead, dtype=torch.int16),
        }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, rain_scale: float) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    predictions, truths, climatologies = [], [], []
    for batch in loader:
        inputs = batch["inputs"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            residual = model(inputs).float().cpu().numpy() * rain_scale
        climatology = batch["climatology"].numpy()
        prediction = np.maximum(0.0, climatology + residual)
        predictions.append(prediction.astype("float32"))
        truths.append(batch["truth"].numpy().astype("float32"))
        climatologies.append(climatology.astype("float32"))
    pred = np.concatenate(predictions).reshape(-1)
    truth = np.concatenate(truths).reshape(-1)
    clim = np.concatenate(climatologies).reshape(-1)
    score = float(np.sqrt(np.mean((pred.astype("float64") - truth) ** 2)))
    return score, pred, truth, clim


def train_unet(
    maps: Mapping[str, Any],
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    output_path: Path,
    max_epochs: int = 30,
    patience: int = 5,
    batch_size: int = 2,
    seed: int = 2026,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = ClimateMapDataset(maps, train_idx, "train", seed)
    valid_data = ClimateMapDataset(maps, valid_idx, "validation", seed)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    model = SmallUNet(len(CHANNEL_NAMES), base=16).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_rmse = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            inputs = batch["inputs"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                prediction = model(inputs)
                loss = F.mse_loss(prediction, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        valid_rmse, pred, truth, clim = evaluate(model, valid_loader, device, maps["rain_scale"])
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "validation_rmse": valid_rmse})
        print(json.dumps(history[-1]), flush=True)
        if valid_rmse < best_rmse - 1e-5:
            best_rmse, best_epoch, stale = valid_rmse, epoch, 0
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "epoch": epoch}, output_path)
            np.savez_compressed(output_path.with_suffix(".oof.npz"), prediction=pred, truth=truth, climatology=clim,
                                lead=np.repeat(np.arange(1, 25), pred.size // 24))
        else:
            stale += 1
            if stale >= patience:
                break
    return {"best_rmse": best_rmse, "best_epoch": best_epoch, "history": history, "device": str(device)}


def validate_unet_fold(
    fold: Fold,
    data_dir: str | Path | None = None,
    max_epochs: int = 30,
    patience: int = 5,
) -> dict[str, Any]:
    data_path = resolve_data_dir(data_dir)
    times = _read_grid(data_path)[0][:-1]
    train_idx, valid_idx = fold_indices(times, fold, 30)
    maps = load_full_maps(data_path, train_idx)
    started = time.perf_counter()
    prefix = EXPERIMENTS_DIR / f"unet_fold{fold.name}"
    result = train_unet(maps, train_idx, valid_idx, prefix.with_suffix(".pt"), max_epochs, patience)
    with np.load(prefix.with_suffix(".oof.npz")) as oof:
        metrics = score_24_months(oof["prediction"], oof["truth"], len(maps["lat"]) * len(maps["lon"]))
    metadata = {
        "fold": asdict(fold), "channels": CHANNEL_NAMES, "base_channels": 16, "levels": 3,
        "batch_size": 2, "optimizer": "AdamW", "max_epochs": max_epochs, "patience": patience,
        "seed": 2026, "result": result, "metrics": metrics,
        "pipeline_code_sha256": code_sha256(), "unet_code_sha256": sha256_file(Path(__file__)),
    }
    prefix.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    append_registry({
        "run_id": prefix.name, "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "code_sha256": metadata["unet_code_sha256"], "config": "unet_base16", "spatial_sample": "full",
        "seed": 2026, "fold": fold.name, "features": len(CHANNEL_NAMES),
        "train_first": str(target_dates_from_origins(times[train_idx])[0].date()),
        "train_last": str(target_dates_from_origins(times[train_idx])[-1].date()),
        "validation_first": fold.first_target, "validation_last": fold.last_target,
        **metrics, "seconds": round(time.perf_counter() - started, 3), "artifacts": str(prefix),
    })
    print(json.dumps(metadata, indent=2), flush=True)
    return metadata


def fit_final_unet(
    epochs: int,
    data_dir: str | Path | None = None,
    output_path: str | Path = ARTIFACTS_DIR / "unet_final.pt",
) -> Path:
    """Ajusta o modelo final pelo número mediano de épocas aprovado nos folds."""
    data_path = resolve_data_dir(data_dir)
    times = _read_grid(data_path)[0][:-1]
    train_idx = _final_train_indices(times)
    maps = load_full_maps(data_path, train_idx)
    torch.manual_seed(2026)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2026)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ClimateMapDataset(maps, train_idx, "train", 2026)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    model = SmallUNet(len(CHANNEL_NAMES), 16).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in loader:
            inputs, target = batch["inputs"].to(device), batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = F.mse_loss(model(inputs), target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        print(f"U-Net final época {epoch}/{epochs}", flush=True)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "channels": CHANNEL_NAMES, "base_channels": 16, "levels": 3, "epochs": epochs,
        "seed": 2026, "train_targets": ["1993-01-01", "2022-12-01"],
        "scalar_stats": maps["scalar_stats"], "rain_mean": maps["rain_mean"], "rain_scale": maps["rain_scale"],
        "pipeline_code_sha256": code_sha256(), "unet_code_sha256": sha256_file(Path(__file__)),
        "model_path": output.name,
    }
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, output)
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output


def _unet_test_tensor(prepared: Mapping[str, Any], month_index: int, metadata: Mapping[str, Any]) -> torch.Tensor:
    date, origin = prepared["dates"][month_index], prepared["origins"][month_index]
    channels: list[np.ndarray] = []
    for name in ATMOS_FILES:
        mean, std = metadata["scalar_stats"][name]
        available = np.concatenate([prepared["history"][name], prepared["test"][name][: month_index + 1]], axis=0)
        current, lag2, lag3 = available[-1], available[-2], available[-3]
        channels.extend([(current - mean) / std, (lag2 - mean) / std, (lag3 - mean) / std,
                         (current - prepared["predictor_mean"][name][origin.month - 1]) / std])
    rain_mean, scale = metadata["rain_mean"], metadata["rain_scale"]
    climatology = prepared["target_mean"][date.month - 1]
    la, lo = np.meshgrid(prepared["lat"], prepared["lon"], indexing="ij")
    channels.extend([
        (climatology - rain_mean) / scale, prepared["target_std"][date.month - 1] / scale,
        (prepared["frozen_tp"][month_index] - rain_mean) / scale,
        la / max(abs(float(prepared["lat"].min())), abs(float(prepared["lat"].max()))),
        (lo - float(prepared["lon"].mean())) / max(float(np.ptp(prepared["lon"])) / 2, 1),
        np.full_like(climatology, np.sin(2 * np.pi * date.month / 12)),
        np.full_like(climatology, np.cos(2 * np.pi * date.month / 12)),
        np.full_like(climatology, (month_index + 1) / 24.0),
    ])
    scalar: dict[str, np.ndarray] = {}
    _add_ocean_columns(scalar, pd.DatetimeIndex([date]), 1, read_ocean_indices())
    for ocean in ("soi", "nino12", "nino3", "nino4"):
        for lag in (1, 2, 3):
            channels.append(np.full_like(climatology, scalar[f"{ocean}_lag{lag}"][0] / 3.0))
    return torch.from_numpy(np.stack(channels).astype("float32"))[None]


@torch.no_grad()
def predict_unet_submission(
    checkpoint: str | Path,
    data_dir: str | Path | None = None,
    output_path: str | Path = ARTIFACTS_DIR / "submission_unet.csv",
) -> Path:
    data_path = resolve_data_dir(data_dir)
    saved = torch.load(checkpoint, map_location="cpu")
    metadata = saved["metadata"]
    model = SmallUNet(len(metadata["channels"]), metadata["base_channels"])
    model.load_state_dict(saved["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    prepared = _prepare_full_inference(data_path)
    predictions = []
    for month_index in range(24):
        tensor = _unet_test_tensor(prepared, month_index, metadata).to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            residual = model(tensor).float().cpu().numpy()[0, 0] * metadata["rain_scale"]
        climatology = prepared["target_mean"][prepared["dates"][month_index].month - 1]
        predictions.append(np.maximum(0.0, climatology + residual).ravel())
    sample = pd.read_csv(data_path / "sample_submission.csv")
    sample["tp_mm_day"] = np.concatenate(predictions).astype("float32")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output, index=False)
    audit_submission(output, data_path / "sample_submission.csv")
    return output


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--fold", choices=FOLDS, default="2015")
    validate.add_argument("--max-epochs", type=int, default=30)
    validate.add_argument("--patience", type=int, default=5)
    fit = sub.add_parser("fit-final")
    fit.add_argument("--epochs", type=int, required=True)
    fit.add_argument("--output", default=str(ARTIFACTS_DIR / "unet_final.pt"))
    predict = sub.add_parser("predict")
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "validate":
        validate_unet_fold(FOLDS[args.fold], args.data_dir, args.max_epochs, args.patience)
    elif args.command == "fit-final":
        fit_final_unet(args.epochs, args.data_dir, args.output)
    elif args.command == "predict":
        predict_unet_submission(args.checkpoint, args.data_dir, args.output)


if __name__ == "__main__":
    main()
