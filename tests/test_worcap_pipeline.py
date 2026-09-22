from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from worcap_pipeline import (  # noqa: E402
    ATMOS_FILES,
    Fold,
    SampledBundle,
    apply_lead_calibration,
    build_table,
    fit_lead_calibration,
    fold_indices,
    evaluate_spatial_average,
    optimize_two_model_blend,
    score_24_months,
    stratified_point_pair,
)


def ocean_series() -> dict[str, pd.Series]:
    dates = pd.date_range("1980-01-01", "2030-12-01", freq="MS")
    return {
        name: pd.Series(np.linspace(-1, 1, len(dates), dtype="float32"), index=dates)
        for name in ("soi", "nino12", "nino3", "nino4")
    }


def synthetic_bundle() -> SampledBundle:
    rng = np.random.default_rng(7)
    times = pd.date_range("1990-01-01", periods=60, freq="MS")
    lat = np.asarray([-1.0, 0.0], dtype="float32")
    lon = np.asarray([-2.0, -1.0], dtype="float32")
    points = np.arange(4)
    series = {"tp_anchor": rng.uniform(0, 5, (60, 4)).astype("float32")}
    for name in ATMOS_FILES:
        series[name] = rng.normal(size=(60, 4)).astype("float32")
    local_mean = {name: series[name] + 0.1 for name in ("cloud_cover", "shum_850", "rel_hum_850", "u_850", "v_850")}
    local_std = {name: np.full((60, 4), 0.2, dtype="float32") for name in local_mean}
    target = rng.uniform(0, 8, (60, 4)).astype("float32")
    clim = np.stack([target[np.arange(60) % 12 == month].mean(axis=0) for month in range(12)])
    clim_std = np.stack([target[np.arange(60) % 12 == month].std(axis=0) for month in range(12)])
    context = {
        "clim_local_mean": clim.copy(),
        "clim_local_std": clim_std.copy(),
        "clim_grad_lat": np.zeros_like(clim),
        "clim_grad_lon": np.zeros_like(clim),
    }
    return SampledBundle(
        times=times,
        lat=lat,
        lon=lon,
        point_flat=points,
        series=series,
        local_mean=local_mean,
        local_std=local_std,
        flux_convergence=rng.normal(size=(60, 4)).astype("float32"),
        target=target,
        target_clim_mean=clim.astype("float32"),
        target_clim_std=clim_std.astype("float32"),
        target_clim_context=context,
    )


class TemporalContractTests(unittest.TestCase):
    def test_fold_has_24_months_and_past_only(self) -> None:
        times = pd.date_range("1960-01-01", "2022-11-01", freq="MS")
        train, valid = fold_indices(times, Fold("x", "2015-01-01", "2016-12-01"))
        targets = times + pd.DateOffset(months=1)
        self.assertEqual(len(valid), 24)
        self.assertLess(targets[train].max(), targets[valid].min())
        self.assertEqual(targets[train].min(), pd.Timestamp("1985-01-01"))

    def test_future_changes_do_not_modify_earlier_features(self) -> None:
        original = synthetic_bundle()
        changed = synthetic_bundle()
        # As duas construções começam idênticas; alteramos apenas uma linha futura.
        changed.series = {name: values.copy() for name, values in original.series.items()}
        changed.target = original.target.copy()
        changed.series["t2"][45] += 1_000
        changed.target[45] += 1_000
        train_idx = np.arange(10, 35)
        valid_idx = np.arange(35, 47)
        left = build_table(original, valid_idx, train_idx, "validation", ocean_series=ocean_series())
        right = build_table(changed, valid_idx, train_idx, "validation", ocean_series=ocean_series())
        points = len(original.point_flat)
        # Meses anteriores não podem depender de uma alteração no mês 11.
        feature_columns = [column for column in left if column != "target"]
        np.testing.assert_allclose(left.iloc[: 10 * points][feature_columns], right.iloc[: 10 * points][feature_columns])
        # Alterar o alvo oculto jamais altera as features.
        target_changed = synthetic_bundle()
        target_changed.target = original.target.copy()
        target_changed.target[45] += 1_000
        only_target = build_table(
            target_changed, valid_idx, train_idx, "validation", ocean_series=ocean_series()
        )
        np.testing.assert_allclose(left[feature_columns], only_target[feature_columns])

    def test_frozen_precipitation_anchor(self) -> None:
        bundle = synthetic_bundle()
        frame = build_table(
            bundle, np.arange(35, 47), np.arange(10, 35), "validation", ocean_series=ocean_series()
        )
        anchor = frame["tp_anchor"].to_numpy().reshape(12, 4)
        for month in range(1, 12):
            np.testing.assert_allclose(anchor[0], anchor[month])


class CalibrationTests(unittest.TestCase):
    def test_recovers_known_bucket_scales(self) -> None:
        rng = np.random.default_rng(9)
        lead = np.repeat(np.arange(1, 25), 20)
        base = rng.uniform(1, 4, len(lead))
        anomaly = rng.normal(0, 0.2, len(lead))
        true_scale = np.asarray([0.4, 0.7, 1.0, 1.3])
        truth = base + anomaly * true_scale[(lead - 1) // 6]
        raw = base + anomaly
        fitted = fit_lead_calibration(base, raw, truth, lead)
        np.testing.assert_allclose(fitted, true_scale, atol=1e-6)
        np.testing.assert_allclose(apply_lead_calibration(base, raw, lead, fitted), truth, atol=1e-6)

    def test_year_metrics(self) -> None:
        truth = np.zeros(24 * 2)
        prediction = np.concatenate([np.ones(12 * 2), np.full(12 * 2, 2)])
        scores = score_24_months(prediction, truth, 2)
        self.assertAlmostEqual(scores["rmse_year1"], 1.0)
        self.assertAlmostEqual(scores["rmse_year2"], 2.0)
        self.assertAlmostEqual(scores["rmse_24"], np.sqrt(2.5))

    def test_oof_blend_selection_and_gates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths_a, paths_b = [], []
            for fold in range(3):
                truth = np.zeros(24 * 3, dtype="float32")
                climatology = np.zeros_like(truth)
                lead = np.repeat(np.arange(1, 25), 3)
                # Erros opostos: a média é melhor que ambos em todos os folds.
                prediction_a = np.full_like(truth, 1.0 + fold * 0.01)
                prediction_b = np.full_like(truth, -1.0 - fold * 0.01)
                path_a = Path(directory) / f"a{fold}.npz"
                path_b = Path(directory) / f"b{fold}.npz"
                np.savez(path_a, prediction=prediction_a, truth=truth, climatology=climatology, lead=lead)
                np.savez(path_b, prediction=prediction_b, truth=truth, climatology=climatology, lead=lead)
                paths_a.append(path_a)
                paths_b.append(path_b)
            spatial = evaluate_spatial_average(paths_a, paths_b)
            self.assertTrue(spatial["accept_average"])
            optimized = optimize_two_model_blend(paths_a, paths_b, step=0.05)
            self.assertTrue(optimized["accept_blend"])
            self.assertAlmostEqual(optimized["weight_a"], 0.5)


class SpatialSamplingTests(unittest.TestCase):
    def test_samples_are_sized_disjoint_and_reproducible(self) -> None:
        lat = np.linspace(-60, 15, 40)
        lon = np.linspace(-90, -25, 50)
        rain = np.arange(2000, dtype="float32").reshape(40, 50) % 17
        first_a, first_b = stratified_point_pair(rain, lat, lon, n_each=300, seed=11)
        second_a, second_b = stratified_point_pair(rain, lat, lon, n_each=300, seed=11)
        self.assertEqual(len(first_a), 300)
        self.assertEqual(len(first_b), 300)
        self.assertEqual(len(np.intersect1d(first_a, first_b)), 0)
        np.testing.assert_array_equal(first_a, second_a)
        np.testing.assert_array_equal(first_b, second_b)


if __name__ == "__main__":
    unittest.main()
