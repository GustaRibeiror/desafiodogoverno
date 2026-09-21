"""Utilitários independentes do experimento residual."""

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
_DATA_ROOT = ROOT / "data"
_NESTED_DATA = _DATA_ROOT / "previsao-climatica-de-precipitacao-sobre-a-america-do-sul"
DATA_DIR = _NESTED_DATA if (_NESTED_DATA / "treino_tp.nc").exists() else _DATA_ROOT


def rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    """Calcula o RMSE global, a mesma métrica usada pela competição."""
    return float(np.sqrt(np.mean((np.asarray(prediction) - np.asarray(target)) ** 2)))
