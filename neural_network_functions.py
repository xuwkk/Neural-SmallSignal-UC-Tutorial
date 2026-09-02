"""Data and evaluation helpers for the regression tutorial."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class RegressionData:
    """One per-unit input dataset and its continuous regression targets."""

    features: torch.Tensor
    targets: torch.Tensor
    feature_names: tuple[str, ...]
    feature_scale: np.ndarray


def prepare_regression_data(
    dataset,
    feature_columns: list[str],
    system_base_mva: float,
    target_column: str = "critical_real",
    binary_columns: tuple[str, ...] = ("u_2", "u_3", "u_6"),
) -> RegressionData:
    """Scale NN inputs and retain the critical eigenvalue real part in 1/s."""

    raw_features = dataset[feature_columns].to_numpy(dtype=np.float32)
    targets = dataset[target_column].to_numpy(dtype=np.float32).reshape(-1, 1)

    feature_scale = np.ones(len(feature_columns), dtype=np.float32)

    for column_index, column_name in enumerate(feature_columns):
        if column_name not in binary_columns:
            feature_scale[column_index] = system_base_mva

    scaled_features = raw_features / feature_scale
    return RegressionData(
        features=torch.tensor(scaled_features, dtype=torch.float32),
        targets=torch.tensor(targets, dtype=torch.float32),
        feature_names=tuple(feature_columns),
        feature_scale=feature_scale,
    )


def regression_metrics(
    model: nn.Module,
    training_data: RegressionData,
    stability_margin: float,
) -> dict[str, float | int]:
    """Calculate in-sample regression errors and false-stable predictions."""

    with torch.no_grad():
        predictions = model(training_data.features).cpu().numpy().ravel()
        targets = training_data.targets.cpu().numpy().ravel()

    errors = predictions - targets
    residual_sum = float(np.sum(errors**2))
    total_sum = float(np.sum((targets - targets.mean()) ** 2))
    actual_unstable = targets > -stability_margin
    predicted_stable = predictions <= -stability_margin

    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "r_squared": 1.0 - residual_sum / total_sum,
        "false_stable": int(np.sum(actual_unstable & predicted_stable)),
    }


def save_regressor(
    path: str | Path,
    model: nn.Module,
    training_data: RegressionData,
) -> Path:
    """Save the regressor weights and input normalization used by the UC."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_names": training_data.feature_names,
            "feature_scale": training_data.feature_scale,
        },
        path,
    )
    return path
