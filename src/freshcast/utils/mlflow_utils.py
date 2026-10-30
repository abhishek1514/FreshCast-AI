"""MLflow tracking and model registry utilities for FreshCast."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

DEFAULT_EXPERIMENT_NAME = "FreshCast-Demand-Forecasting"
DEFAULT_TRACKING_URI = "sqlite:///mlruns.db"


def setup_mlflow(
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    tracking_uri: str | None = None,
) -> str:
    """Configures MLflow tracking URI and ensures experiment exists."""
    uri = tracking_uri or DEFAULT_TRACKING_URI
    mlflow.set_tracking_uri(uri)
    logger.info("Configured MLflow tracking URI: %s", uri)

    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(
            name=experiment_name,
            tags={"project": "FreshCast", "domain": "foodservice-supply-chain"},
        )
        logger.info(
            "Created new MLflow experiment '%s' (ID: %s)",
            experiment_name,
            experiment_id,
        )
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)
    return experiment_id


def log_backtest_run(
    run_name: str,
    params: dict[str, Any],
    summary_metrics: dict[str, float],
    artifacts: dict[str, Path | str] | None = None,
    tags: dict[str, str] | None = None,
) -> str:
    """Logs a walk-forward backtest evaluation run to MLflow."""
    setup_mlflow()

    with mlflow.start_run(run_name=run_name) as run:
        # Log parameters
        for k, v in params.items():
            mlflow.log_param(k, str(v))

        # Log metrics
        for k, v in summary_metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, float(v))

        # Set tags
        if tags:
            mlflow.set_tags(tags)

        # Log artifact files
        if artifacts:
            for art_name, art_path in artifacts.items():
                p = Path(art_path)
                if p.exists():
                    if p.is_file():
                        mlflow.log_artifact(str(p), artifact_path=art_name)
                    elif p.is_dir():
                        mlflow.log_artifacts(str(p), artifact_path=art_name)

        run_id = run.info.run_id
        logger.info("Successfully logged MLflow run '%s' (Run ID: %s)", run_name, run_id)
        return run_id
