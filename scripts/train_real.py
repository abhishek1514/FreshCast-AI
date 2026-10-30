"""Train FreshCast on the prepared FreshRetailNet real-world subset."""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd

from freshcast.data.loader import DataLoader
from freshcast.features.pipeline import TimeSeriesFeaturePipeline
from freshcast.metrics.financial import calculate_supply_chain_financial_loss
from freshcast.metrics.standard import calculate_standard_metrics
from freshcast.models.backtest import WalkForwardBacktester
from freshcast.models.baselines import MovingAverageForecaster, NaiveSeasonalForecaster
from freshcast.models.gbdt import LightGBMForecaster, XGBoostForecaster

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger("freshcast.train_real")
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    data_path = ROOT / "data/raw/foodservice_daily_sales.csv"
    if not data_path.exists():
        raise SystemExit("Real dataset not prepared. Run: python scripts/prepare_real_data.py")
    df = DataLoader().load_raw_data(data_path)
    date_count = df.date.nunique()
    if date_count < 50:
        raise SystemExit(f"Only {date_count} dates available; at least 50 are recommended.")
    LOG.info("Training on %s rows, %s stores, %s products, %s dates", len(df), df.dc_id.nunique(), df.sku_id.nunique(), date_count)
    # FreshRetailNet contains about 90 days per series, so use 3x7-day folds rather than
    # the synthetic project's 3x28-day folds. This preserves chronological validation.
    models = [NaiveSeasonalForecaster(lag=7), MovingAverageForecaster(window=7), LightGBMForecaster(), XGBoostForecaster()]
    backtester = WalkForwardBacktester(n_splits=3, test_window_days=7)
    folds, summary = backtester.benchmark_models(models, df, TimeSeriesFeaturePipeline())
    out = ROOT / "data/processed"
    out.mkdir(parents=True, exist_ok=True)
    folds.to_csv(out / "backtest_results.csv", index=False)
    summary.to_csv(out / "backtest_summary.csv", index=False)
    LOG.info("Walk-forward results:\n%s", summary.to_string(index=False))
    train_df, test_df = DataLoader().split_train_test_by_date(df, test_days=14)
    pipeline = TimeSeriesFeaturePipeline()
    train_t = pipeline.fit_transform(train_df, drop_na=True)
    model = LightGBMForecaster()
    model.fit(train_t[pipeline.feature_cols], train_t[pipeline.TARGET_COL])
    combined = pd.concat([train_df, test_df], ignore_index=True)
    transformed = pipeline.transform(combined, drop_na=False)
    test_t = transformed[transformed.date >= test_df.date.min()].copy()
    preds = model.predict(test_t[pipeline.feature_cols])
    test_t["pred_sales"] = preds
    metrics = calculate_standard_metrics(test_t[pipeline.TARGET_COL].values, preds)
    financial = calculate_supply_chain_financial_loss(test_t, pred_col="pred_sales", actual_col=pipeline.TARGET_COL)
    LOG.info("14-day temporal holdout metrics: %s", metrics)
    LOG.info("Illustrative financial estimate (assumed costs, not actual supplier economics): %s", financial)
    importance = model.get_feature_importance()
    importance.to_csv(out / "feature_importance.csv", index=False)
    bundle = {"model": model, "pipeline": pipeline, "feature_importance": importance, "test_metrics": metrics, "test_financial": financial, "evaluation_df": test_t, "data_source": "FreshRetailNet-50K", "financial_costs_are_assumptions": True}
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(bundle, models_dir / "champion_model.joblib")
    LOG.info("Saved models/champion_model.joblib")

if __name__ == "__main__":
    main()
