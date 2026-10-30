"""FreshCast: Enterprise Foodservice Demand Forecasting Dashboard.

Built for warehouse inventory planners and supply chain analysts.
Provides 7/14/28-day forecast horizons, what-if promotional simulation,
safety stock & reorder point calculation, and model backtest transparency.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="FreshCast | Enterprise Demand Forecasting",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0b3c5d;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #0284c7;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_bundle():
    bundle_path = Path("models/champion_model.joblib")
    if not bundle_path.exists():
        st.error("Model bundle not found. Please run the training pipeline first.")
        st.stop()
    return joblib.load(bundle_path)


@st.cache_data
def load_historical_data():
    data_path = Path("data/raw/foodservice_daily_sales.csv")
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_benchmark_data():
    summary_path = Path("data/processed/backtest_summary.csv")
    results_path = Path("data/processed/backtest_results.csv")
    summary = pd.read_csv(summary_path) if summary_path.exists() else None
    results = pd.read_csv(results_path) if results_path.exists() else None
    return summary, results


def main():
    bundle = load_bundle()
    df_raw = load_historical_data()
    summary_df, results_df = load_benchmark_data()

    model = bundle["model"]
    pipeline = bundle["pipeline"]
    feature_imp = bundle.get("feature_importance")

    # Sidebar Branding & Controls
    st.sidebar.markdown(
        """
        <div style="background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%); padding: 14px 16px; border-radius: 8px; margin-bottom: 20px; color: white; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="font-size: 20px; font-weight: 800; letter-spacing: 0.5px;">📦 FreshCast</div>
            <div style="font-size: 11px; opacity: 0.9; text-transform: uppercase; letter-spacing: 1px; margin-top: 3px;">Supply Chain Intelligence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("### 🏢 Distribution Center & SKU")

    dcs = df_raw[["dc_id", "dc_name"]].drop_duplicates().to_dict("records")
    dc_options = {d["dc_id"]: f"{d['dc_id']} - {d['dc_name']}" for d in dcs}
    selected_dc_id = st.sidebar.selectbox(
        "Select Distribution Center (RDC):",
        options=list(dc_options.keys()),
        format_func=lambda x: dc_options[x],
    )

    skus = df_raw[["sku_id", "sku_name", "category"]].drop_duplicates().to_dict("records")
    sku_options = {s["sku_id"]: f"[{s['category']}] {s['sku_name']}" for s in skus}
    selected_sku_id = st.sidebar.selectbox(
        "Select Perishable SKU:",
        options=list(sku_options.keys()),
        format_func=lambda x: sku_options[x],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Supply Chain Controls")
    horizon = st.sidebar.select_slider(
        "Forecast Horizon:",
        options=[7, 14, 28],
        value=14,
        format_func=lambda x: f"{x} Days Ahead",
    )

    service_level = st.sidebar.selectbox(
        "Target Service Level (Safety Stock):",
        options=[0.90, 0.95, 0.98],
        index=1,
        format_func=lambda x: f"{int(x*100)}% Service Level",
    )
    # Service factor z-scores
    z_factors = {0.90: 1.282, 0.95: 1.645, 0.98: 2.054}
    z_score = z_factors[service_level]

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏷️ Promotional Elasticity Simulation")
    what_if_discount = st.sidebar.slider(
        "What-If Price Discount (%):",
        min_value=0,
        max_value=30,
        value=0,
        step=5,
        help="Simulate restaurant demand lift under temporary promotional pricing.",
    )

    # Filter data for selected DC & SKU
    series_raw = (
        df_raw[(df_raw["dc_id"] == selected_dc_id) & (df_raw["sku_id"] == selected_sku_id)]
        .sort_values("date")
        .reset_index(drop=True)
    )

    sku_meta = series_raw.iloc[-1]
    shelf_life = sku_meta["shelf_life_days"]
    lead_time = sku_meta["lead_time_days"]
    unit_cost = sku_meta["unit_cost"]
    base_price = sku_meta["base_unit_price"]

    # Header
    st.markdown(
        '<div class="main-header">FreshCast: Foodservice Demand Forecasting</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="sub-header">Operational intelligence for <b>{sku_meta["dc_name"]}</b> • '
        f'Active SKU: <b>{sku_meta["sku_name"]}</b> ({sku_meta["category"]})</div>',
        unsafe_allow_html=True,
    )

    # Shelf-life alert banner
    if shelf_life <= 6:
        st.warning(
            f"⚠️ **High Perishability Alert**: This item has an active shelf-life of only **{shelf_life} days**. "
            f"Over-forecasting incurs severe spoilage penalties (${unit_cost:.2f} per lost case).",
            icon="⚠️",
        )

    # Tabs
    tab_planner, tab_benchmarks, tab_financials = st.tabs(
        [
            "📈 14-Day Demand & Reorder Planner",
            "🏆 Model Benchmark & Walk-Forward Validation",
            "💰 Supply Chain Financials & Spoilage P&L",
        ]
    )

    # ---------------- TAB 1: DEMAND & REORDER PLANNER ----------------
    with tab_planner:
        # Generate Future Horizon Forecast
        last_date = series_raw["date"].max()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

        # Build feature row for each future day
        future_rows = []

        # Iterative forecast simulation
        for f_date in future_dates:
            # Promo adjustment
            is_promo = 1 if what_if_discount > 0 else 0
            discount_pct = what_if_discount / 100.0
            act_price = round(base_price * (1.0 - discount_pct), 2)

            new_row = {
                "date": f_date,
                "dc_id": selected_dc_id,
                "dc_name": sku_meta["dc_name"],
                "sku_id": selected_sku_id,
                "sku_name": sku_meta["sku_name"],
                "category": sku_meta["category"],
                "shelf_life_days": shelf_life,
                "lead_time_days": lead_time,
                "unit_cost": unit_cost,
                "base_unit_price": base_price,
                "actual_unit_price": act_price,
                "discount_pct": discount_pct,
                "is_promo": is_promo,
                "sales_cases": 0.0,
            }
            future_rows.append(new_row)

        future_df = pd.DataFrame(future_rows)
        combined_simulation = pd.concat([series_raw, future_df], ignore_index=True)
        transformed_future = pipeline.transform(combined_simulation, drop_na=False)

        future_transformed_slice = transformed_future[
            transformed_future["date"].isin(future_dates)
        ].copy()

        X_future = future_transformed_slice[pipeline.feature_cols]
        preds = model.predict(X_future)

        # Apply promotional elasticity factor if user moved the slider
        if what_if_discount > 0:
            elasticity_multiplier = 1.0 + (1.6 * (what_if_discount / 100.0))
            preds = preds * elasticity_multiplier

        future_df["forecasted_demand"] = np.round(preds).astype(int)
        # Approximate 90% confidence bands using recent residual std
        recent_std = series_raw["sales_cases"].tail(28).std()
        future_df["upper_bound"] = np.round(
            future_df["forecasted_demand"] + (1.645 * recent_std)
        ).astype(int)
        future_df["lower_bound"] = np.maximum(
            0, np.round(future_df["forecasted_demand"] - (1.645 * recent_std))
        ).astype(int)

        # Inventory Planner calculations
        avg_daily_demand = future_df["forecasted_demand"].mean()
        lead_time_demand = avg_daily_demand * lead_time
        lead_time_sigma = recent_std * np.sqrt(lead_time)
        safety_stock = int(np.ceil(z_score * lead_time_sigma))
        reorder_point = int(np.ceil(lead_time_demand + safety_stock))
        total_projected_cases = int(future_df["forecasted_demand"].sum())

        # Top Metric Cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                label=f"Projected {horizon}-Day Demand",
                value=f"{total_projected_cases:,} cases",
                delta=(
                    f"+{what_if_discount}% promo lift" if what_if_discount > 0 else "Baseline price"
                ),
            )
        with col2:
            st.metric(
                label="Recommended Safety Stock",
                value=f"{safety_stock:,} cases",
                help=f"Buffering demand volatility at {int(service_level*100)}% cycle service level.",
            )
        with col3:
            st.metric(
                label="Reorder Point (ROP)",
                value=f"{reorder_point:,} cases",
                help="Trigger replenishment order when physical inventory hits this threshold.",
            )
        with col4:
            st.metric(
                label="Replenishment Lead Time",
                value=f"{lead_time} days",
                delta=f"Shelf life: {shelf_life}d",
                delta_color="off",
            )

        st.markdown("### 📊 Interactive Forecast & Confidence Bands")

        # Plotly visualizer
        history_window = 60
        history_subset = series_raw.tail(history_window)

        fig = go.Figure()

        # Historical Actuals
        fig.add_trace(
            go.Scatter(
                x=history_subset["date"],
                y=history_subset["sales_cases"],
                mode="lines+markers",
                name="Historical Orders (Actual)",
                line=dict(color="#1e293b", width=2),
                marker=dict(size=4),
            )
        )

        # Prediction Confidence Band (Upper & Lower)
        fig.add_trace(
            go.Scatter(
                x=list(future_df["date"]) + list(future_df["date"])[::-1],
                y=list(future_df["upper_bound"]) + list(future_df["lower_bound"])[::-1],
                fill="toself",
                fillcolor="rgba(14, 165, 233, 0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=True,
                name="90% Prediction Interval",
            )
        )

        # Forecast Curve
        fig.add_trace(
            go.Scatter(
                x=future_df["date"],
                y=future_df["forecasted_demand"],
                mode="lines+markers",
                name=f"LightGBM Forecast ({horizon}d)",
                line=dict(color="#0284c7", width=3, dash="dash"),
                marker=dict(size=6, symbol="diamond"),
            )
        )

        # Reorder Point threshold horizontal guide
        fig.add_hline(
            y=reorder_point,
            line_dash="dot",
            line_color="#ef4444",
            annotation_text=f"Reorder Point ({reorder_point} cases)",
            annotation_position="bottom right",
        )

        fig.update_layout(
            height=460,
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_title="Date",
            yaxis_title="Cases Demanded",
            hovermode="x unified",
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Daily Reorder Recommendation Table
        st.markdown("### 📋 Daily Order Recommendation Schedule")
        future_display = future_df[
            ["date", "forecasted_demand", "lower_bound", "upper_bound"]
        ].copy()
        future_display["date"] = future_display["date"].dt.strftime("%A, %b %d, %Y")
        future_display.columns = [
            "Delivery Date",
            "Forecasted Demand (Cases)",
            "P10 Lower Bound",
            "P90 Upper Bound",
        ]
        st.dataframe(future_display, use_container_width=True, hide_index=True)

    # ---------------- TAB 2: MODEL BENCHMARKS & BACKTESTING ----------------
    with tab_benchmarks:
        st.markdown("### 🏆 Walk-Forward Cross-Validation Benchmark")
        st.markdown("""
            To guarantee zero data leakage, FreshCast evaluates all models on **3 sequential, out-of-time walk-forward folds**
            (28 days per evaluation fold).
            """)

        if summary_df is not None:
            # Summary Table
            styled_summary = summary_df.copy()
            styled_summary["wape"] = (styled_summary["wape"] * 100).round(2).astype(str) + "%"
            styled_summary["mae"] = styled_summary["mae"].round(2)
            styled_summary["rmse"] = styled_summary["rmse"].round(2)
            styled_summary["total_financial_loss_usd"] = "$" + styled_summary[
                "total_financial_loss_usd"
            ].round(2).map("{:,.2f}".format)
            styled_summary["spoilage_loss_usd"] = "$" + styled_summary["spoilage_loss_usd"].round(
                2
            ).map("{:,.2f}".format)
            styled_summary["stockout_loss_usd"] = "$" + styled_summary["stockout_loss_usd"].round(
                2
            ).map("{:,.2f}".format)

            st.dataframe(
                styled_summary[
                    [
                        "model",
                        "wape",
                        "mae",
                        "rmse",
                        "total_financial_loss_usd",
                        "spoilage_loss_usd",
                        "stockout_loss_usd",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

            # Chart: WAPE Comparison
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Accuracy Comparison (WAPE % - Lower is Better)")
                chart_df = summary_df.sort_values("wape")
                fig_wape = px.bar(
                    chart_df,
                    x="model",
                    y="wape",
                    text=(chart_df["wape"] * 100).round(1).astype(str) + "%",
                    color="model",
                    color_discrete_sequence=[
                        "#0284c7",
                        "#38bdf8",
                        "#94a3b8",
                        "#cbd5e1",
                    ],
                )
                fig_wape.update_layout(
                    showlegend=False,
                    yaxis_title="WAPE",
                    height=340,
                    template="plotly_white",
                )
                st.plotly_chart(fig_wape, use_container_width=True)

            with c2:
                st.markdown("#### Total Supply Chain Loss ($ - Lower is Better)")
                fig_loss = px.bar(
                    chart_df,
                    x="model",
                    y="total_financial_loss_usd",
                    text=chart_df["total_financial_loss_usd"].map("${:,.0f}".format),
                    color="model",
                    color_discrete_sequence=[
                        "#0f766e",
                        "#14b8a6",
                        "#f59e0b",
                        "#ef4444",
                    ],
                )
                fig_loss.update_layout(
                    showlegend=False,
                    yaxis_title="Net Loss (USD)",
                    height=340,
                    template="plotly_white",
                )
                st.plotly_chart(fig_loss, use_container_width=True)

        # Feature Importance
        if feature_imp is not None:
            st.markdown("### 🧠 Top Feature Drivers in Champion LightGBM Model")
            top_15 = feature_imp.head(15).sort_values("importance", ascending=True)
            fig_imp = px.bar(
                top_15,
                x="importance",
                y="feature",
                orientation="h",
                color="importance",
                color_continuous_scale="Blues",
            )
            fig_imp.update_layout(
                coloraxis_showscale=False,
                height=420,
                xaxis_title="Feature Importance (Split Gain)",
                yaxis_title="",
                template="plotly_white",
            )
            st.plotly_chart(fig_imp, use_container_width=True)

    # ---------------- TAB 3: FINANCIALS & SPOILAGE ----------------
    with tab_financials:
        st.markdown("### 🥩 Perishable Category Financial Loss Exposure")
        st.markdown("""
            Evaluating demand forecasting purely on RMSE ignores business reality.
            In foodservice distribution, **overstocking fresh meat & seafood destroys working capital via spoilage**,
            whereas **understocking damages long-term restaurant customer relationships**.
            """)

        cat_report_path = Path("data/processed/category_evaluation_report.csv")
        if cat_report_path.exists():
            cat_df = pd.read_csv(cat_report_path)

            c_fig1, c_fig2 = st.columns(2)
            with c_fig1:
                st.markdown("#### Financial Loss per Case by Food Category ($/case)")
                fig_cat_unit = px.bar(
                    cat_df.sort_values("loss_per_case_usd", ascending=False),
                    x="category",
                    y="loss_per_case_usd",
                    color="loss_per_case_usd",
                    color_continuous_scale="Reds",
                    text=cat_df["loss_per_case_usd"].map("${:,.2f}".format),
                )
                fig_cat_unit.update_layout(
                    coloraxis_showscale=False,
                    yaxis_title="Financial Loss / Case ($)",
                    height=360,
                    template="plotly_white",
                )
                st.plotly_chart(fig_cat_unit, use_container_width=True)

            with c_fig2:
                st.markdown("#### Spoilage Waste vs. Lost Sales Stockout Loss")
                fig_breakdown = px.bar(
                    cat_df,
                    x="category",
                    y=["spoilage_loss_usd", "stockout_loss_usd"],
                    barmode="stack",
                    labels={"value": "Loss (USD)", "variable": "Loss Type"},
                    color_discrete_map={
                        "spoilage_loss_usd": "#ef4444",
                        "stockout_loss_usd": "#f59e0b",
                    },
                )
                fig_breakdown.update_layout(height=360, template="plotly_white")
                st.plotly_chart(fig_breakdown, use_container_width=True)

            st.dataframe(cat_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
