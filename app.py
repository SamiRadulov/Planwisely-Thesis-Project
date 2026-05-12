import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Planwisely - Sales Forecast",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)

st.title("Planwisely Sales Forecast")
st.markdown(
    "Upload a CSV with sales data. Supports a single date column or split year/month/day columns. "
    "If a **product_id** column is present you can filter by product."
)

# ── Column detection ───────────────────────────────────────────────────────────

def detect_structure(df):
    cols_lower = {c: c.lower() for c in df.columns}

    product_col = next((c for c, l in cols_lower.items() if l in ("product_id", "product", "sku", "item_id")), None)

    sales_candidates = ("sku_sold", "sales", "quantity", "units", "revenue", "amount")
    sales_col = next((c for c, l in cols_lower.items() if l in sales_candidates), None)
    if sales_col is None:
        for c in df.columns:
            if c == product_col:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                sales_col = c
                break

    year_col  = next((c for c, l in cols_lower.items() if l == "year"), None)
    month_col = next((c for c, l in cols_lower.items() if l == "month"), None)
    day_col   = next((c for c, l in cols_lower.items() if l in ("day", "day_of_month")), None)

    if year_col and month_col and day_col:
        return dict(date_mode="split", date_col=None,
                    year_col=year_col, month_col=month_col, day_col=day_col,
                    sales_col=sales_col, product_col=product_col)

    date_col = None
    for c in df.columns:
        if c in (product_col, sales_col):
            continue
        try:
            parsed = pd.to_datetime(df[c], infer_datetime_format=True, errors="coerce")
            if parsed.notna().mean() > 0.8:
                date_col = c
                break
        except Exception:
            pass

    return dict(date_mode="single", date_col=date_col,
                year_col=None, month_col=None, day_col=None,
                sales_col=sales_col, product_col=product_col)


def build_date_series(df, info):
    if info["date_mode"] == "split":
        return pd.to_datetime(dict(
            year=df[info["year_col"]],
            month=df[info["month_col"]],
            day=df[info["day_col"]]
        ))
    return pd.to_datetime(df[info["date_col"]], infer_datetime_format=True, errors="coerce")


# ── Feature engineering ────────────────────────────────────────────────────────

def feature_engineering(df_in):
    df = df_in[["date", "sales"]].copy()
    df = df.sort_values("date").reset_index(drop=True)
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    df = df.dropna(subset=["sales"])

    df["year"]           = df["date"].dt.isocalendar().year.astype(int)
    df["month"]          = df["date"].dt.month
    df["week_of_year"]   = df["date"].dt.isocalendar().week.astype(int)
    df["day_of_week"]    = df["date"].dt.dayofweek
    df["is_month_start"] = (df["date"].dt.day <= 7).astype(int)
    df["is_month_end"]   = (df["date"].dt.day >= 24).astype(int)
    df["quarter"]        = df["date"].dt.quarter
    df["month_sin"]      = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]      = np.cos(2 * np.pi * df["month"] / 12)
    df["week_sin"]       = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["week_cos"]       = np.cos(2 * np.pi * df["week_of_year"] / 52)
    df["dow_sin"]        = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]        = np.cos(2 * np.pi * df["day_of_week"] / 7)

    for lag in [1, 2, 4, 8]:
        df[f"lag_{lag}"] = df["sales"].shift(lag)

    for window in [4, 8]:
        df[f"rolling_mean_{window}"] = df["sales"].shift(1).rolling(window, min_periods=1).mean()
        df[f"rolling_std_{window}"]  = df["sales"].shift(1).rolling(window, min_periods=1).std().fillna(0)

    df["trend"] = np.arange(len(df))
    df = df.dropna(subset=["lag_1", "lag_2", "lag_4", "lag_8"])
    return df


FEATURE_COLS = [
    "trend",
    "year", "month", "week_of_year", "quarter",
    "is_month_start", "is_month_end",
    "month_sin", "month_cos", "week_sin", "week_cos", "dow_sin", "dow_cos",
    "lag_1", "lag_2", "lag_4", "lag_8",
    "rolling_mean_4", "rolling_mean_8", "rolling_std_4", "rolling_std_8",
]


# ── Model ──────────────────────────────────────────────────────────────────────

def train_model(df):
    X = df[FEATURE_COLS]
    y = df["sales"]
    model = LinearRegression()
    model.fit(X, y)
    return model


def forecast_future(model, df, periods=12):
    delta = df["date"].iloc[-1] - df["date"].iloc[-2]
    sales_history = list(df["sales"])
    future_rows = []

    for i in range(periods):
        next_date = df["date"].iloc[-1] + (i + 1) * delta
        row = {
            "date"          : next_date,
            "year"          : next_date.isocalendar()[0],
            "month"         : next_date.month,
            "week_of_year"  : next_date.isocalendar()[1],
            "day_of_week"   : next_date.dayofweek,
            "is_month_start": int(next_date.day <= 7),
            "is_month_end"  : int(next_date.day >= 24),
            "quarter"       : next_date.quarter,
            "trend"         : len(df) + i,
        }
        row["month_sin"] = np.sin(2 * np.pi * row["month"] / 12)
        row["month_cos"] = np.cos(2 * np.pi * row["month"] / 12)
        row["week_sin"]  = np.sin(2 * np.pi * row["week_of_year"] / 52)
        row["week_cos"]  = np.cos(2 * np.pi * row["week_of_year"] / 52)
        row["dow_sin"]   = np.sin(2 * np.pi * row["day_of_week"] / 7)
        row["dow_cos"]   = np.cos(2 * np.pi * row["day_of_week"] / 7)

        for lag in [1, 2, 4, 8]:
            row[f"lag_{lag}"] = sales_history[-lag] if len(sales_history) >= lag else np.nan
        for window in [4, 8]:
            hist = sales_history[-window:]
            row[f"rolling_mean_{window}"] = np.mean(hist)
            row[f"rolling_std_{window}"]  = np.std(hist) if len(hist) > 1 else 0.0

        pred = max(model.predict(np.array([[row[c] for c in FEATURE_COLS]]))[0], 0)
        sales_history.append(pred)
        row["forecast"] = pred
        future_rows.append(row)

    return pd.DataFrame(future_rows)


# ── Plot ───────────────────────────────────────────────────────────────────────

def plot_forecast(df_hist, future, title_suffix=""):
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    # Historical (last 52 points for context)
    hist_tail = df_hist.tail(52)
    ax.plot(hist_tail["date"], hist_tail["sales"],
            color="#4C9BE8", linewidth=1.5, label="Historical sales")

    # Forecast
    ax.plot(future["date"], future["forecast"],
            color="#2DC97E", linewidth=2, label="Forecast")
    ax.fill_between(future["date"],
                    future["forecast"] * 0.85,
                    future["forecast"] * 1.15,
                    color="#2DC97E", alpha=0.15, label="+/-15% band")

    # Divider line
    ax.axvline(df_hist["date"].iloc[-1], color="#888", linewidth=1, linestyle="--")

    title = "Sales Forecast"
    if title_suffix:
        title += f"  |  {title_suffix}"
    ax.set_title(title, color="white", fontsize=13)
    ax.set_ylabel("Sales (sku_sold)", color="white", fontsize=11)
    ax.legend(facecolor="#1C1F26", edgecolor="#444", labelcolor="white", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    plt.tight_layout()
    return fig


# ── UI ─────────────────────────────────────────────────────────────────────────

uploaded = st.file_uploader(
    "Upload your sales CSV",
    type=["csv"],
    help="Needs at least a date column (or year/month/day) and a sales column.",
)

if uploaded is not None:
    with st.spinner("Reading file..."):
        try:
            raw = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()

    info = detect_structure(raw)

    if info["date_mode"] == "split":
        st.info(
            f"Split date detected: **{info['year_col']}** / **{info['month_col']}** / "
            f"**{info['day_col']}** will be combined into a single date."
        )

    # Sales column selector
    sales_col = st.selectbox(
        "Sales column",
        raw.columns.tolist(),
        index=raw.columns.tolist().index(info["sales_col"]) if info["sales_col"] else 0,
    )

    # Product selector
    product_id = None
    if info["product_col"] is not None:
        products = sorted(raw[info["product_col"]].dropna().unique())
        product_id = st.selectbox(
            f"Filter by {info['product_col']}",
            options=products,
            format_func=lambda x: f"Product {x}",
        )

    forecast_periods = st.slider("Forecast periods ahead", min_value=4, max_value=52, value=12, step=4)

    if st.button("Run Forecast", type="primary"):

        df_work = raw.copy()
        if product_id is not None:
            df_work = df_work[df_work[info["product_col"]] == product_id].copy()

        try:
            df_work["date"] = build_date_series(df_work, info)
        except Exception as e:
            st.error(f"Could not parse dates: {e}")
            st.stop()

        df_work["sales"] = pd.to_numeric(df_work[sales_col], errors="coerce")
        df_work = df_work[["date", "sales"]].dropna()

        with st.spinner("Running forecast..."):
            df_feat = feature_engineering(df_work)

            if len(df_feat) < 20:
                st.error(
                    f"Only {len(df_feat)} usable rows — need at least 20. "
                    "Try a product with more history."
                )
                st.stop()

            model = train_model(df_feat)
            future = forecast_future(model, df_feat, periods=forecast_periods)

        suffix = f"Product {product_id}" if product_id is not None else ""

        st.subheader("Forecast")
        st.pyplot(plot_forecast(df_feat, future, suffix), use_container_width=True)

        st.subheader("Forecast Values")
        forecast_display = future[["date", "forecast"]].copy()
        forecast_display["forecast"] = forecast_display["forecast"].round(1)
        forecast_display.columns = ["Date", "Forecasted Sales"]
        st.dataframe(forecast_display, use_container_width=True)

        st.download_button(
            "Download Forecast CSV",
            forecast_display.to_csv(index=False).encode(),
            "forecast.csv",
            "text/csv",
        )
