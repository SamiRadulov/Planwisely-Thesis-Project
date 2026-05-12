import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")

# Page config
st.set_page_config(
    page_title="Planwisely - Sales Forecast",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)

st.title("Planwisely Sales Forecast")
st.markdown(
    "Upload a CSV file with a **date column** and a **sales column** to generate "
    "a forecast using linear regression."
)

# Helper functions

def detect_columns(df):
    date_col = None
    sales_col = None
    for col in df.columns:
        if date_col is None:
            try:
                parsed = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
                if parsed.notna().mean() > 0.8:
                    date_col = col
                    continue
            except Exception:
                pass
        if sales_col is None:
            if pd.api.types.is_numeric_dtype(df[col]):
                sales_col = col
    return date_col, sales_col


def feature_engineering(df, date_col, sales_col):
    df = df[[date_col, sales_col]].copy()
    df.columns = ["date", "sales"]
    df["date"] = pd.to_datetime(df["date"])
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

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["week_sin"]  = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["week_cos"]  = np.cos(2 * np.pi * df["week_of_year"] / 52)
    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)

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
    "month_sin", "month_cos",
    "week_sin", "week_cos",
    "dow_sin", "dow_cos",
    "lag_1", "lag_2", "lag_4", "lag_8",
    "rolling_mean_4", "rolling_mean_8",
    "rolling_std_4", "rolling_std_8",
]


def train_and_evaluate(df):
    X = df[FEATURE_COLS]
    y = df["sales"]

    tscv = TimeSeriesSplit(n_splits=5)
    cv_maes, cv_rmses, cv_r2s = [], [], []

    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        m = LinearRegression()
        m.fit(X_tr, y_tr)
        preds = m.predict(X_val)
        cv_maes.append(mean_absolute_error(y_val, preds))
        cv_rmses.append(np.sqrt(mean_squared_error(y_val, preds)))
        cv_r2s.append(r2_score(y_val, preds))

    model = LinearRegression()
    model.fit(X, y)
    y_hat = model.predict(X)

    metrics = {
        "MAE (CV)"  : np.mean(cv_maes),
        "RMSE (CV)" : np.mean(cv_rmses),
        "R2 (CV)"   : np.mean(cv_r2s),
        "R2 (train)": r2_score(y, y_hat),
    }
    return model, y_hat, metrics


def forecast_future(model, df, periods=12):
    delta = df["date"].iloc[-1] - df["date"].iloc[-2]
    sales_history = list(df["sales"])
    future_rows = []

    for i in range(periods):
        next_date = df["date"].iloc[-1] + (i + 1) * delta
        row = {"date": next_date}
        row["year"]           = next_date.isocalendar()[0]
        row["month"]          = next_date.month
        row["week_of_year"]   = next_date.isocalendar()[1]
        row["day_of_week"]    = next_date.dayofweek
        row["is_month_start"] = int(next_date.day <= 7)
        row["is_month_end"]   = int(next_date.day >= 24)
        row["quarter"]        = next_date.quarter
        row["month_sin"]      = np.sin(2 * np.pi * row["month"] / 12)
        row["month_cos"]      = np.cos(2 * np.pi * row["month"] / 12)
        row["week_sin"]       = np.sin(2 * np.pi * row["week_of_year"] / 52)
        row["week_cos"]       = np.cos(2 * np.pi * row["week_of_year"] / 52)
        row["dow_sin"]        = np.sin(2 * np.pi * row["day_of_week"] / 7)
        row["dow_cos"]        = np.cos(2 * np.pi * row["day_of_week"] / 7)
        row["trend"]          = len(df) + i

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


def plot_results(df, y_hat, future):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [2, 1]})
    fig.patch.set_facecolor("#0E1117")
    for ax in axes:
        ax.set_facecolor("#0E1117")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    ax = axes[0]
    ax.plot(df["date"], df["sales"], color="#4C9BE8", linewidth=1.5, label="Actual")
    ax.plot(df["date"], y_hat, color="#F4A261", linewidth=1.5, linestyle="--", label="Fitted")
    ax.plot(future["date"], future["forecast"], color="#2DC97E", linewidth=2, label="Forecast")
    ax.fill_between(future["date"], future["forecast"] * 0.85, future["forecast"] * 1.15,
                    color="#2DC97E", alpha=0.15, label="+/-15% band")
    ax.set_title("Actual vs Fitted + Forecast", color="white", fontsize=13)
    ax.set_ylabel("Sales", color="white", fontsize=11)
    ax.legend(facecolor="#1C1F26", edgecolor="#444", labelcolor="white", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()

    ax2 = axes[1]
    residuals = df["sales"].values - y_hat
    ax2.bar(df["date"], residuals,
            color=["#E8574C" if r < 0 else "#4C9BE8" for r in residuals], width=5, alpha=0.7)
    ax2.axhline(0, color="white", linewidth=0.8, linestyle="--")
    ax2.set_title("Residuals (Actual - Fitted)", color="white", fontsize=11)
    ax2.set_ylabel("Residual", color="white", fontsize=10)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    plt.tight_layout()
    return fig


def plot_feature_importance(model):
    coefs = pd.Series(np.abs(model.coef_), index=FEATURE_COLS).sort_values(ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.barh(coefs.index, coefs.values, color="#4C9BE8")
    ax.set_title("Top Feature Importances (|coefficient|)", color="white", fontsize=12)
    plt.tight_layout()
    return fig


# UI

uploaded = st.file_uploader(
    "Upload your sales CSV (date column + sales column)",
    type=["csv"],
    help="The file should have at least one date column and one numeric sales column.",
)

if uploaded is not None:
    with st.spinner("Reading file..."):
        try:
            raw = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()

    st.subheader("Preview")
    st.dataframe(raw.head(10), use_container_width=True)

    date_col, sales_col = detect_columns(raw)

    col1, col2 = st.columns(2)
    with col1:
        date_col = st.selectbox("Date column", raw.columns.tolist(),
                                index=raw.columns.tolist().index(date_col) if date_col else 0)
    with col2:
        sales_col = st.selectbox("Sales column", raw.columns.tolist(),
                                 index=raw.columns.tolist().index(sales_col) if sales_col else 0)

    forecast_periods = st.slider("Forecast periods ahead", min_value=4, max_value=52, value=12, step=4)

    if st.button("Run Forecast", type="primary"):
        with st.spinner("Engineering features..."):
            df_feat = feature_engineering(raw, date_col, sales_col)

        if len(df_feat) < 20:
            st.error("Not enough data after feature engineering (need at least 20 rows). "
                     "Please upload a longer time series.")
            st.stop()

        with st.spinner("Training linear regression..."):
            model, y_hat, metrics = train_and_evaluate(df_feat)

        with st.spinner("Generating forecast..."):
            future = forecast_future(model, df_feat, periods=forecast_periods)

        st.subheader("Model Performance")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("MAE (CV)",   f"{metrics['MAE (CV)']:.1f}")
        mc2.metric("RMSE (CV)",  f"{metrics['RMSE (CV)']:.1f}")
        mc3.metric("R2 (CV)",    f"{metrics['R2 (CV)']:.3f}")
        mc4.metric("R2 (train)", f"{metrics['R2 (train)']:.3f}")

        st.subheader("Actual vs Fitted + Forecast")
        st.pyplot(plot_results(df_feat, y_hat, future), use_container_width=True)

        st.subheader("Feature Importance")
        st.pyplot(plot_feature_importance(model), use_container_width=True)

        st.subheader("Forecast Values")
        forecast_display = future[["date", "forecast"]].copy()
        forecast_display["forecast"] = forecast_display["forecast"].round(1)
        forecast_display.columns = ["Date", "Forecasted Sales"]
        st.dataframe(forecast_display, use_container_width=True)

        st.download_button(
            "Download Forecast CSV",
            forecast_display.to_csv(index=False).encode(),
            "forecast.csv",
            "text/csv"
        )
