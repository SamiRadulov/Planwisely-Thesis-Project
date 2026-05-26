import io
import warnings
import pathlib
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
from xgboost import XGBRegressor
import shap
from interpret.glassbox import ExplainableBoostingRegressor
warnings.filterwarnings("ignore")

LOGO   = "https://planwisely.ai/wp-content/uploads/2024/08/Planwisely01_1765553.png"
NAVY   = "#1B3A6B"
BLUE   = "#2563EB"
GREEN  = "#10B981"
RED    = "#EF4444"
AMBER  = "#F59E0B"
CARD   = "#FFFFFF"
BORDER = "#E2E8F0"

_HERE = pathlib.Path(__file__).parent
try:
    _WEATHER_RAW = pd.read_csv(_HERE / "weather_weekly.csv")
    _HOLIDAY_RAW = pd.read_csv(_HERE / "holiday_weekly.csv")
    HAS_EXTERNAL = True
    EXT_WEATHER_COLS = ["temp_mean","temp_min","temp_max","precip_sum",
                        "sunshine_sum","temp_anomaly","heavy_rain"]
    EXT_HOLIDAY_COLS = ["has_holiday","min_days_to_holiday","hol_ascension",
                        "hol_christmas","hol_easter","hol_kings_day",
                        "hol_liberation_day","hol_new_year","hol_pentecost"]
    _WEATHER_AVGS = _WEATHER_RAW.groupby("week")[EXT_WEATHER_COLS].mean().reset_index()
except Exception:
    _WEATHER_RAW = _HOLIDAY_RAW = _WEATHER_AVGS = None
    HAS_EXTERNAL = False
    EXT_WEATHER_COLS = []
    EXT_HOLIDAY_COLS = []

EXT_COLS   = EXT_WEATHER_COLS + EXT_HOLIDAY_COLS
FCOLS_BASE = [
    "trend","month","week_of_year","quarter",
    "is_month_start","is_month_end",
    "month_sin","month_cos","week_sin","week_cos",
    "lag_1","lag_2","lag_4","lag_8",
    "rolling_mean_4","rolling_mean_8","rolling_std_4","rolling_std_8",
]
FCOLS = FCOLS_BASE + EXT_COLS

FEAT_LABELS = {
    "lag_1":"Last period sales","lag_2":"Sales 2 periods ago",
    "lag_4":"Sales 4 periods ago","lag_8":"Sales 8 periods ago",
    "rolling_mean_4":"4-period avg","rolling_mean_8":"8-period avg",
    "rolling_std_4":"4-period volatility","rolling_std_8":"8-period volatility",
    "trend":"Long-term trend","month":"Month of year",
    "week_of_year":"Week of year","quarter":"Quarter",
    "is_month_start":"Month start","is_month_end":"Month end",
    "month_sin":"Seasonality (month)","month_cos":"Seasonality (month)",
    "week_sin":"Seasonality (week)","week_cos":"Seasonality (week)",
    "temp_mean":"Avg temperature","temp_min":"Min temperature",
    "temp_max":"Max temperature","precip_sum":"Precipitation",
    "sunshine_sum":"Sunshine hours","temp_anomaly":"Temp anomaly",
    "heavy_rain":"Heavy rain","has_holiday":"Holiday week",
    "min_days_to_holiday":"Days to holiday","hol_ascension":"Ascension Day",
    "hol_christmas":"Christmas","hol_easter":"Easter",
    "hol_kings_day":"King's Day","hol_liberation_day":"Liberation Day",
    "hol_new_year":"New Year","hol_pentecost":"Pentecost",
}

st.set_page_config(page_title="Planwisely - Sales Forecast", page_icon="??", layout="wide")

st.markdown(f"""
<style>
  .stApp {{ background:{NAVY}; }}
  .block-container {{ padding:0.9rem 2rem 2rem; max-width:100%; }}

  .stButton>button {{
      background:{BLUE}!important; color:#fff!important; border:none!important;
      border-radius:8px!important; padding:0.45rem 1.2rem!important;
      font-weight:600!important; font-size:15px!important; width:100%;
      height:52px!important;
  }}
  .stButton>button:hover   {{ background:#1D4ED8!important; }}
  .stButton>button:disabled {{ background:#4B6A9B!important; opacity:0.6!important; }}

  /* FILE UPLOADER */
  .stFileUploader {{
      background: transparent !important;
      padding: 0 !important;
  }}

  .stFileUploader section {{
      padding: 0 !important;
      min-height: 52px !important;
      height: 52px !important;
      border-radius: 8px !important;
      display: flex !important;
      align-items: center !important;
  }}

  [data-testid="stFileUploaderDropzone"] {{
      padding: 0 10px !important;
      min-height: 52px !important;
      height: 52px !important;
      display: flex !important;
      align-items: center !important;
      gap: 10px !important;
      width: 100% !important;
  }}

  [data-testid="stFileUploaderDropzone"] svg {{
      display: none !important;
  }}

  [data-testid="stFileUploaderDropzone"] span,
  [data-testid="stFileUploaderDropzone"] small {{
      font-size: 13px !important;
      line-height: 1.2 !important;
  }}

  [data-testid="stFileUploaderFile"] {{
      margin-top: 0 !important;
      padding: 4px 8px !important;
      min-height: 38px !important;
      height: 38px !important;
      display: flex !important;
      align-items: center !important;
  }}

  [data-testid="stFileUploaderFile"] > div {{
      padding-top: 0 !important;
      margin-top: 0 !important;
      align-items: center !important;
  }}

  [data-testid="stFileUploaderFile"] svg {{
      margin-top: 0 !important;
      align-self: center !important;
  }}

  [data-testid="stFileUploaderFileName"] {{
      padding-top: 0 !important;
      margin-top: 0 !important;
      line-height: 1.1 !important;
  }}

  /* SELECTBOXES */
  div[data-baseweb="select"] {{
      min-height: 52px !important;
  }}
  div[data-baseweb="select"] > div:first-child {{
      min-height: 52px !important;
      height: 52px !important;
      align-items: center !important;
  }}

  .tile {{
      background:{CARD}; border-radius:12px;
      padding:18px 22px; border:1px solid {BORDER}; height:100%;
  }}
  .tile-label {{
      font-size:12px; font-weight:700; letter-spacing:.12em;
      text-transform:uppercase; color:#94A3B8; margin-bottom:2px;
  }}
  .tile-title {{
      font-size:17px; font-weight:700; color:{NAVY}; margin-bottom:10px;
  }}
  .tile-divider {{ border:none; border-top:1px solid {BORDER}; margin:14px 0; }}

  .kpi-row {{ display:flex; gap:0; }}
  .kpi-cell {{
      flex:1; padding-right:14px; margin-right:14px;
      border-right:1px solid {BORDER};
  }}
  .kpi-cell:last-child {{ border-right:none; padding-right:0; margin-right:0; }}
  .kpi-num {{ font-size:33px; font-weight:800; color:{NAVY}; line-height:1; }}
  .kpi-sub {{ font-size:13px; color:#64748B; margin-top:3px; }}

  .ptile {{
      background:{CARD}; border-radius:12px; border:1px solid {BORDER};
      padding:18px 14px; text-align:center; height:100%;
      display:flex; flex-direction:column; align-items:center; justify-content:center;
  }}
  .ptile-period {{ font-size:12px; font-weight:700; letter-spacing:.12em;
      text-transform:uppercase; color:#94A3B8; margin-bottom:4px; }}
  .ptile-date   {{ font-size:14px; color:#64748B; margin-bottom:8px; }}
  .ptile-num    {{ font-size:35px; font-weight:800; color:{NAVY}; line-height:1; }}
  .ptile-sub    {{ font-size:13px; color:#64748B; margin-top:4px; margin-bottom:8px; }}

  .badge {{ display:inline-block; padding:3px 9px; border-radius:9999px;
            font-size:13px; font-weight:700; }}
  .b-up   {{ background:#D1FAE5; color:#065F46; }}
  .b-down {{ background:#FEE2E2; color:#991B1B; }}
  .b-flat {{ background:#FEF3C7; color:#78350F; }}

  .tbl {{ width:100%; border-collapse:collapse; font-size:14px; }}
  .tbl th {{ padding:7px 10px; background:{NAVY}; color:#fff;
             font-weight:600; text-align:left; font-size:13px; }}
  .tbl td {{ padding:5px 10px; border-bottom:1px solid {BORDER}; color:#374151; }}
  .tbl tr:last-child td {{ border-bottom:none; }}

  .div  {{ border:none; border-top:1px solid rgba(255,255,255,0.15); margin:10px 0; }}
  .note {{ font-size:13px; color:#94A3B8; margin-top:6px; }}
  label {{ color:rgba(255,255,255,0.85) !important; }}
  .stSelectbox > div > div {{ background:{CARD}; border-radius:8px; }}

  .sec-head {{
      text-align:center; color:#fff;
      font-size:24px; font-weight:800; letter-spacing:0.03em;
      margin:18px 0 4px;
  }}
  .sec-sub {{
      text-align:center; color:rgba(255,255,255,0.5);
      font-size:13px; margin-bottom:14px;
  }}
</style>
""", unsafe_allow_html=True)

ext_pill = ('<span style="background:#10B981;color:#fff;font-size:9px;font-weight:700;'
            'padding:2px 8px;border-radius:9999px;margin-left:10px;vertical-align:middle">'
            'Weather and Holiday data loaded</span>') if HAS_EXTERNAL else ""
st.markdown(f"""
<div style="display:flex;align-items:center;gap:16px;padding:2px 0 12px">
  <img src="{LOGO}" style="width:140px;height:auto;filter:brightness(0) invert(1);">
  <div>
    <div style="font-size:24px;font-weight:800;color:#fff;line-height:1.1">
      Sales Demand Forecast{ext_pill}
    </div>
    <div style="font-size:13px;color:rgba(255,255,255,0.5);margin-top:3px">
      Upload your sales CSV - feature engineering, external data merging, and forecasting run automatically.
    </div>
  </div>
</div>
<hr class='div'>
""", unsafe_allow_html=True)

def detect_structure(df):
    cl = {c: c.lower() for c in df.columns}
    product_col = next((c for c,l in cl.items() if l in ("product_id","product","sku","item_id")), None)
    sales_col   = next((c for c,l in cl.items() if l in ("sku_sold","sales","quantity","units","revenue","amount")), None)
    if sales_col is None:
        sales_col = next((c for c in df.columns if c != product_col and pd.api.types.is_numeric_dtype(df[c])), None)
    year_col  = next((c for c,l in cl.items() if l == "year"),  None)
    month_col = next((c for c,l in cl.items() if l == "month"), None)
    day_col   = next((c for c,l in cl.items() if l in ("day","day_of_month")), None)
    if year_col and month_col and day_col:
        return dict(date_mode="split", date_col=None, year_col=year_col,
                    month_col=month_col, day_col=day_col,
                    sales_col=sales_col, product_col=product_col)
    date_col = None
    for c in df.columns:
        if c in (product_col, sales_col): continue
        try:
            if pd.to_datetime(df[c], infer_datetime_format=True, errors="coerce").notna().mean() > 0.8:
                date_col = c; break
        except Exception: pass
    return dict(date_mode="single", date_col=date_col, year_col=None,
                month_col=None, day_col=None, sales_col=sales_col, product_col=product_col)

def build_date_series(df, info):
    if info["date_mode"] == "split":
        return pd.to_datetime(dict(year=df[info["year_col"]],
                                   month=df[info["month_col"]], day=df[info["day_col"]]))
    return pd.to_datetime(df[info["date_col"]], infer_datetime_format=True, errors="coerce")

def feature_engineering(df_in):
    df = df_in[["date","sales"]].copy().sort_values("date").reset_index(drop=True)
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    df = df.dropna(subset=["sales"])
    df["year"]           = df["date"].dt.isocalendar().year.astype(int)
    df["month"]          = df["date"].dt.month
    df["week_of_year"]   = df["date"].dt.isocalendar().week.astype(int)
    df["day_of_week"]    = df["date"].dt.dayofweek
    df["is_month_start"] = (df["date"].dt.day <= 7).astype(int)
    df["is_month_end"]   = (df["date"].dt.day >= 24).astype(int)
    df["quarter"]        = df["date"].dt.quarter
    df["month_sin"] = np.sin(2*np.pi*df["month"]/12)
    df["month_cos"] = np.cos(2*np.pi*df["month"]/12)
    df["week_sin"]  = np.sin(2*np.pi*df["week_of_year"]/52)
    df["week_cos"]  = np.cos(2*np.pi*df["week_of_year"]/52)
    for lag in [1,2,4,8]:
        df[f"lag_{lag}"] = df["sales"].shift(lag)
    for w in [4,8]:
        df[f"rolling_mean_{w}"] = df["sales"].shift(1).rolling(w,min_periods=1).mean()
        df[f"rolling_std_{w}"]  = df["sales"].shift(1).rolling(w,min_periods=1).std().fillna(0)
    df["trend"] = np.arange(len(df))
    if HAS_EXTERNAL:
        wdf = _WEATHER_RAW.rename(columns={"week":"week_of_year"})
        df  = df.merge(wdf[["year","week_of_year"]+EXT_WEATHER_COLS], on=["year","week_of_year"], how="left")
        hdf = _HOLIDAY_RAW.rename(columns={"week":"week_of_year"})
        df  = df.merge(hdf[["year","week_of_year"]+EXT_HOLIDAY_COLS], on=["year","week_of_year"], how="left")
        for col in EXT_WEATHER_COLS:
            if col in df.columns: df[col] = df[col].fillna(df[col].mean())
        for col in EXT_HOLIDAY_COLS:
            if col in df.columns: df[col] = df[col].fillna(0)
    return df.dropna(subset=["lag_1","lag_2","lag_4","lag_8"])

def _ext_for_date(nd):
    if not HAS_EXTERNAL: return {}
    week = int(nd.isocalendar()[1]); year = int(nd.isocalendar()[0])
    r = {}
    w_row = _WEATHER_AVGS[_WEATHER_AVGS["week"] == week]
    for col in EXT_WEATHER_COLS:
        r[col] = float(w_row[col].values[0]) if len(w_row) > 0 else 0.0
    h_row = _HOLIDAY_RAW[(_HOLIDAY_RAW["year"]==year) & (_HOLIDAY_RAW["week"]==week)]
    for col in EXT_HOLIDAY_COLS:
        r[col] = float(h_row[col].values[0]) if len(h_row) > 0 else 0.0
    return r

def get_feature_importance(model, fcols):
    support = model.named_steps["var"].get_support()
    kept    = [f for f, s in zip(fcols, support) if s]
    reg     = model.named_steps["reg"]
    if isinstance(reg, XGBRegressor):
        scores = reg.get_booster().get_score(importance_type="gain")
        result = [(f, scores.get(f"f{i}", 0.0)) for i, f in enumerate(kept)]
        return sorted(result, key=lambda x: abs(x[1]), reverse=True)
    if isinstance(reg, ExplainableBoostingRegressor):
        data   = reg.explain_global().data()
        # With set_output("pandas"), EBM uses real feature names directly
        names  = data["names"]
        scores = [float(s) for s in data["scores"]]
        return sorted(zip(names, scores), key=lambda x: abs(x[1]), reverse=True)
    coef = reg.coef_
    return sorted([(f, float(c)) for f, c in zip(kept, coef)],
                  key=lambda x: abs(x[1]), reverse=True)

def train_model(df, model_type="Ridge"):
    fcols = [c for c in FCOLS if c in df.columns]
    if model_type == "Lasso":
        reg = Lasso(alpha=1.0, max_iter=10000)
    elif model_type == "XGBoost":
        reg = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                           subsample=0.8, colsample_bytree=0.8, random_state=42,
                           verbosity=0)
    elif model_type == "EBM":
        # EBM is a tree-based method — no StandardScaler needed.
        # set_output("pandas") ensures VarianceThreshold passes a named DataFrame
        # so EBM picks up real feature names in explain_global().
        var = VarianceThreshold(threshold=0.0).set_output(transform="pandas")
        reg = ExplainableBoostingRegressor(random_state=42, n_jobs=-1,
                                           max_bins=128, interactions=5,
                                           max_rounds=500, learning_rate=0.05)
        model = Pipeline([("var", var), ("reg", reg)])
        model.fit(df[fcols], df["sales"])
        return model, fcols
    else:
        reg = Ridge(alpha=10.0)
    model = Pipeline([
        ("var",    VarianceThreshold(threshold=0.0)),
        ("scaler", StandardScaler()),
        ("reg",    reg),
    ])
    model.fit(df[fcols], df["sales"])
    return model, fcols

<<<<<<< HEAD
=======
def compute_shap(model, fcols, df):
    """Compute SHAP values for an XGBoost pipeline. Returns (pairs, shap_vals, kept_features)."""
    support  = model.named_steps["var"].get_support()
    kept     = [f for f, s in zip(fcols, support) if s]
    X_var    = model.named_steps["var"].transform(df[fcols])
    X_scaled = model.named_steps["scaler"].transform(X_var)
    explainer  = shap.TreeExplainer(model.named_steps["reg"])
    shap_vals  = explainer.shap_values(X_scaled)
    mean_abs   = np.abs(shap_vals).mean(axis=0)
    pairs = sorted(zip(kept, mean_abs.tolist()), key=lambda x: x[1], reverse=True)
    return pairs, shap_vals, kept

def generate_shap_text(pairs, fcast, hist, sel_pid):
    """Produce a natural-language explanation from SHAP feature importances."""
    fc_avg   = fcast["forecast"].mean()
    prev_avg = hist["sales"].iloc[-4:].mean() if len(hist) >= 4 else hist["sales"].mean()
    pct      = (fc_avg - prev_avg) / prev_avg * 100 if prev_avg else 0
    direction = "increase" if pct > 0 else "decrease"

    top3 = [FEAT_LABELS.get(f, f) for f, _ in pairs[:3]]
    if len(top3) == 1:
        drivers_str = top3[0]
    else:
        drivers_str = ", ".join(top3[:-1]) + f", and {top3[-1]}"

    text = (f"Demand for Product {sel_pid} is forecast to <b>{direction} "
            f"by {abs(pct):.1f}%</b> versus the previous 4 periods. "
            f"The primary drivers identified by the model are: <b>{drivers_str}</b>. ")

    lag_feats     = [f for f, _ in pairs[:6] if "lag" in f]
    rolling_feats = [f for f, _ in pairs[:6] if "rolling" in f]
    weather_feats = [f for f, _ in pairs[:6] if f in EXT_WEATHER_COLS]
    holiday_feats = [f for f, _ in pairs[:6] if f in EXT_HOLIDAY_COLS]
    trend_feats   = [f for f, _ in pairs[:6] if f == "trend"]

    if lag_feats or rolling_feats:
        text += "Recent sales history is the strongest signal — the model is largely extrapolating momentum from prior periods. "
    if weather_feats:
        wlabel = FEAT_LABELS.get(weather_feats[0], weather_feats[0]).lower()
        text += f"Weather conditions (especially {wlabel}) are contributing meaningfully to the forecast. "
    if holiday_feats:
        text += "Upcoming or recent public holidays are also factored into the prediction. "
    if trend_feats:
        text += "A detectable long-term trend in this product's sales is shaping the outlook. "

    return text

>>>>>>> dashboard
def forecast_4(model, fcols, df):
    delta    = df["date"].iloc[-1] - df["date"].iloc[-2]
    hist     = list(df["sales"]); rows = []
    hist_max = df["sales"].max() * 3
    hist_p10 = max(0, df["sales"].quantile(0.10) * 0.5)
    for i in range(4):
        nd = df["date"].iloc[-1] + (i+1)*delta
        r  = {"date":nd,"year":int(nd.isocalendar()[0]),"month":nd.month,
              "week_of_year":int(nd.isocalendar()[1]),"day_of_week":nd.dayofweek,
              "is_month_start":int(nd.day<=7),"is_month_end":int(nd.day>=24),
              "quarter":nd.quarter,"trend":len(df)+i}
        r["month_sin"] = np.sin(2*np.pi*r["month"]/12)
        r["month_cos"] = np.cos(2*np.pi*r["month"]/12)
        r["week_sin"]  = np.sin(2*np.pi*r["week_of_year"]/52)
        r["week_cos"]  = np.cos(2*np.pi*r["week_of_year"]/52)
        for lag in [1,2,4,8]:
            r[f"lag_{lag}"] = hist[-lag] if len(hist)>=lag else np.nan
        for w in [4,8]:
            h2=hist[-w:]; r[f"rolling_mean_{w}"]=np.mean(h2)
            r[f"rolling_std_{w}"]=np.std(h2) if len(h2)>1 else 0.0
        r.update(_ext_for_date(nd))
<<<<<<< HEAD
        pred = model.predict(np.array([[r[c] for c in fcols]]))[0]
=======
        pred = model.predict(pd.DataFrame([[r[c] for c in fcols]], columns=fcols))[0]
>>>>>>> dashboard
        pred = float(np.clip(pred, hist_p10, hist_max))
        hist.append(pred); r["forecast"]=pred; rows.append(r)
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False)
def run_all(raw_bytes, sales_col, date_mode, date_col, year_col, month_col,
            day_col, product_col, model_type):
    df   = pd.read_csv(io.BytesIO(raw_bytes))
    info = dict(date_mode=date_mode, date_col=date_col, year_col=year_col,
                month_col=month_col, day_col=day_col, product_col=product_col)
    out  = {}
    pids = [None] if product_col is None else \
           df.groupby(product_col).filter(lambda x: len(x)>=20)[product_col].unique().tolist()
    for pid in pids:
        sub = (df if pid is None else df[df[product_col]==pid]).copy()
        sub["date"]  = build_date_series(sub, info)
        sub["sales"] = pd.to_numeric(sub[sales_col], errors="coerce")
        feat = feature_engineering(sub[["date","sales"]].dropna())
        if len(feat) < 20: continue
        mdl, fcols = train_model(feat, model_type)
        fcast      = forecast_4(mdl, fcols, feat)
        r2_val     = float(r2_score(feat["sales"], mdl.predict(feat[fcols])))
        imp        = get_feature_importance(mdl, fcols)
<<<<<<< HEAD
        out[pid if pid is not None else "all"] = {
            "history":feat, "forecast":fcast, "importance":imp, "r2":r2_val,
=======
        xai_data = None
        if model_type == "XGBoost":
            shap_pairs, shap_vals, shap_kept = compute_shap(mdl, fcols, feat)
            xai_data = {"type": "shap", "pairs": shap_pairs,
                        "vals": shap_vals, "kept": shap_kept}
        elif model_type == "EBM":
            ebm_pairs = list(get_feature_importance(mdl, fcols))
            xai_data  = {"type": "ebm", "pairs": ebm_pairs}
        out[pid if pid is not None else "all"] = {
            "history":feat, "forecast":fcast, "importance":imp,
            "r2":r2_val, "xai":xai_data,
>>>>>>> dashboard
        }
    return out

def white_ax(figsize=(9,3)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(CARD); ax.set_facecolor(CARD)
    ax.tick_params(colors="#374151", labelsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
    return fig, ax

def badge(pc):
    if pc >  5: return f"<span class='badge b-up'>+{pc:.1f}%</span>"
    if pc < -5: return f"<span class='badge b-down'>{pc:.1f}%</span>"
    return             f"<span class='badge b-flat'>{pc:+.1f}%</span>"

def arrow(pc):
    if pc >  5: return f"<span style='color:{GREEN};font-weight:800'>up</span>"
    if pc < -5: return f"<span style='color:{RED};font-weight:800'>down</span>"
    return             f"<span style='color:{AMBER};font-weight:800'>flat</span>"

# UPLOAD ROW
uL, uM, uR, uRR = st.columns([2.3, 1, 1, 0.7], gap="medium")
with uL:
    uploaded = st.file_uploader("Upload sales CSV", type=["csv"], label_visibility="visible")
with uM:
<<<<<<< HEAD
    model_type_sel = st.selectbox("Model", ["Ridge", "Lasso"], index=0)
=======
    model_type_sel = st.selectbox("Model", ["Ridge", "Lasso", "XGBoost", "EBM"], index=0)
>>>>>>> dashboard
with uR:
    sales_col_sel = None
    info = {}
    if uploaded:
        raw  = pd.read_csv(uploaded)
        info = detect_structure(raw)
        opts = raw.columns.tolist()
        sales_col_sel = st.selectbox(
            "Sales column", opts,
            index=opts.index(info["sales_col"]) if info.get("sales_col") in opts else 0
        )
with uRR:
    st.markdown("<div style='padding-top:24px'>", unsafe_allow_html=True)
    run_btn = st.button("Run", type="primary", disabled=(uploaded is None),
                        use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

if uploaded and run_btn:
    st.session_state.update(rb=uploaded.getvalue(), sc=sales_col_sel,
                             inf=info, mtype=model_type_sel)

if "rb" in st.session_state and uploaded is not None:
    _mt = model_type_sel
    if st.session_state.get("mtype") != _mt:
        st.session_state["mtype"] = _mt
    feat_note = "time-series + weather + holiday" if HAS_EXTERNAL else "time-series"
    with st.spinner(f"Running {_mt} on {feat_note} features..."):
        results = run_all(
            st.session_state["rb"], st.session_state["sc"],
            st.session_state["inf"]["date_mode"], st.session_state["inf"]["date_col"],
            st.session_state["inf"]["year_col"],  st.session_state["inf"]["month_col"],
            st.session_state["inf"]["day_col"],   st.session_state["inf"]["product_col"],
            _mt,
        )
    st.session_state["results"] = results

if st.session_state.get("results"):
    results = st.session_state["results"]
    if not results:
        st.error("No products with 20 or more usable records found."); st.stop()

    smry_rows = []
    for pid, r in results.items():
        f, h  = r["forecast"], r["history"]
        ftot  = f["forecast"].sum(); favg = f["forecast"].mean()
        prev  = h["sales"].iloc[-4:].mean() if len(h)>=4 else h["sales"].mean()
        pct   = (favg-prev)/prev*100 if prev else 0
        smry_rows.append(dict(product_id=pid, forecast_total=ftot,
                              forecast_avg=favg, prev_avg=prev, pct_change=pct,
                              r2=r["r2"]))
    smry        = pd.DataFrame(smry_rows)
    n_prod      = len(smry)
    total_dem   = smry["forecast_total"].sum()
    avg_prod    = smry["forecast_avg"].mean()
    period_tots = np.array([
        sum(r["forecast"]["forecast"].iloc[i] for r in results.values())
        for i in range(4)
    ])
    up_c = (smry["pct_change"] >  5).sum()
    dn_c = (smry["pct_change"] < -5).sum()
    st_c = n_prod - up_c - dn_c

    # SECTION 1 — PRODUCT DETAIL
    st.markdown("<div class='sec-head'>Product Detail</div>", unsafe_allow_html=True)

    all_pids = sorted(results.keys(), key=lambda x: str(x))
    sel_pid  = st.selectbox("Select a product", options=all_pids,
                            format_func=lambda x: f"Product {x}")

    if sel_pid is not None:
        pr       = results[sel_pid]
        hist     = pr["history"]
        fcast    = pr["forecast"]
        imp      = pr["importance"]
        r2_val   = pr["r2"]
        prev_avg = hist["sales"].iloc[-4:].mean() if len(hist)>=4 else hist["sales"].mean()
        fc_avg   = fcast["forecast"].mean()
        ov_pct   = (fc_avg - prev_avg) / prev_avg * 100 if prev_avg else 0

        p1, p2, p3, p4 = st.columns(4, gap="medium")
        for col, i in zip([p1,p2,p3,p4], range(4)):
            row      = fcast.iloc[i]
            date_str = pd.to_datetime(row["date"]).strftime("%d %b %Y")
            dp       = (row["forecast"] - prev_avg) / prev_avg * 100 if prev_avg else 0
            tcls     = "b-up" if dp > 5 else ("b-down" if dp < -5 else "b-flat")
            tsign    = f"+{dp:.1f}%" if dp > 0 else f"{dp:.1f}%"
            col.markdown(f"""
            <div class='ptile'>
              <div class='ptile-period'>Period {i+1}</div>
              <div class='ptile-date'>{date_str}</div>
              <div class='ptile-num'>{row['forecast']:,.0f}</div>
              <div class='ptile-sub'>units forecasted</div>
              <span class='badge {tcls}'>{tsign} vs prev 4</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="text-align:center;margin-bottom:12px">
          <div style="font-size:24px;font-weight:800;color:#fff;">
            Product {sel_pid} - History + Forecast
          </div>
        </div>""", unsafe_allow_html=True)

        chart_col, xai_col = st.columns([6, 4], gap="medium")

        with chart_col:
            fig2, ax2 = white_ax((9, 3.8))
            tail = hist.tail(52)
            ax2.plot(tail["date"], tail["sales"], color=NAVY, lw=1.8, label="Historical", zorder=3)
            ax2.plot(fcast["date"], fcast["forecast"], color=BLUE, lw=2.2,
                     marker="o", markersize=5, label="Forecast", zorder=4)
            ax2.fill_between(fcast["date"],
                             fcast["forecast"]*0.85, fcast["forecast"]*1.15,
                             color=BLUE, alpha=0.12, label="15% band")
            ax2.axvline(hist["date"].iloc[-1], color=BORDER, lw=1.2, linestyle="--")
            ax2.yaxis.grid(True, color=BORDER, zorder=0); ax2.set_axisbelow(True)
            ax2.set_ylabel("Units", color="#64748B", fontsize=9)
            ax2.legend(facecolor=CARD, edgecolor=BORDER, fontsize=8)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            fig2.autofmt_xdate()
            plt.tight_layout(pad=0.4)
            st.pyplot(fig2, use_container_width=True)

        with xai_col:
<<<<<<< HEAD
            st.markdown(f"""
            <div class='tile' style='min-height:300px'>
              <div style='font-size:20px;font-weight:800;color:{NAVY};
                          border-bottom:2px solid {BORDER};padding-bottom:10px;
                          margin-bottom:10px'>
                Explainability
              </div>
            </div>""", unsafe_allow_html=True)
=======
            xai_data = pr.get("xai")
            if xai_data and _mt in ("XGBoost", "EBM"):
                pairs   = xai_data["pairs"]
                top10   = pairs[:10]
                feat_names_xai  = [FEAT_LABELS.get(f, f) for f, _ in top10][::-1]
                xai_scores      = [float(v) for _, v in top10][::-1]
                xai_label       = "Mean |SHAP value|" if _mt == "XGBoost" else "EBM Feature Importance"
                imp_title       = "Top Feature Importances (SHAP)" if _mt == "XGBoost" \
                                  else "Top Feature Importances (EBM)"

                fig_xai = go.Figure(go.Bar(
                    x=xai_scores, y=feat_names_xai,
                    orientation="h",
                    marker_color=BLUE,
                    text=[f"{v:.3f}" for v in xai_scores],
                    textposition="outside",
                    textfont=dict(size=11, color=NAVY),
                    cliponaxis=False,
                ))
                fig_xai.update_layout(
                    paper_bgcolor=CARD, plot_bgcolor=CARD,
                    height=320,
                    margin=dict(l=10, r=50, t=10, b=30),
                    xaxis=dict(title=xai_label,
                               title_font=dict(size=11, color="#64748B"),
                               tickfont=dict(size=10, color="#374151"),
                               gridcolor=BORDER, zeroline=False),
                    yaxis=dict(tickfont=dict(size=11, color=NAVY), showgrid=False),
                    showlegend=False,
                )

                xai_text = generate_shap_text(pairs, fcast, hist, sel_pid)

                st.markdown(f"""
                <div class='tile' style='min-height:300px'>
                  <div style='font-size:20px;font-weight:800;color:{NAVY};
                              border-bottom:2px solid {BORDER};padding-bottom:10px;
                              margin-bottom:12px'>
                    Explainability
                  </div>
                  <p style='font-size:13px;color:#374151;line-height:1.6;margin-bottom:14px'>
                    {xai_text}
                  </p>
                  <div style='font-size:12px;font-weight:700;color:#94A3B8;
                              letter-spacing:.1em;text-transform:uppercase;margin-bottom:6px'>
                    {imp_title}
                  </div>
                </div>""", unsafe_allow_html=True)
                st.plotly_chart(fig_xai, use_container_width=True)
            else:
                st.markdown(f"""
                <div class='tile' style='min-height:300px'>
                  <div style='font-size:20px;font-weight:800;color:{NAVY};
                              border-bottom:2px solid {BORDER};padding-bottom:10px;
                              margin-bottom:10px'>
                    Explainability
                  </div>
                  <p style='font-size:12px;color:#94A3B8;margin-top:8px'>
                    Select <b>XGBoost</b> or <b>EBM</b> as the model and re-run to see explanations.
                  </p>
                </div>""", unsafe_allow_html=True)
>>>>>>> dashboard

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # SECTION 2 — SUMMARY AND ANALYSIS
    st.markdown("""
    <div class='sec-head'>Summary and Analysis</div>
    <div class='sec-sub'>All products - Next 4 forecast periods</div>
    """, unsafe_allow_html=True)

    # Row 1: Overview+Prioritisation | Total Demand chart
    r1c1, r1c2 = st.columns([1, 1], gap="medium")

    with r1c1:
        top5     = smry.nlargest(5,"forecast_total").reset_index(drop=True)
        rows_top = "".join(f"""
          <tr>
            <td><b>#{i+1}</b></td>
            <td>Product {r['product_id']}</td>
            <td>{r['forecast_total']:,.0f}</td>
            <td>{arrow(r['pct_change'])}</td>
          </tr>""" for i, (_, r) in enumerate(top5.iterrows()))
        st.markdown(f"""
        <div class='tile' style='min-height:560px'>
          <div class='tile-label'>Overview</div>
          <div class='tile-title'>Key Numbers</div>
          <div class='kpi-row'>
            <div class='kpi-cell'>
              <div class='kpi-num'>{n_prod}</div>
              <div class='kpi-sub'>Products forecasted</div>
            </div>
            <div class='kpi-cell'>
              <div class='kpi-num'>{total_dem:,.0f}</div>
              <div class='kpi-sub'>Total demand (4 periods)</div>
            </div>
            <div class='kpi-cell'>
              <div class='kpi-num'>{avg_prod:,.0f}</div>
              <div class='kpi-sub'>Avg / product / period</div>
            </div>
          </div>
          <div class='kpi-row' style='margin-top:10px'>
            <div class='kpi-cell'>
              <span class='badge b-up'>{up_c} up</span>
              <div class='kpi-sub' style='margin-top:4px'>Trending up</div>
            </div>
            <div class='kpi-cell'>
              <span class='badge b-flat'>{st_c} flat</span>
              <div class='kpi-sub' style='margin-top:4px'>Stable</div>
            </div>
            <div class='kpi-cell'>
              <span class='badge b-down'>{dn_c} down</span>
              <div class='kpi-sub' style='margin-top:4px'>Trending down</div>
            </div>
          </div>
          <hr class='tile-divider'>
          <div class='tile-label'>Prioritisation</div>
          <div class='tile-title'>Top 5 by Forecasted Volume (Total units across all 4 periods)</div>
          <table class='tbl'>
            <tr><th>#</th><th>Product</th><th>Total forecast</th><th>Trend</th></tr>
            {rows_top}
          </table>
        </div>""", unsafe_allow_html=True)

    with r1c2:
        plabels = ["Period 1", "Period 2", "Period 3", "Period 4"]
        ymax = max(period_tots) * 1.18

        fig_pl = go.Figure(go.Bar(
            x=plabels,
            y=period_tots,
            marker_color=BLUE,
            width=0.45,
            text=[f"{v:,.0f}" for v in period_tots],
            textposition="outside",
            textfont=dict(size=14, color=NAVY, family="Arial Black"),
            cliponaxis=False,
        ))

        fig_pl.update_layout(
            paper_bgcolor=CARD,
            plot_bgcolor=CARD,
            height=560,
            margin=dict(l=60, r=20, t=90, b=40),
            yaxis=dict(
                title="Units",
                title_font=dict(size=12, color="#64748B"),
                tickfont=dict(size=12, color="#374151"),
                gridcolor=BORDER,
                range=[0, ymax],
                zeroline=False,
            ),
            xaxis=dict(
                tickfont=dict(size=14, color=NAVY),
                showgrid=False,
            ),
            annotations=[
                dict(x=0.0, y=1.13, xref="paper", yref="paper",
                     text="FORECAST", showarrow=False,
                     font=dict(size=10, color="#94A3B8", family="Arial"),
                     xanchor="left"),
                dict(x=0.0, y=1.07, xref="paper", yref="paper",
                     text="Total Demand - Next 4 Periods", showarrow=False,
                     font=dict(size=14, color=NAVY, family="Arial Black"),
                     xanchor="left"),
            ],
            showlegend=False,
        )

        st.plotly_chart(fig_pl, use_container_width=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Row 2: Biggest Movers | Period Trend
    r2c1, r2c2 = st.columns(2, gap="medium")

    with r2c1:
        movers  = pd.concat([smry.nlargest(5,"pct_change"),
                              smry.nsmallest(5,"pct_change")]) \
                    .drop_duplicates("product_id").sort_values("pct_change", ascending=False)
        rows_mv = "".join(f"""
          <tr>
            <td><b>#{str(r['product_id'])}</b></td>
            <td>{r['prev_avg']:.0f}</td>
            <td>{r['forecast_avg']:.0f}</td>
            <td>{badge(r['pct_change'])}</td>
          </tr>""" for _, r in movers.iterrows())
        st.markdown(f"""
        <div class='tile' style='min-height:560px'>
          <div class='tile-label'>Analysis</div>
          <div class='tile-title'>Biggest Movers vs Previous Period</div>
          <table class='tbl'>
            <tr><th>Product</th><th>Prev 4 avg</th><th>Forecast avg</th><th>Change</th></tr>
            {rows_mv}
          </table>
        </div>""", unsafe_allow_html=True)

    with r2c2:
        p_changes = [None] + [
            (period_tots[i] - period_tots[i-1]) / period_tots[i-1] * 100
            for i in range(1, 4)
        ]
        rows_pt = ""
        for i, (v, ch) in enumerate(zip(period_tots, p_changes)):
            chg_html = ("<span style='color:#94A3B8;font-size:10px'>baseline</span>"
                        if ch is None else badge(ch))
            rows_pt += f"""
          <tr>
            <td><b>Period {i+1}</b></td>
            <td>{v:,.0f}</td>
            <td>{chg_html}</td>
          </tr>"""

        p1v, p4v = period_tots[0], period_tots[3]
        traj_pct = (p4v - p1v) / p1v * 100 if p1v else 0
        traj_lbl = "Growing" if traj_pct > 2 else ("Declining" if traj_pct < -2 else "Flat")
        traj_cls = "b-up" if traj_pct > 2 else ("b-down" if traj_pct < -2 else "b-flat")
        tsign    = f"+{traj_pct:.1f}%" if traj_pct > 0 else f"{traj_pct:.1f}%"

        # min-height set to match Biggest Movers tile (10 rows ~ 420px)
        st.markdown(f"""
        <div class='tile' style='min-height:560px'>
          <div class='tile-label'>Demand Trajectory</div>
          <div class='tile-title'>Period-by-Period Trend</div>
          <div style='margin-bottom:12px'>
            <span class='badge {traj_cls}' style='font-size:14px;padding:4px 12px'>
              {traj_lbl} &nbsp; P1 to P4: {tsign}
            </span>
          </div>
          <table class='tbl'>
            <tr><th>Period</th><th>Total Demand</th><th>vs Prev Period</th></tr>
            {rows_pt}
          </table>
          <p class='note'>Compares each forecast period to the previous one across all products.</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    dl = smry[["product_id","forecast_total","forecast_avg","prev_avg","pct_change"]].copy()
    dl.columns = ["Product ID","Total Forecast (4 periods)","Avg per Period",
                  "Prev 4 Period Avg","% Change"]
    st.download_button(
        "Download Full Forecast CSV",
        dl.round(1).to_csv(index=False).encode(),
        "planwisely_forecast.csv", "text/csv",
    )
