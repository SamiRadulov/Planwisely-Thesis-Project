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
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
warnings.filterwarnings("ignore")

# ── Brand ──────────────────────────────────────────────────────────────────────
LOGO   = "https://planwisely.ai/wp-content/uploads/2024/08/Planwisely01_1765553.png"
NAVY   = "#1B3A6B"
BLUE   = "#2563EB"
GREEN  = "#10B981"
RED    = "#EF4444"
AMBER  = "#F59E0B"
CARD   = "#FFFFFF"
BGCARD = "#F1F5F9"
BORDER = "#E2E8F0"

# ── External data (weather + holidays) ────────────────────────────────────────
_HERE = pathlib.Path(__file__).parent
try:
    _WEATHER_RAW = pd.read_csv(_HERE / "weather_weekly.csv")
    _HOLIDAY_RAW = pd.read_csv(_HERE / "holiday_weekly.csv")
    HAS_EXTERNAL = True
    EXT_WEATHER_COLS = ["temp_mean","temp_min","temp_max","precip_sum","sunshine_sum","temp_anomaly","heavy_rain"]
    EXT_HOLIDAY_COLS = ["has_holiday","min_days_to_holiday","hol_ascension","hol_christmas",
                        "hol_easter","hol_kings_day","hol_liberation_day","hol_new_year","hol_pentecost"]
    # Weekly historical averages for weather (used when forecasting future weeks)
    _WEATHER_AVGS = _WEATHER_RAW.groupby("week")[EXT_WEATHER_COLS].mean().reset_index()
except Exception:
    _WEATHER_RAW = _HOLIDAY_RAW = _WEATHER_AVGS = None
    HAS_EXTERNAL = False
    EXT_WEATHER_COLS = []
    EXT_HOLIDAY_COLS = []

EXT_COLS = EXT_WEATHER_COLS + EXT_HOLIDAY_COLS

# ── Feature columns ────────────────────────────────────────────────────────────
FCOLS_BASE = [
    "trend","month","week_of_year","quarter",
    "is_month_start","is_month_end",
    "month_sin","month_cos","week_sin","week_cos",
    "lag_1","lag_2","lag_4","lag_8",
    "rolling_mean_4","rolling_mean_8","rolling_std_4","rolling_std_8",
]
FCOLS = FCOLS_BASE + EXT_COLS

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Planwisely – Sales Forecast", page_icon="📈", layout="wide")

st.markdown(f"""
<style>
  .stApp {{ background:{NAVY}; }}
  .block-container {{ padding:1rem 2rem 2rem; max-width:100%; }}

  .stButton>button {{
      background:{BLUE}!important; color:#fff!important; border:none!important;
      border-radius:8px!important; padding:0.5rem 1.5rem!important;
      font-weight:600!important; font-size:14px!important; width:100%;
  }}
  .stButton>button:hover {{ background:#1D4ED8!important; }}
  .stButton>button:disabled {{ background:#4B6A9B!important; opacity:0.6!important; }}

  .tile {{
      background:{CARD}; border-radius:12px;
      padding:18px 22px; border:1px solid {BORDER};
      height:100%;
  }}
  .tile-label {{
      font-size:10px; font-weight:700; letter-spacing:.1em;
      text-transform:uppercase; color:#94A3B8; margin-bottom:2px;
  }}
  .tile-title {{
      font-size:15px; font-weight:700; color:{NAVY}; margin-bottom:12px;
  }}

  .kpi-row {{ display:flex; gap:0; }}
  .kpi-cell {{
      flex:1; padding-right:16px; margin-right:16px;
      border-right:1px solid {BORDER};
  }}
  .kpi-cell:last-child {{ border-right:none; padding-right:0; margin-right:0; }}
  .kpi-num {{ font-size:34px; font-weight:800; color:{NAVY}; line-height:1; }}
  .kpi-sub {{ font-size:11px; color:#64748B; margin-top:3px; }}

  .badge {{ display:inline-block; padding:2px 8px; border-radius:9999px; font-size:11px; font-weight:700; }}
  .b-up   {{ background:#D1FAE5; color:#065F46; }}
  .b-down {{ background:#FEE2E2; color:#991B1B; }}
  .b-flat {{ background:#FEF3C7; color:#78350F; }}

  .tbl {{ width:100%; border-collapse:collapse; font-size:12px; }}
  .tbl th {{ padding:7px 10px; background:{NAVY}; color:#fff; font-weight:600; text-align:left; font-size:11px; }}
  .tbl td {{ padding:6px 10px; border-bottom:1px solid {BORDER}; color:#374151; }}
  .tbl tr:last-child td {{ border-bottom:none; }}

  .div {{ border:none; border-top:1px solid rgba(255,255,255,0.15); margin:12px 0; }}
  .note {{ font-size:10px; color:#94A3B8; margin-top:6px; }}
  label {{ color:rgba(255,255,255,0.85) !important; }}
  .stSelectbox > div > div {{ background:{CARD}; border-radius:8px; }}
  .stFileUploader {{ background:{CARD}; border-radius:10px; padding:10px; }}
  .sec-head {{
      color:rgba(255,255,255,0.9); font-size:12px; font-weight:700;
      letter-spacing:.08em; text-transform:uppercase; margin:10px 0 8px;
  }}
  .gap8  {{ height:8px; }}
  .gap14 {{ height:14px; }}
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
ext_badge = ('<span style="background:#10B981;color:#fff;font-size:10px;font-weight:700;'
             'padding:2px 8px;border-radius:9999px;margin-left:10px">'
             '✓ Weather &amp; Holiday features loaded</span>') if HAS_EXTERNAL else ""
st.markdown(f"""
<div style="display:flex;align-items:center;gap:18px;padding:4px 0 14px">
  <img src="{LOGO}" style="width:145px;height:auto;filter:brightness(0) invert(1);">
  <div>
    <div style="font-size:23px;font-weight:800;color:#fff;line-height:1.15">
      Sales Demand Forecast{ext_badge}
    </div>
    <div style="font-size:12px;color:rgba(255,255,255,0.6);margin-top:4px">
      Upload your sales CSV — feature engineering, external data merging, and forecasting run automatically.
    </div>
  </div>
</div>
<hr class='div'>
""", unsafe_allow_html=True)

# ── Core helpers ───────────────────────────────────────────────────────────────
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
                    month_col=month_col, day_col=day_col, sales_col=sales_col, product_col=product_col)
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
    """Build time-series features and merge external weather/holiday data automatically."""
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
    # ── Auto-merge external features ──────────────────────────────────────────
    if HAS_EXTERNAL:
        w = _WEATHER_RAW.rename(columns={"week":"week_of_year"})
        df = df.merge(w[["year","week_of_year"]+EXT_WEATHER_COLS], on=["year","week_of_year"], how="left")
        h = _HOLIDAY_RAW.rename(columns={"week":"week_of_year"})
        df = df.merge(h[["year","week_of_year"]+EXT_HOLIDAY_COLS], on=["year","week_of_year"], how="left")
        for col in EXT_WEATHER_COLS:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mean())
        for col in EXT_HOLIDAY_COLS:
            if col in df.columns:
                df[col] = df[col].fillna(0)
    return df.dropna(subset=["lag_1","lag_2","lag_4","lag_8"])

def _ext_for_date(nd):
    """Return external features for a forecast date (weather=week avg, holidays=exact lookup)."""
    if not HAS_EXTERNAL:
        return {}
    week = int(nd.isocalendar()[1])
    year = int(nd.isocalendar()[0])
    r = {}
    # Weather: use historical week-of-year average
    w_row = _WEATHER_AVGS[_WEATHER_AVGS["week"] == week]
    for col in EXT_WEATHER_COLS:
        r[col] = float(w_row[col].values[0]) if len(w_row) > 0 else 0.0
    # Holidays: exact year+week lookup
    h_row = _HOLIDAY_RAW[(_HOLIDAY_RAW["year"]==year) & (_HOLIDAY_RAW["week"]==week)]
    for col in EXT_HOLIDAY_COLS:
        r[col] = float(h_row[col].values[0]) if len(h_row) > 0 else 0.0
    return r

def train_model(df, model_type="Ridge"):
    """Train Ridge or Lasso pipeline on available feature columns."""
    fcols = [c for c in FCOLS if c in df.columns]
    if model_type == "Lasso":
        reg = Lasso(alpha=1.0, max_iter=10000)
    else:
        reg = Ridge(alpha=10.0)
    model = Pipeline([
        ("var",    VarianceThreshold(threshold=0.0)),
        ("scaler", StandardScaler()),
        ("reg",    reg),
    ])
    model.fit(df[fcols], df["sales"])
    return model, fcols

def forecast_4(model, fcols, df):
    """Recursively forecast 4 periods ahead with clipping."""
    delta    = df["date"].iloc[-1] - df["date"].iloc[-2]
    hist     = list(df["sales"]); rows = []
    hist_max = df["sales"].max() * 3
    hist_p10 = max(0, df["sales"].quantile(0.10) * 0.5)
    for i in range(4):
        nd = df["date"].iloc[-1] + (i+1)*delta
        r  = {"date":nd, "year":int(nd.isocalendar()[0]), "month":nd.month,
              "week_of_year":int(nd.isocalendar()[1]), "day_of_week":nd.dayofweek,
              "is_month_start":int(nd.day<=7), "is_month_end":int(nd.day>=24),
              "quarter":nd.quarter, "trend":len(df)+i}
        r["month_sin"] = np.sin(2*np.pi*r["month"]/12)
        r["month_cos"] = np.cos(2*np.pi*r["month"]/12)
        r["week_sin"]  = np.sin(2*np.pi*r["week_of_year"]/52)
        r["week_cos"]  = np.cos(2*np.pi*r["week_of_year"]/52)
        for lag in [1,2,4,8]:
            r[f"lag_{lag}"] = hist[-lag] if len(hist)>=lag else np.nan
        for w in [4,8]:
            h2 = hist[-w:]
            r[f"rolling_mean_{w}"] = np.mean(h2)
            r[f"rolling_std_{w}"]  = np.std(h2) if len(h2)>1 else 0.0
        r.update(_ext_for_date(nd))
        pred = model.predict(np.array([[r[c] for c in fcols]]))[0]
        pred = float(np.clip(pred, hist_p10, hist_max))
        hist.append(pred); r["forecast"] = pred; rows.append(r)
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False)
def run_all(raw_bytes, sales_col, date_mode, date_col, year_col, month_col, day_col,
            product_col, model_type):
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
        out[pid if pid is not None else "all"] = {"history":feat, "forecast":fcast}
    return out

# ── Chart helper ───────────────────────────────────────────────────────────────
def white_ax(figsize=(9,3)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(CARD)
    ax.set_facecolor(CARD)
    ax.tick_params(colors="#374151", labelsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
    return fig, ax

# ── Badge / arrow helpers ──────────────────────────────────────────────────────
def badge(pc):
    if pc >  5: return f"<span class='badge b-up'>▲ +{pc:.1f}%</span>"
    if pc < -5: return f"<span class='badge b-down'>▼ {pc:.1f}%</span>"
    return             f"<span class='badge b-flat'>▶ {pc:+.1f}%</span>"

def arrow(pc):
    if pc >  5: return f"<span style='color:{GREEN};font-size:15px;font-weight:800'>▲</span>"
    if pc < -5: return f"<span style='color:{RED};font-size:15px;font-weight:800'>▼</span>"
    return             f"<span style='color:{AMBER};font-size:15px;font-weight:800'>▶</span>"

# ── Upload row ─────────────────────────────────────────────────────────────────
uL, uM, uR, uRR = st.columns([3.5, 1.5, 1.5, 1], gap="medium")
with uL:
    uploaded = st.file_uploader("Upload sales CSV", type=["csv"], label_visibility="visible")
with uM:
    sales_col_sel = None
    info = {}
    if uploaded:
        raw  = pd.read_csv(uploaded)
        info = detect_structure(raw)
        opts = raw.columns.tolist()
        sales_col_sel = st.selectbox("Sales column", opts,
                                     index=opts.index(info["sales_col"]) if info.get("sales_col") in opts else 0)
with uR:
    model_type_sel = st.selectbox("Model", ["Ridge", "Lasso"], index=0)
with uRR:
    st.markdown("<div style='padding-top:26px'>", unsafe_allow_html=True)
    run_btn = st.button("Run →", type="primary", disabled=(uploaded is None), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

if uploaded and info.get("date_mode") == "split":
    st.markdown(f"<p style='color:rgba(255,255,255,0.6);font-size:11px;margin-top:-6px'>"
                f"Split date detected: <b style='color:#fff'>{info['year_col']} / "
                f"{info['month_col']} / {info['day_col']}</b> — combined automatically.</p>",
                unsafe_allow_html=True)

st.markdown("<hr class='div'>", unsafe_allow_html=True)

# ── Session state & execution ──────────────────────────────────────────────────
if uploaded and run_btn:
    st.session_state["rb"]    = uploaded.getvalue()
    st.session_state["sc"]    = sales_col_sel
    st.session_state["inf"]   = info
    st.session_state["mtype"] = model_type_sel

if "rb" in st.session_state and uploaded is not None:
    # Auto-recompute if model type changed
    _mtype = model_type_sel
    if st.session_state.get("mtype") != _mtype:
        st.session_state["mtype"] = _mtype
    with st.spinner(f"Running {_mtype} forecasts with {'weather + holiday + ' if HAS_EXTERNAL else ''}time-series features…"):
        results = run_all(
            st.session_state["rb"], st.session_state["sc"],
            st.session_state["inf"]["date_mode"], st.session_state["inf"]["date_col"],
            st.session_state["inf"]["year_col"],  st.session_state["inf"]["month_col"],
            st.session_state["inf"]["day_col"],   st.session_state["inf"]["product_col"],
            _mtype,
        )
    st.session_state["results"] = results

# ── Results ────────────────────────────────────────────────────────────────────
if st.session_state.get("results"):
    results = st.session_state["results"]
    if not results:
        st.error("No products with ≥20 usable records found.")
        st.stop()

    # Summary stats
    rows = []
    for pid, r in results.items():
        f, h = r["forecast"], r["history"]
        ftot = f["forecast"].sum(); favg = f["forecast"].mean()
        prev = h["sales"].iloc[-4:].mean() if len(h)>=4 else h["sales"].mean()
        pct  = (favg-prev)/prev*100 if prev else 0
        rows.append(dict(product_id=pid, forecast_total=ftot,
                         forecast_avg=favg, prev_avg=prev, pct_change=pct))
    smry        = pd.DataFrame(rows)
    n_prod      = len(smry)
    total_dem   = smry["forecast_total"].sum()
    avg_prod    = smry["forecast_avg"].mean()
    period_tots = np.array([sum(r["forecast"]["forecast"].iloc[i] for r in results.values()) for i in range(4)])
    up_c = (smry["pct_change"] >  5).sum()
    dn_c = (smry["pct_change"] < -5).sum()
    st_c = n_prod - up_c - dn_c

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — Product Detail (shown first)
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("<div class='sec-head'>Product Detail</div>", unsafe_allow_html=True)

    all_pids = sorted(results.keys(), key=lambda x: str(x))
    sel_pid  = st.selectbox("Select a product to inspect", options=all_pids,
                            format_func=lambda x: f"Product {x}")

    if sel_pid is not None:
        pr       = results[sel_pid]
        hist     = pr["history"]
        fcast    = pr["forecast"]
        prev_avg = hist["sales"].iloc[-4:].mean() if len(hist)>=4 else hist["sales"].mean()

        # 4 period tiles
        d1, d2, d3, d4 = st.columns(4, gap="medium")
        for col, i in zip([d1,d2,d3,d4], range(4)):
            row       = fcast.iloc[i]
            date_str  = pd.to_datetime(row["date"]).strftime("%d %b %Y")
            dp        = (row["forecast"] - prev_avg) / prev_avg * 100 if prev_avg else 0
            tcls      = "b-up" if dp > 5 else ("b-down" if dp < -5 else "b-flat")
            tsym      = "▲" if dp > 5 else ("▼" if dp < -5 else "▶")
            col.markdown(f"""
            <div class='tile' style='text-align:center;padding:16px 12px'>
              <div class='tile-label'>Period {i+1}</div>
              <div style='font-size:11px;color:#64748B;margin-bottom:5px'>{date_str}</div>
              <div class='kpi-num' style='font-size:28px'>{row['forecast']:,.0f}</div>
              <div class='kpi-sub'>units forecasted</div>
              <span class='badge {tcls}' style='margin-top:7px;display:inline-block'>
                {tsym} {dp:+.1f}% vs prev 4
              </span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div class='gap8'></div>", unsafe_allow_html=True)

        # History + forecast chart
        st.markdown("<div class='tile'>", unsafe_allow_html=True)
        st.markdown(f"<div class='tile-label'>Product {sel_pid} — History + Forecast</div>"
                    f"<div class='tile-title'>Last 52 Periods + Next 4 (±15% confidence band)</div>",
                    unsafe_allow_html=True)
        fig2, ax2 = white_ax((14, 3.4))
        tail = hist.tail(52)
        ax2.plot(tail["date"], tail["sales"], color=NAVY, linewidth=1.8, label="Historical", zorder=3)
        ax2.plot(fcast["date"], fcast["forecast"], color=BLUE, linewidth=2.2,
                 marker="o", markersize=6, label="Forecast", zorder=4)
        ax2.fill_between(fcast["date"], fcast["forecast"]*0.85, fcast["forecast"]*1.15,
                         color=BLUE, alpha=0.12, label="±15% band")
        ax2.axvline(hist["date"].iloc[-1], color=BORDER, linewidth=1.2, linestyle="--")
        ax2.yaxis.grid(True, color=BORDER, zorder=0); ax2.set_axisbelow(True)
        ax2.set_ylabel("Units", color="#64748B", fontsize=9)
        ax2.legend(facecolor=CARD, edgecolor=BORDER, fontsize=9)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        fig2.autofmt_xdate()
        plt.tight_layout(pad=0.4)
        st.pyplot(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='gap14'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — Summary KPIs + Demand Bar Chart
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("<div class='sec-head'>Summary</div>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1], gap="medium")

    with c1:
        st.markdown(f"""
        <div class='tile'>
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
              <div class='kpi-sub'>Avg per product / period</div>
            </div>
          </div>
          <hr style='border:none;border-top:1px solid {BORDER};margin:12px 0 10px'>
          <div class='kpi-row'>
            <div class='kpi-cell'>
              <span class='badge b-up'>▲ {up_c} products</span>
              <div class='kpi-sub' style='margin-top:4px'>Trending up (&gt;5%)</div>
            </div>
            <div class='kpi-cell'>
              <span class='badge b-flat'>▶ {st_c} products</span>
              <div class='kpi-sub' style='margin-top:4px'>Stable (±5%)</div>
            </div>
            <div class='kpi-cell'>
              <span class='badge b-down'>▼ {dn_c} products</span>
              <div class='kpi-sub' style='margin-top:4px'>Trending down (&gt;5%)</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"<div class='tile'><div class='tile-label'>Forecast</div>"
                    f"<div class='tile-title'>Total Demand — Next 4 Periods</div>",
                    unsafe_allow_html=True)
        fig, ax = white_ax((8, 3))
        plabels = ["Period 1","Period 2","Period 3","Period 4"]
        bars = ax.bar(plabels, period_tots, color=BLUE, width=0.45, zorder=3)
        ax.yaxis.grid(True, color=BORDER, zorder=0); ax.set_axisbelow(True)
        for b, v in zip(bars, period_tots):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+total_dem*0.0008,
                    f"{v:,.0f}", ha="center", va="bottom", fontsize=9, color=NAVY, fontweight="bold")
        ax.set_ylabel("Units", color="#64748B", fontsize=9)
        ax.tick_params(axis="x", colors=NAVY)
        plt.tight_layout(pad=0.3)
        st.pyplot(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='gap14'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — Analysis: Movers + Top 5
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("<div class='sec-head'>Analysis</div>", unsafe_allow_html=True)
    c3, c4 = st.columns([1, 1], gap="medium")

    with c3:
        movers = pd.concat([smry.nlargest(5,"pct_change"), smry.nsmallest(5,"pct_change")]) \
                   .drop_duplicates("product_id").sort_values("pct_change", ascending=False)
        rows_h = "".join(f"""
          <tr>
            <td><b>Product {r['product_id']}</b></td>
            <td>{r['prev_avg']:.0f}</td>
            <td>{r['forecast_avg']:.0f}</td>
            <td>{badge(r['pct_change'])}</td>
          </tr>""" for _, r in movers.iterrows())
        st.markdown(f"""
        <div class='tile'>
          <div class='tile-label'>Change Analysis</div>
          <div class='tile-title'>Biggest Movers vs Previous Period</div>
          <table class='tbl'>
            <tr><th>Product</th><th>Prev 4 avg</th><th>Forecast avg</th><th>Change</th></tr>
            {rows_h}
          </table>
          <p class='note'>Top 5 rising + top 5 falling vs last 4 actual periods.</p>
        </div>""", unsafe_allow_html=True)

    with c4:
        top5    = smry.nlargest(5,"forecast_total").reset_index(drop=True)
        rows_h2 = "".join(f"""
          <tr>
            <td><b>#{i+1}</b></td>
            <td>Product {r['product_id']}</td>
            <td>{r['forecast_total']:,.0f}</td>
            <td>{arrow(r['pct_change'])}</td>
          </tr>""" for i, (_, r) in enumerate(top5.iterrows()))
        st.markdown(f"""
        <div class='tile'>
          <div class='tile-label'>Prioritisation</div>
          <div class='tile-title'>Top 5 by Forecasted Volume</div>
          <table class='tbl'>
            <tr><th>#</th><th>Product</th><th>Total forecast</th><th>Trend</th></tr>
            {rows_h2}
          </table>
          <p class='note'>Total units across all 4 periods. Use to prioritise production capacity.</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='gap14'></div>", unsafe_allow_html=True)

    # ── Download ───────────────────────────────────────────────────────────────
    dl = smry[["product_id","forecast_total","forecast_avg","prev_avg","pct_change"]].copy()
    dl.columns = ["Product ID","Total Forecast (4 periods)","Avg per Period","Prev 4 Period Avg","% Change"]
    st.download_button(
        "Download Full Forecast CSV",
        dl.round(1).to_csv(index=False).encode(),
        "planwisely_forecast.csv", "text/csv",
    )
