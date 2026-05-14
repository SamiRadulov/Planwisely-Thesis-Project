import io
import warnings
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
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
BGCARD = "#F1F5F9"
BORDER = "#E2E8F0"

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Planwisely – Sales Forecast", page_icon="📈", layout="wide")

st.markdown(f"""
<style>
  .stApp {{ background:#ffffff; }}
  .block-container {{ padding-top:1.2rem; padding-bottom:2rem; max-width:1200px; }}
  h1,h2,h3 {{ color:{NAVY} !important; }}
  .stButton>button {{
      background:{BLUE}!important; color:#fff!important; border:none!important;
      border-radius:8px!important; padding:0.55rem 2rem!important;
      font-weight:600!important; font-size:15px!important;
  }}
  .stButton>button:hover {{ background:#1D4ED8!important; }}
  /* ── Tile card ── */
  .tile {{
      background:{BGCARD}; border-radius:14px;
      padding:24px 28px; border:1px solid {BORDER};
  }}
  .tile-label {{
      font-size:11px; font-weight:700; letter-spacing:.1em;
      text-transform:uppercase; color:#94A3B8; margin-bottom:6px;
  }}
  .tile-title {{
      font-size:16px; font-weight:700; color:{NAVY}; margin-bottom:18px;
  }}
  /* ── KPI numbers ── */
  .kpi-row {{ display:flex; gap:0; }}
  .kpi-cell {{
      flex:1; padding-right:20px; margin-right:20px;
      border-right:1px solid {BORDER};
  }}
  .kpi-cell:last-child {{ border-right:none; padding-right:0; margin-right:0; }}
  .kpi-num {{ font-size:38px; font-weight:800; color:{NAVY}; line-height:1; }}
  .kpi-sub {{ font-size:12px; color:#64748B; margin-top:5px; }}
  /* ── Badges ── */
  .badge {{
      display:inline-block; padding:2px 9px;
      border-radius:9999px; font-size:12px; font-weight:700;
  }}
  .b-up   {{ background:#D1FAE5; color:#065F46; }}
  .b-down {{ background:#FEE2E2; color:#991B1B; }}
  .b-flat {{ background:#FEF3C7; color:#78350F; }}
  /* ── Table ── */
  .tbl {{ width:100%; border-collapse:collapse; font-size:13px; }}
  .tbl th {{
      padding:9px 12px; background:{NAVY}; color:#fff;
      font-weight:600; text-align:left; font-size:12px;
  }}
  .tbl td {{ padding:8px 12px; border-bottom:1px solid {BORDER}; color:#374151; }}
  .tbl tr:last-child td {{ border-bottom:none; }}
  /* ── Upload zone ── */
  .upload-wrap {{
      background:{BGCARD}; border-radius:14px;
      padding:26px 28px; border:1px solid {BORDER}; margin-bottom:22px;
  }}
  .divider {{ border:none; border-top:1px solid {BORDER}; margin:18px 0 22px; }}
  .gap {{ height:16px; }}
  .note {{ font-size:11px; color:#94A3B8; margin-top:10px; }}
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
hL, hR = st.columns([1, 5])
with hL:
    st.image(LOGO, width=175)
with hR:
    st.markdown(f"<h1 style='padding-top:10px'>Sales Demand Forecast</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748B;margin-top:-8px'>Upload your sales data and get a 4-period demand forecast across all products.</p>", unsafe_allow_html=True)
st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ── Column detection ───────────────────────────────────────────────────────────
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
        return dict(date_mode="split", date_col=None,
                    year_col=year_col, month_col=month_col, day_col=day_col,
                    sales_col=sales_col, product_col=product_col)
    date_col = None
    for c in df.columns:
        if c in (product_col, sales_col): continue
        try:
            if pd.to_datetime(df[c], infer_datetime_format=True, errors="coerce").notna().mean() > 0.8:
                date_col = c; break
        except Exception: pass
    return dict(date_mode="single", date_col=date_col,
                year_col=None, month_col=None, day_col=None,
                sales_col=sales_col, product_col=product_col)

def build_date_series(df, info):
    if info["date_mode"] == "split":
        return pd.to_datetime(dict(year=df[info["year_col"]],
                                   month=df[info["month_col"]],
                                   day=df[info["day_col"]]))
    return pd.to_datetime(df[info["date_col"]], infer_datetime_format=True, errors="coerce")

# ── Feature engineering ────────────────────────────────────────────────────────
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
    df["dow_sin"]   = np.sin(2*np.pi*df["day_of_week"]/7)
    df["dow_cos"]   = np.cos(2*np.pi*df["day_of_week"]/7)
    for lag in [1,2,4,8]:
        df[f"lag_{lag}"] = df["sales"].shift(lag)
    for w in [4,8]:
        df[f"rolling_mean_{w}"] = df["sales"].shift(1).rolling(w, min_periods=1).mean()
        df[f"rolling_std_{w}"]  = df["sales"].shift(1).rolling(w, min_periods=1).std().fillna(0)
    df["trend"] = np.arange(len(df))
    return df.dropna(subset=["lag_1","lag_2","lag_4","lag_8"])

FCOLS = [
    "trend","year","month","week_of_year","quarter",
    "is_month_start","is_month_end",
    "month_sin","month_cos","week_sin","week_cos","dow_sin","dow_cos",
    "lag_1","lag_2","lag_4","lag_8",
    "rolling_mean_4","rolling_mean_8","rolling_std_4","rolling_std_8",
]

def train_model(df):
    m = LinearRegression()
    m.fit(df[FCOLS], df["sales"])
    return m

def forecast_4(model, df):
    delta = df["date"].iloc[-1] - df["date"].iloc[-2]
    hist  = list(df["sales"])
    rows  = []
    for i in range(4):
        nd = df["date"].iloc[-1] + (i+1)*delta
        r  = {
            "date":nd, "year":nd.isocalendar()[0], "month":nd.month,
            "week_of_year":nd.isocalendar()[1], "day_of_week":nd.dayofweek,
            "is_month_start":int(nd.day<=7), "is_month_end":int(nd.day>=24),
            "quarter":nd.quarter, "trend":len(df)+i,
        }
        r["month_sin"]=np.sin(2*np.pi*r["month"]/12)
        r["month_cos"]=np.cos(2*np.pi*r["month"]/12)
        r["week_sin"] =np.sin(2*np.pi*r["week_of_year"]/52)
        r["week_cos"] =np.cos(2*np.pi*r["week_of_year"]/52)
        r["dow_sin"]  =np.sin(2*np.pi*r["day_of_week"]/7)
        r["dow_cos"]  =np.cos(2*np.pi*r["day_of_week"]/7)
        for lag in [1,2,4,8]:
            r[f"lag_{lag}"] = hist[-lag] if len(hist)>=lag else np.nan
        for w in [4,8]:
            h2=hist[-w:]; r[f"rolling_mean_{w}"]=np.mean(h2)
            r[f"rolling_std_{w}"]=np.std(h2) if len(h2)>1 else 0.0
        pred = max(model.predict(np.array([[r[c] for c in FCOLS]]))[0], 0)
        hist.append(pred); r["forecast"]=pred; rows.append(r)
    return pd.DataFrame(rows)

# ── Cached runner ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_all(raw_bytes, sales_col, date_mode, date_col, year_col, month_col, day_col, product_col):
    df   = pd.read_csv(io.BytesIO(raw_bytes))
    info = dict(date_mode=date_mode, date_col=date_col, year_col=year_col,
                month_col=month_col, day_col=day_col, product_col=product_col)
    out  = {}
    pids = [None] if product_col is None else \
           df.groupby(product_col).filter(lambda x: len(x)>=20)[product_col].unique().tolist()
    for pid in pids:
        sub = df if pid is None else df[df[product_col]==pid].copy()
        sub = sub.copy()
        sub["date"]  = build_date_series(sub, info)
        sub["sales"] = pd.to_numeric(sub[sales_col], errors="coerce")
        feat = feature_engineering(sub[["date","sales"]].dropna())
        if len(feat) < 20: continue
        mdl  = train_model(feat)
        fcast= forecast_4(mdl, feat)
        out[pid if pid is not None else "all"] = {"history":feat, "forecast":fcast}
    return out

# ── Chart helper ───────────────────────────────────────────────────────────────
def light_ax(figsize=(9,3.5)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BGCARD)
    ax.set_facecolor(BGCARD)
    ax.tick_params(colors="#374151", labelsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
    return fig, ax

# ── Upload section ─────────────────────────────────────────────────────────────
st.markdown("<div class='upload-wrap'>", unsafe_allow_html=True)
uL, uR = st.columns([4, 1])
with uL:
    uploaded = st.file_uploader(
        "Upload sales CSV — needs a date column (or year/month/day) and a sales column",
        type=["csv"], label_visibility="visible"
    )
with uR:
    st.markdown("<div style='padding-top:26px'>", unsafe_allow_html=True)
    run_btn = st.button("Run Forecast →", type="primary", use_container_width=True,
                        disabled=(uploaded is None))
    st.markdown("</div>", unsafe_allow_html=True)

if uploaded:
    raw = pd.read_csv(uploaded)
    info = detect_structure(raw)
    col_opts = raw.columns.tolist()
    sales_col = st.selectbox("Sales column", col_opts,
                             index=col_opts.index(info["sales_col"]) if info["sales_col"] else 0)
    if info["date_mode"] == "split":
        st.info(f"Split date detected: **{info['year_col']}** / **{info['month_col']}** / **{info['day_col']}** combined automatically.")
st.markdown("</div>", unsafe_allow_html=True)

# ── Session state: persist results across reruns ───────────────────────────────
if uploaded and run_btn:
    st.session_state["results"]   = None          # clear while loading
    st.session_state["run_bytes"] = uploaded.getvalue()
    st.session_state["run_sales"] = sales_col
    st.session_state["run_info"]  = info

if "run_bytes" in st.session_state and uploaded is not None:
    rb   = st.session_state["run_bytes"]
    sc   = st.session_state["run_sales"]
    inf  = st.session_state["run_info"]

    with st.spinner("Running forecasts for all eligible products…"):
        results = run_all(rb, sc,
                          inf["date_mode"], inf["date_col"],
                          inf["year_col"],  inf["month_col"],
                          inf["day_col"],   inf["product_col"])
    st.session_state["results"] = results

if st.session_state.get("results"):
    results = st.session_state["results"]

    if not results:
        st.error("No products with ≥20 usable records found.")
        st.stop()

    # ── Aggregate stats ────────────────────────────────────────────────────────
    rows = []
    for pid, r in results.items():
        f, h = r["forecast"], r["history"]
        ftot = f["forecast"].sum()
        favg = f["forecast"].mean()
        prev = h["sales"].iloc[-4:].mean() if len(h)>=4 else h["sales"].mean()
        pct  = (favg-prev)/prev*100 if prev else 0
        rows.append(dict(product_id=pid, forecast_total=ftot,
                         forecast_avg=favg, prev_avg=prev, pct_change=pct,
                         periods=f["forecast"].tolist()))
    smry = pd.DataFrame(rows)

    n_prod      = len(smry)
    total_dem   = smry["forecast_total"].sum()
    avg_prod    = smry["forecast_avg"].mean()
    period_tots = np.array([sum(r["forecast"]["forecast"].iloc[i] for r in results.values()) for i in range(4)])
    up_count    = (smry["pct_change"] >  5).sum()
    dn_count    = (smry["pct_change"] < -5).sum()
    st_count    = n_prod - up_count - dn_count

    # ── ROW 1: KPI tile + Forecast chart ──────────────────────────────────────
    c1, c2 = st.columns(2, gap="large")

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
              <div class='kpi-sub'>Total demand — next 4 periods</div>
            </div>
            <div class='kpi-cell'>
              <div class='kpi-num'>{avg_prod:,.0f}</div>
              <div class='kpi-sub'>Avg per product / period</div>
            </div>
          </div>
          <hr style='border:none;border-top:1px solid {BORDER};margin:20px 0 14px'>
          <div class='kpi-row'>
            <div class='kpi-cell'>
              <span class='badge b-up'>&#9650; {up_count}</span>
              <div class='kpi-sub' style='margin-top:4px'>Trending up (&gt;5%)</div>
            </div>
            <div class='kpi-cell'>
              <span class='badge b-flat'>&#9654; {st_count}</span>
              <div class='kpi-sub' style='margin-top:4px'>Stable (±5%)</div>
            </div>
            <div class='kpi-cell'>
              <span class='badge b-down'>&#9660; {dn_count}</span>
              <div class='kpi-sub' style='margin-top:4px'>Trending down (&gt;5%)</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"<div class='tile'><div class='tile-label'>Forecast</div><div class='tile-title'>Total Demand — Next 4 Periods</div>", unsafe_allow_html=True)
        fig, ax = light_ax((7, 3.2))
        plabels = [f"Period {i+1}" for i in range(4)]
        bars = ax.bar(plabels, period_tots, color=BLUE, width=0.45, zorder=3)
        ax.yaxis.grid(True, color=BORDER, zorder=0); ax.set_axisbelow(True)
        for b, v in zip(bars, period_tots):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+total_dem*0.003,
                    f"{v:,.0f}", ha="center", va="bottom", fontsize=9,
                    color=NAVY, fontweight="bold")
        ax.set_ylabel("Units", color="#64748B", fontsize=9)
        ax.tick_params(axis="x", colors=NAVY)
        plt.tight_layout(pad=0.4)
        st.pyplot(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='gap'></div>", unsafe_allow_html=True)

    # ── ROW 2: Biggest Movers + Top 5 by Volume ────────────────────────────────
    c3, c4 = st.columns(2, gap="large")

    with c3:
        movers = pd.concat([
            smry.nlargest(5, "pct_change"),
            smry.nsmallest(5, "pct_change")
        ]).drop_duplicates("product_id").sort_values("pct_change", ascending=False)

        def badge(pc):
            if pc > 5:  return f"<span class='badge b-up'>+{pc:.1f}%</span>"
            if pc < -5: return f"<span class='badge b-down'>{pc:.1f}%</span>"
            return      f"<span class='badge b-flat'>{pc:+.1f}%</span>"

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
            <tr><th>Product</th><th>Prev avg</th><th>Forecast avg</th><th>Change</th></tr>
            {rows_h}
          </table>
          <p class='note'>Comparing average of next 4 forecast periods vs last 4 actual periods.</p>
        </div>""", unsafe_allow_html=True)

    with c4:
        top5 = smry.nlargest(5, "forecast_total").reset_index(drop=True)

        def arrow(pc):
            if pc >  5: return f"<span style='color:{GREEN};font-size:16px;font-weight:700'>&#9650;</span>"
            if pc < -5: return f"<span style='color:{RED};font-size:16px;font-weight:700'>&#9660;</span>"
            return              f"<span style='color:{AMBER};font-size:16px;font-weight:700'>&#9654;</span>"

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
          <p class='note'>Total units forecasted across all 4 periods. Use this to prioritise production capacity.</p>
        </div>""", unsafe_allow_html=True)

    # ── Download ───────────────────────────────────────────────────────────────
    st.markdown("<div class='gap'></div>", unsafe_allow_html=True)
    dl = smry[["product_id","forecast_total","forecast_avg","prev_avg","pct_change"]].copy()
    dl.columns = ["Product ID","Total Forecast (4 periods)","Avg per Period","Prev 4 Period Avg","% Change"]
    st.download_button("Download Full Forecast CSV", dl.round(1).to_csv(index=False).encode(),
                       "planwisely_forecast.csv", "text/csv")
