import io
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
from sklearn.metrics import r2_score

from feature_engineering import (
    build_features, get_external_for_date,
    HAS_WEATHER, HAS_HOLIDAY, HAS_COVID, HAS_SCHOOL,
    WEATHER_COLS, HOLIDAY_COLS, COVID_COLS, SCHOOL_COLS,
)
from models import FCOLS, FEAT_LABELS, train_model, get_feature_importance
from shap_utils import compute_shap, generate_shap_text
from lime_utils import compute_lime, generate_lime_text

warnings.filterwarnings("ignore")

LOGO   = "https://planwisely.ai/wp-content/uploads/2024/08/Planwisely01_1765553.png"
NAVY   = "#1B3A6B"
BLUE   = "#2563EB"
GREEN  = "#10B981"
RED    = "#EF4444"
AMBER  = "#F59E0B"
CARD   = "#FFFFFF"
BORDER = "#E2E8F0"

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

ext_parts = []
if HAS_WEATHER: ext_parts.append("Weather")
if HAS_HOLIDAY: ext_parts.append("Holiday")
if HAS_COVID:   ext_parts.append("COVID")
if HAS_SCHOOL:  ext_parts.append("School holidays")

ext_pill = (
    f'<span style="background:#10B981;color:#fff;font-size:9px;font-weight:700;'
    f'padding:2px 8px;border-radius:9999px;margin-left:10px;vertical-align:middle">'
    f'{" + ".join(ext_parts)} data loaded</span>'
) if ext_parts else ""

st.markdown(f"""
<div style="display:flex;align-items:center;gap:16px;padding:2px 0 12px">
  <img src="{LOGO}" style="width:140px;height:auto;filter:brightness(0) invert(1);">
  <div>
    <div style="font-size:24px;font-weight:800;color:#fff;line-height:1.1">
      Sales Demand Forecast{ext_pill}
    </div>
    <div style="font-size:13px;color:rgba(255,255,255,0.5);margin-top:3px">
      Upload your sales CSV — feature engineering, external data merging, and forecasting run automatically.
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

def forecast_4(model, fcols, df):
    """Iteratively predict 4 periods ahead, feeding each prediction as a lag."""
    delta    = df["date"].iloc[-1] - df["date"].iloc[-2]
    hist     = list(df["sales"])
    hist_max = df["sales"].max() * 3
    hist_p10 = max(0, df["sales"].quantile(0.10) * 0.5)
    rows     = []
    for i in range(4):
        nd = df["date"].iloc[-1] + (i + 1) * delta
        r  = {
            "date": nd,
            "year": int(nd.isocalendar()[0]),
            "month": nd.month,
            "week_of_year": int(nd.isocalendar()[1]),
            "day_of_week": nd.dayofweek,
            "is_month_start": int(nd.day <= 7),
            "is_month_end": int(nd.day >= 24),
            "quarter": nd.quarter,
            "trend": len(df) + i,
        }
        r["month_sin"] = np.sin(2 * np.pi * r["month"] / 12)
        r["month_cos"] = np.cos(2 * np.pi * r["month"] / 12)
        r["week_sin"]  = np.sin(2 * np.pi * r["week_of_year"] / 52)
        r["week_cos"]  = np.cos(2 * np.pi * r["week_of_year"] / 52)
        for lag in [1, 2, 4, 8]:
            r[f"lag_{lag}"] = hist[-lag] if len(hist) >= lag else np.nan
        for w in [4, 8]:
            h2 = hist[-w:]
            r[f"rolling_mean_{w}"] = np.mean(h2)
            r[f"rolling_std_{w}"]  = np.std(h2) if len(h2) > 1 else 0.0
        r.update(get_external_for_date(nd))
        pred = model.predict(pd.DataFrame([[r[c] for c in fcols]], columns=fcols))[0]
        pred = float(np.clip(pred, hist_p10, hist_max))
        hist.append(pred)
        r["forecast"] = pred
        rows.append(r)
    return pd.DataFrame(rows)


def _contribs(mdl, fcols, model_type, rows):
    """Summed per-feature contribution over `rows` for the given model."""
    if model_type == "EBM":
        var, reg = mdl.named_steps["var"], mdl.named_steps["reg"]
        Xv  = var.transform(rows[fcols])
        loc = reg.explain_local(Xv, reg.predict(Xv))
        agg = {}
        for i in range(len(rows)):
            d = loc.data(i)
            for nm, sc in zip(d["names"], d["scores"]):
                if nm in fcols:
                    agg[nm] = agg.get(nm, 0.0) + float(sc)
        return agg
    from shap_utils import compute_shap_lightgbm
    fn = compute_shap if model_type == "XGBoost" else compute_shap_lightgbm
    _, vals, kept = fn(mdl, fcols, rows)
    return dict(zip(kept, np.asarray(vals).sum(axis=0).tolist()))


def vs_last_year(mdl, fcols, model_type, feat, fcast):
    """Compare the 4-period forecast with the same ISO weeks one year earlier and
    decompose the change into per-feature contribution deltas (tree/EBM models)."""
    h = feat.copy()
    iso = h["date"].dt.isocalendar()
    h["iy"], h["iw"] = iso.year.astype(int), iso.week.astype(int)
    key2idx = {}
    for idx, iy, iw in zip(h.index, h["iy"], h["iw"]):
        key2idx[(int(iy), int(iw))] = idx
    fc = fcast.copy()
    fiso = fc["date"].dt.isocalendar()
    fc["iy"], fc["iw"] = fiso.year.astype(int), fiso.week.astype(int)
    keep_fc, ly_ix = [], []
    for idx, iy, iw in zip(fc.index, fc["iy"], fc["iw"]):
        k = (int(iy) - 1, int(iw))
        if k in key2idx:
            keep_fc.append(idx); ly_ix.append(key2idx[k])
    if not ly_ix:
        # No matching weeks one year back (e.g. a product whose history doesn't
        # cover the same period last year). Surface this honestly in the UI.
        return {"no_data": True}
    fc_rows, ly_rows = fc.loc[keep_fc], h.loc[ly_ix]
    fc_total = float(fc_rows["forecast"].sum())
    ly_total = float(ly_rows["sales"].sum())
    change_pct = (fc_total - ly_total) / ly_total * 100 if ly_total else 0.0
    out = {"no_data": False, "fc_total": fc_total, "ly_total": ly_total,
           "change_pct": change_pct, "n_matched": len(ly_ix), "drivers": None}
    if model_type == "EBM":
        # EBM contributions are cheap (explain_local) and reuse the trained model,
        # so compute the change drivers here.
        cf = _contribs(mdl, fcols, "EBM", fc_rows)
        cl = _contribs(mdl, fcols, "EBM", ly_rows)
        delta = {f: cf.get(f, 0.0) - cl.get(f, 0.0) for f in set(cf) | set(cl)}
        out["drivers"] = sorted(delta.items(), key=lambda kv: -abs(kv[1]))[:8]
    elif model_type in ("XGBoost", "LightGBM"):
        # SHAP is the slow part; keep the matched rows and compute the drivers on
        # demand for the selected product only (done in the render code below).
        out["fc_rows"], out["ly_rows"] = fc_rows, ly_rows
    return out


@st.cache_data(show_spinner=False)
def run_all(raw_bytes, sales_col, date_mode, date_col, year_col, month_col,
            day_col, product_col, model_type):
    df   = pd.read_csv(io.BytesIO(raw_bytes))
    info = dict(date_mode=date_mode, date_col=date_col, year_col=year_col,
                month_col=month_col, day_col=day_col, product_col=product_col)
    out  = {}
    pids = [None] if product_col is None else df[product_col].unique().tolist()
    for pid in pids:
        sub          = (df if pid is None else df[df[product_col] == pid]).copy()
        sub["date"]  = build_date_series(sub, info)
        sub["sales"] = pd.to_numeric(sub[sales_col], errors="coerce")
        sub          = sub[["date", "sales"]].dropna().sort_values("date")

        # Match the thesis preprocessing: keep only products with at least 52
        # weeks of actual history (~114 products), and aggregate the series to
        # weekly demand so the dashboard forecasts weekly regardless of whether
        # the uploaded file is daily or weekly.
        if sub["date"].dt.to_period("W").nunique() < 52:
            continue
        # Aggregate to one row per active week (sum within the ISO week). We do
        # not zero-fill empty weeks: that matches the thesis weekly series
        # (~110 weeks/product) and keeps EBM training fast.
        sub["_wk"] = sub["date"].dt.to_period("W").dt.start_time
        wk   = (sub.groupby("_wk", as_index=False)["sales"].sum()
                   .rename(columns={"_wk": "date"}))
        feat = build_features(wk)
        if len(feat) < 20:
            continue
        mdl, fcols = train_model(feat, model_type)
        fcast      = forecast_4(mdl, fcols, feat)
        preds      = mdl.predict(feat[fcols])
        feat["fitted"] = preds                       # in-sample model fit, for the chart overlay
        r2_val     = float(r2_score(feat["sales"], preds))
        imp        = get_feature_importance(mdl, fcols)
        xai_data = None
        if model_type in ("XGBoost", "LightGBM"):
            if model_type == "XGBoost":
                shap_pairs, shap_vals, shap_kept = compute_shap(mdl, fcols, feat)
            else:
                from shap_utils import compute_shap_lightgbm
                shap_pairs, shap_vals, shap_kept = compute_shap_lightgbm(mdl, fcols, feat)
            # signed mean SHAP per feature (direction, not just magnitude)
            mean_signed  = np.asarray(shap_vals).mean(axis=0)
            signed_map   = dict(zip(shap_kept, mean_signed.tolist()))
            pairs_signed = [(f, v, signed_map.get(f, 0.0)) for f, v in shap_pairs]
            xai_data = {"type": "shap", "pairs": pairs_signed,
                        "vals": shap_vals, "kept": shap_kept}
        elif model_type == "EBM":
            xai_data = {"type": "ebm", "pairs": list(imp)}
        vly = vs_last_year(mdl, fcols, model_type, feat, fcast)
        out[pid if pid is not None else "all"] = {
            "history": feat, "forecast": fcast,
            "importance": imp, "r2": r2_val, "xai": xai_data,
            "vs_last_year": vly,
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

def _xai_header(title, text, label):
    """Inner HTML for an explanation card (rendered inside a bordered container)."""
    return (f"<div style='font-size:20px;font-weight:800;color:{NAVY};"
            f"border-bottom:2px solid {BORDER};padding-bottom:10px;margin-bottom:10px'>{title}</div>"
            f"<p style='font-size:13px;color:#374151;line-height:1.6;margin-bottom:8px'>{text}</p>"
            f"<div style='font-size:12px;font-weight:700;color:#94A3B8;"
            f"letter-spacing:.1em;text-transform:uppercase'>{label}</div>")

def _render_lime_panel(pairs, fcast, hist, sel_pid):
    """Render the LIME local-explanation card (text + bar) in one bordered box."""
    labels  = [p[0] for p in pairs[:10]]
    weights = [p[1] for p in pairs[:10]]
    colors  = [GREEN if w >= 0 else RED for w in weights]
    fig_lime = go.Figure(go.Bar(
        x=weights, y=labels, orientation="h", marker_color=colors,
        text=[f"+{v:.2f}" if v >= 0 else f"{v:.2f}" for v in weights],
        textposition="outside", textfont=dict(size=11, color=NAVY), cliponaxis=False,
    ))
    fig_lime.update_layout(
        paper_bgcolor=CARD, plot_bgcolor=CARD, height=300,
        margin=dict(l=10, r=60, t=6, b=28),
        xaxis=dict(title="LIME weight (impact on prediction)",
                   title_font=dict(size=11, color="#64748B"),
                   tickfont=dict(size=10, color="#374151"),
                   gridcolor=BORDER, zeroline=True, zerolinecolor=NAVY, zerolinewidth=1.5),
        yaxis=dict(tickfont=dict(size=11, color=NAVY), showgrid=False),
        showlegend=False,
    )
    lime_text = generate_lime_text(pairs, fcast, hist, sel_pid)
    with st.container(border=True):
        st.markdown(_xai_header("Local Explainability (LIME)", lime_text,
                                "Top Local Feature Contributions (LIME)"), unsafe_allow_html=True)
        st.plotly_chart(fig_lime, use_container_width=True)

def _render_vs_last_year(vly, drivers=None):
    """Render the 'forecast vs same weeks last year' card (text + change-driver bar)."""
    if vly.get("no_data"):
        with st.container(border=True):
            st.markdown(
                f"<div style='font-size:18px;font-weight:800;color:{NAVY};"
                f"border-bottom:2px solid {BORDER};padding-bottom:8px;margin-bottom:10px'>"
                f"Forecast vs last year</div>"
                f"<p style='font-size:13px;color:#94A3B8;line-height:1.6'>"
                f"No recorded sales for the same period last year, so a year-over-year "
                f"comparison isn't available for this product.</p>",
                unsafe_allow_html=True)
        return
    pct   = vly["change_pct"]
    color = GREEN if pct >= 0 else RED
    word  = "higher" if pct >= 0 else "lower"
    drv_label = ("<div style='font-size:12px;font-weight:700;color:#94A3B8;"
                 "letter-spacing:.1em;text-transform:uppercase'>"
                 "What's driving the change</div>") if drivers else ""
    body = (f"<div style='font-size:18px;font-weight:800;color:{NAVY};"
            f"border-bottom:2px solid {BORDER};padding-bottom:8px;margin-bottom:10px'>"
            f"Forecast vs last year</div>"
            f"<p style='font-size:13px;color:#374151;line-height:1.6;margin-bottom:10px'>"
            f"The next 4 periods are forecast at <b>{vly['fc_total']:,.0f}</b> units versus "
            f"<b>{vly['ly_total']:,.0f}</b> in the same weeks last year, i.e. "
            f"<b style='color:{color}'>{abs(pct):.1f}% {word}</b>.</p>{drv_label}")
    with st.container(border=True):
        st.markdown(body, unsafe_allow_html=True)
        if drivers:
            top    = sorted(drivers, key=lambda kv: abs(kv[1]))[-6:]
            labels = [FEAT_LABELS.get(f, f) for f, _ in top]
            vals   = [v for _, v in top]
            colors = [GREEN if v >= 0 else RED for v in vals]
            fig = go.Figure(go.Bar(
                x=vals, y=labels, orientation="h", marker_color=colors,
                text=[f"+{v:.1f}" if v >= 0 else f"{v:.1f}" for v in vals],
                textposition="outside", textfont=dict(size=11, color=NAVY), cliponaxis=False,
            ))
            fig.update_layout(
                paper_bgcolor=CARD, plot_bgcolor=CARD, height=260,
                margin=dict(l=10, r=60, t=6, b=24),
                xaxis=dict(title="Contribution to the change vs last year",
                           title_font=dict(size=11, color="#64748B"),
                           tickfont=dict(size=10, color="#374151"),
                           gridcolor=BORDER, zeroline=True, zerolinecolor=NAVY, zerolinewidth=1.5),
                yaxis=dict(tickfont=dict(size=11, color=NAVY), showgrid=False),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

# UPLOAD ROW
# Fixed 5-column layout so the row never changes shape between reruns. A changing
# column count made Streamlit leave a half-transparent "ghost" Run button behind
# when switching models. The Explainability dropdown renders only for the
# black-box models (XGBoost / LightGBM); for EBM (native) and the linear
# baselines its slot is simply left empty.
uL, uM, uS, uXAI, uRR = st.columns([2.3, 1, 1, 0.9, 0.7], gap="medium")

with uL:
    uploaded = st.file_uploader("Upload sales CSV", type=["csv"], label_visibility="visible")
with uM:
    model_type_sel = st.selectbox("Model", ["Ridge", "Lasso", "XGBoost", "EBM", "LightGBM"],
                                  key="model_sel")
with uS:
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
with uXAI:
    if model_type_sel in ("XGBoost", "LightGBM"):
        xai_toggle = st.selectbox("Explainability", ["SHAP", "LIME"],
                                  help="SHAP: signed feature contributions (global). "
                                       "LIME: local explanation of the latest prediction.")
        st.session_state["xai_mode"] = xai_toggle
    else:
        # EBM (native) and the linear baselines have no post-hoc choice.
        st.session_state["xai_mode"] = "SHAP"
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
    ext_note = " + ".join(ext_parts) if ext_parts else "time-series only"
    with st.spinner(f"Running {_mt} — features: time-series + {ext_note}..."):
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

    all_pids = sorted(
        results.keys(),
        key=lambda x: (float(x) if str(x).lstrip("-").replace(".", "", 1).isdigit()
                       else float("inf"), str(x)),
    )
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
            with st.container(border=True):
                st.markdown(
                    f"<div style='font-size:16px;font-weight:800;color:{NAVY};"
                    f"margin-bottom:4px'>Sales history &amp; 4-period forecast</div>",
                    unsafe_allow_html=True)
                fig2, ax2 = white_ax((9, 4.7))
                tail = hist.tail(52)
                ax2.plot(tail["date"], tail["sales"], color=NAVY, lw=1.8, label="Actual sales", zorder=3)
                if "fitted" in tail.columns:
                    ax2.plot(tail["date"], tail["fitted"], color=GREEN, lw=1.5,
                             linestyle="--", label="Model fit (in-sample)", zorder=3.5)
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
            xai_data = pr.get("xai")
            xai_mode = st.session_state.get("xai_mode", "SHAP")

            # SHAP: signed feature contributions for XGBoost / LightGBM
            if xai_mode == "SHAP" and xai_data and _mt in ("XGBoost", "LightGBM"):
                pairs          = xai_data["pairs"]              # (feat, mean_abs, mean_signed)
                top10          = sorted(pairs[:10], key=lambda x: abs(x[2]))
                feat_names_xai = [FEAT_LABELS.get(f, f) for f, _, _ in top10]
                xai_signed     = [float(s) for _, _, s in top10]
                bar_colors     = [GREEN if s >= 0 else RED for s in xai_signed]

                fig_xai = go.Figure(go.Bar(
                    x=xai_signed, y=feat_names_xai, orientation="h",
                    marker_color=bar_colors,
                    text=[f"+{v:.2f}" if v >= 0 else f"{v:.2f}" for v in xai_signed],
                    textposition="outside", textfont=dict(size=11, color=NAVY), cliponaxis=False,
                ))
                fig_xai.update_layout(
                    paper_bgcolor=CARD, plot_bgcolor=CARD, height=320,
                    margin=dict(l=10, r=60, t=10, b=30),
                    xaxis=dict(title="SHAP value (impact on prediction)",
                               title_font=dict(size=11, color="#64748B"),
                               tickfont=dict(size=10, color="#374151"),
                               gridcolor=BORDER, zeroline=True,
                               zerolinecolor=NAVY, zerolinewidth=1.5),
                    yaxis=dict(tickfont=dict(size=11, color=NAVY), showgrid=False),
                    showlegend=False,
                )
                xai_text = generate_shap_text(pairs, fcast, hist, sel_pid)
                with st.container(border=True):
                    st.markdown(_xai_header("Explainability (SHAP)", xai_text,
                                            "Top Feature Contributions (SHAP)"), unsafe_allow_html=True)
                    st.plotly_chart(fig_xai, use_container_width=True)

            # SHAP mode for EBM: show its native (glass-box) importances
            elif xai_mode == "SHAP" and xai_data and _mt == "EBM":
                pairs          = xai_data["pairs"]
                top10          = pairs[:10]
                feat_names_xai = [FEAT_LABELS.get(f, f) for f, _ in top10][::-1]
                xai_scores     = [float(v) for _, v in top10][::-1]
                fig_xai = go.Figure(go.Bar(
                    x=xai_scores, y=feat_names_xai, orientation="h", marker_color=BLUE,
                    text=[f"{v:.3f}" for v in xai_scores],
                    textposition="outside", textfont=dict(size=11, color=NAVY), cliponaxis=False,
                ))
                fig_xai.update_layout(
                    paper_bgcolor=CARD, plot_bgcolor=CARD, height=320,
                    margin=dict(l=10, r=50, t=10, b=30),
                    xaxis=dict(title="EBM feature importance",
                               title_font=dict(size=11, color="#64748B"),
                               tickfont=dict(size=10, color="#374151"),
                               gridcolor=BORDER, zeroline=False),
                    yaxis=dict(tickfont=dict(size=11, color=NAVY), showgrid=False),
                    showlegend=False,
                )
                xai_text = generate_shap_text(pairs, fcast, hist, sel_pid)
                with st.container(border=True):
                    st.markdown(_xai_header("Explainability (EBM)", xai_text,
                                            "Top Feature Importances (EBM)"), unsafe_allow_html=True)
                    st.plotly_chart(fig_xai, use_container_width=True)

            # LIME: local explanation for the latest prediction (XGBoost / LightGBM)
            elif xai_mode == "LIME" and _mt in ("XGBoost", "LightGBM"):
                with st.spinner("Computing LIME explanation..."):
                    try:
                        # Retrain just this product's model on the fly (fast for
                        # tree models) so we don't have to cache every model.
                        mdl_sel, fcols_sel = train_model(hist, _mt)
                        avail      = [c for c in fcols_sel if c in hist.columns]
                        last_row   = hist[avail].iloc[[-1]]
                        lime_pairs = compute_lime(mdl_sel, fcols_sel, hist[avail], last_row)
                        _render_lime_panel(lime_pairs, fcast, hist, sel_pid)
                    except Exception as e:
                        st.warning(f"LIME could not be computed: {e}")

            elif xai_mode == "LIME" and _mt == "EBM":
                with st.container(border=True):
                    st.markdown(f"""
                      <div style='font-size:20px;font-weight:800;color:{NAVY};
                                  border-bottom:2px solid {BORDER};padding-bottom:10px;margin-bottom:8px'>
                        Explainability</div>
                      <p style='font-size:12px;color:#94A3B8;margin-top:4px'>
                        EBM is already a glass-box model, use <b>SHAP</b> mode to see its built-in explanations.
                      </p>""", unsafe_allow_html=True)

            else:
                with st.container(border=True):
                    st.markdown(f"""
                      <div style='font-size:20px;font-weight:800;color:{NAVY};
                                  border-bottom:2px solid {BORDER};padding-bottom:10px;margin-bottom:8px'>
                        Explainability</div>
                      <p style='font-size:12px;color:#94A3B8;margin-top:4px'>
                        Select <b>XGBoost</b>, <b>EBM</b>, or <b>LightGBM</b> and re-run to see explanations.
                      </p>""", unsafe_allow_html=True)

            vly = pr.get("vs_last_year")
            if vly:
                drivers = vly.get("drivers")
                # XGBoost/LightGBM change-drivers (SHAP) are computed on demand for
                # the selected product only, to keep run_all fast.
                if (not vly.get("no_data") and drivers is None
                        and vly.get("fc_rows") is not None and _mt in ("XGBoost", "LightGBM")):
                    with st.spinner("Computing change drivers..."):
                        try:
                            mdl_sel, fcols_sel = train_model(hist, _mt)
                            cf = _contribs(mdl_sel, fcols_sel, _mt, vly["fc_rows"])
                            cl = _contribs(mdl_sel, fcols_sel, _mt, vly["ly_rows"])
                            delta = {f: cf.get(f, 0.0) - cl.get(f, 0.0)
                                     for f in set(cf) | set(cl)}
                            drivers = sorted(delta.items(), key=lambda kv: -abs(kv[1]))[:8]
                        except Exception:
                            drivers = None
                _render_vs_last_year(vly, drivers)

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
        # Period 0 = the last actually observed period (summed across products),
        # used as the baseline so every forecast period (P1-P4) has a comparison.
        period0_tot = sum(r["history"]["sales"].iloc[-1] for r in results.values())
        traj_tots   = [period0_tot, *period_tots]
        traj_labels = ["Period 0 (observed)", "Period 1", "Period 2", "Period 3", "Period 4"]

        rows_pt = ""
        for i, v in enumerate(traj_tots):
            if i == 0:
                chg_html = "<span style='color:#94A3B8;font-size:10px'>baseline</span>"
            else:
                prev     = traj_tots[i-1]
                ch       = (v - prev) / prev * 100 if prev else 0
                chg_html = badge(ch)
            rows_pt += f"""
          <tr>
            <td><b>{traj_labels[i]}</b></td>
            <td>{v:,.0f}</td>
            <td>{chg_html}</td>
          </tr>"""

        p0v, p4v = traj_tots[0], traj_tots[-1]
        traj_pct = (p4v - p0v) / p0v * 100 if p0v else 0
        traj_lbl = "Growing" if traj_pct > 2 else ("Declining" if traj_pct < -2 else "Flat")
        traj_cls = "b-up" if traj_pct > 2 else ("b-down" if traj_pct < -2 else "b-flat")
        tsign    = f"+{traj_pct:.1f}%" if traj_pct > 0 else f"{traj_pct:.1f}%"

        st.markdown(f"""
        <div class='tile' style='min-height:560px'>
          <div class='tile-label'>Demand Trajectory</div>
          <div class='tile-title'>Period-by-Period Trend</div>
          <div style='margin-bottom:12px'>
            <span class='badge {traj_cls}' style='font-size:14px;padding:4px 12px'>
              {traj_lbl} &nbsp; P0 to P4: {tsign}
            </span>
          </div>
          <table class='tbl'>
            <tr><th>Period</th><th>Total Demand</th><th>vs Prev Period</th></tr>
            {rows_pt}
          </table>
          <p class='note'>Period 0 is the last observed period; each forecast period is compared to the previous one across all products.</p>
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
