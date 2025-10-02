# app.py
# ------------------------------------------------------------
# AUB Online Learning (Hub) – Enrollments App (Branded)
# Design: 3-color system + executive layout
# Features:
#   - KPI cards
#   - Actual vs Forecast (Prophet or Linear) + Backtest WAPE + Scenarios
#   - COR (country) allocation view
#   - Business-first EDA (non-repetitive)
# Files (optional, read if present in working dir):
#   - actual_enrollments.csv   (columns: sem_date, enrollments)
#   - forecast_prophet.csv     (columns incl. ds/sem_date, yhat[, yhat_lower, yhat_upper])
#   - forecast_linear.csv      (columns incl. ds/sem_date, yhat or pred_linear)
#   - forecast_cor.csv         (columns incl. sem_date, Country of Residence (or country), prop/prop_smooth, pred_total/pred)
# ------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# ---------------------- BRAND THEME --------------------------
PRIMARY = "#268b8b"   # teal
ACCENT  = "#f3c417"   # sunflower
PURPLE  = "#87189D"   # purple
TEXT    = "#1C1C1C"
LIGHTBG = "#FFFFFF"
SECBG   = "#F8F8F8"

# Plotly template (consistent colors)
pio.templates["hub"] = pio.templates["plotly_white"]
pio.templates["hub"].layout.update(
    colorway=[PRIMARY, ACCENT, PURPLE, "#8D99AE", "#6c757d"],
    paper_bgcolor=LIGHTBG,
    plot_bgcolor=LIGHTBG,
    font=dict(color=TEXT, family="Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial")
)
pio.templates.default = "hub"

# Polished CSS
st.markdown(f"""
<style>
.block-container {{max-width: 1180px; padding-top: 1rem; padding-bottom: 2rem;}}
h1, h2, h3 {{letter-spacing:.2px;}}
div[data-testid="stMetric"] {{
  border: 1px solid #eaeaea; background:{SECBG}; padding:14px; border-radius:14px;
}}
.stButton>button {{
  border-radius:10px; padding:8px 14px; font-weight:600; background:{PRIMARY}; color:white; border:0;
}}
.stButton>button:hover {{ filter: brightness(0.95); }}
span.tag {{background:{ACCENT}22; color:{PRIMARY}; padding:4px 10px; border-radius:999px; font-size:12px; font-weight:600;}}
hr {{margin: 1rem 0 1.2rem 0;}}
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="AUB Online Learning – Enrollments", page_icon="📈", layout="wide")

# ---------------------- HELPERS ------------------------------
@st.cache_data
def read_csv_if_exists(path, parse_dates=None):
    if os.path.exists(path):
        try:
            return pd.read_csv(path, parse_dates=parse_dates)
        except Exception:
            # fallback if parse fails
            return pd.read_csv(path)
    return None

def _coerce_sem_date(df):
    """Ensure a 'sem_date' datetime column exists (from 'sem_date' or 'ds')."""
    if df is None or len(df) == 0:
        return df
    df = df.copy()
    if "sem_date" in df.columns:
        df["sem_date"] = pd.to_datetime(df["sem_date"])
    elif "ds" in df.columns:
        df["sem_date"] = pd.to_datetime(df["ds"])
    else:
        # try to infer any date-like first column
        for c in df.columns:
            try:
                dt = pd.to_datetime(df[c])
                df["sem_date"] = dt
                break
            except Exception:
                pass
        if "sem_date" not in df.columns:
            raise ValueError("Could not find or infer a date column (sem_date/ds).")
    return df

def _coerce_y_columns(df):
    """Normalize prediction columns to yhat / yhat_lower / yhat_upper if available."""
    if df is None or len(df) == 0:
        return df
    df = df.copy()
    # map common names to yhat
    rename_map = {}
    if "yhat" not in df.columns:
        for alt in ["pred_linear", "pred_total", "pred", "y_pred", "forecast"]:
            if alt in df.columns:
                rename_map[alt] = "yhat"
                break
    if "yhat_lower" not in df.columns:
        for alt in ["lower", "lo80", "lo95"]:
            if alt in df.columns:
                rename_map[alt] = "yhat_lower"
                break
    if "yhat_upper" not in df.columns:
        for alt in ["upper", "hi80", "hi95"]:
            if alt in df.columns:
                rename_map[alt] = "yhat_upper"
                break
    if rename_map:
        df = df.rename(columns=rename_map)
    if "yhat" not in df.columns:
        raise ValueError("Forecast file must have a prediction column (yhat or known alias).")
    return df

def _mk_age_band(s):
    bins = [0, 24, 34, 44, 200]
    labels = ["<25","25–34","35–44","45+"]
    try:
        return pd.cut(pd.to_numeric(s, errors="coerce"), bins=bins, labels=labels, right=True)
    except Exception:
        return pd.Series([np.nan]*len(s))

def _format_int(x):
    try:
        return f"{int(round(x)):,}"
    except Exception:
        return "—"

def _season_label(d: pd.Timestamp):
    # For display: "Spring YYYY" if month=1, "Fall YYYY" if month=8 (or use Month Name)
    if d.month in (1,):
        return f"Spring {d.year}"
    if d.month in (8,):
        return f"Fall {d.year}"
    return d.strftime("%b %Y")

# ---------------------- LOAD DATA ----------------------------
df_actual = read_csv_if_exists("actual_enrollments.csv", parse_dates=["sem_date"])
df_prophet = read_csv_if_exists("forecast_prophet.csv")
df_linear  = read_csv_if_exists("forecast_linear.csv")
df_cor     = read_csv_if_exists("forecast_cor.csv")

# Normalize shapes
if df_actual is not None and len(df_actual):
    df_actual = _coerce_sem_date(df_actual)
    if "enrollments" not in df_actual.columns:
        # try alternate names
        for alt in ["y", "actual", "value", "count", "n"]:
            if alt in df_actual.columns:
                df_actual = df_actual.rename(columns={alt: "enrollments"})
                break

# Choose forecast source: prefer Prophet if present
df_forecast = None
src = None
if df_prophet is not None and len(df_prophet):
    df_prophet = _coerce_sem_date(df_prophet)
    df_prophet = _coerce_y_columns(df_prophet)
    df_forecast = df_prophet.copy()
    src = "Prophet (seasonality-aware)"
elif df_linear is not None and len(df_linear):
    df_linear = _coerce_sem_date(df_linear)
    df_linear = _coerce_y_columns(df_linear)
    df_forecast = df_linear.copy()
    src = "Linear baseline"

# Align forecast columns to minimal set
if df_forecast is not None:
    keep_cols = [c for c in ["sem_date","yhat","yhat_lower","yhat_upper"] if c in df_forecast.columns]
    df_forecast = df_forecast[keep_cols].drop_duplicates(subset=["sem_date"]).sort_values("sem_date")

# ---------- APP HEADER ----------
st.title("AUB Online Learning – Enrollments & Forecasts")
st.markdown('<span class="tag">Hub-branded • Business-first</span>', unsafe_allow_html=True)

# -------------- KPI CARDS ----------------
total_hist = df_actual["enrollments"].sum() if df_actual is not None and "enrollments" in df_actual else np.nan
future_next = np.nan
if df_forecast is not None and len(df_forecast):
    # next future point after last actual
    if df_actual is not None and len(df_actual):
        last_actual_date = df_actual["sem_date"].max()
        fut = df_forecast[df_forecast["sem_date"] > last_actual_date]
        if len(fut):
            future_next = fut.sort_values("sem_date").iloc[0]["yhat"]
    else:
        future_next = df_forecast.sort_values("sem_date").iloc[0]["yhat"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Enrollments (historical)", _format_int(total_hist))
c2.metric("Next Semester Forecast", _format_int(future_next) if not np.isnan(future_next) else "—")
c3.metric("Graduation Rate", "—")   # populate if you have it in another file
c4.metric("Drop Rate", "—")         # populate if you have it in another file
st.divider()

# ----------------- TABS ------------------
tab1, tab2, tab3 = st.tabs(["📈 Forecast", "🌍 Countries (COR)", "🔎 EDA (Business)"])

# ===================== TAB 1: FORECAST =======================
with tab1:
    st.subheader("Actual vs Forecast")
    if df_actual is None or len(df_actual) == 0:
        st.warning("`actual_enrollments.csv` not found or empty.")
    if df_forecast is None or len(df_forecast) == 0:
        st.warning("No forecast file found. Add `forecast_prophet.csv` or `forecast_linear.csv`.")
    if (df_actual is not None and len(df_actual)) or (df_forecast is not None and len(df_forecast)):
        # Main chart with optional MA3 for actuals
        show_ma = st.toggle("Show 3-month moving average (visual aid)", value=False)
        fig = go.Figure()

        # Actuals
        if df_actual is not None and len(df_actual):
            dfa = df_actual.sort_values("sem_date").copy()
            fig.add_trace(go.Scatter(
                x=dfa["sem_date"], y=dfa["enrollments"], mode="lines+markers", name="Actual"
            ))
            if show_ma and len(dfa) >= 2:
                dfa["MA3"] = dfa["enrollments"].rolling(3, min_periods=1).mean()
                fig.add_trace(go.Scatter(
                    x=dfa["sem_date"], y=dfa["MA3"], mode="lines", name="MA3 (Actual)"
                ))

        # Forecast (base, before scenarios)
        if df_forecast is not None and len(df_forecast):
            dff = df_forecast.sort_values("sem_date").copy()
            # --- Backtest on last K using a quick linear fit over actuals ---
            st.subheader("Forecast quality & scenarios")
            K = 2
            if df_actual is not None and len(df_actual) > (K + 2):
                tr = df_actual.sort_values("sem_date").copy()
                tr = tr.assign(t=np.arange(len(tr)))
                b, a = np.polyfit(tr["t"].iloc[:-K], tr["enrollments"].iloc[:-K], 1)
                holdout = tr.iloc[-K:].copy()
                holdout["pred_linear"] = a + b * holdout["t"]
                wape = (np.abs(holdout["enrollments"] - holdout["pred_linear"]).sum()
                        / max(holdout["enrollments"].sum(), 1))
                st.write(f"**Backtest (holdout {K} semesters):** WAPE ≈ **{wape:.1%}**")
                st.caption("Use ± this % as a planning buffer around the seasonal forecast.")
            else:
                st.info("Not enough history to backtest (need > K+2 actual points).")

            # --- Scenario sliders ---
            colA, colB = st.columns(2)
            push = colA.slider("Marketing push impact", -20, 30, 0, help="Percent change vs base forecast (%)")
            price = colB.slider("Scholarship/price sensitivity", -10, 15, 0, help="Percent change vs base forecast (%)")
            factor = (1 + push/100) * (1 + price/100)
            st.write(f"**Scenario applied:** × {factor:.2f}")
            dff_scn = dff.copy()
            if df_actual is not None and len(df_actual):
                last_a = df_actual["sem_date"].max()
                mask_future = dff_scn["sem_date"] > last_a
            else:
                mask_future = np.ones(len(dff_scn)).astype(bool)
            dff_scn.loc[mask_future, "yhat"] *= factor
            if "yhat_lower" in dff_scn.columns:
                dff_scn.loc[mask_future, "yhat_lower"] *= factor
            if "yhat_upper" in dff_scn.columns:
                dff_scn.loc[mask_future, "yhat_upper"] *= factor

            # Confidence band if provided
            if {"yhat_lower","yhat_upper"}.issubset(dff_scn.columns):
                fig.add_traces([
                    go.Scatter(
                        x=dff_scn["sem_date"], y=dff_scn["yhat_upper"],
                        line=dict(width=0), showlegend=False, hoverinfo="skip"
                    ),
                    go.Scatter(
                        x=dff_scn["sem_date"], y=dff_scn["yhat_lower"],
                        fill="tonexty", mode="lines", line=dict(width=0),
                        name="Forecast range"
                    )
                ])

            fig.add_trace(go.Scatter(
                x=dff_scn["sem_date"], y=dff_scn["yhat"],
                mode="lines+markers", name=f"Forecast ({'Prophet' if src.startswith('Prophet') else 'Linear'})"
            ))

            fig.update_layout(
                xaxis_title="Semester",
                yaxis_title="Enrollments",
                legend_title="Series",
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Forecast source: **{src or '—'}**. Fall mapped to August, Spring to January (business timing alignment).")

# ===================== TAB 2: COR ============================
with tab2:
    st.subheader("Where growth will come from (COR split)")
    if df_cor is None or len(df_cor) == 0:
        st.info("Add `forecast_cor.csv` to view the country allocation (top-down split).")
    else:
        df_cor = _coerce_sem_date(df_cor)

        # normalize column names
        corr_col_candidates = ["Country of Residence", "country", "Country", "COR"]
        cor_col = None
        for c in corr_col_candidates:
            if c in df_cor.columns:
                cor_col = c
                break
        if cor_col is None:
            st.error("`forecast_cor.csv` must include a country column (e.g., 'Country of Residence').")
        else:
            # Pred total & proportion columns
            pred_total_col = None
            for c in ["pred_total", "yhat", "forecast_total", "pred"]:
                if c in df_cor.columns:
                    pred_total_col = c
                    break
            prop_col = None
            for c in ["prop_smooth", "prop"]:
                if c in df_cor.columns:
                    prop_col = c
                    break

            # If no per-country prediction, derive from total * share
            if "pred_count" not in df_cor.columns:
                if (pred_total_col is not None) and (prop_col is not None):
                    # Ensure total-by-semester is same across rows, then allocate
                    # (Assumes rows are per (sem_date, country))
                    dtmp = df_cor.copy()
                    dtmp["pred_count"] = dtmp[pred_total_col] * dtmp[prop_col]
                    df_cor = dtmp
                else:
                    st.warning("No 'pred_count' and missing total/share columns; cannot compute allocation.")

            # pick a future semester to view
            df_cor = df_cor.sort_values("sem_date")
            all_semesters = df_cor["sem_date"].drop_duplicates().tolist()
            if len(all_semesters) == 0:
                st.warning("No semesters in COR file.")
            else:
                sem_choice = st.select_slider(
                    "Select semester", options=all_semesters, value=all_semesters[-1],
                    format_func=_season_label
                )
                df_view = df_cor[df_cor["sem_date"] == sem_choice].copy()
                # Normalize display column
                disp = df_view[[cor_col, "pred_count"]].dropna().sort_values("pred_count", ascending=False)
                disp_top = disp.head(12)

                fig_cor = px.bar(
                    disp_top, x="pred_count", y=cor_col, orientation="h",
                    labels={"pred_count": "Forecasted enrollments", cor_col: "Country"},
                )
                fig_cor.update_layout(margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_cor, use_container_width=True)
                st.caption("Top countries by forecasted enrollments (totals reconcile to the headline forecast).")

# ===================== TAB 3: EDA (Business) =================
with tab3:
    st.subheader("Dataset snapshot")
    if df_actual is not None and len(df_actual):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", _format_int(len(df_actual)))
        c2.metric("Columns", str(df_actual.shape[1]))
        c3.metric("Dataset", "actual_enrollments.csv")
        c4.metric("Snapshot", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"))
        with st.expander("Preview (first 5 rows)"):
            st.dataframe(df_actual.head())
    else:
        st.info("`actual_enrollments.csv` not found or empty.")

    st.subheader("Enrollments over time")
    if df_actual is not None and len(df_actual):
        show_ma2 = st.toggle("Show 3-month moving average (EDA)", value=False)
        dfa = df_actual.sort_values("sem_date").copy()
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=dfa["sem_date"], y=dfa["enrollments"], mode="lines+markers", name="Actual"))
        if show_ma2 and len(dfa) >= 2:
            dfa["MA3"] = dfa["enrollments"].rolling(3, min_periods=1).mean()
            fig2.add_trace(go.Scatter(x=dfa["sem_date"], y=dfa["MA3"], mode="lines", name="MA3"))
        fig2.update_layout(xaxis_title="Semester", yaxis_title="Enrollments", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)
    st.caption("Business alignment: Fall ≈ August, Spring ≈ January (marketing windows).")
    st.divider()

    # ----- Business-first profile (OPTIONAL; requires richer student-level df) -----
    st.subheader("Who enrolls? (quick profile)")
    st.caption("If you connect a student-level file, this section will auto-summarize by Country, Age Band, Employment.")
    uploaded = st.file_uploader("Optional: upload a student-level CSV (with columns like Country of Residence, Age, Employment, Final Status)", type=["csv"])
    if uploaded:
        df_main = pd.read_csv(uploaded)
        # Country
        cor_col = None
        for c in ["Country of Residence","country","Country","COR"]:
            if c in df_main.columns:
                cor_col = c; break
        if cor_col:
            top_c = (df_main[cor_col].value_counts(normalize=True).head(5)*100).round(1).rename("Share %")
        else:
            top_c = pd.Series(dtype=float)

        # Age bands
        age_col = None
        for c in ["Age","age"]:
            if c in df_main.columns:
                age_col = c; break
        age_share = pd.Series(dtype=float)
        if age_col:
            age_share = _mk_age_band(df_main[age_col]).value_counts(normalize=True).sort_index()*100
            age_share = age_share.round(1).rename("Share %")

        # Employment
        emp_col = None
        for c in ["Employment","employment","Employment status"]:
            if c in df_main.columns:
                emp_col = c; break
        emp_share = pd.Series(dtype=float)
        if emp_col:
            emp_share = (df_main[emp_col].value_counts(normalize=True).head(5)*100).round(1).rename("Share %")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.write("**Top Countries (share)**")
            st.dataframe(top_c)
        with c2:
            st.write("**Age Bands (share)**")
            st.dataframe(age_share)
        with c3:
            st.write("**Employment (share)**")
            st.dataframe(emp_share)

        # Outcomes at a glance
        if "Final Status" in df_main.columns:
            st.subheader("Outcomes at a glance")
            out = df_main["Final Status"].value_counts(normalize=True).mul(100).round(1).rename("Share %")
            st.dataframe(out)

        # Readiness badges (tiny)
        st.subheader("Data readiness (quick checks)")
        badges = []
        miss_ok = (df_main.isna().mean().max() < 0.20)
        badges.append("Missingness ✅" if miss_ok else "Missingness ⚠️")
        dups = int(df_main.duplicated().sum())
        badges.append("Duplicates ✅" if dups == 0 else f"Duplicates ⚠️ ({dups})")
        if "Final Status" in df_main.columns:
            cls = df_main["Final Status"].value_counts(normalize=True)
            badges.append("Class balance ✅" if (len(cls)>0 and cls.min()>0.15) else "Class balance ⚠️")
        st.write(" • ".join(badges))
        st.caption("These checks justify preprocessing decisions (impute, dedupe, class weights/SMOTE).")

# ----------------- FOOTER -------------------
st.divider()
st.caption("© AUB Online Learning (Hub) — Branded Streamlit app • Built for business storytelling.")

