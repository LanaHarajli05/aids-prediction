# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Diploma Forecast", layout="wide")
st.title("AI & DS – Forecast Dashboard")

BASE = Path(__file__).parent

# ---------- Helpers ----------
def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(f"Missing file: {path.name}. Please add it to the repo root.")
        st.stop()
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Could not read {path.name}: {e}")
        st.stop()

def coerce_sem_date(df: pd.DataFrame, possible_cols=("sem_date", "ds", "date", "semester")) -> pd.DataFrame:
    # Find a column that represents the semester/date
    col = None
    for c in possible_cols:
        if c in df.columns:
            col = c
            break
    if col is None:
        st.error(f"Could not find a date column in {list(df.columns)}.\n"
                 f"Expected one of: {possible_cols}")
        st.stop()
    if col != "sem_date":
        df = df.rename(columns={col: "sem_date"})
    # Parse to datetime; be flexible with formats
    df["sem_date"] = pd.to_datetime(df["sem_date"], errors="coerce")
    if df["sem_date"].isna().all():
        st.error("Could not parse any 'sem_date' values as dates.")
        st.stop()
    return df

def ensure_columns(df: pd.DataFrame, required: dict) -> pd.DataFrame:
    """
    required: mapping {desired_name: [list of acceptable aliases]}
    Renames the first alias found to desired_name; errors if none found.
    """
    rename_map = {}
    for want, aliases in required.items():
        found = None
        for a in [want] + aliases:
            if a in df.columns:
                found = a
                break
        if found is None:
            st.error(f"Missing required column for '{want}'. "
                     f"Looked for aliases: {aliases + [want]}")
            st.stop()
        if found != want:
            rename_map[found] = want
    if rename_map:
        df = df.rename(columns=rename_map)
    return df

def pick_forecast_file():
    """Prefer prophet file if present, else linear."""
    fp_prophet = BASE / "forecast_prophet.csv"
    fp_linear  = BASE / "forecast_linear.csv"
    if fp_prophet.exists():
        return fp_prophet, "prophet"
    elif fp_linear.exists():
        return fp_linear, "linear"
    else:
        st.error("No forecast file found. Add 'forecast_prophet.csv' or 'forecast_linear.csv' to the repo root.")
        st.stop()

def detect_country_col(df: pd.DataFrame) -> str:
    # Likely names for the country column
    candidates = ["COR", "Country", "Country of Residence", "country", "Country_of_Residence"]
    for c in candidates:
        if c in df.columns:
            return c
    # otherwise guess: the first non-date, non-numeric column
    for c in df.columns:
        if c == "sem_date":
            continue
        if df[c].dtype == "object":
            return c
    st.error("Could not detect the country column in COR forecast file.")
    st.stop()

# ---------- Load data (root) ----------
actual_path = BASE / "actual_enrollments.csv"
cor_path    = BASE / "forecast_cor.csv"
fc_path, fc_kind = pick_forecast_file()

actual = read_csv_safe(actual_path)
fc     = read_csv_safe(fc_path)
cor    = read_csv_safe(cor_path)

# Standardize/validate Actuals
actual = coerce_sem_date(actual)
actual = ensure_columns(actual, {"enrollments": ["count", "total", "Enrollments"]})
actual = actual[["sem_date", "enrollments"]].sort_values("sem_date")

# Standardize/validate Forecast
fc = coerce_sem_date(fc)

# Handle prophet vs linear columns
if fc_kind == "prophet":
    # Expect yhat and maybe intervals; if missing, create placeholders
    if "yhat" not in fc.columns and "pred_total" in fc.columns:
        fc = fc.rename(columns={"pred_total": "yhat"})
    if "yhat" not in fc.columns and "pred_linear" in fc.columns:
        fc = fc.rename(columns={"pred_linear": "yhat"})
else:
    # linear: often 'pred_linear'
    if "pred_linear" in fc.columns and "yhat" not in fc.columns:
        fc = fc.rename(columns={"pred_linear": "yhat"})

# Ensure core columns
for c in ["yhat"]:
    if c not in fc.columns:
        st.error(f"Forecast file is missing '{c}'. Columns are: {list(fc.columns)}")
        st.stop()

# Create intervals if absent
if "yhat_lower" not in fc.columns:
    fc["yhat_lower"] = np.nan
if "yhat_upper" not in fc.columns:
    fc["yhat_upper"] = np.nan

fc = fc[["sem_date", "yhat", "yhat_lower", "yhat_upper"]].sort_values("sem_date")

# Standardize/validate COR forecast
cor = coerce_sem_date(cor)
# We need a country column and a predicted count
country_col = detect_country_col(cor)

# If pred_count missing, attempt to compute from 'pred_total' and 'prop_smooth' or 'prop'
if "pred_count" not in cor.columns:
    if "pred_total" in cor.columns and ("prop_smooth" in cor.columns or "prop" in cor.columns):
        prop_col = "prop_smooth" if "prop_smooth" in cor.columns else "prop"
        cor["pred_count"] = (cor["pred_total"] * cor[prop_col]).round().astype("Int64")
    else:
        st.error("COR file missing 'pred_count' and can't compute it (need either 'pred_total' & 'prop_smooth'/'prop').")
        st.stop()

# Keep only needed columns
keep_cols = ["sem_date", country_col, "pred_count"]
cor = cor[keep_cols].rename(columns={country_col: "Country"}).sort_values(["sem_date", "pred_count"], ascending=[True, False])

# ---------- UI ----------
tab1, tab2 = st.tabs(["Enrollments Forecast", "COR Forecast"])

# ===== Tab 1: Enrollments =====
with tab1:
    st.metric("Actual Total Enrollments", int(actual["enrollments"].sum()))

    # Plot Actual vs Forecast
    plot_df = pd.concat([
        actual.rename(columns={"enrollments": "value"}).assign(kind="Actual")[["sem_date", "value", "kind"]],
        fc.rename(columns={"yhat": "value"})[["sem_date", "value"]].assign(kind="Forecast")
    ])

    fig = px.line(plot_df, x="sem_date", y="value", color="kind", markers=True,
                  title="Actual vs. Forecasted Enrollments")
    st.plotly_chart(fig, use_container_width=True)

    # Forecast table (with intervals if present)
    st.subheader("Forecast Table")
    ft = fc.copy()
    ft = ft.rename(columns={"sem_date": "Semester", "yhat": "Pred", "yhat_lower": "Low", "yhat_upper": "High"})
    st.dataframe(ft, use_container_width=True)

# ===== Tab 2: COR =====
with tab2:
    st.caption("Forecasted enrollments by Country of Residence for future semesters.")
    future_sems = sorted(cor["sem_date"].unique())
    if not future_sems:
        st.warning("No future semesters found in COR forecast.")
    else:
        sem_sel = st.selectbox(
            "Select a future semester",
            future_sems,
            format_func=lambda d: pd.to_datetime(d).strftime("%b %Y")
        )

        cor_sub = cor[cor["sem_date"] == sem_sel].sort_values("pred_count", ascending=False)
        if cor_sub.empty:
            st.info("No COR data for the selected semester.")
        else:
            fig2 = px.bar(
                cor_sub,
                x="pred_count", y="Country",
                orientation="h",
                title=f"Future COR Breakdown – {pd.to_datetime(sem_sel).strftime('%b %Y')}"
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.dataframe(
                cor_sub.rename(columns={"sem_date": "Semester", "pred_count": "Predicted Enrollments"}),
                use_container_width=True
            )

# Footer tip
st.caption("Tip: If you change the CSVs in the repo, click 'Rerun' on Streamlit to refresh.")
