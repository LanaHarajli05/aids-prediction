# app.py — AI & DS: EDA & Forecast Dashboard (Unnamed cols hidden, PII redacted, total=Count max)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import re

st.set_page_config(page_title="AI & DS – EDA & Forecast", layout="wide")
st.title("AI & DS – EDA & Forecast Dashboard")

BASE = Path(__file__).parent

# ---------- helpers ----------
def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(f"Missing file: {path.name}.")
        st.stop()
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Could not read {path.name}: {e}")
        st.stop()

def coerce_sem_date(df: pd.DataFrame, possible_cols=("sem_date", "ds", "date", "semester")) -> pd.DataFrame:
    col = next((c for c in possible_cols if c in df.columns), None)
    if col is None:
        st.error(f"No date/semester column found in {list(df.columns)}; expected one of {possible_cols}.")
        st.stop()
    if col != "sem_date":
        df = df.rename(columns={col: "sem_date"})
    df["sem_date"] = pd.to_datetime(df["sem_date"], errors="coerce")
    if df["sem_date"].isna().all():
        st.error("Could not parse any 'sem_date' values as dates.")
        st.stop()
    return df

def ensure_columns(df: pd.DataFrame, required: dict) -> pd.DataFrame:
    rename_map = {}
    for want, aliases in required.items():
        found = next((c for c in [want] + aliases if c in df.columns), None)
        if not found:
            st.error(f"Missing column '{want}' (aliases tried: {aliases}). Columns: {list(df.columns)}")
            st.stop()
        if found != want:
            rename_map[found] = want
    return df.rename(columns=rename_map) if rename_map else df

def pick_forecast_file():
    p_prophet = BASE / "forecast_prophet.csv"
    p_linear  = BASE / "forecast_linear.csv"
    if p_prophet.exists(): return p_prophet, "prophet"
    if p_linear.exists():  return p_linear, "linear"
    st.error("No forecast CSV found. Add 'forecast_prophet.csv' or 'forecast_linear.csv' to the repo root.")
    st.stop()

def detect_country_col(df: pd.DataFrame) -> str:
    for c in ["COR", "Country", "Country of Residence", "country", "Country_of_Residence"]:
        if c in df.columns: return c
    for c in df.columns:
        if c != "sem_date" and df[c].dtype == "object": return c
    st.error("Could not detect the country column in COR forecast file.")
    st.stop()

def std_country(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.upper()
    mapping = {
        "LEBANON": "Lebanon",
        "UAE": "UAE", "UNITED ARAB EMIRATES": "UAE", "EMIRATES": "UAE",
        "KSA": "Saudi Arabia", "SAUDI ARABIA": "Saudi Arabia",
        "QATAR": "Qatar", "KUWAIT": "Kuwait",
        "USA": "USA", "UNITED STATES": "USA",
    }
    return s.map(mapping).fillna(s.str.title())

# --- display mapping: Fall→Aug 1, Spring→Jan 1
def adjust_sem_month(dt: pd.Timestamp) -> pd.Timestamp:
    if pd.isna(dt): return dt
    y, m = dt.year, dt.month
    if m == 10:  # Fall encoded as Oct -> show Aug
        return pd.Timestamp(y, 8, 1)
    if m == 3:   # Spring encoded as Mar -> show Jan
        return pd.Timestamp(y, 1, 1)
    return dt

def apply_sem_adjustment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sem_date"] = df["sem_date"].map(adjust_sem_month)
    return df

# ---------- load CSV artifacts ----------
actual_path = BASE / "actual_enrollments.csv"
cor_path    = BASE / "forecast_cor.csv"
fc_path, fc_kind = pick_forecast_file()

actual = read_csv_safe(actual_path)
fc     = read_csv_safe(fc_path)
cor    = read_csv_safe(cor_path)

# actuals
actual = coerce_sem_date(actual)
actual = ensure_columns(actual, {"enrollments": ["count", "total", "Enrollments"]})
actual = apply_sem_adjustment(actual).sort_values("sem_date")[["sem_date","enrollments"]]

# forecast
fc = coerce_sem_date(fc)
if fc_kind == "prophet":
    if "yhat" not in fc.columns and "pred_total" in fc.columns: fc = fc.rename(columns={"pred_total": "yhat"})
    if "yhat" not in fc.columns and "pred_linear" in fc.columns: fc = fc.rename(columns={"pred_linear": "yhat"})
else:
    if "pred_linear" in fc.columns and "yhat" not in fc.columns: fc = fc.rename(columns={"pred_linear": "yhat"})
if "yhat" not in fc.columns:
    st.error(f"Forecast file missing 'yhat'. Columns: {list(fc.columns)}")
    st.stop()
if "yhat_lower" not in fc.columns: fc["yhat_lower"] = np.nan
if "yhat_upper" not in fc.columns: fc["yhat_upper"] = np.nan
fc = apply_sem_adjustment(fc).sort_values("sem_date")[["sem_date","yhat","yhat_lower","yhat_upper"]]

# COR
cor = coerce_sem_date(cor)
country_col = detect_country_col(cor)
if "pred_count" not in cor.columns:
    if "pred_total" in cor.columns and ("prop_smooth" in cor.columns or "prop" in cor.columns):
        prop_col = "prop_smooth" if "prop_smooth" in cor.columns else "prop"
        cor["pred_count"] = (cor["pred_total"] * cor[prop_col]).round().astype("Int64")
    else:
        st.error("COR CSV missing 'pred_count' (and cannot compute from 'pred_total' * 'prop').")
        st.stop()
cor["Country"] = std_country(cor[country_col])
cor = apply_sem_adjustment(cor)
cor["pred_count"] = pd.to_numeric(cor["pred_count"], errors="coerce").fillna(0).astype(int)
cor = (cor.groupby(["sem_date","Country"], as_index=False)["pred_count"]
         .sum()
         .sort_values(["sem_date","pred_count"], ascending=[True, False]))

# ---------- load Excel for EDA & headline total ----------
excel_file = BASE / "AI & DS Enrolled Students Course .xlsx"
raw_df = pd.DataFrame()
true_total = None

if excel_file.exists():
    try:
        try:
            raw_df = pd.read_excel(excel_file, sheet_name="All Enrolled (2)")
        except Exception:
            raw_df = pd.read_excel(excel_file, sheet_name="All Enrolled")
        # drop Unnamed before anything else
        raw_df = raw_df.loc[:, ~raw_df.columns.str.startswith("Unnamed:")]
        # compute true_total
        if "Count" in raw_df.columns:
            count_series = pd.to_numeric(raw_df["Count"], errors="coerce")
            if count_series.notna().any():
                true_total = int(count_series.max())  # -> 425 for your sheet
        if true_total is None:
            # fallback: count rows that have any non-null in a few identity columns
            cols_for_presence = [c for c in ["Gender","COR","Age","University","Major","Degree Type"] if c in raw_df.columns]
            if cols_for_presence:
                mask = raw_df[cols_for_presence].notna().any(axis=1)
                true_total = int(mask.sum())
            else:
                true_total = int(len(raw_df))
    except Exception:
        true_total = None

# ---- PII redaction for preview ----
def redact_pii(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = []
    for c in df.columns:
        lc = c.lower()
        # keep "Name of University", drop other name-like columns
        if ("name" in lc and "university" not in lc) and any(k in lc for k in ["name", "full"]):
            drop_cols.append(c); continue
        if any(k in lc for k in ["email", "phone", "mobile", "whatsapp"]):
            drop_cols.append(c)
    # also ensure all Unnamed are hidden (if any slipped through)
    drop_cols.extend([c for c in df.columns if c.startswith("Unnamed:")])
    return df.drop(columns=list(set(drop_cols)), errors="ignore")

# ---------- UI ----------
tab_eda, tab1, tab2 = st.tabs(["EDA", "Enrollments Forecast", "COR Forecast"])

# ===== TAB: EDA =====
with tab_eda:
    st.subheader("Exploratory Data Analysis (AI & DS)")

    if raw_df.empty:
        st.warning("Raw Excel not found in repo root or sheet name mismatch (expected 'All Enrolled (2)').")
    else:
        # 1) Preview FIRST (PII & Unnamed hidden)
        st.dataframe(redact_pii(raw_df.copy()).head(50), use_container_width=True)

        # 2) Overview KPIs
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(raw_df):,}")
        c2.metric("Columns", f"{raw_df.shape[1]:,}")
        c3.metric("Total Enrollments", f"{true_total:,}" if true_total is not None else "—")

        # 3) Demographics snapshot (exclude English test fields; ignore Unnamed)
        demo_pool = []
        for c in raw_df.columns:
            if c.startswith("Unnamed:"):  # hide from options
                continue
            lc = c.lower()
            if any(k in lc for k in ["gender","age","job","occupation","degree","education","major","country","city","cohort","admit"]):
                if not ("english" in lc and "test" in lc):
                    demo_pool.append(c)

        st.markdown("### Demographics at a Glance")
        default_demo = [c for c in demo_pool if any(k in c.lower() for k in ["gender","age","degree","country"])][:5]
        demo_cols = st.multiselect(
            "Pick demographic columns to summarize",
            sorted(set(demo_pool)),
            default=default_demo
        )

        for col in demo_cols:
            series = raw_df[col].astype(str).str.strip()
            series = series.replace({"nan": np.nan, "None": np.nan}).dropna()
            vc = series.value_counts().head(15)
            if vc.empty:
                continue
            if "gender" in col.lower():
                fig = px.pie(names=vc.index, values=vc.values, title=f"Distribution — {col}")
            else:
                fig = px.bar(vc[::-1], orientation="h", title=f"Top values — {col}")
            st.plotly_chart(fig, use_container_width=True)

        # 4) Cohort/Admit timeline (Fall→Aug, Spring→Jan, Summer→Jul)
        def clean_cohort_text(s):
            if pd.isna(s): return s
            s = str(s).strip().replace(" -", "-").replace("- ", "-")
            return s
        def cohort_to_date(cohort):
            if pd.isna(cohort): return pd.NaT
            txt = str(cohort).strip()
            m = re.match(r'^(Fall|Spring|Summer)\s+(\d{2})-(\d{2})$', txt)
            if not m: return pd.NaT
            season, y1, y2 = m.groups()
            y1, y2 = 2000 + int(y1), 2000 + int(y2)
            if season == "Fall":   return pd.Timestamp(y1, 8, 1)
            if season == "Spring": return pd.Timestamp(y2, 1, 1)
            if season == "Summer": return pd.Timestamp(y2, 7, 1)
            return pd.NaT

        coh_col = next((c for c in raw_df.columns if "cohort" in c.lower() or ("admit" in c.lower() and "semester" in c.lower())), None)
        if coh_col:
            tmp = raw_df.copy()
            tmp["Cohort_clean"] = tmp[coh_col].map(clean_cohort_text)
            tmp["sem_date"] = tmp["Cohort_clean"].map(cohort_to_date)
            ts = (tmp.dropna(subset=["sem_date"])
                    .groupby("sem_date").size().rename("enrollments")
                    .reset_index().sort_values("sem_date"))
            if not ts.empty:
                fig_ts = px.line(ts, x="sem_date", y="enrollments", markers=True, title="Enrollments by Cohort")
                st.plotly_chart(fig_ts, use_container_width=True)

# ===== TAB 1: Enrollments Forecast (unchanged) =====
with tab1:
    st.metric("Actual Total Enrollments", int(true_total) if true_total is not None else int(actual["enrollments"].sum()))

    plot_df = pd.concat([
        actual.rename(columns={"enrollments":"value"}).assign(kind="Actual")[["sem_date","value","kind"]],
        fc.rename(columns={"yhat":"value"})[["sem_date","value"]].assign(kind="Forecast")
    ], ignore_index=True)

    all_sems = sorted(plot_df["sem_date"].unique())
    if all_sems:
        default_range = (all_sems[0], all_sems[-1])
        sem_range = st.select_slider(
            "Show range",
            options=all_sems,
            value=default_range,
            format_func=lambda d: pd.to_datetime(d).strftime("%b %Y"),
        )
        mask = (plot_df["sem_date"] >= sem_range[0]) & (plot_df["sem_date"] <= sem_range[1])
        fig = px.line(plot_df[mask], x="sem_date", y="value", color="kind", markers=True,
                      title="Actual vs. Forecasted Enrollments")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for plotting.")

    last_actual = actual["sem_date"].max() if not actual.empty else None
    future_only = fc[fc["sem_date"] > last_actual] if last_actual is not None else fc.copy()

    if not future_only.empty:
        target_sem = st.selectbox(
            "Target semester",
            list(future_only["sem_date"]),
            format_func=lambda d: pd.to_datetime(d).strftime("%b %Y"),
        )
        target_val = int(future_only.loc[future_only["sem_date"] == target_sem, "yhat"].iloc[0])
        st.metric(f"Predicted Enrollments – {pd.to_datetime(target_sem).strftime('%b %Y')}", target_val)

        st.subheader("Next 4 Semesters")
        next4 = future_only.head(4).copy()
        next4_disp = (next4.assign(Semester=next4["sem_date"].dt.strftime("%b %Y"))
                             [["Semester","yhat"]]
                             .rename(columns={"yhat":"Predicted Enrollments"}))
        st.dataframe(next4_disp, use_container_width=True, hide_index=True)
    else:
        st.info("No future forecast rows found.")

# ===== TAB 2: COR Forecast (unchanged visuals) =====
with tab2:
    future_sems = sorted(cor["sem_date"].unique())
    if future_sems:
        sem_sel = st.selectbox(
            "Select a future semester",
            future_sems,
            format_func=lambda d: pd.to_datetime(d).strftime("%b %Y"),
        )
        cor_sub = cor[cor["sem_date"] == sem_sel].sort_values("pred_count", ascending=False)
        fig2 = px.bar(
            cor_sub, x="pred_count", y="Country",
            orientation="h",
            title=f"Future COR Breakdown – {pd.to_datetime(sem_sel).strftime('%b %Y')}",
        )
        st.plotly_chart(fig2, use_container_width=True)

        tbl = (cor_sub.assign(Semester=pd.to_datetime(sem_sel).strftime("%b %Y"))
                        [["Semester","Country","pred_count"]]
                        .rename(columns={"pred_count":"Predicted Enrollments"}))
        st.dataframe(tbl, use_container_width=True, hide_index=True)
    else:
        st.info("No COR forecast data found.")
