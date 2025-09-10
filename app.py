import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# ---------- File paths ----------
BASE = Path(__file__).parent
actual_path = BASE / "actual_enrollments.csv"
fc_path = BASE / "forecast_prophet.csv"   # or use forecast_linear.csv if you prefer
cor_path = BASE / "forecast_cor.csv"

# ---------- Load data ----------
actual = pd.read_csv(actual_path, parse_dates=["sem_date"])
fc     = pd.read_csv(fc_path, parse_dates=["sem_date"])
cor    = pd.read_csv(cor_path, parse_dates=["sem_date"])

# ---------- Streamlit layout ----------
st.set_page_config(page_title="Diploma Forecast", layout="wide")
st.title("AI & DS – Forecast Dashboard")

tab1, tab2 = st.tabs(["Enrollments Forecast", "COR Forecast"])

# ---------- Tab 1: Enrollments ----------
with tab1:
    st.metric("Actual Total Enrollments", int(actual["enrollments"].sum()))  # e.g., 425

    # Merge actual + forecast for plotting
    df_plot = pd.concat([
        actual.rename(columns={"enrollments":"value"}).assign(kind="Actual")[["sem_date","value","kind"]],
        fc.rename(columns={"yhat":"value"})[["sem_date","value"]].assign(kind="Forecast")
    ])

    fig = px.line(df_plot, x="sem_date", y="value", color="kind", markers=True,
                  title="Actual vs Forecasted Enrollments")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast Table")
    st.dataframe(
        fc[["sem_date","yhat","yhat_lower","yhat_upper"]]
        .rename(columns={"sem_date":"Semester","yhat":"Pred","yhat_lower":"Low","yhat_upper":"High"})
    )

# ---------- Tab 2: COR ----------
with tab2:
    sems = sorted(cor["sem_date"].unique())
    sem = st.selectbox("Select future semester", sems, format_func=lambda d: pd.to_datetime(d).strftime("%b %Y"))
    sub = cor[cor["sem_date"] == sem].sort_values("pred_count", ascending=False)

    fig2 = px.bar(sub, x="pred_count", y=sub.columns.tolist()[1], orientation="h",
                  title=f"Future COR Breakdown – {pd.to_datetime(sem).strftime('%b %Y')}")
    st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(
        sub.rename(columns={"sem_date":"Semester","pred_count":"Predicted Enrollments"})
    )
