%%writefile app.py
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Diploma Forecast", layout="wide")
st.title("AI & DS – Forecast Dashboard")

tab1, tab2 = st.tabs(["Enrollments Forecast", "COR Forecast"])

with tab1:
    actual = pd.read_csv("outputs/actual_enrollments.csv", parse_dates=["sem_date"])
    fc = pd.read_csv("outputs/forecast_prophet.csv", parse_dates=["sem_date"])  # or linear
    st.metric("Actual Total Enrollments", int(actual["enrollments"].sum()))  # 425

    df_plot = pd.concat([
        actual.rename(columns={"enrollments":"value"}).assign(kind="Actual")[["sem_date","value","kind"]],
        fc.rename(columns={"yhat":"value"})[["sem_date","value"]].assign(kind="Forecast")
    ])
    fig = px.line(df_plot, x="sem_date", y="value", color="kind", markers=True)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast Table")
    st.dataframe(fc[["sem_date","yhat","yhat_lower","yhat_upper"]].rename(
        columns={"sem_date":"Semester","yhat":"Pred","yhat_lower":"Low","yhat_upper":"High"}))

with tab2:
    cor = pd.read_csv("outputs/forecast_cor.csv", parse_dates=["sem_date"])
    sem = st.selectbox("Select future semester", sorted(cor["sem_date"].unique()))
    sub = cor[cor["sem_date"]==sem].sort_values("pred_count", ascending=False)
    fig2 = px.bar(sub, x="pred_count", y=sub.columns.tolist()[1], orientation="h",
                  title=f"Future COR Breakdown – {sem.date()}")
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(sub.rename(columns={"sem_date":"Semester","pred_count":"Predicted Enrollments"}))
