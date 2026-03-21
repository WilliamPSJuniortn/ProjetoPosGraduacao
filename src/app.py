import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pipeline import carregar_dados, treinar_modelo, prever_evasao, importancia_features

st.set_page_config(
    page_title="Análise de Evasão Escolar",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Sistema de Análise e Predição de Evasão Escolar")
st.markdown("Dashboard interativo para identificar alunos em risco de evasão.")

# --- Carregar dados e modelo ---
@st.cache_data
def get_dados():
    return carregar_dados()

@st.cache_resource
def get_modelo():
    modelo, encoders, relatorio, matriz = treinar_modelo()
    return modelo, encoders, relatorio, matriz

df = get_dados()
modelo, encoders, relatorio, matriz = get_modelo()

# --- Sidebar ---
st.sidebar.header("Filtros")
cursos = ["Todos"] + sorted(df["curso"].unique().tolist())
curso_selecionado = st.sidebar.selectbox("Curso", cursos)

if curso_selecionado != "Todos":
    df_filtrado = df[df["curso"] == curso_selecionado]
else:
    df_filtrado = df

# --- KPIs ---
total = len(df_filtrado)
evadidos = df_filtrado["evadiu"].sum()
taxa_evasao = round(evadidos / total * 100, 1) if total > 0 else 0
cr_medio = round(df_filtrado["cr"].mean(), 2)
faltas_media = round(df_filtrado["faltas"].mean(), 1)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total de Alunos", total)
col2.metric("Evasões", int(evadidos))
col3.metric("Taxa de Evasão", f"{taxa_evasao}%")
col4.metric("CR Médio", cr_medio)

st.markdown("---")

# --- Gráficos ---
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Evasão por Curso")
    evasao_curso = df.groupby("curso")["evadiu"].mean().reset_index()
    evasao_curso.columns = ["Curso", "Taxa de Evasão"]
    evasao_curso["Taxa de Evasão"] = (evasao_curso["Taxa de Evasão"] * 100).round(1)
    fig1 = px.bar(evasao_curso, x="Curso", y="Taxa de Evasão", color="Taxa de Evasão",
                  color_continuous_scale="RdYlGn_r", text="Taxa de Evasão")
    fig1.update_traces(texttemplate="%{text}%")
    st.plotly_chart(fig1, use_container_width=True)

with col_b:
    st.subheader("CR vs Faltas (por situação de evasão)")
    fig2 = px.scatter(df_filtrado, x="cr", y="faltas", color=df_filtrado["evadiu"].map({0: "Não Evadiu", 1: "Evadiu"}),
                      color_discrete_map={"Não Evadiu": "#2ecc71", "Evadiu": "#e74c3c"},
                      labels={"color": "Situação"}, hover_data=["nome", "curso"])
    st.plotly_chart(fig2, use_container_width=True)

col_c, col_d = st.columns(2)

with col_c:
    st.subheader("Evasão por Situação Financeira")
    fin_ev = df.groupby("situacao_financeira")["evadiu"].mean().reset_index()
    fin_ev.columns = ["Situação Financeira", "Taxa de Evasão"]
    fin_ev["Taxa de Evasão"] = (fin_ev["Taxa de Evasão"] * 100).round(1)
    fig3 = px.pie(fin_ev, names="Situação Financeira", values="Taxa de Evasão",
                  color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig3, use_container_width=True)

with col_d:
    st.subheader("Importância das Variáveis no Modelo")
    imp = importancia_features()
    fig4 = px.bar(imp, x="importancia", y="feature", orientation="h",
                  color="importancia", color_continuous_scale="Blues")
    fig4.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# --- Tabela de alunos em risco ---
st.subheader("Alunos em Risco de Evasão")
em_risco = df_filtrado[df_filtrado["evadiu"] == 1][
    ["nome", "curso", "periodo", "cr", "faltas", "situacao_financeira", "reprovacoes"]
].rename(columns={
    "nome": "Nome", "curso": "Curso", "periodo": "Período",
    "cr": "CR", "faltas": "Faltas", "situacao_financeira": "Sit. Financeira",
    "reprovacoes": "Reprovações"
})
st.dataframe(em_risco, use_container_width=True)

st.markdown("---")

# --- Predição individual ---
st.subheader("🔍 Predição Individual de Evasão")
st.markdown("Preencha os dados de um aluno para prever o risco de evasão.")

with st.form("form_predicao"):
    c1, c2, c3 = st.columns(3)
    with c1:
        curso = st.selectbox("Curso", sorted(df["curso"].unique().tolist()))
        periodo = st.slider("Período", 1, 8, 2)
        idade = st.number_input("Idade", min_value=16, max_value=60, value=20)
    with c2:
        cr = st.slider("Coeficiente de Rendimento (CR)", 0.0, 10.0, 6.0, step=0.1)
        faltas = st.slider("Número de Faltas", 0, 60, 10)
    with c3:
        reprovacoes = st.slider("Reprovações", 0, 10, 0)
        situacao_financeira = st.selectbox("Situação Financeira", ["estavel", "intermediaria", "dificuldade"])

    submitted = st.form_submit_button("Prever Evasão")

if submitted:
    resultado = prever_evasao({
        "curso": curso,
        "periodo": periodo,
        "cr": cr,
        "faltas": faltas,
        "reprovacoes": reprovacoes,
        "idade": idade,
        "situacao_financeira": situacao_financeira,
    })

    prob = resultado["probabilidade_evasao"]

    if resultado["evadiu"]:
        st.error(f"⚠️ **Alto risco de evasão** — Probabilidade: {prob}%")
    else:
        st.success(f"✅ **Baixo risco de evasão** — Probabilidade: {prob}%")

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        title={"text": "Risco de Evasão (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#e74c3c" if prob > 50 else "#2ecc71"},
            "steps": [
                {"range": [0, 30], "color": "#d5f5e3"},
                {"range": [30, 60], "color": "#fdebd0"},
                {"range": [60, 100], "color": "#fadbd8"},
            ],
        }
    ))
    st.plotly_chart(fig_gauge, use_container_width=True)
