import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
)

# ---------------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Predição de Evasão Escolar",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS customizado
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Fundo e fonte geral */
    .main { background-color: #f8f9fb; }
    h1 { font-size: 2rem !important; }
    h2 { font-size: 1.35rem !important; color: #2c3e50; }
    h3 { font-size: 1.1rem !important; color: #34495e; }

    /* KPI cards */
    [data-testid="metric-container"] {
        background: white;
        border: 1px solid #e8ecf0;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    [data-testid="metric-container"] label {
        font-size: 0.8rem !important;
        color: #7f8c8d !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.9rem !important;
        font-weight: 700 !important;
        color: #2c3e50 !important;
    }

    /* Badge de sprint */
    .badge-sprint1 {
        background: #d5e8d4; color: #27ae60;
        border-radius: 20px; padding: 3px 12px;
        font-size: 0.75rem; font-weight: 700;
        display: inline-block; margin-bottom: 4px;
    }
    .badge-sprint2 {
        background: #dae8fc; color: #2980b9;
        border-radius: 20px; padding: 3px 12px;
        font-size: 0.75rem; font-weight: 700;
        display: inline-block; margin-bottom: 4px;
    }

    /* Tabela de métricas */
    .metric-table th {
        background: #2c3e50; color: white;
        padding: 10px 16px; text-align: center;
    }
    .metric-table td {
        padding: 8px 16px; text-align: center; border-bottom: 1px solid #ecf0f1;
    }
    .metric-table tr:hover { background: #f4f6f8; }

    /* Alerta de risco */
    .risco-alto   { background:#fde8e8; border-left:4px solid #e74c3c;
                    padding:12px 16px; border-radius:6px; margin:8px 0; }
    .risco-baixo  { background:#e8f8f0; border-left:4px solid #2ecc71;
                    padding:12px 16px; border-radius:6px; margin:8px 0; }
    .risco-medio  { background:#fef9e7; border-left:4px solid #f39c12;
                    padding:12px 16px; border-radius:6px; margin:8px 0; }

    /* Divider com label */
    .section-label {
        font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 1px; color: #95a5a6; margin: 4px 0 10px 0;
    }

    /* Tab styling */
    button[data-baseweb="tab"] { font-size: 0.95rem !important; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_LIMPO  = os.path.join(BASE_DIR, "data", "alunos_limpo.csv")
DATA_BRUTO  = os.path.join(BASE_DIR, "data", "alunos.csv")
PREP_PATH   = os.path.join(BASE_DIR, "models", "preprocessador.pkl")
LR_PATH     = os.path.join(BASE_DIR, "models", "modelo_regressao_logistica.pkl")
RF_PATH     = os.path.join(BASE_DIR, "models", "modelo_random_forest.pkl")

COLUNAS_NUMERICAS   = ["periodo", "cr", "faltas", "reprovacoes", "idade"]
COLUNAS_CATEGORICAS = ["curso", "situacao_financeira"]

# ---------------------------------------------------------------------------
# Carregamento de dados e modelos (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def get_dados():
    return pd.read_csv(DATA_LIMPO)

@st.cache_resource
def get_sprint2():
    preprocessador = joblib.load(PREP_PATH)
    lr = joblib.load(LR_PATH)
    rf = joblib.load(RF_PATH)
    return preprocessador, lr, rf

@st.cache_data
def get_metricas_sprint2():
    df = get_dados()
    preprocessador, lr, rf = get_sprint2()

    X = df.drop(columns=["id_aluno", "nome", "evadiu"])
    y = df["evadiu"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_test_prep = preprocessador.transform(X_test)

    resultados = {}
    for nome, modelo in [("Regressão Logística", lr), ("Random Forest", rf)]:
        y_pred  = modelo.predict(X_test_prep)
        y_proba = modelo.predict_proba(X_test_prep)[:, 1]
        resultados[nome] = {
            "y_pred" : y_pred,
            "y_proba": y_proba,
            "y_test" : y_test.values,
            "metricas": {
                "Acurácia" : round(accuracy_score(y_test, y_pred), 4),
                "Precisão" : round(precision_score(y_test, y_pred, zero_division=0), 4),
                "Recall"   : round(recall_score(y_test, y_pred, zero_division=0), 4),
                "F1-Score" : round(f1_score(y_test, y_pred, zero_division=0), 4),
                "ROC-AUC"  : round(roc_auc_score(y_test, y_proba), 4),
            },
        }
    return resultados

# ---------------------------------------------------------------------------
# Header principal
# ---------------------------------------------------------------------------
st.markdown("""
<div style="display:flex; align-items:center; gap:16px; margin-bottom:8px;">
    <span style="font-size:2.6rem;">🎓</span>
    <div>
        <h1 style="margin:0; color:#2c3e50;">Sistema de Predição de Evasão Escolar</h1>
        <p style="margin:0; color:#7f8c8d; font-size:0.95rem;">
            Análise exploratória · Modelagem de classificação · Predição individual
        </p>
    </div>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🔧 Configurações")
    st.markdown('<p class="section-label">Filtro de dados</p>', unsafe_allow_html=True)

    df = get_dados()
    cursos_disponiveis = ["Todos"] + sorted(df["curso"].unique().tolist())
    curso_selecionado  = st.selectbox("Curso", cursos_disponiveis)

    st.markdown('<p class="section-label">Modelo para predição</p>', unsafe_allow_html=True)
    modelo_nome = st.radio(
        "Selecione o modelo",
        ["Regressão Logística", "Random Forest"],
        index=0,
        help="Ambos treinados na Sprint 2 com StandardScaler + OneHotEncoder",
    )

    st.markdown("---")
    st.markdown("#### 📁 Sprints")
    st.markdown("""
    <div class='badge-sprint1'>✅ Sprint 1 — EDA</div><br>
    <div class='badge-sprint2'>✅ Sprint 2 — Modelagem</div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📊 Dataset")
    st.caption(f"**{len(df)}** registros · **{len(df.columns)}** colunas")
    st.caption(f"Taxa de evasão global: **{df['evadiu'].mean()*100:.1f}%**")

# Filtro
df_filtrado = df if curso_selecionado == "Todos" else df[df["curso"] == curso_selecionado]

# ---------------------------------------------------------------------------
# KPIs globais
# ---------------------------------------------------------------------------
total      = len(df_filtrado)
evadidos   = int(df_filtrado["evadiu"].sum())
taxa       = round(evadidos / total * 100, 1) if total > 0 else 0
cr_medio   = round(df_filtrado["cr"].mean(), 2)
faltas_med = round(df_filtrado["faltas"].mean(), 1)
repr_med   = round(df_filtrado["reprovacoes"].mean(), 1)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("👥 Total de Alunos",  total)
k2.metric("🚨 Alunos Evadidos",  evadidos)
k3.metric("📉 Taxa de Evasão",   f"{taxa}%",    delta=f"{taxa - df['evadiu'].mean()*100:.1f}% vs global" if curso_selecionado != "Todos" else None)
k4.metric("📚 CR Médio",         cr_medio)
k5.metric("📋 Faltas Médias",    faltas_med)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📊 Análise Exploratória (Sprint 1)",
    "🤖 Comparação de Modelos (Sprint 2)",
    "🔍 Predição Individual",
])

# ============================================================
# TAB 1 — Análise Exploratória
# ============================================================
with tab1:
    st.markdown('<p class="section-label">Sprint 1 — Análise Exploratória dos Dados</p>', unsafe_allow_html=True)

    # --- Linha 1 ---
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Evasão por Curso")
        ec = df.groupby("curso")["evadiu"].mean().reset_index()
        ec.columns = ["Curso", "Taxa"]
        ec["Taxa %"] = (ec["Taxa"] * 100).round(1)
        fig1 = px.bar(
            ec.sort_values("Taxa %", ascending=False),
            x="Curso", y="Taxa %",
            color="Taxa %",
            color_continuous_scale="RdYlGn_r",
            text="Taxa %",
            template="plotly_white",
        )
        fig1.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig1.update_layout(
            coloraxis_showscale=False,
            margin=dict(t=10, b=10),
            xaxis_title="",
            yaxis_title="Taxa de Evasão (%)",
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col_b:
        st.markdown("#### CR vs Faltas")
        fig2 = px.scatter(
            df_filtrado,
            x="cr", y="faltas",
            color=df_filtrado["evadiu"].map({0: "Não Evadiu", 1: "Evadiu"}),
            color_discrete_map={"Não Evadiu": "#2ecc71", "Evadiu": "#e74c3c"},
            opacity=0.65,
            labels={"color": "Situação", "cr": "CR", "faltas": "Faltas"},
            hover_data=["nome", "curso"],
            template="plotly_white",
        )
        fig2.update_layout(margin=dict(t=10, b=10), legend_title_text="Situação")
        st.plotly_chart(fig2, use_container_width=True)

    # --- Linha 2 ---
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown("#### Evasão por Situação Financeira")
        fin = df.groupby("situacao_financeira")["evadiu"].mean().reset_index()
        fin.columns = ["Situação", "Taxa"]
        fin["Taxa %"] = (fin["Taxa"] * 100).round(1)
        fig3 = px.bar(
            fin.sort_values("Taxa %", ascending=False),
            x="Situação", y="Taxa %",
            color="Situação",
            color_discrete_map={
                "dificuldade": "#e74c3c",
                "intermediaria": "#f39c12",
                "estavel": "#2ecc71",
            },
            text="Taxa %",
            template="plotly_white",
        )
        fig3.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig3.update_layout(
            showlegend=False, margin=dict(t=10, b=10),
            xaxis_title="", yaxis_title="Taxa de Evasão (%)",
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        st.markdown("#### Distribuição do CR por Evasão")
        df_box = df_filtrado.copy()
        df_box["Situação"] = df_box["evadiu"].map({0: "Não Evadiu", 1: "Evadiu"})
        fig4 = px.box(
            df_box,
            x="Situação", y="cr",
            color="Situação",
            color_discrete_map={"Não Evadiu": "#2ecc71", "Evadiu": "#e74c3c"},
            points="outliers",
            template="plotly_white",
            labels={"cr": "Coeficiente de Rendimento"},
        )
        fig4.update_layout(showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig4, use_container_width=True)

    # --- Correlações ---
    st.markdown("#### Correlações com a Variável Alvo")
    numericas = df[["periodo", "cr", "faltas", "reprovacoes", "idade", "evadiu"]]
    corr = numericas.corr()["evadiu"].drop("evadiu").sort_values()

    cores_corr = ["#e74c3c" if v > 0 else "#2980b9" for v in corr.values]
    fig5 = go.Figure(go.Bar(
        x=corr.index, y=corr.values,
        marker_color=cores_corr,
        text=[f"{v:+.4f}" for v in corr.values],
        textposition="outside",
    ))
    fig5.update_layout(
        template="plotly_white",
        yaxis_title="Correlação de Pearson com evadiu",
        xaxis_title="",
        margin=dict(t=10, b=10),
        showlegend=False,
    )
    fig5.add_hline(y=0, line_width=1, line_color="black")
    st.plotly_chart(fig5, use_container_width=True)

    # --- Tabela de alunos em risco ---
    st.markdown("#### Alunos em Risco de Evasão")
    em_risco = df_filtrado[df_filtrado["evadiu"] == 1][[
        "nome", "curso", "periodo", "cr", "faltas", "situacao_financeira", "reprovacoes"
    ]].rename(columns={
        "nome": "Nome", "curso": "Curso", "periodo": "Período",
        "cr": "CR", "faltas": "Faltas",
        "situacao_financeira": "Sit. Financeira", "reprovacoes": "Reprovações",
    })
    st.dataframe(em_risco, use_container_width=True, hide_index=True)


# ============================================================
# TAB 2 — Comparação de Modelos (Sprint 2)
# ============================================================
with tab2:
    st.markdown('<p class="section-label">Sprint 2 — Treinamento e Comparação de Modelos</p>', unsafe_allow_html=True)

    resultados = get_metricas_sprint2()

    # --- Info do pré-processamento ---
    with st.expander("ℹ️ Detalhes do Pré-processamento (Sprint 2)", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.info("**Divisão:** 80% treino / 20% teste  \n`stratify=y, random_state=42`")
        c2.info("**Numéricas:** `StandardScaler`  \n`periodo, cr, faltas, reprovacoes, idade`")
        c3.info("**Categóricas:** `OneHotEncoder`  \n`curso, situacao_financeira`")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Tabela de métricas ---
    st.markdown("#### Comparação de Métricas")

    df_met = pd.DataFrame({
        nome: dados["metricas"] for nome, dados in resultados.items()
    }).T

    # Destaque visual por métrica
    col_met1, col_met2 = st.columns(2)
    for col_ui, (nome_modelo, dados) in zip([col_met1, col_met2], resultados.items()):
        with col_ui:
            m = dados["metricas"]
            melhor_auc = nome_modelo == max(resultados, key=lambda n: resultados[n]["metricas"]["ROC-AUC"])
            badge = "🥇 Melhor ROC-AUC" if melhor_auc else ""
            st.markdown(f"**{nome_modelo}** {badge}")
            sub1, sub2, sub3 = st.columns(3)
            sub1.metric("Acurácia",  f"{m['Acurácia']:.4f}")
            sub2.metric("F1-Score",  f"{m['F1-Score']:.4f}")
            sub3.metric("ROC-AUC",   f"{m['ROC-AUC']:.4f}")
            sub4, sub5 = st.columns(2)
            sub4.metric("Precisão",  f"{m['Precisão']:.4f}")
            sub5.metric("Recall",    f"{m['Recall']:.4f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Gráfico de barras comparativo ---
    st.markdown("#### Comparação Visual das Métricas")
    metricas_nomes  = list(next(iter(resultados.values()))["metricas"].keys())
    fig_comp = go.Figure()
    cores_modelos = {"Regressão Logística": "#3498db", "Random Forest": "#e67e22"}

    for nome, dados in resultados.items():
        vals = [dados["metricas"][m] for m in metricas_nomes]
        fig_comp.add_trace(go.Bar(
            name=nome,
            x=metricas_nomes,
            y=vals,
            text=[f"{v:.4f}" for v in vals],
            textposition="outside",
            marker_color=cores_modelos[nome],
        ))

    fig_comp.update_layout(
        barmode="group",
        template="plotly_white",
        yaxis=dict(range=[0.87, 1.01], title="Valor"),
        xaxis_title="",
        legend_title_text="Modelo",
        margin=dict(t=20, b=10),
        height=380,
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    # --- Matrizes de confusão ---
    st.markdown("#### Matrizes de Confusão")
    col_cm1, col_cm2 = st.columns(2)
    rotulos = ["Não Evadiu", "Evadiu"]

    for col_ui, (nome, dados) in zip([col_cm1, col_cm2], resultados.items()):
        with col_ui:
            cm = confusion_matrix(dados["y_test"], dados["y_pred"])
            fig_cm = px.imshow(
                cm,
                labels=dict(x="Predito", y="Real", color="Contagem"),
                x=rotulos, y=rotulos,
                text_auto=True,
                color_continuous_scale="Blues",
                title=nome,
                aspect="auto",
            )
            fig_cm.update_traces(textfont_size=18)
            fig_cm.update_layout(
                margin=dict(t=40, b=10),
                coloraxis_showscale=False,
                height=300,
            )
            st.plotly_chart(fig_cm, use_container_width=True)

    # --- Curvas ROC ---
    st.markdown("#### Curvas ROC")
    fig_roc = go.Figure()

    for nome, dados in resultados.items():
        fpr, tpr, _ = roc_curve(dados["y_test"], dados["y_proba"])
        auc_val = dados["metricas"]["ROC-AUC"]
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode="lines",
            name=f"{nome} (AUC={auc_val:.4f})",
            line=dict(width=2.5, color=cores_modelos[nome]),
            fill="tozeroy",
            fillcolor=f"rgba{tuple(int(cores_modelos[nome].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.06,)}",
        ))

    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Aleatório", line=dict(dash="dash", color="gray", width=1.5),
    ))
    fig_roc.update_layout(
        template="plotly_white",
        xaxis_title="Taxa de Falsos Positivos (FPR)",
        yaxis_title="Taxa de Verdadeiros Positivos (TPR)",
        legend=dict(x=0.55, y=0.05),
        height=400, margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    # --- Importância de features (RF) ---
    st.markdown("#### Importância das Features — Random Forest")
    preprocessador, _, rf = get_sprint2()

    nomes_cat = list(
        preprocessador.named_transformers_["cat"]
        .named_steps["onehot"]
        .get_feature_names_out(COLUNAS_CATEGORICAS)
    )
    nomes_features = COLUNAS_NUMERICAS + list(nomes_cat)

    df_imp = pd.DataFrame({
        "Feature": nomes_features,
        "Importância": rf.feature_importances_,
    }).sort_values("Importância", ascending=True)

    cores_imp = ["#e74c3c" if v >= df_imp["Importância"].quantile(0.75) else "#3498db"
                 for v in df_imp["Importância"]]

    fig_imp = px.bar(
        df_imp,
        x="Importância", y="Feature",
        orientation="h",
        text=df_imp["Importância"].map(lambda v: f"{v:.4f}"),
        template="plotly_white",
        color=df_imp["Importância"],
        color_continuous_scale=[[0, "#d6eaf8"], [0.5, "#3498db"], [1, "#e74c3c"]],
    )
    fig_imp.update_traces(textposition="outside")
    fig_imp.update_layout(
        coloraxis_showscale=False,
        margin=dict(t=10, b=10),
        height=420,
        xaxis_title="Importância (Gini)",
        yaxis_title="",
    )
    st.plotly_chart(fig_imp, use_container_width=True)


# ============================================================
# TAB 3 — Predição Individual
# ============================================================
with tab3:
    st.markdown('<p class="section-label">Predição individual de risco de evasão</p>', unsafe_allow_html=True)

    st.markdown(f"""
    Modelo ativo: **{modelo_nome}**
    Pré-processamento: `StandardScaler` + `OneHotEncoder` (Sprint 2)
    """)

    with st.form("form_predicao"):
        st.markdown("#### Dados do Aluno")
        c1, c2, c3 = st.columns(3)

        with c1:
            curso = st.selectbox("Curso", sorted(df["curso"].unique().tolist()))
            periodo = st.slider("Período", 1, 8, 3)
            idade = st.number_input("Idade", min_value=16, max_value=60, value=21)

        with c2:
            cr = st.slider("Coeficiente de Rendimento (CR)", 0.0, 10.0, 6.5, step=0.1)
            faltas = st.slider("Número de Faltas", 0, 60, 8)

        with c3:
            reprovacoes = st.slider("Reprovações", 0, 10, 0)
            situacao_financeira = st.selectbox(
                "Situação Financeira",
                ["estavel", "intermediaria", "dificuldade"],
            )

        submitted = st.form_submit_button("🔍 Prever Risco de Evasão", use_container_width=True)

    if submitted:
        preprocessador, lr, rf = get_sprint2()
        modelo_ativo = lr if modelo_nome == "Regressão Logística" else rf

        entrada = pd.DataFrame([{
            "periodo": periodo, "cr": cr, "faltas": faltas,
            "reprovacoes": reprovacoes, "idade": idade,
            "curso": curso, "situacao_financeira": situacao_financeira,
        }])

        entrada_prep = preprocessador.transform(entrada)
        prob = float(modelo_ativo.predict_proba(entrada_prep)[0][1]) * 100
        pred = modelo_ativo.predict(entrada_prep)[0]

        # Banner de resultado
        if prob >= 60:
            st.markdown(f"""
            <div class="risco-alto">
                <strong>⚠️ Alto risco de evasão</strong><br>
                Probabilidade estimada: <strong>{prob:.1f}%</strong> — Intervenção recomendada.
            </div>""", unsafe_allow_html=True)
        elif prob >= 35:
            st.markdown(f"""
            <div class="risco-medio">
                <strong>⚡ Risco moderado de evasão</strong><br>
                Probabilidade estimada: <strong>{prob:.1f}%</strong> — Monitoramento recomendado.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="risco-baixo">
                <strong>✅ Baixo risco de evasão</strong><br>
                Probabilidade estimada: <strong>{prob:.1f}%</strong> — Situação estável.
            </div>""", unsafe_allow_html=True)

        # Gauge + fatores de risco lado a lado
        col_g, col_f = st.columns([1, 1])

        with col_g:
            cor_gauge = "#e74c3c" if prob >= 60 else ("#f39c12" if prob >= 35 else "#2ecc71")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob,
                number={"suffix": "%", "font": {"size": 42}},
                delta={"reference": 30, "suffix": "%", "prefix": "vs base "},
                title={"text": "Risco de Evasão", "font": {"size": 18}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": cor_gauge, "thickness": 0.25},
                    "bgcolor": "white",
                    "borderwidth": 2,
                    "steps": [
                        {"range": [0, 35],  "color": "#eafaf1"},
                        {"range": [35, 60], "color": "#fef9e7"},
                        {"range": [60, 100],"color": "#fdf2f2"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.75,
                        "value": 30,
                    },
                },
            ))
            fig_gauge.update_layout(height=300, margin=dict(t=30, b=20, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_f:
            st.markdown("#### Fatores de Risco Identificados")
            fatores = []

            if cr < 5.0:
                fatores.append(("🔴", f"CR muito baixo ({cr:.1f} < 5.0)"))
            elif cr < 6.5:
                fatores.append(("🟡", f"CR abaixo da média ({cr:.1f})"))

            if faltas > 15:
                fatores.append(("🔴", f"Alto número de faltas ({int(faltas)})"))
            elif faltas > 8:
                fatores.append(("🟡", f"Faltas acima da média ({int(faltas)})"))

            if reprovacoes >= 3:
                fatores.append(("🔴", f"Reprovações críticas ({int(reprovacoes)})"))
            elif reprovacoes >= 1:
                fatores.append(("🟡", f"Possui reprovações ({int(reprovacoes)})"))

            if situacao_financeira == "dificuldade":
                fatores.append(("🔴", "Situação financeira em dificuldade"))
            elif situacao_financeira == "intermediaria":
                fatores.append(("🟡", "Situação financeira intermediária"))

            if not fatores:
                st.success("Nenhum fator de risco crítico identificado.")
            else:
                for icone, desc in fatores:
                    st.markdown(f"{icone} {desc}")

            st.markdown("---")
            st.caption(f"Modelo: **{modelo_nome}**")
            st.caption(f"Classe predita: **{'Evadiu' if pred else 'Não Evadiu'}**")
