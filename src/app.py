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
    .main { background-color: #f8f9fb; }
    [data-testid="metric-container"] {
        background: white;
        border: 1px solid #e8ecf0;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    [data-testid="metric-container"] label {
        font-size: 0.78rem !important;
        color: #7f8c8d !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.9rem !important;
        font-weight: 700 !important;
        color: #2c3e50 !important;
    }
    .section-label {
        font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 1px; color: #95a5a6; margin: 4px 0 12px 0;
    }
    .risco-alto  { background:#fde8e8; border-left:4px solid #e74c3c;
                   padding:14px 18px; border-radius:8px; margin:10px 0; }
    .risco-medio { background:#fef9e7; border-left:4px solid #f39c12;
                   padding:14px 18px; border-radius:8px; margin:10px 0; }
    .risco-baixo { background:#e8f8f0; border-left:4px solid #2ecc71;
                   padding:14px 18px; border-radius:8px; margin:10px 0; }
    .badge { border-radius:20px; padding:3px 12px; font-size:0.75rem;
             font-weight:700; display:inline-block; margin:2px 0; }
    .badge-ok  { background:#d5e8d4; color:#27ae60; }
    .badge-s2  { background:#dae8fc; color:#2980b9; }
    button[data-baseweb="tab"] { font-size: 0.95rem !important; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_LIMPO = os.path.join(BASE_DIR, "data", "alunos_limpo.csv")
PREP_PATH  = os.path.join(BASE_DIR, "models", "preprocessador.pkl")
LR_PATH    = os.path.join(BASE_DIR, "models", "modelo_regressao_logistica.pkl")
RF_PATH    = os.path.join(BASE_DIR, "models", "modelo_random_forest.pkl")

COLUNAS_NUM = ["periodo", "cr", "faltas", "reprovacoes", "idade"]
COLUNAS_CAT = ["curso", "situacao_financeira"]
CORES_MODELO = {"Regressão Logística": "#3498db", "Random Forest": "#e67e22"}

# ---------------------------------------------------------------------------
# Carregamento (cache separado — sem aninhamento)
# ---------------------------------------------------------------------------
@st.cache_data
def load_dados():
    return pd.read_csv(DATA_LIMPO)

@st.cache_resource
def load_modelos():
    prep = joblib.load(PREP_PATH)
    lr   = joblib.load(LR_PATH)
    rf   = joblib.load(RF_PATH)
    return prep, lr, rf

# ---------------------------------------------------------------------------
# Carrega dados e modelos no topo do script (uma vez por sessão)
# ---------------------------------------------------------------------------
try:
    df = load_dados()
    prep, lr, rf = load_modelos()
    modelos_ok = True
except Exception as e:
    st.error(f"Erro ao carregar modelos: {e}\nExecute `python main_sprint2.py` para gerar os modelos.")
    st.stop()

# ---------------------------------------------------------------------------
# Pré-computa métricas do conjunto de teste (rápido — 300 amostras)
# Feito fora de qualquer cache para evitar aninhamento de caches
# ---------------------------------------------------------------------------
X_full = df.drop(columns=["id_aluno", "nome", "evadiu"])
y_full = df["evadiu"]
_, X_test_df, _, y_test = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
)
X_test_prep = prep.transform(X_test_df)

resultados = {}
for nome, modelo in [("Regressão Logística", lr), ("Random Forest", rf)]:
    y_pred  = modelo.predict(X_test_prep)
    y_proba = modelo.predict_proba(X_test_prep)[:, 1]
    resultados[nome] = {
        "y_pred" : y_pred,
        "y_proba": y_proba,
        "metricas": {
            "Acurácia" : round(accuracy_score(y_test, y_pred), 4),
            "Precisão" : round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall"   : round(recall_score(y_test, y_pred, zero_division=0), 4),
            "F1-Score" : round(f1_score(y_test, y_pred, zero_division=0), 4),
            "ROC-AUC"  : round(roc_auc_score(y_test, y_proba), 4),
        },
    }

# Nomes das features após OHE
nomes_cat      = list(prep.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(COLUNAS_CAT))
nomes_features = COLUNAS_NUM + nomes_cat

# Melhor modelo por ROC-AUC
melhor_modelo  = max(resultados, key=lambda n: resultados[n]["metricas"]["ROC-AUC"])

# Session state para persistir predição ao trocar de tab
if "pred_resultado" not in st.session_state:
    st.session_state.pred_resultado = None

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎓 Painel de Controle")

    st.markdown('<p class="section-label">Filtro de curso</p>', unsafe_allow_html=True)
    cursos = ["Todos"] + sorted(df["curso"].unique().tolist())
    curso_sel = st.selectbox("Curso", cursos, key="filtro_curso")

    st.markdown('<p class="section-label">Modelo ativo (predição)</p>', unsafe_allow_html=True)
    modelo_sel = st.radio(
        "Modelo",
        list(CORES_MODELO.keys()),
        index=0,
        key="modelo_sel",
        help="Modelo utilizado na aba de Predição Individual",
    )
    auc_sel = resultados[modelo_sel]["metricas"]["ROC-AUC"]
    st.caption(f"ROC-AUC: **{auc_sel:.4f}**")

    st.markdown("---")
    st.markdown("#### Sprints")
    st.markdown("""
    <span class='badge badge-ok'>✅ Sprint 1 — EDA</span><br>
    <span class='badge badge-s2'>✅ Sprint 2 — Modelagem</span>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption(f"Dataset: **{len(df)}** registros")
    st.caption(f"Taxa de evasão global: **{df['evadiu'].mean()*100:.1f}%**")
    st.caption(f"Melhor modelo: **{melhor_modelo}**")

# Dados filtrados
df_f = df if curso_sel == "Todos" else df[df["curso"] == curso_sel]

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("# 🎓 Sistema de Predição de Evasão Escolar")
    st.caption("Análise exploratória · Modelagem Sprint 2 · Predição individual")
with col_h2:
    st.markdown(f"""
    <div style='text-align:right; padding-top:10px;'>
        <span class='badge badge-ok' style='font-size:0.8rem;'>Sprint 1 ✅</span>&nbsp;
        <span class='badge badge-s2' style='font-size:0.8rem;'>Sprint 2 ✅</span>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------
total      = len(df_f)
evadidos   = int(df_f["evadiu"].sum())
taxa       = round(evadidos / total * 100, 1) if total > 0 else 0
cr_medio   = round(df_f["cr"].mean(), 2)
faltas_med = round(df_f["faltas"].mean(), 1)
delta_taxa = f"{taxa - df['evadiu'].mean()*100:.1f}% vs global" if curso_sel != "Todos" else None

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("👥 Total de Alunos",  total)
k2.metric("🚨 Alunos Evadidos",  evadidos)
k3.metric("📉 Taxa de Evasão",   f"{taxa}%", delta=delta_taxa)
k4.metric("📚 CR Médio",         cr_medio)
k5.metric("📋 Faltas Médias",    faltas_med)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📊  Análise Exploratória (Sprint 1)",
    "🤖  Comparação de Modelos (Sprint 2)",
    "🔍  Predição Individual",
])

# ============================================================
# TAB 1 — Análise Exploratória
# ============================================================
with tab1:
    st.markdown('<p class="section-label">Sprint 1 — Exploração dos Dados</p>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("##### Evasão por Curso")
        ec = (
            df.groupby("curso")["evadiu"].mean()
            .mul(100).round(1).reset_index()
        )
        ec.columns = ["Curso", "Taxa (%)"]
        fig1 = px.bar(
            ec.sort_values("Taxa (%)", ascending=False),
            x="Curso", y="Taxa (%)",
            color="Taxa (%)",
            color_continuous_scale="RdYlGn_r",
            text="Taxa (%)",
            template="plotly_white",
        )
        fig1.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig1.update_layout(coloraxis_showscale=False, xaxis_title="",
                           yaxis_title="Taxa (%)", margin=dict(t=10, b=10))
        st.plotly_chart(fig1, use_container_width=True)

    with col_b:
        st.markdown("##### CR vs Faltas")
        df_sc = df_f.copy()
        df_sc["Situação"] = df_sc["evadiu"].map({0: "Não Evadiu", 1: "Evadiu"})
        fig2 = px.scatter(
            df_sc, x="cr", y="faltas", color="Situação",
            color_discrete_map={"Não Evadiu": "#2ecc71", "Evadiu": "#e74c3c"},
            opacity=0.6, template="plotly_white",
            labels={"cr": "CR", "faltas": "Faltas"},
            hover_data=["nome", "curso"],
        )
        fig2.update_layout(margin=dict(t=10, b=10), legend_title_text="Situação")
        st.plotly_chart(fig2, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown("##### Distribuição do CR por Evasão")
        df_bx = df_f.copy()
        df_bx["Situação"] = df_bx["evadiu"].map({0: "Não Evadiu", 1: "Evadiu"})
        fig3 = px.box(
            df_bx, x="Situação", y="cr", color="Situação",
            color_discrete_map={"Não Evadiu": "#2ecc71", "Evadiu": "#e74c3c"},
            points="outliers", template="plotly_white",
            labels={"cr": "Coeficiente de Rendimento"},
        )
        fig3.update_layout(showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        st.markdown("##### Evasão por Situação Financeira")
        fin = (
            df.groupby("situacao_financeira")["evadiu"].mean()
            .mul(100).round(1).reset_index()
        )
        fin.columns = ["Situação", "Taxa (%)"]
        fig4 = px.bar(
            fin.sort_values("Taxa (%)", ascending=False),
            x="Situação", y="Taxa (%)",
            color="Situação",
            color_discrete_map={
                "dificuldade": "#e74c3c",
                "intermediaria": "#f39c12",
                "estavel": "#2ecc71",
            },
            text="Taxa (%)", template="plotly_white",
        )
        fig4.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig4.update_layout(showlegend=False, xaxis_title="",
                           yaxis_title="Taxa (%)", margin=dict(t=10, b=10))
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("##### Correlações com a Variável Alvo (evadiu)")
    corr = (
        df[["periodo", "cr", "faltas", "reprovacoes", "idade", "evadiu"]]
        .corr()["evadiu"].drop("evadiu").sort_values()
    )
    fig5 = go.Figure(go.Bar(
        x=corr.index, y=corr.values,
        marker_color=["#e74c3c" if v > 0 else "#2980b9" for v in corr.values],
        text=[f"{v:+.4f}" for v in corr.values],
        textposition="outside",
    ))
    fig5.add_hline(y=0, line_width=1, line_color="black")
    fig5.update_layout(
        template="plotly_white", yaxis_title="Correlação de Pearson",
        xaxis_title="", margin=dict(t=10, b=10), showlegend=False,
    )
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown("##### Alunos em Risco de Evasão")
    em_risco = df_f[df_f["evadiu"] == 1][[
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

    # Info pré-processamento
    with st.expander("ℹ️ Detalhes do Pré-processamento", expanded=False):
        ci1, ci2, ci3 = st.columns(3)
        ci1.info("**Divisão**\n\n80% treino / 20% teste\n\n`stratify=y · random_state=42`")
        ci2.info("**Numéricas**\n\n`StandardScaler`\n\nperiodo · cr · faltas · reprovacoes · idade")
        ci3.info("**Categóricas**\n\n`OneHotEncoder`\n\ncurso · situacao_financeira → 8 colunas binárias")

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------- Métricas lado a lado ----------
    st.markdown("#### Métricas no Conjunto de Teste")
    col_m1, col_m2 = st.columns(2)

    for col_ui, nome in zip([col_m1, col_m2], resultados):
        m = resultados[nome]["metricas"]
        with col_ui:
            crown = " 🥇" if nome == melhor_modelo else ""
            st.markdown(f"**{nome}{crown}**")
            s1, s2, s3 = st.columns(3)
            s1.metric("Acurácia",  f"{m['Acurácia']:.4f}")
            s2.metric("F1-Score",  f"{m['F1-Score']:.4f}")
            s3.metric("ROC-AUC",   f"{m['ROC-AUC']:.4f}")
            s4, s5 = st.columns(2)
            s4.metric("Precisão",  f"{m['Precisão']:.4f}")
            s5.metric("Recall",    f"{m['Recall']:.4f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------- Gráfico comparativo de métricas ----------
    st.markdown("#### Comparação Visual das Métricas")
    metricas_nomes = list(next(iter(resultados.values()))["metricas"].keys())
    fig_comp = go.Figure()
    for nome in resultados:
        vals = [resultados[nome]["metricas"][m] for m in metricas_nomes]
        fig_comp.add_trace(go.Bar(
            name=nome, x=metricas_nomes, y=vals,
            text=[f"{v:.4f}" for v in vals],
            textposition="outside",
            marker_color=CORES_MODELO[nome],
        ))
    fig_comp.update_layout(
        barmode="group", template="plotly_white",
        yaxis=dict(range=[0.87, 1.01], title="Valor"),
        xaxis_title="", legend_title_text="Modelo",
        margin=dict(t=20, b=10), height=370,
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    # ---------- Matrizes de Confusão ----------
    st.markdown("#### Matrizes de Confusão")
    col_cm1, col_cm2 = st.columns(2)
    rotulos = ["Não Evadiu", "Evadiu"]

    for col_ui, nome in zip([col_cm1, col_cm2], resultados):
        with col_ui:
            cm = confusion_matrix(y_test, resultados[nome]["y_pred"])
            fig_cm = px.imshow(
                cm, labels=dict(x="Predito", y="Real", color="Contagem"),
                x=rotulos, y=rotulos,
                text_auto=True,
                color_continuous_scale="Blues",
                title=nome, aspect="auto",
            )
            fig_cm.update_traces(textfont_size=20)
            fig_cm.update_layout(
                coloraxis_showscale=False,
                margin=dict(t=40, b=10), height=280,
            )
            st.plotly_chart(fig_cm, use_container_width=True)

    # ---------- Curvas ROC ----------
    st.markdown("#### Curvas ROC")
    fig_roc = go.Figure()
    for nome in resultados:
        fpr, tpr, _ = roc_curve(y_test, resultados[nome]["y_proba"])
        auc_val = resultados[nome]["metricas"]["ROC-AUC"]
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{nome}  (AUC = {auc_val:.4f})",
            line=dict(width=2.5, color=CORES_MODELO[nome]),
        ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Aleatório",
        line=dict(dash="dash", color="gray", width=1.5),
    ))
    fig_roc.update_layout(
        template="plotly_white",
        xaxis_title="Taxa de Falsos Positivos (FPR)",
        yaxis_title="Taxa de Verdadeiros Positivos (TPR)",
        legend=dict(x=0.4, y=0.08),
        height=400, margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    # ---------- Importância de features — RF ----------
    st.markdown("#### Importância das Features — Random Forest")
    df_imp = pd.DataFrame({
        "Feature": nomes_features,
        "Importância": rf.feature_importances_,
    }).sort_values("Importância", ascending=True)

    fig_imp = px.bar(
        df_imp, x="Importância", y="Feature", orientation="h",
        text=df_imp["Importância"].map(lambda v: f"{v:.4f}"),
        color="Importância",
        color_continuous_scale=[[0, "#d6eaf8"], [0.5, "#3498db"], [1, "#e74c3c"]],
        template="plotly_white",
    )
    fig_imp.update_traces(textposition="outside")
    fig_imp.update_layout(
        coloraxis_showscale=False, height=430,
        xaxis_title="Importância (Gini)", yaxis_title="",
        margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    # ---------- Coeficientes — Regressão Logística ----------
    st.markdown("#### Coeficientes — Regressão Logística")
    st.caption("Valores positivos aumentam o risco · valores negativos reduzem o risco")
    df_coef = pd.DataFrame({
        "Feature": nomes_features,
        "Coeficiente": lr.coef_[0],
    }).sort_values("Coeficiente", ascending=True)

    fig_coef = px.bar(
        df_coef, x="Coeficiente", y="Feature", orientation="h",
        color="Coeficiente",
        color_continuous_scale=[[0, "#2ecc71"], [0.5, "#ecf0f1"], [1, "#e74c3c"]],
        template="plotly_white",
        text=df_coef["Coeficiente"].map(lambda v: f"{v:+.3f}"),
    )
    fig_coef.update_traces(textposition="outside")
    fig_coef.add_vline(x=0, line_width=1.5, line_color="black")
    fig_coef.update_layout(
        coloraxis_showscale=False, height=430,
        xaxis_title="Coeficiente (log-odds)", yaxis_title="",
        margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig_coef, use_container_width=True)


# ============================================================
# TAB 3 — Predição Individual
# ============================================================
with tab3:
    st.markdown('<p class="section-label">Preencha os dados do aluno para estimar o risco de evasão</p>', unsafe_allow_html=True)

    col_info, col_modelo = st.columns([3, 1])
    with col_info:
        st.markdown(f"Modelo ativo: **{modelo_sel}** · ROC-AUC: **{auc_sel:.4f}**")
        st.caption("Altere o modelo na barra lateral esquerda.")
    with col_modelo:
        st.markdown(f"<div style='text-align:right;'><span class='badge badge-s2'>{modelo_sel}</span></div>",
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.form("form_predicao", clear_on_submit=False):
        st.markdown("##### Dados do Aluno")
        c1, c2, c3 = st.columns(3)
        with c1:
            p_curso    = st.selectbox("Curso", sorted(df["curso"].unique().tolist()))
            p_periodo  = st.slider("Período", 1, 8, 3)
            p_idade    = st.number_input("Idade", min_value=16, max_value=60, value=21)
        with c2:
            p_cr       = st.slider("Coeficiente de Rendimento (CR)", 0.0, 10.0, 6.5, step=0.1)
            p_faltas   = st.slider("Número de Faltas", 0, 60, 8)
        with c3:
            p_reprov   = st.slider("Reprovações", 0, 10, 0)
            p_financ   = st.selectbox("Situação Financeira",
                                      ["estavel", "intermediaria", "dificuldade"])

        submitted = st.form_submit_button("🔍  Prever Risco de Evasão", use_container_width=True, type="primary")

    # Computa e salva em session_state ao submeter
    if submitted:
        modelo_ativo = lr if modelo_sel == "Regressão Logística" else rf
        entrada = pd.DataFrame([{
            "periodo": p_periodo, "cr": p_cr, "faltas": p_faltas,
            "reprovacoes": p_reprov, "idade": p_idade,
            "curso": p_curso, "situacao_financeira": p_financ,
        }])
        ep   = prep.transform(entrada)
        prob = float(modelo_ativo.predict_proba(ep)[0][1]) * 100
        pred = int(modelo_ativo.predict(ep)[0])

        fatores = []
        if p_cr < 5.0:
            fatores.append(("🔴", f"CR muito baixo ({p_cr:.1f} < 5.0)"))
        elif p_cr < 6.5:
            fatores.append(("🟡", f"CR abaixo da média ({p_cr:.1f})"))
        if p_faltas > 15:
            fatores.append(("🔴", f"Alto número de faltas ({p_faltas})"))
        elif p_faltas > 8:
            fatores.append(("🟡", f"Faltas acima da média ({p_faltas})"))
        if p_reprov >= 3:
            fatores.append(("🔴", f"Reprovações críticas ({p_reprov})"))
        elif p_reprov >= 1:
            fatores.append(("🟡", f"Possui reprovações ({p_reprov})"))
        if p_financ == "dificuldade":
            fatores.append(("🔴", "Situação financeira em dificuldade"))
        elif p_financ == "intermediaria":
            fatores.append(("🟡", "Situação financeira intermediária"))

        st.session_state.pred_resultado = {
            "prob": prob, "pred": pred, "modelo": modelo_sel,
            "fatores": fatores, "aluno": {
                "Curso": p_curso, "Período": p_periodo, "CR": p_cr,
                "Faltas": p_faltas, "Reprovações": p_reprov,
                "Situação Financeira": p_financ, "Idade": p_idade,
            }
        }

    # Exibe resultado (persiste ao trocar de tab)
    if st.session_state.pred_resultado:
        r    = st.session_state.pred_resultado
        prob = r["prob"]

        if prob >= 60:
            st.markdown(f"""<div class="risco-alto">
                <strong>⚠️ Alto risco de evasão</strong> — Probabilidade: <strong>{prob:.1f}%</strong><br>
                <small>Intervenção recomendada com urgência.</small></div>""", unsafe_allow_html=True)
        elif prob >= 35:
            st.markdown(f"""<div class="risco-medio">
                <strong>⚡ Risco moderado de evasão</strong> — Probabilidade: <strong>{prob:.1f}%</strong><br>
                <small>Monitoramento recomendado.</small></div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="risco-baixo">
                <strong>✅ Baixo risco de evasão</strong> — Probabilidade: <strong>{prob:.1f}%</strong><br>
                <small>Situação estável.</small></div>""", unsafe_allow_html=True)

        col_g, col_f = st.columns([1, 1])

        with col_g:
            cor_g = "#e74c3c" if prob >= 60 else ("#f39c12" if prob >= 35 else "#2ecc71")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob,
                number={"suffix": "%", "font": {"size": 44}},
                delta={"reference": 30, "suffix": "% vs base",
                       "increasing": {"color": "#e74c3c"},
                       "decreasing": {"color": "#2ecc71"}},
                title={"text": "Risco de Evasão", "font": {"size": 18}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": cor_g, "thickness": 0.25},
                    "bgcolor": "white",
                    "steps": [
                        {"range": [0, 35],   "color": "#eafaf1"},
                        {"range": [35, 60],  "color": "#fef9e7"},
                        {"range": [60, 100], "color": "#fdf2f2"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.75, "value": 30,
                    },
                },
            ))
            fig_gauge.update_layout(height=290, margin=dict(t=30, b=20, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_f:
            st.markdown("##### Fatores de Risco")
            fatores = r["fatores"]
            if fatores:
                for icone, desc in fatores:
                    st.markdown(f"{icone} {desc}")
            else:
                st.success("Nenhum fator de risco crítico identificado.")

            st.markdown("---")
            st.markdown("##### Dados Utilizados")
            aluno_df = pd.DataFrame([r["aluno"]])
            st.dataframe(aluno_df, use_container_width=True, hide_index=True)
            st.caption(f"Modelo: **{r['modelo']}**")
