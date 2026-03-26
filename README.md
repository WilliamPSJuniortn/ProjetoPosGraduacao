# Sistema de Análise e Predição de Evasão Escolar

Dashboard interativo com modelos de Machine Learning para identificar alunos em risco de evasão, desenvolvido como projeto de pós-graduação.

## Visão Geral

O projeto analisa dados acadêmicos e socioeconômicos de alunos para prever a probabilidade de evasão escolar. Foram implementadas duas sprints: a Sprint 1 preparou os dados com limpeza e análise exploratória; a Sprint 2 treinou e comparou dois classificadores (Regressão Logística e Random Forest) com pré-processamento padronizado.

## Estrutura do Projeto

```
DevProjPos/
├── data/
│   ├── alunos.csv                        # Dataset bruto (1500 registros)
│   ├── alunos_limpo.csv                  # Dataset após limpeza e tratamento
│   └── dicionario_dados.csv              # Documentação das colunas
├── models/
│   ├── preprocessador.pkl                # ColumnTransformer (StandardScaler + OneHotEncoder)
│   ├── modelo_regressao_logistica.pkl    # Regressão Logística treinada (Sprint 2)
│   ├── modelo_random_forest.pkl          # Random Forest treinado (Sprint 2)
│   ├── modelo_evasao.pkl                 # Modelo legado (Sprint 1)
│   └── encoder.pkl                       # Encoder legado (Sprint 1)
├── notebooks/
│   ├── exploracao.ipynb                  # EDA — Sprint 1 (10 seções)
│   └── sprint2_modelagem.ipynb           # Modelagem — Sprint 2 (12 seções)
├── src/
│   ├── app.py                            # Dashboard Streamlit (3 abas)
│   └── pipeline.py                       # Pipeline legado Sprint 1
├── tests/
│   └── test_app_logic.py                 # 19 testes automatizados
├── gerar_dataset.py                      # Geração do dataset sintético
├── limpeza_dados.py                      # Limpeza e tratamento dos dados
├── dicionario_dados.py                   # Geração do dicionário de dados
├── preprocessamento.py                   # Pré-processamento Sprint 2
├── modelagem.py                          # Treinamento e comparação Sprint 2
├── main_sprint1.py                       # Pipeline completa da Sprint 1
├── main_sprint2.py                       # Pipeline completa da Sprint 2
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Dataset

| Coluna | Tipo | Descrição |
|---|---|---|
| id_aluno | Numérica | Identificador único |
| nome | Texto | Nome do aluno |
| curso | Categórica | Ciência da Computação, Engenharia Civil, Administração, Direito, Medicina |
| periodo | Numérica | Período atual (1 a 8) |
| cr | Numérica | Coeficiente de Rendimento (0.0 a 10.0) |
| faltas | Numérica | Número de faltas |
| situacao_financeira | Categórica | estavel, intermediaria, dificuldade |
| reprovacoes | Numérica | Quantidade de reprovações |
| idade | Numérica | Idade do aluno |
| evadiu | Binária | 0 = permaneceu, 1 = evadiu (variável alvo) |

**Taxa de evasão:** 30% | **Registros:** 1500

## Como Executar

### Com Docker (recomendado)

```bash
docker compose up --build -d
```

| Serviço | URL |
|---|---|
| Dashboard Streamlit | http://localhost:8501 |
| Jupyter Lab | http://localhost:8888 |

```bash
# Parar os containers
docker compose down
```

### Localmente

```bash
pip install -r requirements.txt

# Sprint 1 — gera dados, limpeza e dicionário
python main_sprint1.py

# Sprint 2 — pré-processamento, treino e comparação dos modelos
python main_sprint2.py

# Iniciar o dashboard
streamlit run src/app.py
```

### Executar testes

```bash
python tests/test_app_logic.py
```

---

## Sprint 1 — Preparação dos Dados

```bash
python main_sprint1.py
```

Executa em sequência:
1. `gerar_dataset.py` — gera 1500 registros com distribuições realistas (seed=42)
2. `limpeza_dados.py` — trata ausentes (mediana/moda), duplicatas e outliers via IQR
3. `dicionario_dados.py` — documenta todas as colunas

**Resultados:**
- Ausentes tratados: 225 valores (~3% por coluna numérica)
- Outliers tratados: 139 valores (IQR em `cr`, `faltas`, `reprovacoes`)
- Top correlações com evasão: `faltas` (+0.73) · `cr` (−0.73) · `reprovacoes` (+0.53)

**Notebook:** `notebooks/exploracao.ipynb` — 10 seções com visualizações e insights

---

## Sprint 2 — Modelagem e Comparação de Classificadores

```bash
python main_sprint2.py
```

Executa em sequência:
1. `preprocessamento.py` — separa X/y, aplica `StandardScaler` nas numéricas e `OneHotEncoder` nas categóricas via `ColumnTransformer`, divide treino/teste (80/20 estratificado)
2. `modelagem.py` — treina Regressão Logística e Random Forest, calcula e compara métricas

### Pré-processamento

| Etapa | Técnica | Colunas |
|---|---|---|
| Padronização | `StandardScaler` | periodo, cr, faltas, reprovacoes, idade |
| Codificação | `OneHotEncoder` | curso, situacao_financeira |
| Resultado | 13 features | 5 numéricas + 8 binárias (one-hot) |

### Resultados

| Métrica | Regressão Logística | Random Forest |
|---|---|---|
| Acurácia | 0.9633 | 0.9633 |
| Precisão | 0.9540 | 0.9647 |
| Recall | 0.9222 | 0.9111 |
| F1-Score | 0.9379 | 0.9371 |
| **ROC-AUC** | **0.9925** | 0.9845 |

**Melhor modelo:** Regressão Logística (ROC-AUC superior)

**Top 3 features (Random Forest):**
1. `cr` — 40.1%
2. `faltas` — 30.8%
3. `reprovacoes` — 12.3%

**Notebook:** `notebooks/sprint2_modelagem.ipynb` — 12 seções incluindo curvas ROC, matrizes de confusão e coeficientes da Regressão Logística

---

## Dashboard (Streamlit)

O dashboard possui 3 abas:

| Aba | Conteúdo |
|---|---|
| 📊 Análise Exploratória (Sprint 1) | KPIs, evasão por curso/situação financeira, CR vs Faltas, correlações, tabela de alunos em risco |
| 🤖 Comparação de Modelos (Sprint 2) | Métricas lado a lado, gráfico comparativo, matrizes de confusão interativas, curvas ROC, importância de features, coeficientes LR |
| 🔍 Predição Individual | Formulário com dados do aluno, gauge de probabilidade, fatores de risco identificados |

A barra lateral permite filtrar por curso e selecionar o modelo ativo (Regressão Logística ou Random Forest) para a predição.

---

## Tecnologias

- **Python 3.11**
- **ML:** Scikit-learn (LogisticRegression, RandomForestClassifier, StandardScaler, OneHotEncoder, ColumnTransformer)
- **Dashboard:** Streamlit · Plotly
- **Análise:** Pandas · NumPy · Seaborn · Matplotlib
- **Infra:** Docker · Jupyter Lab
