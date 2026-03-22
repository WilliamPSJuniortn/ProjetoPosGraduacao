# Sistema de Análise e Predição de Evasão Escolar

Dashboard interativo com modelo de Machine Learning para identificar alunos em risco de evasão, desenvolvido como projeto de pós-graduação.

## Visão Geral

O projeto analisa dados acadêmicos e socioeconômicos de alunos para prever a probabilidade de evasão escolar, utilizando um modelo Random Forest treinado sobre variáveis como coeficiente de rendimento, frequência, situação financeira e histórico de reprovações.

## Estrutura do Projeto

```
DevProjPos/
├── data/
│   ├── alunos.csv              # Dataset bruto (1500 registros, ~3% ausentes)
│   ├── alunos_limpo.csv        # Dataset após limpeza e tratamento
│   └── dicionario_dados.csv    # Documentação das colunas
├── models/
│   ├── modelo_evasao.pkl       # Modelo Random Forest treinado
│   └── encoder.pkl             # Encoders das variáveis categóricas
├── notebooks/
│   └── exploracao.ipynb        # Análise exploratória (EDA) — 10 seções
├── src/
│   ├── app.py                  # Dashboard Streamlit
│   └── pipeline.py             # Pipeline de pré-processamento e ML
├── gerar_dataset.py            # Geração do dataset sintético (seed=42)
├── limpeza_dados.py            # Limpeza e tratamento dos dados
├── dicionario_dados.py         # Geração do dicionário de dados
├── main_sprint1.py             # Pipeline completa da Sprint 1
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
docker compose up -d
```

| Serviço | URL |
|---|---|
| Dashboard Streamlit | http://localhost:8501 |
| Jupyter Lab | http://localhost:8888 |

Para parar:
```bash
docker compose down
```

### Localmente

```bash
pip install -r requirements.txt

# Executar pipeline da Sprint 1 (gera dados, limpeza e dicionário)
python main_sprint1.py

# Iniciar o dashboard
streamlit run src/app.py
```

## Sprint 1 — Preparação dos Dados

Execute a pipeline completa:

```bash
python main_sprint1.py
```

O script executa em sequência:
1. `gerar_dataset.py` — gera 1500 registros com distribuições realistas
2. `limpeza_dados.py` — trata ausentes (mediana/moda), duplicatas e outliers via IQR
3. `dicionario_dados.py` — documenta todas as colunas

**Resultados da Sprint 1:**
- Ausentes tratados: 225 valores (3% por coluna numérica)
- Outliers tratados: 139 valores (IQR em `cr`, `faltas`, `reprovacoes`)
- Top correlações com evasão: `faltas` (+0.74) · `cr` (−0.72) · `reprovacoes` (+0.52)

## Modelo

- **Algoritmo:** Random Forest Classifier (100 estimadores)
- **Divisão:** 80% treino / 20% teste (estratificado)
- **Features:** periodo, cr, faltas, reprovacoes, idade, curso, situacao_financeira

## Tecnologias

- Python 3.11
- Streamlit · Plotly · Seaborn · Matplotlib
- Scikit-learn · Pandas · NumPy
- Docker · Jupyter Lab
