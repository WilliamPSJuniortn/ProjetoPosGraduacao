"""
dicionario_dados.py
Gera o dicionário de dados em data/dicionario_dados.csv comparando
alunos.csv (antes da limpeza) e alunos_limpo.csv (após a limpeza).
"""

import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
BASE_DIR     = os.path.dirname(__file__)
BRUTO_PATH   = os.path.join(BASE_DIR, "data", "alunos.csv")
LIMPO_PATH   = os.path.join(BASE_DIR, "data", "alunos_limpo.csv")
OUTPUT_PATH  = os.path.join(BASE_DIR, "data", "dicionario_dados.csv")

# ---------------------------------------------------------------------------
# Carregar datasets
# ---------------------------------------------------------------------------
df_bruto = pd.read_csv(BRUTO_PATH)
df_limpo = pd.read_csv(LIMPO_PATH)

# ---------------------------------------------------------------------------
# Definições manuais de cada coluna
# ---------------------------------------------------------------------------
DEFINICOES = {
    "id_aluno": {
        "descricao": "Identificador único do aluno",
        "valores_possiveis": f"1 a {len(df_bruto)}",
    },
    "nome": {
        "descricao": "Nome completo do aluno",
        "valores_possiveis": "Texto livre",
    },
    "curso": {
        "descricao": "Curso de vinculação do aluno",
        "valores_possiveis": "Ciência da Computação | Engenharia Civil | Administração | Direito | Medicina",
    },
    "periodo": {
        "descricao": "Período atual do aluno no curso",
        "valores_possiveis": f"min={int(df_limpo['periodo'].min())} | max={int(df_limpo['periodo'].max())}",
    },
    "cr": {
        "descricao": "Coeficiente de Rendimento Acadêmico",
        "valores_possiveis": f"min={df_limpo['cr'].min():.1f} | max={df_limpo['cr'].max():.1f}",
    },
    "faltas": {
        "descricao": "Número total de faltas do aluno",
        "valores_possiveis": f"min={int(df_limpo['faltas'].min())} | max={int(df_limpo['faltas'].max())}",
    },
    "situacao_financeira": {
        "descricao": "Situação financeira declarada pelo aluno",
        "valores_possiveis": "estavel | intermediaria | dificuldade",
    },
    "reprovacoes": {
        "descricao": "Quantidade de reprovações acumuladas",
        "valores_possiveis": f"min={int(df_limpo['reprovacoes'].min())} | max={int(df_limpo['reprovacoes'].max())}",
    },
    "idade": {
        "descricao": "Idade do aluno em anos",
        "valores_possiveis": f"min={int(df_limpo['idade'].min())} | max={int(df_limpo['idade'].max())}",
    },
    "evadiu": {
        "descricao": "Variável alvo — indica se o aluno evadiu",
        "valores_possiveis": "0 = Não evadiu | 1 = Evadiu",
    },
}

# ---------------------------------------------------------------------------
# Montar dicionário
# ---------------------------------------------------------------------------
registros = []
for col in df_bruto.columns:
    tipo_python = str(df_bruto[col].dtype)

    pct_ausentes_antes = df_bruto[col].isnull().mean() * 100
    pct_ausentes_depois = df_limpo[col].isnull().mean() * 100 if col in df_limpo.columns else 0.0

    defn = DEFINICOES.get(col, {"descricao": "—", "valores_possiveis": "—"})

    registros.append({
        "coluna":                   col,
        "tipo_python":              tipo_python,
        "descricao":                defn["descricao"],
        "valores_possiveis":        defn["valores_possiveis"],
        "pct_ausentes_antes (%)":   round(pct_ausentes_antes, 2),
        "pct_ausentes_depois (%)":  round(pct_ausentes_depois, 2),
    })

dicionario = pd.DataFrame(registros)
dicionario.to_csv(OUTPUT_PATH, index=False)

print(f"[SPRINT 1] Dicionário de dados gerado: {len(dicionario)} colunas documentadas")
print(f"[SPRINT 1] Arquivo salvo em: {OUTPUT_PATH}")
