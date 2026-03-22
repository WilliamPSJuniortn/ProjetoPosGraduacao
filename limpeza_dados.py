"""
limpeza_dados.py
Carrega data/alunos.csv, realiza limpeza e salva em data/alunos_limpo.csv.
Etapas: tratamento de ausentes, remoção de duplicatas, tratamento de outliers (IQR).
"""

import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(__file__)
INPUT_PATH  = os.path.join(BASE_DIR, "data", "alunos.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "alunos_limpo.csv")

# ---------------------------------------------------------------------------
# Carregar dados
# ---------------------------------------------------------------------------
df = pd.read_csv(INPUT_PATH)
linhas_inicial = len(df)

print("[SPRINT 1] === Relatório de Limpeza ===")
print(f"[SPRINT 1] Registros carregados: {linhas_inicial}")

# ---------------------------------------------------------------------------
# 1. Valores ausentes
# ---------------------------------------------------------------------------
ausentes_por_coluna = df.isnull().sum()
total_ausentes = int(ausentes_por_coluna.sum())

print("\n[SPRINT 1] Valores ausentes por coluna:")
for col, qtd in ausentes_por_coluna.items():
    if qtd > 0:
        pct = qtd / linhas_inicial * 100
        print(f"           {col:<22} -> {qtd:>3} ({pct:.1f}%)")

# Tratamento: mediana para numéricas, moda para categóricas
COLUNAS_NUMERICAS    = ["periodo", "cr", "faltas", "reprovacoes", "idade"]
COLUNAS_CATEGORICAS  = ["curso", "situacao_financeira"]

for col in COLUNAS_NUMERICAS:
    mediana = df[col].median()
    df[col] = df[col].fillna(mediana)

for col in COLUNAS_CATEGORICAS:
    moda = df[col].mode()[0]
    df[col] = df[col].fillna(moda)

print(f"\n[SPRINT 1] Ausentes tratados: {total_ausentes} valores")
print(f"           Numéricas -> mediana | Categóricas -> moda")

# ---------------------------------------------------------------------------
# 2. Duplicatas
# ---------------------------------------------------------------------------
duplicatas = df.duplicated().sum()
df.drop_duplicates(inplace=True)
linhas_apos_dup = len(df)

print(f"\n[SPRINT 1] Duplicatas removidas: {duplicatas}")

# ---------------------------------------------------------------------------
# 3. Outliers via IQR nas colunas cr, faltas, reprovacoes
# ---------------------------------------------------------------------------
COLUNAS_OUTLIER = ["cr", "faltas", "reprovacoes"]
total_outliers = 0

print("\n[SPRINT 1] Tratamento de outliers (IQR):")
for col in COLUNAS_OUTLIER:
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lim_inf = Q1 - 1.5 * IQR
    lim_sup = Q3 + 1.5 * IQR

    mascara_outlier = (df[col] < lim_inf) | (df[col] > lim_sup)
    n_outliers = int(mascara_outlier.sum())
    total_outliers += n_outliers

    # Substituir outliers pelos limites (winsorização)
    df.loc[df[col] < lim_inf, col] = lim_inf
    df.loc[df[col] > lim_sup, col] = lim_sup

    print(f"           {col:<12} -> {n_outliers:>3} outliers "
          f"| limites [{lim_inf:.2f}, {lim_sup:.2f}]")

# ---------------------------------------------------------------------------
# Garantir tipos corretos após tratamento
# ---------------------------------------------------------------------------
df["periodo"]    = df["periodo"].round(0).astype(int)
df["faltas"]     = df["faltas"].round(0).astype(int)
df["reprovacoes"] = df["reprovacoes"].round(0).astype(int)
df["idade"]      = df["idade"].round(0).astype(int)
df["cr"]         = df["cr"].round(1)

# ---------------------------------------------------------------------------
# Salvar
# ---------------------------------------------------------------------------
df.to_csv(OUTPUT_PATH, index=False)
linhas_final = len(df)

print(f"\n[SPRINT 1] === Resumo Final ===")
print(f"[SPRINT 1] Linhas antes da limpeza : {linhas_inicial}")
print(f"[SPRINT 1] Linhas após  a limpeza  : {linhas_final}")
print(f"[SPRINT 1] Ausentes tratados       : {total_ausentes}")
print(f"[SPRINT 1] Outliers tratados       : {total_outliers}")
print(f"[SPRINT 1] Arquivo salvo em        : {OUTPUT_PATH}")
