"""
main_sprint1.py
Pipeline principal da Sprint 1 — executa todos os scripts em sequência
e imprime o resumo final consolidado.
"""

import subprocess
import sys
import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def executar_script(nome_script):
    """Executa um script Python da raiz do projeto via subprocess."""
    caminho = os.path.join(BASE_DIR, nome_script)
    resultado = subprocess.run(
        [sys.executable, caminho],
        capture_output=True, text=True, cwd=BASE_DIR,
    )
    if resultado.stdout:
        print(resultado.stdout, end="")
    if resultado.returncode != 0:
        print(f"[SPRINT 1] ERRO ao executar {nome_script}:")
        print(resultado.stderr)
        sys.exit(1)


def contar_outliers_iqr(series):
    """Conta outliers pelo método IQR em uma Series pandas."""
    Q1  = series.quantile(0.25)
    Q3  = series.quantile(0.75)
    IQR = Q3 - Q1
    return int(((series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)).sum())


# ---------------------------------------------------------------------------
# Etapa 1 — Geração do dataset
# ---------------------------------------------------------------------------
print("[SPRINT 1] Iniciando geração do dataset...")
executar_script("gerar_dataset.py")

# ---------------------------------------------------------------------------
# Etapa 2 — Limpeza dos dados
# ---------------------------------------------------------------------------
print("\n[SPRINT 1] Iniciando limpeza dos dados...")
executar_script("limpeza_dados.py")

# ---------------------------------------------------------------------------
# Etapa 3 — Dicionário de dados
# ---------------------------------------------------------------------------
print("\n[SPRINT 1] Gerando dicionário de dados...")
executar_script("dicionario_dados.py")

# ---------------------------------------------------------------------------
# Coletar estatísticas para o resumo final
# ---------------------------------------------------------------------------
df_bruto = pd.read_csv(os.path.join(BASE_DIR, "data", "alunos.csv"))
df_limpo = pd.read_csv(os.path.join(BASE_DIR, "data", "alunos_limpo.csv"))

# Contagem de ausentes no dataset bruto
total_ausentes = int(df_bruto.isnull().sum().sum())

# Contagem de outliers nas colunas tratadas (calculado sobre o dataset bruto)
colunas_outlier = ["cr", "faltas", "reprovacoes"]
total_outliers = sum(
    contar_outliers_iqr(df_bruto[col].dropna())
    for col in colunas_outlier
)

# Taxa de evasão (dataset limpo)
taxa_evasao = df_limpo["evadiu"].mean() * 100

# Top 3 correlações com evadiu
numericas = df_limpo[["periodo", "cr", "faltas", "reprovacoes", "idade", "evadiu"]]
correlacoes = (
    numericas.corr()["evadiu"]
    .drop("evadiu")
    .reindex(numericas.corr()["evadiu"].drop("evadiu").abs().sort_values(ascending=False).index)
)
top3 = correlacoes.head(3)

# ---------------------------------------------------------------------------
# Resumo final
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("[SPRINT 1] CONCLUÍDA — Base pronta para modelagem")
print("=" * 60)
print(f"[SPRINT 1] Dataset bruto  : {len(df_bruto)} registros | {len(df_bruto.columns)} colunas")
print(f"[SPRINT 1] Dataset limpo  : {len(df_limpo)} registros   | {len(df_limpo.columns)} colunas")
print(f"[SPRINT 1] Taxa de evasão : {taxa_evasao:.1f}%")
print(f"[SPRINT 1] Ausentes tratados: {total_ausentes} valores")
print(f"[SPRINT 1] Outliers tratados: {total_outliers} valores")
print(f"[SPRINT 1] Top 3 correlações com evasão:")
for i, (col, r) in enumerate(top3.items(), start=1):
    print(f"           {i}. {col:<16} -> r = {r:+.2f}")
print("[SPRINT 1] Arquivos gerados:")
print("           • data/alunos.csv")
print("           • data/alunos_limpo.csv")
print("           • data/dicionario_dados.csv")
print("=" * 60)
