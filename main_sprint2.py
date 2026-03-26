"""
main_sprint2.py
Pipeline principal da Sprint 2 — pré-processamento, treinamento de dois
modelos de classificação e comparação de desempenho.
"""

import subprocess
import sys
import os

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
        print(f"[SPRINT 2] ERRO ao executar {nome_script}:")
        print(resultado.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Verificação de dependência: alunos_limpo.csv precisa existir
# ---------------------------------------------------------------------------
limpo_path = os.path.join(BASE_DIR, "data", "alunos_limpo.csv")
if not os.path.exists(limpo_path):
    print("[SPRINT 2] Dataset limpo não encontrado — executando Sprint 1 antes...")
    executar_script("main_sprint1.py")

# ---------------------------------------------------------------------------
# Sprint 2 — Pré-processamento + Modelagem
# ---------------------------------------------------------------------------
print("=" * 60)
print("[SPRINT 2] Iniciando Sprint 2 — Modelagem")
print("=" * 60)

print("\n[SPRINT 2] Etapa 1: Pré-processamento...")
executar_script("preprocessamento.py")

print("\n[SPRINT 2] Etapa 2: Treinamento e comparação dos modelos...")
executar_script("modelagem.py")

# ---------------------------------------------------------------------------
# Resumo final
# ---------------------------------------------------------------------------
import joblib
import pandas as pd

lr_path = os.path.join(BASE_DIR, "models", "modelo_regressao_logistica.pkl")
rf_path = os.path.join(BASE_DIR, "models", "modelo_random_forest.pkl")

print()
print("=" * 60)
print("[SPRINT 2] CONCLUÍDA — Modelos treinados e comparados")
print("=" * 60)
print("[SPRINT 2] Arquivos gerados:")
print("           • models/preprocessador.pkl")
print("           • models/modelo_regressao_logistica.pkl")
print("           • models/modelo_random_forest.pkl")
print("[SPRINT 2] Scripts criados:")
print("           • preprocessamento.py")
print("           • modelagem.py")
print("           • main_sprint2.py")
print("=" * 60)
