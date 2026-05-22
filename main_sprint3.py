"""
main_sprint3.py
Pipeline principal da Sprint 3 — Otimização de hiperparâmetros e avaliação
avançada. Executa os scripts em sequência e imprime o resumo final consolidado.
"""

import subprocess
import sys
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

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
        print(f"[SPRINT 3] ERRO ao executar {nome_script}:")
        print(resultado.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Verificar dependências das sprints anteriores
# ---------------------------------------------------------------------------
lr_base_path = os.path.join(BASE_DIR, "models", "modelo_regressao_logistica.pkl")
rf_base_path = os.path.join(BASE_DIR, "models", "modelo_random_forest.pkl")

if not os.path.exists(lr_base_path) or not os.path.exists(rf_base_path):
    print("[SPRINT 3] Modelos da Sprint 2 não encontrados. Executando main_sprint2.py...")
    executar_script("main_sprint2.py")

# ---------------------------------------------------------------------------
# Etapa 1 — Otimização de hiperparâmetros (GridSearchCV)
# ---------------------------------------------------------------------------
print("[SPRINT 3] Iniciando otimização de hiperparâmetros...")
executar_script("otimizacao.py")

# ---------------------------------------------------------------------------
# Etapa 2 — Avaliação avançada (CV, learning curves, calibração, PR)
# ---------------------------------------------------------------------------
print("\n[SPRINT 3] Iniciando avaliação avançada dos modelos...")
executar_script("avaliacao_avancada.py")

# ---------------------------------------------------------------------------
# Coletar estatísticas para o resumo final
# ---------------------------------------------------------------------------
df_limpo      = pd.read_csv(os.path.join(BASE_DIR, "data", "alunos_limpo.csv"))
resultados_s3 = pd.read_csv(os.path.join(BASE_DIR, "data", "resultados_sprint3.csv"))
cv_resultados = pd.read_csv(os.path.join(BASE_DIR, "data", "cv_resultados_sprint3.csv"))

prep   = joblib.load(os.path.join(BASE_DIR, "models", "preprocessador.pkl"))
lr_opt = joblib.load(os.path.join(BASE_DIR, "models", "modelo_lr_otimizado.pkl"))
rf_opt = joblib.load(os.path.join(BASE_DIR, "models", "modelo_rf_otimizado.pkl"))

X = df_limpo.drop(columns=["id_aluno", "nome", "evadiu"])
y = df_limpo["evadiu"]
_, X_test_df, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_test_prep = prep.transform(X_test_df)

auc_lr = round(roc_auc_score(y_test, lr_opt.predict_proba(X_test_prep)[:, 1]), 4)
auc_rf = round(roc_auc_score(y_test, rf_opt.predict_proba(X_test_prep)[:, 1]), 4)
ap_lr  = round(average_precision_score(y_test, lr_opt.predict_proba(X_test_prep)[:, 1]), 4)
ap_rf  = round(average_precision_score(y_test, rf_opt.predict_proba(X_test_prep)[:, 1]), 4)

melhor     = "Random Forest" if auc_rf >= auc_lr else "Regressão Logística"
auc_melhor = max(auc_lr, auc_rf)

cv_lr = cv_resultados[cv_resultados["modelo"] == "Regressão Logística"].iloc[0]
cv_rf = cv_resultados[cv_resultados["modelo"] == "Random Forest"].iloc[0]

# ---------------------------------------------------------------------------
# Resumo final
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("[SPRINT 3] CONCLUÍDA — Modelos otimizados e avaliados")
print("=" * 60)
print(f"[SPRINT 3] Método            : GridSearchCV + StratifiedKFold (5 folds)")
print(f"[SPRINT 3] Métrica de seleção: ROC-AUC")
print()
print(f"[SPRINT 3] {'Modelo':<35} {'ROC-AUC':>9} {'Avg-Prec':>9}")
print(f"[SPRINT 3] {'-' * 55}")
print(f"[SPRINT 3] {'Regressão Logística (otimizada)':<35} {auc_lr:>9.4f} {ap_lr:>9.4f}")
print(f"[SPRINT 3] {'Random Forest (otimizado)':<35} {auc_rf:>9.4f} {ap_rf:>9.4f}")
print()
print(f"[SPRINT 3] Validação Cruzada — ROC-AUC médio (5 folds):")
print(
    f"           Regressão Logística : "
    f"{cv_lr['roc_auc_media']:.4f} ± {cv_lr['roc_auc_std']:.4f}"
)
print(
    f"           Random Forest       : "
    f"{cv_rf['roc_auc_media']:.4f} ± {cv_rf['roc_auc_std']:.4f}"
)
print()
print(f"[SPRINT 3] Melhor modelo final : {melhor}  (ROC-AUC = {auc_melhor:.4f})")
print(f"[SPRINT 3] Arquivos gerados:")
print("           • models/modelo_lr_otimizado.pkl")
print("           • models/modelo_rf_otimizado.pkl")
print("           • data/resultados_sprint3.csv")
print("           • data/cv_resultados_sprint3.csv")
print("           • data/learning_curves_sprint3.csv")
print("=" * 60)
