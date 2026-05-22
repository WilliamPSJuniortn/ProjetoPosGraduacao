"""
otimizacao.py
Sprint 3 — Otimização de hiperparâmetros via GridSearchCV:
  • Regressão Logística otimizada
  • Random Forest otimizado
Validação cruzada estratificada (StratifiedKFold, 5 folds).
Métrica de seleção: ROC-AUC.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from preprocessamento import executar_preprocessamento

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR       = os.path.join(BASE_DIR, "models")
LR_OPT_PATH     = os.path.join(MODEL_DIR, "modelo_lr_otimizado.pkl")
RF_OPT_PATH     = os.path.join(MODEL_DIR, "modelo_rf_otimizado.pkl")
RESULTADOS_PATH = os.path.join(BASE_DIR, "data", "resultados_sprint3.csv")


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------
def calcular_metricas(y_true, y_pred, y_proba):
    return {
        "Acurácia":  round(accuracy_score(y_true, y_pred), 4),
        "Precisão":  round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1-Score":  round(f1_score(y_true, y_pred, zero_division=0), 4),
        "ROC-AUC":   round(roc_auc_score(y_true, y_proba), 4),
    }


def imprimir_grid_resultado(nome, grid_search):
    print(f"\n[SPRINT 3] --- {nome} ---")
    print(f"           Melhores parâmetros:")
    for param, valor in grid_search.best_params_.items():
        print(f"             {param:<25}: {valor}")
    print(f"           Melhor ROC-AUC (CV): {grid_search.best_score_:.4f}")


# ---------------------------------------------------------------------------
# Otimização principal
# ---------------------------------------------------------------------------
def otimizar_modelos():
    # Pré-processamento
    X_train, X_test, y_train, y_test, preprocessador, nomes_features = (
        executar_preprocessamento()
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n[SPRINT 3] === Otimização de Hiperparâmetros (GridSearchCV) ===")
    print(f"[SPRINT 3] Validação cruzada: StratifiedKFold (5 folds)")
    print(f"[SPRINT 3] Métrica de seleção: ROC-AUC")

    # ------------------------------------------------------------------
    # Modelo 1: Regressão Logística — GridSearchCV
    # ------------------------------------------------------------------
    print("\n[SPRINT 3] Otimizando Regressão Logística...")
    grade_lr = {
        "C":           [0.01, 0.1, 1, 10, 100],
        "solver":      ["lbfgs", "liblinear"],
        "penalty":     ["l2"],
        "max_iter":    [1000],
        "random_state":[42],
    }
    grid_lr = GridSearchCV(
        LogisticRegression(), grade_lr,
        scoring="roc_auc", cv=cv, n_jobs=-1, verbose=0,
    )
    grid_lr.fit(X_train, y_train)

    lr_otimizado = grid_lr.best_estimator_
    y_pred_lr    = lr_otimizado.predict(X_test)
    y_proba_lr   = lr_otimizado.predict_proba(X_test)[:, 1]
    metricas_lr  = calcular_metricas(y_test, y_pred_lr, y_proba_lr)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(lr_otimizado, LR_OPT_PATH)
    imprimir_grid_resultado("Regressão Logística Otimizada", grid_lr)
    print(f"[SPRINT 3] Modelo salvo em: {LR_OPT_PATH}")

    # ------------------------------------------------------------------
    # Modelo 2: Random Forest — GridSearchCV
    # ------------------------------------------------------------------
    print("\n[SPRINT 3] Otimizando Random Forest...")
    grade_rf = {
        "n_estimators":      [50, 100, 200],
        "max_depth":         [None, 5, 10, 15],
        "min_samples_split": [2, 5, 10],
        "random_state":      [42],
    }
    grid_rf = GridSearchCV(
        RandomForestClassifier(), grade_rf,
        scoring="roc_auc", cv=cv, n_jobs=-1, verbose=0,
    )
    grid_rf.fit(X_train, y_train)

    rf_otimizado = grid_rf.best_estimator_
    y_pred_rf    = rf_otimizado.predict(X_test)
    y_proba_rf   = rf_otimizado.predict_proba(X_test)[:, 1]
    metricas_rf  = calcular_metricas(y_test, y_pred_rf, y_proba_rf)

    joblib.dump(rf_otimizado, RF_OPT_PATH)
    imprimir_grid_resultado("Random Forest Otimizado", grid_rf)
    print(f"[SPRINT 3] Modelo salvo em: {RF_OPT_PATH}")

    # ------------------------------------------------------------------
    # Carregar modelos base para comparação
    # ------------------------------------------------------------------
    lr_base = joblib.load(os.path.join(MODEL_DIR, "modelo_regressao_logistica.pkl"))
    rf_base = joblib.load(os.path.join(MODEL_DIR, "modelo_random_forest.pkl"))

    m_lr_base = calcular_metricas(
        y_test, lr_base.predict(X_test), lr_base.predict_proba(X_test)[:, 1]
    )
    m_rf_base = calcular_metricas(
        y_test, rf_base.predict(X_test), rf_base.predict_proba(X_test)[:, 1]
    )

    # ------------------------------------------------------------------
    # Comparação: base vs otimizado
    # ------------------------------------------------------------------
    print("\n[SPRINT 3] === Ganho com Otimização ===")
    cabecalho = f"[SPRINT 3] {'Modelo':<35} {'Acurácia':>9} {'F1-Score':>9} {'ROC-AUC':>9}"
    print(cabecalho)
    print(f"[SPRINT 3] {'-' * 64}")
    for rotulo, m in [
        ("Regressão Logística (base)",      m_lr_base),
        ("Regressão Logística (otimizada)", metricas_lr),
        ("Random Forest (base)",            m_rf_base),
        ("Random Forest (otimizado)",       metricas_rf),
    ]:
        print(
            f"[SPRINT 3] {rotulo:<35} "
            f"{m['Acurácia']:>9.4f} {m['F1-Score']:>9.4f} {m['ROC-AUC']:>9.4f}"
        )

    # ------------------------------------------------------------------
    # Vencedor
    # ------------------------------------------------------------------
    melhor = (
        "Random Forest"
        if metricas_rf["ROC-AUC"] >= metricas_lr["ROC-AUC"]
        else "Regressão Logística"
    )
    print(f"\n[SPRINT 3] Melhor modelo otimizado (ROC-AUC): {melhor}")

    # ------------------------------------------------------------------
    # Salvar tabela de resultados
    # ------------------------------------------------------------------
    linhas = [
        {"modelo": "LR Base",      "tipo": "base",      **m_lr_base},
        {"modelo": "LR Otimizado", "tipo": "otimizado", **metricas_lr},
        {"modelo": "RF Base",      "tipo": "base",      **m_rf_base},
        {"modelo": "RF Otimizado", "tipo": "otimizado", **metricas_rf},
    ]
    pd.DataFrame(linhas).to_csv(RESULTADOS_PATH, index=False)
    print(f"[SPRINT 3] Resultados salvos em: {RESULTADOS_PATH}")

    return {
        "lr": {"modelo": lr_otimizado, "metricas": metricas_lr, "grid": grid_lr},
        "rf": {"modelo": rf_otimizado, "metricas": metricas_rf, "grid": grid_rf},
        "melhor": melhor,
    }


# ---------------------------------------------------------------------------
# Execução direta
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    otimizar_modelos()
    print("\n[SPRINT 3] Otimização concluída.")
