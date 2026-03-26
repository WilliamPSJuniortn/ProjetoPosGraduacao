"""
modelagem.py
Sprint 2 — Treinamento e comparação de dois modelos de classificação:
  • Regressão Logística
  • Random Forest
Métricas: Acurácia, Precisão, Recall, F1-Score, ROC-AUC.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from preprocessamento import executar_preprocessamento

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
LR_PATH   = os.path.join(MODEL_DIR, "modelo_regressao_logistica.pkl")
RF_PATH   = os.path.join(MODEL_DIR, "modelo_random_forest.pkl")


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


def imprimir_metricas(nome, metricas):
    print(f"\n[SPRINT 2] --- {nome} ---")
    for metrica, valor in metricas.items():
        print(f"           {metrica:<12}: {valor:.4f}")


def imprimir_matriz_confusao(nome, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n[SPRINT 2] Matriz de Confusão — {nome}:")
    print(f"           {'':>14} Pred: Não Evadiu  Pred: Evadiu")
    print(f"           Real: Não Evadiu    {cm[0][0]:>5}            {cm[0][1]:>5}")
    print(f"           Real: Evadiu        {cm[1][0]:>5}            {cm[1][1]:>5}")


# ---------------------------------------------------------------------------
# Treinamento
# ---------------------------------------------------------------------------
def treinar_modelos():
    # Pré-processamento
    X_train, X_test, y_train, y_test, preprocessador, nomes_features = (
        executar_preprocessamento()
    )

    print("\n[SPRINT 2] === Treinamento dos Modelos ===")

    # ------------------------------------------------------------------
    # Modelo 1: Regressão Logística
    # ------------------------------------------------------------------
    print("\n[SPRINT 2] Treinando Regressão Logística...")
    lr = LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs")
    lr.fit(X_train, y_train)

    y_pred_lr   = lr.predict(X_test)
    y_proba_lr  = lr.predict_proba(X_test)[:, 1]
    metricas_lr = calcular_metricas(y_test, y_pred_lr, y_proba_lr)

    joblib.dump(lr, LR_PATH)
    print(f"[SPRINT 2] Modelo salvo em: {LR_PATH}")

    # ------------------------------------------------------------------
    # Modelo 2: Random Forest
    # ------------------------------------------------------------------
    print("\n[SPRINT 2] Treinando Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    y_pred_rf   = rf.predict(X_test)
    y_proba_rf  = rf.predict_proba(X_test)[:, 1]
    metricas_rf = calcular_metricas(y_test, y_pred_rf, y_proba_rf)

    joblib.dump(rf, RF_PATH)
    print(f"[SPRINT 2] Modelo salvo em: {RF_PATH}")

    # ------------------------------------------------------------------
    # Comparação de desempenho
    # ------------------------------------------------------------------
    print("\n[SPRINT 2] === Comparação de Desempenho ===")
    imprimir_metricas("Regressão Logística", metricas_lr)
    imprimir_metricas("Random Forest", metricas_rf)

    imprimir_matriz_confusao("Regressão Logística", y_test, y_pred_lr)
    imprimir_matriz_confusao("Random Forest", y_test, y_pred_rf)

    # ------------------------------------------------------------------
    # Relatório detalhado
    # ------------------------------------------------------------------
    print("\n[SPRINT 2] Relatório Detalhado — Regressão Logística:")
    print(classification_report(y_test, y_pred_lr, target_names=["Não Evadiu", "Evadiu"]))

    print("[SPRINT 2] Relatório Detalhado — Random Forest:")
    print(classification_report(y_test, y_pred_rf, target_names=["Não Evadiu", "Evadiu"]))

    # ------------------------------------------------------------------
    # Importância de features — Random Forest
    # ------------------------------------------------------------------
    importancias = pd.DataFrame({
        "feature":    nomes_features,
        "importancia": rf.feature_importances_,
    }).sort_values("importancia", ascending=False).head(10)

    print("[SPRINT 2] Top 10 Features — Random Forest:")
    for _, row in importancias.iterrows():
        barra = "#" * int(row["importancia"] * 50)
        print(f"           {row['feature']:<30} {row['importancia']:.4f}  {barra}")

    # ------------------------------------------------------------------
    # Vencedor
    # ------------------------------------------------------------------
    melhor = (
        "Random Forest"
        if metricas_rf["ROC-AUC"] >= metricas_lr["ROC-AUC"]
        else "Regressão Logística"
    )
    print(f"\n[SPRINT 2] Melhor modelo (ROC-AUC): {melhor}")

    return {
        "lr": {"modelo": lr, "metricas": metricas_lr},
        "rf": {"modelo": rf, "metricas": metricas_rf},
        "melhor": melhor,
    }


# ---------------------------------------------------------------------------
# Execução direta
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    treinar_modelos()
