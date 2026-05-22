"""
avaliacao_avancada.py
Sprint 3 — Avaliação avançada dos modelos otimizados:
  • Validação cruzada estratificada (5 folds, múltiplas métricas)
  • Curvas de aprendizado (learning curves)
  • Curva de calibração (reliability diagram)
  • Curva Precisão-Recall + Average Precision Score
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold, learning_curve
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
)

from preprocessamento import executar_preprocessamento

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(BASE_DIR, "models")
LR_OPT_PATH = os.path.join(MODEL_DIR, "modelo_lr_otimizado.pkl")
RF_OPT_PATH = os.path.join(MODEL_DIR, "modelo_rf_otimizado.pkl")
CV_PATH     = os.path.join(BASE_DIR, "data", "cv_resultados_sprint3.csv")
LC_PATH     = os.path.join(BASE_DIR, "data", "learning_curves_sprint3.csv")


def executar_avaliacao_avancada():
    # Pré-processamento
    X_train, X_test, y_train, y_test, preprocessador, nomes_features = (
        executar_preprocessamento()
    )

    lr_otimizado = joblib.load(LR_OPT_PATH)
    rf_otimizado = joblib.load(RF_OPT_PATH)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metricas_cv = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    # ------------------------------------------------------------------
    # Validação cruzada
    # ------------------------------------------------------------------
    print("\n[SPRINT 3] === Validação Cruzada (5-Fold Estratificada) ===")

    linhas_cv = []
    for nome, modelo in [
        ("Regressão Logística", lr_otimizado),
        ("Random Forest",       rf_otimizado),
    ]:
        scores = cross_validate(
            modelo, X_train, y_train,
            cv=cv, scoring=metricas_cv, n_jobs=-1,
        )
        print(f"\n[SPRINT 3] --- {nome} ---")
        row = {"modelo": nome}
        for m in metricas_cv:
            vals = scores[f"test_{m}"]
            print(f"           {m:<12}: {vals.mean():.4f} ± {vals.std():.4f}")
            row[f"{m}_media"] = round(float(vals.mean()), 4)
            row[f"{m}_std"]   = round(float(vals.std()), 4)
        linhas_cv.append(row)

    pd.DataFrame(linhas_cv).to_csv(CV_PATH, index=False)
    print(f"\n[SPRINT 3] Resultados CV salvos em: {CV_PATH}")

    # ------------------------------------------------------------------
    # Curvas de aprendizado
    # ------------------------------------------------------------------
    print("\n[SPRINT 3] === Curvas de Aprendizado (ROC-AUC) ===")

    X_full = np.vstack([X_train, X_test])
    y_full = np.concatenate([np.array(y_train), np.array(y_test)])
    tamanhos_rel = np.linspace(0.1, 1.0, 8)

    linhas_lc = []
    for nome, modelo in [
        ("Regressão Logística", lr_otimizado),
        ("Random Forest",       rf_otimizado),
    ]:
        tamanhos, scores_treino, scores_val = learning_curve(
            modelo, X_full, y_full,
            train_sizes=tamanhos_rel,
            cv=cv, scoring="roc_auc", n_jobs=-1,
        )
        print(f"\n[SPRINT 3] {nome} — ROC-AUC por tamanho de treino:")
        for t, st, sv in zip(tamanhos, scores_treino.mean(axis=1), scores_val.mean(axis=1)):
            barra = "#" * int(sv * 30)
            print(f"           Treino={t:>4}  AUC_val={sv:.4f}  {barra}")
            linhas_lc.append({
                "modelo":          nome,
                "tamanho_treino":  int(t),
                "auc_val_media":   round(float(sv), 4),
                "auc_val_std":     round(float(scores_val.std(axis=1)[list(tamanhos).index(t)]), 4),
                "auc_treino_media":round(float(st), 4),
                "auc_treino_std":  round(float(scores_treino.std(axis=1)[list(tamanhos).index(t)]), 4),
            })

    pd.DataFrame(linhas_lc).to_csv(LC_PATH, index=False)
    print(f"\n[SPRINT 3] Curvas de aprendizado salvas em: {LC_PATH}")

    # ------------------------------------------------------------------
    # Curva Precisão-Recall
    # ------------------------------------------------------------------
    print("\n[SPRINT 3] === Curva Precisão-Recall ===")
    for nome, modelo in [
        ("Regressão Logística", lr_otimizado),
        ("Random Forest",       rf_otimizado),
    ]:
        y_proba = modelo.predict_proba(X_test)[:, 1]
        ap      = average_precision_score(y_test, y_proba)
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        print(
            f"[SPRINT 3] {nome:<35} "
            f"Average Precision: {ap:.4f}  |  Pontos na curva: {len(prec)}"
        )

    # ------------------------------------------------------------------
    # Calibração dos modelos
    # ------------------------------------------------------------------
    print("\n[SPRINT 3] === Calibração dos Modelos ===")
    print("[SPRINT 3] (Comparação entre probabilidade prevista e frequência real)")
    for nome, modelo in [
        ("Regressão Logística", lr_otimizado),
        ("Random Forest",       rf_otimizado),
    ]:
        y_proba = modelo.predict_proba(X_test)[:, 1]
        frac_pos, media_prev = calibration_curve(y_test, y_proba, n_bins=5)
        print(f"\n[SPRINT 3] {nome}:")
        print(f"           {'Prev. Médio':>12} {'Freq. Real':>12} {'Desvio':>10}")
        for mp, fp in zip(media_prev, frac_pos):
            desvio = fp - mp
            print(f"           {mp:>12.4f} {fp:>12.4f} {desvio:>+10.4f}")

    # ------------------------------------------------------------------
    # Conclusão
    # ------------------------------------------------------------------
    ap_lr = average_precision_score(
        y_test, lr_otimizado.predict_proba(X_test)[:, 1]
    )
    ap_rf = average_precision_score(
        y_test, rf_otimizado.predict_proba(X_test)[:, 1]
    )
    melhor_ap = "Random Forest" if ap_rf >= ap_lr else "Regressão Logística"
    print(f"\n[SPRINT 3] Melhor Average Precision: {melhor_ap}")

    return {
        "resultados_cv": linhas_cv,
        "lr":            lr_otimizado,
        "rf":            rf_otimizado,
        "X_test":        X_test,
        "y_test":        y_test,
    }


# ---------------------------------------------------------------------------
# Execução direta
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    executar_avaliacao_avancada()
    print("\n[SPRINT 3] Avaliação avançada concluída.")
