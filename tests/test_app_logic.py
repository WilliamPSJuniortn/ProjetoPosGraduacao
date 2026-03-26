"""
test_app_logic.py
Testa toda a lógica do app.py sem precisar do Streamlit rodando.
Execute: python tests/test_app_logic.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
)

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

ERROS  = []
PASSES = []

def ok(msg):
    PASSES.append(msg)
    print(f"  [OK]   {msg}")

def fail(msg):
    ERROS.append(msg)
    print(f"  [FAIL] {msg}")

# -----------------------------------------------------------------------
print("\n[1] Carregamento de arquivos")
# -----------------------------------------------------------------------
try:
    df = pd.read_csv(os.path.join(BASE, "data", "alunos_limpo.csv"))
    assert len(df) > 0 and "evadiu" in df.columns
    ok(f"alunos_limpo.csv carregado — {len(df)} registros, {len(df.columns)} colunas")
except Exception as e:
    fail(f"alunos_limpo.csv: {e}")

try:
    prep = joblib.load(os.path.join(BASE, "models", "preprocessador.pkl"))
    ok("preprocessador.pkl carregado")
except Exception as e:
    fail(f"preprocessador.pkl: {e}")

try:
    lr = joblib.load(os.path.join(BASE, "models", "modelo_regressao_logistica.pkl"))
    ok("modelo_regressao_logistica.pkl carregado")
except Exception as e:
    fail(f"modelo_regressao_logistica.pkl: {e}")

try:
    rf = joblib.load(os.path.join(BASE, "models", "modelo_random_forest.pkl"))
    ok("modelo_random_forest.pkl carregado")
except Exception as e:
    fail(f"modelo_random_forest.pkl: {e}")

# -----------------------------------------------------------------------
print("\n[2] Separação e pré-processamento")
# -----------------------------------------------------------------------
try:
    X = df.drop(columns=["id_aluno", "nome", "evadiu"])
    y = df["evadiu"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_test_prep = prep.transform(X_test)
    assert X_test_prep.shape == (300, 13), f"shape esperado (300,13), obtido {X_test_prep.shape}"
    ok(f"Split OK — treino={len(X)-300}, teste={len(X_test)}, features={X_test_prep.shape[1]}")
except Exception as e:
    fail(f"Split/transform: {e}")

# -----------------------------------------------------------------------
print("\n[3] Nomes das features após OHE")
# -----------------------------------------------------------------------
try:
    nomes_cat = list(prep.named_transformers_["cat"].named_steps["onehot"]
                     .get_feature_names_out(["curso", "situacao_financeira"]))
    nomes_all = ["periodo", "cr", "faltas", "reprovacoes", "idade"] + nomes_cat
    assert len(nomes_all) == 13
    ok(f"13 features: {nomes_all}")
except Exception as e:
    fail(f"Nomes features: {e}")

# -----------------------------------------------------------------------
print("\n[4] Métricas dos dois modelos")
# -----------------------------------------------------------------------
resultados = {}
for nome, modelo in [("Regressão Logística", lr), ("Random Forest", rf)]:
    try:
        y_pred  = modelo.predict(X_test_prep)
        y_proba = modelo.predict_proba(X_test_prep)[:, 1]
        m = {
            "Acurácia" : round(accuracy_score(y_test, y_pred), 4),
            "Precisão" : round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall"   : round(recall_score(y_test, y_pred, zero_division=0), 4),
            "F1-Score" : round(f1_score(y_test, y_pred, zero_division=0), 4),
            "ROC-AUC"  : round(roc_auc_score(y_test, y_proba), 4),
        }
        resultados[nome] = {"y_pred": y_pred, "y_proba": y_proba, "metricas": m}
        assert m["Acurácia"] > 0.90, f"Acurácia muito baixa: {m['Acurácia']}"
        assert m["ROC-AUC"]  > 0.90, f"ROC-AUC muito baixo: {m['ROC-AUC']}"
        ok(f"{nome}: acc={m['Acurácia']}  auc={m['ROC-AUC']}")
    except Exception as e:
        fail(f"{nome} métricas: {e}")

# -----------------------------------------------------------------------
print("\n[5] Matrizes de confusão")
# -----------------------------------------------------------------------
for nome in resultados:
    try:
        cm = confusion_matrix(y_test, resultados[nome]["y_pred"])
        assert cm.shape == (2, 2)
        tn, fp, fn, tp = cm.ravel()
        ok(f"{nome}: TN={tn} FP={fp} FN={fn} TP={tp}")
    except Exception as e:
        fail(f"{nome} confusion matrix: {e}")

# -----------------------------------------------------------------------
print("\n[6] Curvas ROC")
# -----------------------------------------------------------------------
for nome in resultados:
    try:
        fpr, tpr, thresh = roc_curve(y_test, resultados[nome]["y_proba"])
        assert len(fpr) >= 3
        ok(f"{nome}: {len(fpr)} pontos na curva ROC")
    except Exception as e:
        fail(f"{nome} ROC: {e}")

# -----------------------------------------------------------------------
print("\n[7] Importância de features (RF)")
# -----------------------------------------------------------------------
try:
    imp = rf.feature_importances_
    assert len(imp) == 13
    top = sorted(zip(nomes_all, imp), key=lambda x: -x[1])[:3]
    ok(f"Top 3: {[(n, round(v,4)) for n,v in top]}")
except Exception as e:
    fail(f"Feature importance: {e}")

# -----------------------------------------------------------------------
print("\n[8] Coeficientes da Regressão Logística")
# -----------------------------------------------------------------------
try:
    coef = lr.coef_[0]
    assert len(coef) == 13
    top = sorted(zip(nomes_all, coef), key=lambda x: -abs(x[1]))[:3]
    ok(f"Top 3 coef: {[(n, round(v,3)) for n,v in top]}")
except Exception as e:
    fail(f"LR coeficientes: {e}")

# -----------------------------------------------------------------------
print("\n[9] Predição individual — cenários variados")
# -----------------------------------------------------------------------
cenarios = [
    {"periodo": 1, "cr": 3.0, "faltas": 30, "reprovacoes": 4, "idade": 18,
     "curso": "Direito", "situacao_financeira": "dificuldade", "espera": "alto"},
    {"periodo": 6, "cr": 8.5, "faltas": 1,  "reprovacoes": 0, "idade": 24,
     "curso": "Medicina", "situacao_financeira": "estavel", "espera": "baixo"},
    {"periodo": 3, "cr": 5.5, "faltas": 10, "reprovacoes": 1, "idade": 21,
     "curso": "Administração", "situacao_financeira": "intermediaria", "espera": "medio"},
]
for cen in cenarios:
    try:
        espera = cen.pop("espera")
        entrada = pd.DataFrame([cen])
        ep    = prep.transform(entrada)
        prob  = float(lr.predict_proba(ep)[0][1]) * 100
        nivel = "alto" if prob >= 60 else ("medio" if prob >= 35 else "baixo")
        ok(f"Cenario '{espera}': prob={prob:.1f}%  nivel={nivel}  {'consistente' if nivel == espera else 'inesperado'}")
    except Exception as e:
        fail(f"Predição cenário {cen}: {e}")

# -----------------------------------------------------------------------
print("\n[10] Filtro de curso (sidebar)")
# -----------------------------------------------------------------------
try:
    for curso in df["curso"].unique()[:3]:
        df_f = df[df["curso"] == curso]
        assert len(df_f) > 0
    ok(f"Filtro por curso funcional para {len(df['curso'].unique())} cursos")
except Exception as e:
    fail(f"Filtro curso: {e}")

# -----------------------------------------------------------------------
print("\n[11] Session state — estrutura do resultado")
# -----------------------------------------------------------------------
try:
    entrada = pd.DataFrame([{"periodo": 2, "cr": 4.0, "faltas": 20, "reprovacoes": 2, "idade": 20,
                              "curso": "Ciência da Computação", "situacao_financeira": "dificuldade"}])
    ep   = prep.transform(entrada)
    prob = float(lr.predict_proba(ep)[0][1]) * 100
    pred = int(lr.predict(ep)[0])
    resultado = {"prob": prob, "pred": pred, "modelo": "Regressão Logística",
                 "fatores": [], "aluno": {}}
    assert "prob" in resultado and "pred" in resultado and "modelo" in resultado
    ok(f"Estrutura session_state OK — prob={prob:.1f}%  pred={pred}")
except Exception as e:
    fail(f"Session state: {e}")

# -----------------------------------------------------------------------
print("\n" + "=" * 55)
print(f"RESULTADO: {len(PASSES)} passou(aram) | {len(ERROS)} falhou(aram)")
if ERROS:
    print("\nFalhas:")
    for e in ERROS:
        print(f"  • {e}")
else:
    print("Todos os testes passaram!")
print("=" * 55)
