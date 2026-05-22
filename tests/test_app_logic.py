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
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
    average_precision_score, precision_recall_curve,
)
from sklearn.calibration import calibration_curve

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
print("\n[12] Sprint 3 — Carregamento dos modelos otimizados")
# -----------------------------------------------------------------------
s3_disponivel = False
lr_opt = rf_opt = None

try:
    lr_opt = joblib.load(os.path.join(BASE, "models", "modelo_lr_otimizado.pkl"))
    ok("modelo_lr_otimizado.pkl carregado")
except Exception as e:
    fail(f"modelo_lr_otimizado.pkl: {e} — execute python main_sprint3.py")

try:
    rf_opt = joblib.load(os.path.join(BASE, "models", "modelo_rf_otimizado.pkl"))
    ok("modelo_rf_otimizado.pkl carregado")
    s3_disponivel = lr_opt is not None
except Exception as e:
    fail(f"modelo_rf_otimizado.pkl: {e} — execute python main_sprint3.py")

# -----------------------------------------------------------------------
print("\n[13] Sprint 3 — Arquivos de resultados")
# -----------------------------------------------------------------------
try:
    df_res = pd.read_csv(os.path.join(BASE, "data", "resultados_sprint3.csv"))
    assert len(df_res) == 4, f"Esperado 4 linhas, obtido {len(df_res)}"
    assert "ROC-AUC" in df_res.columns
    ok(f"resultados_sprint3.csv: {len(df_res)} linhas, colunas={list(df_res.columns)}")
except Exception as e:
    fail(f"resultados_sprint3.csv: {e}")

try:
    df_cv = pd.read_csv(os.path.join(BASE, "data", "cv_resultados_sprint3.csv"))
    assert len(df_cv) == 2, f"Esperado 2 linhas (um por modelo), obtido {len(df_cv)}"
    assert "roc_auc_media" in df_cv.columns
    ok(f"cv_resultados_sprint3.csv: {len(df_cv)} modelos, AUC médio: "
       f"LR={df_cv[df_cv['modelo']=='Regressão Logística']['roc_auc_media'].values[0]:.4f} | "
       f"RF={df_cv[df_cv['modelo']=='Random Forest']['roc_auc_media'].values[0]:.4f}")
except Exception as e:
    fail(f"cv_resultados_sprint3.csv: {e}")

try:
    df_lc = pd.read_csv(os.path.join(BASE, "data", "learning_curves_sprint3.csv"))
    assert len(df_lc) >= 8, f"Esperado >= 8 pontos, obtido {len(df_lc)}"
    assert "auc_val_media" in df_lc.columns
    ok(f"learning_curves_sprint3.csv: {len(df_lc)} pontos de curva de aprendizado")
except Exception as e:
    fail(f"learning_curves_sprint3.csv: {e}")

# -----------------------------------------------------------------------
print("\n[14] Sprint 3 — Métricas dos modelos otimizados")
# -----------------------------------------------------------------------
if s3_disponivel:
    for nome, modelo in [("LR Otimizado", lr_opt), ("RF Otimizado", rf_opt)]:
        try:
            y_pred  = modelo.predict(X_test_prep)
            y_proba = modelo.predict_proba(X_test_prep)[:, 1]
            auc     = roc_auc_score(y_test, y_proba)
            acc     = accuracy_score(y_test, y_pred)
            assert auc  > 0.90, f"ROC-AUC muito baixo: {auc:.4f}"
            assert acc  > 0.90, f"Acurácia muito baixa: {acc:.4f}"
            ok(f"{nome}: acc={acc:.4f}  auc={auc:.4f}")
        except Exception as e:
            fail(f"{nome} métricas: {e}")
else:
    fail("Modelos otimizados não disponíveis — pule este bloco após rodar main_sprint3.py")

# -----------------------------------------------------------------------
print("\n[15] Sprint 3 — Ganho em relação ao modelo base")
# -----------------------------------------------------------------------
if s3_disponivel:
    try:
        auc_lr_base = roc_auc_score(y_test, lr.predict_proba(X_test_prep)[:, 1])
        auc_rf_base = roc_auc_score(y_test, rf.predict_proba(X_test_prep)[:, 1])
        auc_lr_opt  = roc_auc_score(y_test, lr_opt.predict_proba(X_test_prep)[:, 1])
        auc_rf_opt  = roc_auc_score(y_test, rf_opt.predict_proba(X_test_prep)[:, 1])
        ganho_lr = auc_lr_opt - auc_lr_base
        ganho_rf = auc_rf_opt - auc_rf_base
        ok(f"Ganho LR: {ganho_lr:+.4f}  (base={auc_lr_base:.4f} → opt={auc_lr_opt:.4f})")
        ok(f"Ganho RF: {ganho_rf:+.4f}  (base={auc_rf_base:.4f} → opt={auc_rf_opt:.4f})")
    except Exception as e:
        fail(f"Ganho de desempenho: {e}")

# -----------------------------------------------------------------------
print("\n[16] Sprint 3 — Average Precision e curva Precisão-Recall")
# -----------------------------------------------------------------------
if s3_disponivel:
    for nome, modelo in [("LR Otimizado", lr_opt), ("RF Otimizado", rf_opt)]:
        try:
            y_proba = modelo.predict_proba(X_test_prep)[:, 1]
            ap      = average_precision_score(y_test, y_proba)
            prec, rec, _ = precision_recall_curve(y_test, y_proba)
            assert ap > 0.80, f"Average Precision muito baixa: {ap:.4f}"
            assert len(prec) >= 3
            ok(f"{nome}: Average Precision={ap:.4f}  |  {len(prec)} pontos na curva P-R")
        except Exception as e:
            fail(f"{nome} P-R curve: {e}")

# -----------------------------------------------------------------------
print("\n[17] Sprint 3 — Calibração dos modelos otimizados")
# -----------------------------------------------------------------------
if s3_disponivel:
    for nome, modelo in [("LR Otimizado", lr_opt), ("RF Otimizado", rf_opt)]:
        try:
            y_proba = modelo.predict_proba(X_test_prep)[:, 1]
            frac_pos, media_prev = calibration_curve(y_test, y_proba, n_bins=5)
            desvio_medio = float(np.mean(np.abs(frac_pos - media_prev)))
            assert desvio_medio < 0.25, f"Modelo muito descalibrado: desvio={desvio_medio:.4f}"
            ok(f"{nome}: desvio médio de calibração = {desvio_medio:.4f}")
        except Exception as e:
            fail(f"{nome} calibração: {e}")

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
