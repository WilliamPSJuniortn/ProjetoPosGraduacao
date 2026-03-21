import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/alunos.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/modelo_evasao.pkl")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "../models/encoder.pkl")


def carregar_dados(caminho=DATA_PATH):
    df = pd.read_csv(caminho)
    return df


def preprocessar(df):
    df = df.copy()

    le = LabelEncoder()
    df["curso_enc"] = le.fit_transform(df["curso"])
    df["financeiro_enc"] = le.fit_transform(df["situacao_financeira"])

    encoders = {
        "curso": LabelEncoder().fit(df["curso"]),
        "situacao_financeira": LabelEncoder().fit(df["situacao_financeira"]),
    }

    features = ["periodo", "cr", "faltas", "reprovacoes", "idade", "curso_enc", "financeiro_enc"]
    X = df[features]
    y = df["evadiu"]

    return X, y, encoders


def treinar_modelo(df=None):
    if df is None:
        df = carregar_dados()

    X, y, encoders = preprocessar(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(modelo, MODEL_PATH)
    joblib.dump(encoders, ENCODER_PATH)

    y_pred = modelo.predict(X_test)
    relatorio = classification_report(y_test, y_pred, target_names=["Não Evadiu", "Evadiu"])
    matriz = confusion_matrix(y_test, y_pred)

    return modelo, encoders, relatorio, matriz


def carregar_modelo():
    if not os.path.exists(MODEL_PATH):
        treinar_modelo()
    modelo = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODER_PATH)
    return modelo, encoders


def prever_evasao(dados: dict):
    modelo, encoders = carregar_modelo()

    curso_enc = encoders["curso"].transform([dados["curso"]])[0]
    financeiro_enc = encoders["situacao_financeira"].transform([dados["situacao_financeira"]])[0]

    X = pd.DataFrame([{
        "periodo": dados["periodo"],
        "cr": dados["cr"],
        "faltas": dados["faltas"],
        "reprovacoes": dados["reprovacoes"],
        "idade": dados["idade"],
        "curso_enc": curso_enc,
        "financeiro_enc": financeiro_enc,
    }])

    probabilidade = modelo.predict_proba(X)[0][1]
    predicao = modelo.predict(X)[0]

    return {
        "evadiu": bool(predicao),
        "probabilidade_evasao": round(float(probabilidade) * 100, 1),
    }


def importancia_features():
    modelo, _ = carregar_modelo()
    features = ["periodo", "cr", "faltas", "reprovacoes", "idade", "curso", "sit. financeira"]
    importancias = modelo.feature_importances_
    return pd.DataFrame({"feature": features, "importancia": importancias}).sort_values(
        "importancia", ascending=False
    )


if __name__ == "__main__":
    print("Treinando modelo...")
    _, _, relatorio, _ = treinar_modelo()
    print(relatorio)
