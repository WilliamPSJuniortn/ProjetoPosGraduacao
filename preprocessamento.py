"""
preprocessamento.py
Sprint 2 — Separação X/y, padronização numérica (StandardScaler)
e codificação one-hot para variáveis categóricas (OneHotEncoder).
Retorna conjuntos de treino e teste prontos para modelagem.
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH      = os.path.join(BASE_DIR, "data", "alunos_limpo.csv")
PREP_PATH       = os.path.join(BASE_DIR, "models", "preprocessador.pkl")

# ---------------------------------------------------------------------------
# Definição das colunas
# ---------------------------------------------------------------------------
COLUNAS_DESCARTAR   = ["id_aluno", "nome"]
COLUNA_ALVO         = "evadiu"
COLUNAS_NUMERICAS   = ["periodo", "cr", "faltas", "reprovacoes", "idade"]
COLUNAS_CATEGORICAS = ["curso", "situacao_financeira"]


def carregar_dados(caminho=INPUT_PATH):
    df = pd.read_csv(caminho)
    return df


def separar_features(df):
    """Separa variáveis preditoras (X) da variável alvo (y)."""
    X = df.drop(columns=COLUNAS_DESCARTAR + [COLUNA_ALVO])
    y = df[COLUNA_ALVO]
    return X, y


def construir_preprocessador():
    """Cria ColumnTransformer com StandardScaler + OneHotEncoder."""
    transformador_num = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    transformador_cat = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessador = ColumnTransformer(transformers=[
        ("num", transformador_num, COLUNAS_NUMERICAS),
        ("cat", transformador_cat, COLUNAS_CATEGORICAS),
    ])

    return preprocessador


def executar_preprocessamento(caminho=INPUT_PATH):
    """Pipeline completo: carrega, separa, divide e ajusta o pré-processador."""
    df = carregar_dados(caminho)

    print("[SPRINT 2] === Pré-processamento ===")
    print(f"[SPRINT 2] Registros carregados: {len(df)}")

    # Separação X / y
    X, y = separar_features(df)

    print(f"\n[SPRINT 2] Variável alvo         : {COLUNA_ALVO}")
    print(f"[SPRINT 2] Variáveis numéricas   : {COLUNAS_NUMERICAS}")
    print(f"[SPRINT 2] Variáveis categóricas : {COLUNAS_CATEGORICAS}")
    print(f"[SPRINT 2] Distribuição da classe alvo:")
    contagem = y.value_counts()
    for classe, qtd in contagem.items():
        label = "Evadiu" if classe == 1 else "Não Evadiu"
        print(f"           {label} ({classe}) -> {qtd} ({qtd/len(y)*100:.1f}%)")

    # Divisão treino / teste estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n[SPRINT 2] Divisão treino/teste  : 80% / 20%")
    print(f"           Treino: {len(X_train)} amostras | Teste: {len(X_test)} amostras")

    # Ajustar pré-processador apenas no treino
    preprocessador = construir_preprocessador()
    X_train_prep = preprocessador.fit_transform(X_train)
    X_test_prep  = preprocessador.transform(X_test)

    # Nomes das features após transformação
    nomes_num = COLUNAS_NUMERICAS
    nomes_cat = list(
        preprocessador.named_transformers_["cat"]
        .named_steps["onehot"]
        .get_feature_names_out(COLUNAS_CATEGORICAS)
    )
    nomes_features = nomes_num + nomes_cat

    print(f"\n[SPRINT 2] Features após pré-processamento: {len(nomes_features)}")
    print(f"           Numéricas (padronizadas): {len(nomes_num)}")
    print(f"           Categóricas (one-hot)   : {len(nomes_cat)}")

    # Salvar pré-processador
    os.makedirs(os.path.dirname(PREP_PATH), exist_ok=True)
    joblib.dump(preprocessador, PREP_PATH)
    print(f"\n[SPRINT 2] Pré-processador salvo em: {PREP_PATH}")

    return X_train_prep, X_test_prep, y_train, y_test, preprocessador, nomes_features


# ---------------------------------------------------------------------------
# Execução direta
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    executar_preprocessamento()
    print("\n[SPRINT 2] Pré-processamento concluído.")
