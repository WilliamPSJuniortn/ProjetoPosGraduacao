"""
gerar_dataset.py
Gera o dataset de alunos com 1500 registros para a Sprint 1.
Utiliza seed=42 para reprodutibilidade.
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)

N = 1500
N_EVADIDOS = int(N * 0.30)   # 450 evadidos  → taxa de evasão = 30%
N_NAO_EVADIDOS = N - N_EVADIDOS  # 1050

# ---------------------------------------------------------------------------
# Listas de nomes brasileiros
# ---------------------------------------------------------------------------
PRIMEIROS_NOMES = [
    "Ana", "Bruno", "Carla", "Diego", "Elena", "Fábio", "Gabriela", "Henrique",
    "Isabella", "João", "Karla", "Lucas", "Marina", "Nathan", "Olivia", "Paulo",
    "Renata", "Samuel", "Tatiana", "Victor", "Wanessa", "Yasmin", "Alexandre",
    "Beatriz", "Carlos", "Daniela", "Eduardo", "Fernanda", "Gustavo", "Helena",
    "Igor", "Juliana", "Kevin", "Larissa", "Marcelo", "Natalia", "Patricia",
    "Rafael", "Sabrina", "Thiago", "Ursula", "Vanessa", "Wagner", "Ximena",
    "Adriana", "Bernardo", "Camila", "Danilo", "Eloisa", "Felipe",
    "Giovanna", "Hugo", "Isabela", "Joana", "Leonardo", "Mariana", "Nicolas",
    "Otávio", "Priscila", "Roberto", "Simone", "Tiago", "Valentina",
    "Wesley", "Aline", "Bruna", "Caio", "Débora", "Elias", "Flávia",
    "Guilherme", "Hana", "Iara", "José", "Luana", "Matheus", "Nara",
    "Pedro", "Queila", "Rodrigo", "Sônia", "Talita", "Ulisses", "Vera",
]

SOBRENOMES = [
    "Lima", "Souza", "Mendes", "Ferreira", "Costa", "Rocha", "Nunes", "Alves",
    "Martins", "Oliveira", "Santos", "Pereira", "Gomes", "Castro", "Barbosa",
    "Dias", "Torres", "Freitas", "Lopes", "Cunha", "Borges", "Silva", "Azevedo",
    "Matos", "Vieira", "Leal", "Pinto", "Cruz", "Nascimento", "Teixeira",
    "Andrade", "Fonseca", "Ribeiro", "Campos", "Cardoso", "Araújo", "Guimarães",
    "Monteiro", "Carvalho", "Pires", "Almeida", "Rodrigues", "Moreira", "Ramos",
    "Cavalcanti", "Moura", "Correia", "Xavier", "Gonçalves", "Rezende",
    "Medeiros", "Neto", "Machado", "Amaral", "Batista", "Duarte", "Coelho",
    "Vargas", "Lemos", "Queiroz", "Magalhães", "Braga", "Siqueira", "Nogueira",
    "Dantas", "Leite", "Viana", "Barros", "Melo", "Assis", "Brito",
]

CURSOS = [
    "Ciência da Computação", "Engenharia Civil",
    "Administração", "Direito", "Medicina",
]

SITUACOES = ["estavel", "intermediaria", "dificuldade"]


# ---------------------------------------------------------------------------
# Funções auxiliares de geração
# ---------------------------------------------------------------------------

def gerar_nomes(n):
    """Gera nomes brasileiros compostos (primeiro + sobrenome)."""
    primeiros = np.random.choice(PRIMEIROS_NOMES, n)
    ultimos   = np.random.choice(SOBRENOMES, n)
    return [f"{p} {s}" for p, s in zip(primeiros, ultimos)]


def gerar_idades(n):
    """Distribuição beta escalada para [17, 45], concentrada em 18-28."""
    raw = np.random.beta(2, 6, n)          # média ≈ 0.25, moda ≈ 0.17
    return np.clip((raw * 28 + 17).astype(int), 17, 45)


def gerar_evadidos():
    """Gera features para os alunos que evadiram."""
    n = N_EVADIDOS

    cr          = np.clip(np.random.normal(4.2, 1.0, n), 2.0, 7.5).round(1)
    faltas      = np.clip(np.random.gamma(2.5, 8.8, n).astype(int), 0, 40)
    reprovacoes = np.random.choice(
        [0, 1, 2, 3, 4, 5, 6], n,
        p=[0.15, 0.20, 0.25, 0.20, 0.12, 0.06, 0.02],
    )
    # Distribuição financeira calibrada para P(evadiu|dificuldade)≈60%
    # P(evadiu|intermediaria)≈30%, P(evadiu|estavel)≈10%
    situacao = np.random.choice(
        SITUACOES, n,
        p=[0.10, 0.50, 0.40],   # estavel, intermediaria, dificuldade
    )

    return cr, faltas, reprovacoes, situacao


def gerar_nao_evadidos():
    """Gera features para os alunos que não evadiram."""
    n = N_NAO_EVADIDOS

    cr          = np.clip(np.random.normal(6.5, 1.0, n), 3.5, 10.0).round(1)
    faltas      = np.clip(np.random.gamma(1.5, 3.3, n).astype(int), 0, 40)
    reprovacoes = np.random.choice(
        [0, 1, 2, 3, 4, 5, 6], n,
        p=[0.55, 0.28, 0.12, 0.04, 0.01, 0.00, 0.00],
    )
    situacao = np.random.choice(
        SITUACOES, n,
        p=[0.386, 0.50, 0.114],   # estavel, intermediaria, dificuldade
    )

    return cr, faltas, reprovacoes, situacao


# ---------------------------------------------------------------------------
# Geração principal
# ---------------------------------------------------------------------------

# Features por grupo
cr_ev,    fal_ev,    rep_ev,    sit_ev    = gerar_evadidos()
cr_nev,   fal_nev,   rep_nev,   sit_nev   = gerar_nao_evadidos()

# Concatenar grupos
evadiu_arr      = np.array([1] * N_EVADIDOS + [0] * N_NAO_EVADIDOS)
cr_arr          = np.concatenate([cr_ev,  cr_nev])
faltas_arr      = np.concatenate([fal_ev, fal_nev])
reprov_arr      = np.concatenate([rep_ev, rep_nev])
situacao_arr    = np.concatenate([sit_ev, sit_nev])

# Campos independentes de evasão
periodo_arr = np.random.randint(1, 9, N)         # uniforme 1–8
curso_arr   = np.random.choice(CURSOS, N)
idade_arr   = gerar_idades(N)

# Embaralhar para que evadidos não fiquem agrupados no início
indices = np.random.permutation(N)
evadiu_arr   = evadiu_arr[indices]
cr_arr       = cr_arr[indices]
faltas_arr   = faltas_arr[indices]
reprov_arr   = reprov_arr[indices]
situacao_arr = situacao_arr[indices]
periodo_arr  = periodo_arr[indices]
curso_arr    = curso_arr[indices]
idade_arr    = idade_arr[indices]

# Nomes (gerados em ordem aleatória diretamente)
nomes = gerar_nomes(N)

# Montar DataFrame
df = pd.DataFrame({
    "id_aluno":            range(1, N + 1),
    "nome":                nomes,
    "curso":               curso_arr,
    "periodo":             periodo_arr.astype(float),   # float para suportar NaN
    "cr":                  cr_arr,
    "faltas":              faltas_arr.astype(float),
    "situacao_financeira": situacao_arr,
    "reprovacoes":         reprov_arr.astype(float),
    "idade":               idade_arr.astype(float),
    "evadiu":              evadiu_arr,
})

# ---------------------------------------------------------------------------
# Introduzir ~3% de valores ausentes nas colunas numéricas
# ---------------------------------------------------------------------------
COLUNAS_NUMERICAS = ["periodo", "cr", "faltas", "reprovacoes", "idade"]
TAXA_AUSENTES = 0.03

for col in COLUNAS_NUMERICAS:
    n_ausentes = int(N * TAXA_AUSENTES)
    indices_ausentes = np.random.choice(N, size=n_ausentes, replace=False)
    df.loc[indices_ausentes, col] = np.nan

# ---------------------------------------------------------------------------
# Salvar
# ---------------------------------------------------------------------------
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "data", "alunos.csv")
df.to_csv(OUTPUT_PATH, index=False)

taxa_evasao = df["evadiu"].mean() * 100
print(f"[SPRINT 1] Dataset gerado: {len(df)} registros | Taxa de evasão: {taxa_evasao:.1f}%")
