import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency

# CARREGAMENTO DOS DADOS

df = pd.read_csv("data/raw/pnad_2015.csv")

df = df[["Income", "Years_of_study"]]

df = df.dropna()

print("=" * 60)
print("HIPÓTESE 4 - ASSOCIAÇÃO ENTRE ESCOLARIDADE E RENDA")
print("=" * 60)

condicoes = [
    df["Years_of_study"] <= 8,
    (df["Years_of_study"] >= 9) & (df["Years_of_study"] <= 11),
    df["Years_of_study"] == 12,
    df["Years_of_study"] >= 13
]

grupos = [
    "Fundamental incompleto",
    "Fundamental completo",
    "Ensino médio",
    "Ensino superior"
]

df["Education_Level"] = np.select(
    condicoes,
    grupos,
    default="Não classificado"
)


df["Income_Level"] = pd.qcut(
    df["Income"],
    q=3,
    labels=[
        "Baixa renda",
        "Média renda",
        "Alta renda"
    ]
)

print("\nFREQUÊNCIA DAS CATEGORIAS EDUCACIONAIS\n")
print(df["Education_Level"].value_counts())

print("\nFREQUÊNCIA DAS FAIXAS DE RENDA\n")
print(df["Income_Level"].value_counts())

tabela = pd.crosstab(
    df["Education_Level"],
    df["Income_Level"]
)

print("\nTABELA DE CONTINGÊNCIA\n")
print(tabela)

chi2, p_valor, gl, esperados = chi2_contingency(tabela)

print("\n" + "=" * 60)
print("TESTE QUI-QUADRADO")
print("=" * 60)

print(f"Qui-quadrado: {chi2:.4f}")
print(f"Graus de liberdade: {gl}")
print(f"P-valor: {p_valor:.10f}")

n = tabela.values.sum()

min_dim = min(
    tabela.shape[0] - 1,
    tabela.shape[1] - 1
)

cramer_v = np.sqrt(
    chi2 / (n * min_dim)
)

print("\nV DE CRAMÉR")
print(f"{cramer_v:.4f}")

if cramer_v < 0.10:
    intensidade = "Muito fraca"
elif cramer_v < 0.30:
    intensidade = "Fraca"
elif cramer_v < 0.50:
    intensidade = "Moderada"
else:
    intensidade = "Forte"

print(f"Associação: {intensidade}")

print("\n" + "=" * 60)
print("CONCLUSÃO")
print("=" * 60)

if p_valor < 0.05:

    print(
        "\nResultado: REJEITA-SE H0 "
        "(nível de significância de 5%)."
    )

    print(
        "Existe associação estatisticamente "
        "significativa entre escolaridade "
        "e faixa de renda."
    )

else:

    print(
        "\nResultado: NÃO SE REJEITA H0 "
        "(nível de significância de 5%)."
    )

    print(
        "Não foram encontradas evidências "
        "de associação entre escolaridade "
        "e faixa de renda."
    )

tabela.plot(
    kind="bar",
    stacked=True,
    figsize=(10, 6)
)

plt.title(
    "Escolaridade x Faixa de Renda"
)

plt.xlabel("Escolaridade")
plt.ylabel("Quantidade")

plt.legend(
    title="Faixa de renda"
)

plt.tight_layout()

plt.savefig(
    "outputs/graphs/barras_empilhadas_escolaridade_renda.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

plt.figure(figsize=(8, 6))

plt.imshow(
    tabela,
    aspect="auto"
)

plt.colorbar(
    label="Frequência"
)

plt.xticks(
    range(len(tabela.columns)),
    tabela.columns,
    rotation=45
)

plt.yticks(
    range(len(tabela.index)),
    tabela.index
)

plt.title(
    "Heatmap - Escolaridade x Faixa de Renda"
)

plt.tight_layout()

plt.savefig(
    "outputs/graphs/heatmap_escolaridade_renda.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nArquivos gerados:")
print(
    "- outputs/graphs/barras_empilhadas_escolaridade_renda.png"
)
print(
    "- outputs/graphs/heatmap_escolaridade_renda.png"
)
 