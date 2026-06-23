import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import (
    shapiro,
    levene,
    ttest_ind,
    mannwhitneyu
)

df = pd.read_csv("data/raw/pnad_2015.csv")

# Selecionar apenas as colunas necessárias
df = df[["Income", "Sex"]]

# Remover valores ausentes
df = df.dropna()

print("=" * 50)
print("HIPÓTESE 3 - DIFERENÇA SALARIAL ENTRE SEXOS")
print("=" * 50)


homens = df[df["Sex"] == "Male"]["Income"]
mulheres = df[df["Sex"] == "Female"]["Income"]


print("\nCONTAGEM POR SEXO")
print(df["Sex"].value_counts())

print("\nESTATÍSTICAS DESCRITIVAS DA RENDA POR SEXO")

estatisticas = (
    df.groupby("Sex")["Income"]
    .agg([
        "count",
        "mean",
        "median",
        "std",
        "var",
        "min",
        "max"
    ])
)

print(estatisticas)

plt.figure(figsize=(10, 6))

plt.hist(
    homens,
    bins=30,
    alpha=0.5,
    label="Homens"
)

plt.hist(
    mulheres,
    bins=30,
    alpha=0.5,
    label="Mulheres"
)

plt.title("Distribuição da renda por sexo")
plt.xlabel("Renda")
plt.ylabel("Frequência")
plt.legend()

plt.savefig(
    "outputs/graphs/histograma_renda_sexo.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

plt.figure(figsize=(8, 6))

plt.boxplot(
    [homens, mulheres],
    tick_labels=["Homens", "Mulheres"]
)

plt.title("Distribuição da renda por sexo")
plt.ylabel("Renda")

plt.savefig(
    "outputs/graphs/boxplot_renda_sexo.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\n" + "=" * 50)
print("TESTE DE NORMALIDADE")
print("=" * 50)

homens_amostra = homens.sample(
    min(5000, len(homens)),
    random_state=42
)

mulheres_amostra = mulheres.sample(
    min(5000, len(mulheres)),
    random_state=42
)

shapiro_homens = shapiro(homens_amostra)
shapiro_mulheres = shapiro(mulheres_amostra)

print(
    f"Homens -> Estatística: {shapiro_homens.statistic:.4f} | "
    f"p-valor: {shapiro_homens.pvalue:.6f}"
)

print(
    f"Mulheres -> Estatística: {shapiro_mulheres.statistic:.4f} | "
    f"p-valor: {shapiro_mulheres.pvalue:.6f}"
)

normalidade = (
    shapiro_homens.pvalue > 0.05 and
    shapiro_mulheres.pvalue > 0.05
)

if normalidade:
    print("\nOs grupos apresentam distribuição normal.")
else:
    print("\nOs grupos NÃO apresentam distribuição normal.")

print("\n" + "=" * 50)
print("TESTE DE HOMOGENEIDADE DAS VARIÂNCIAS")
print("=" * 50)

lev_stat, lev_p = levene(
    homens,
    mulheres
)

print(f"Estatística de Levene: {lev_stat:.4f}")
print(f"P-valor: {lev_p:.6f}")

variancias_iguais = lev_p > 0.05

if variancias_iguais:
    print("\nAs variâncias podem ser consideradas iguais.")
else:
    print("\nAs variâncias NÃO podem ser consideradas iguais.")


print("\n" + "=" * 50)
print("TESTE DE COMPARAÇÃO ENTRE HOMENS E MULHERES")
print("=" * 50)

if normalidade and variancias_iguais:

    print("\nTeste escolhido: t de Student para amostras independentes")

    estatistica, p_valor = ttest_ind(
        homens,
        mulheres,
        equal_var=True
    )

    print(f"Estatística t: {estatistica:.4f}")
    print(f"P-valor: {p_valor:.10f}")

else:

    print("\nTeste escolhido: Mann-Whitney U")

    estatistica, p_valor = mannwhitneyu(
        homens,
        mulheres,
        alternative="two-sided"
    )

    print(f"Estatística U: {estatistica:.4f}")
    print(f"P-valor: {p_valor:.10f}")

# INTERPRETAÇÃO

print("\n" + "=" * 50)
print("CONCLUSÃO")
print("=" * 50)

media_homens = homens.mean()
media_mulheres = mulheres.mean()

print(f"\nRenda média dos homens: R$ {media_homens:.2f}")
print(f"Renda média das mulheres: R$ {media_mulheres:.2f}")

if p_valor < 0.05:

    print(
        "\nResultado: REJEITA-SE H0 "
        "(nível de significância de 5%)."
    )

    print(
        "Existe diferença estatisticamente significativa "
        "entre as rendas de homens e mulheres."
    )

else:

    print(
        "\nResultado: NÃO SE REJEITA H0 "
        "(nível de significância de 5%)."
    )

    print(
        "Não foram encontradas evidências suficientes "
        "para afirmar que existe diferença de renda "
        "entre homens e mulheres."
    )

print("\nArquivos gerados:")
print("- outputs/graphs/histograma_renda_sexo.png")
print("- outputs/graphs/boxplot_renda_sexo.png")
