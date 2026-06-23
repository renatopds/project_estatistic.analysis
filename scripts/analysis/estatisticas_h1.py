import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


df = pd.read_csv(
    "data/processed/pnad_processed.correl.estud.e.renda.csv"
)

print(df.head())

#estatísticas descritivas#
print(df.describe())

plt.scatter(df["Years_of_study"], df["Income"])
plt.xlabel("Anos de estudo")
plt.ylabel("Renda")
plt.title("Relação entre anos de estudo e renda")
plt.show()

##adicionando gráfico no diretório de gráficos##
plt.scatter(df["Years_of_study"], df["Income"])
plt.xlabel("Anos de estudo")
plt.ylabel("Renda")
plt.title("Relação entre anos de estudo e renda")

plt.savefig(
    "outputs/graphs/persao_estudo_renda.png",
    dpi=300,
    bbox_inches="tight"
)
##histograma de renda##
plt.figure(figsize=(8,5))
plt.hist(df["Income"], bins=30)

plt.xlabel("Renda")
plt.ylabel("Frequência")
plt.title("Distribuição da renda")

plt.savefig(
    "outputs/graphs/histograma_renda.png",
    dpi=300,
    bbox_inches="tight"
)
##boxplot para visualizar a distribuição da renda##
plt.figure(figsize=(8,5))
plt.boxplot(df["Income"])

plt.title("Boxplot da renda")

plt.savefig(
    "outputs/graphs/boxplot_renda.png",
    dpi=300,
    bbox_inches="tight"
)
##histograma de anos de estudo##
plt.figure(figsize=(10, 6))

plt.hist(df["Years_of_study"], bins=16)

plt.title("Distribuição dos Anos de Estudo")
plt.xlabel("Anos de Estudo")
plt.ylabel("Frequência")

plt.savefig(
    "outputs/graphs/histograma_anos_estudo.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()
##boxplot para visualizar a distribuição dos anos de estudo##
plt.figure(figsize=(8, 6))

plt.boxplot(df["Years_of_study"])

plt.title("Boxplot dos Anos de Estudo")
plt.ylabel("Anos de Estudo")

plt.savefig(
    "outputs/graphs/boxplot_anos_estudo.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()
##grafico de dispersão (anos de estudo vs renda)##
plt.figure(figsize=(10, 6))

plt.scatter(
    df["Years_of_study"],
    df["Income"],
    alpha=0.3
)

plt.title("Relação entre Anos de Estudo e Renda")
plt.xlabel("Anos de Estudo")
plt.ylabel("Renda")

plt.savefig(
    "outputs/graphs/dispersao_estudo_renda.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()
##Parametro##
alpha=0.3

r, p = pearsonr(df["Years_of_study"], df["Income"])

print(f"Correlação (r): {r:.4f}")
print(f"P-valor: {p:.6f}")