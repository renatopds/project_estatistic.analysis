#importando a biblioteca pandas para análise de dados#
import pandas as pd

df = pd.read_csv(
    "data/processed/pnad_processed.correl.estud.e.renda.csv"
)

print(df.head())

#estatísticas descritivas#
print(df.describe())

#Grafico de dispersão para visualizar a relação entre anos de estudo e renda#
import matplotlib.pyplot as plt

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