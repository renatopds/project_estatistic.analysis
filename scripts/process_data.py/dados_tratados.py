import pandas as pd

# leitura do dado bruto
df = pd.read_csv("data/raw/pnad_2015.csv")

# seleção das colunas
df = df[["Income", "Years_of_study"]]

# remoção de valores nulos
df = df.dropna()

# salvar dado tratado em 'processed'
df.to_csv("data/processed/pnad_processed.correl.estud.e.renda.csv", index=False)
