import pandas as pd
import numpy as np
# Analise de variância (ANOVA)##
from scipy.stats import f_oneway

# Carregar dados
df = pd.read_csv("data/processed/pnad_processed.correl.estud.e.renda.csv")

# Criar grupos educacionais
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

df["Education_Level"] = np.select(condicoes, grupos, default="Não classificado")

# Conferir quantidade por grupo
print(df["Education_Level"].value_counts())


fund_inc = df[df["Education_Level"]=="Fundamental incompleto"]["Income"]
fund_comp = df[df["Education_Level"]=="Fundamental completo"]["Income"]
medio = df[df["Education_Level"]=="Ensino médio"]["Income"]
superior = df[df["Education_Level"]=="Ensino superior"]["Income"]

f_stat, p_valor = f_oneway(
    fund_inc,
    fund_comp,
    medio,
    superior
)

print(f"Estatística F: {f_stat:.4f}")
print(f"P-valor: {p_valor:.10f}")