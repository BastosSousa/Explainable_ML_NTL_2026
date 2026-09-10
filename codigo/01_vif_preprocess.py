# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 18:13:58 2026

@author: Natalia
"""
import warnings
import os
import platform
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

# %%
diretorio_atual = os.getcwd()
nome_sistema_operacional = platform.system()

if nome_sistema_operacional == 'Linux':
    barra = "/"
    volta_nivel = "../"
else:
    barra = "\\"
    volta_nivel = "..\\"

print(nome_sistema_operacional)
print(volta_nivel)
print(barra)

# %%
out_dir = diretorio_atual + barra + volta_nivel + barra + 'saidas'
os.makedirs(out_dir, exist_ok=True)


caminho_dados_gdf_df_pivot = diretorio_atual + barra + volta_nivel + barra + 'entradas' + barra + 'gdf_df_pivot.csv'

# carregar seu dataset já preparado
df = pd.read_csv(caminho_dados_gdf_df_pivot)

# remover colunas irrelevantes
drop_cols = ["geometry", "GEN", "latitude", "longitude"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna(axis=1, how='all')
df = df.fillna(df.median())

# remover correlação alta
corr = df.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [c for c in upper.columns if any(upper[c] > 0.95)]
df = df.drop(columns=to_drop)

# VIF iterativo
def calc_vif(df, thresh=5):
    variables = df.columns.tolist()
    while True:
        vif = pd.Series(
            [variance_inflation_factor(df.values, i) for i in range(df.shape[1])],
            index=df.columns
        )
        max_vif = vif.max()
        if max_vif < thresh:
            break
        drop_var = vif.idxmax()
        df = df.drop(columns=[drop_var])
        variables.remove(drop_var)
    return df

df_vif = calc_vif(df)

df_vif.to_csv(f"{out_dir}/dados_vif.csv", index=False)

print(" VIF finished")