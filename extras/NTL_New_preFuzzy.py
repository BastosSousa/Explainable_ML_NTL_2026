# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 11:03:42 2025

@author: Natalia
"""

import os
import platform
import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
import numpy as np
#%%
diretorio_atual = os.getcwd()
# barra="\\"
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


#%%
caminho_dados_final1 = diretorio_atual+barra + \
    volta_nivel+barra+'saidas'+barra+'data_final1.xlsx'
    

#%%
dados = pd.read_csv(caminho_dados_final1,sep=';')

#%%

df = dados.copy()

#%%

wide = df.groupby("h3_cell")["Perdas"].apply(list).reset_index()

#%%
max_len = wide["Perdas"].apply(len).max()
wide["Perdas_padded"] = wide["Perdas"].apply(lambda x: x + [np.nan]*(max_len - len(x)))
X = np.vstack(wide["Perdas_padded"].to_numpy())

#%%
X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)

#%%

pca = PCA(n_components=1)
scores = pca.fit_transform(X)

#%%
wide["pca_score"] = scores
print(wide[["h3_cell", "pca_score"]])

#%%
wide2 = wide[['h3_cell']]

#%%

for column in df.columns:
    if column not in ['h3_cell','geometry','clusters']:
        #print(f'{column}_padded')
        wide = df.groupby("h3_cell")[f'{column}'].apply(list).reset_index()
        max_len = wide[f'{column}'].apply(len).max()
        wide[f'{column}_padded'] = wide[f'{column}'].apply(lambda x: x + [np.nan]*(max_len - len(x)))
        X = np.vstack(wide[f'{column}_padded'].to_numpy())
        X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
        pca = PCA(n_components=1)
        scores = pca.fit_transform(X)
        wide2[f'{column}_pca_score'] = scores




    








