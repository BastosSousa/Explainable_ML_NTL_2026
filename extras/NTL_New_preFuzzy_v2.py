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
import skfuzzy as fuzz
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
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

scaler = MinMaxScaler()
arr_scaled = scaler.fit_transform(df.iloc[:, 0:92]) 

#%%
df_scaled = pd.DataFrame(arr_scaled, columns=df.iloc[:, 0:92].columns,index=df.index)

df_scaled['h3_cell'] = df['h3_cell'].values
df_scaled['geometry'] = df['geometry'].values
df_scaled['clusters'] = df['clusters'].values

#%%  PCA scores

for column in df_scaled.columns:
    if column not in ['h3_cell','geometry','clusters']:
        #print(f'{column}_padded')
        wide = df_scaled.groupby("h3_cell")[f'{column}'].apply(list).reset_index()
        max_len = wide[f'{column}'].apply(len).max()
        wide[f'{column}_padded'] = wide[f'{column}'].apply(lambda x: x + [np.nan]*(max_len - len(x)))
        X = np.vstack(wide[f'{column}_padded'].to_numpy())
        X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)

        pca = PCA(n_components=1)
       
        scores = pca.fit_transform(X)
        wide2[f'{column}_pca_score'] = scores

#%%

wide3 = wide2[['h3_cell']]

#%%  Rescaling PCA scores

for column in wide2.columns:
    if column not in ['h3_cell','geometry','clusters']:
        #print(f'{column}')
        scores = wide2[[f'{column}']].values 
        scaler = MinMaxScaler(feature_range=(0, 1))
        wide3[f'{column}_index'] = scaler.fit_transform(scores)


#%%
df_scaled_2 = df_scaled[['h3_cell','geometry','clusters']]


wide4 = df_scaled_2.merge(wide3, on='h3_cell', how='left').drop_duplicates()

#%%
x = np.linspace(0, 1, 100)
low = fuzz.trimf(x, [0, 0, 0.5])       # triangle: peak at 0
medium = fuzz.trimf(x, [0.25, 0.5, 0.75])  # peak at 0.5
high = fuzz.trimf(x, [0.5, 1, 1])      # peak at 1

#%%

def fuzzy_classify(score):
    low_m = fuzz.interp_membership(x, low, score)
    med_m = fuzz.interp_membership(x, medium, score)
    high_m = fuzz.interp_membership(x, high, score)
    memberships = {"Low": low_m, "Medium": med_m, "High": high_m}
    return max(memberships, key=memberships.get)

#%%

wide5 = wide4[['h3_cell']]

#%%
for column in wide4.columns:
    if column not in ['h3_cell','geometry','clusters']:
        wide5[f"{column}_risk"] = wide4[f'{column}'].apply(fuzzy_classify)



#%%

# # Plot the fuzzy sets
# plt.figure(figsize=(8, 5))
# plt.plot(x, low, 'b', linewidth=2, label='Low Risk')
# plt.plot(x, medium, 'g', linewidth=2, label='Medium Risk')
# plt.plot(x, high, 'r', linewidth=2, label='High Risk')

# plt.title('Fuzzy Membership Functions for Risk Classification')
# plt.xlabel('Score (scaled 0–1)')
# plt.ylabel('Membership Degree')
# plt.legend()
# plt.grid(True)
# plt.show()