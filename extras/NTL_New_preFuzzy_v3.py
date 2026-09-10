# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 11:03:42 2025

@author: Natalia
"""
import h3
import os
import platform
import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
import numpy as np
import skfuzzy as fuzz
from sklearn.preprocessing import MinMaxScaler
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

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
df_scaled_2 = df_scaled[['h3_cell','clusters']]


wide4 = df_scaled_2.merge(wide3, on='h3_cell', how='left').drop_duplicates()

#%%

def cell_to_shapely(cell):
    coords = h3.cell_to_boundary(cell)
    flipped = tuple(coord[::-1] for coord in coords)
    return Polygon(flipped)

#%%
h3_geoms = wide4['h3_cell'].apply(lambda x: cell_to_shapely(x))
wide5 = gpd.GeoDataFrame(data=wide4, geometry=h3_geoms, crs=4326)

#%%

fig, ax = plt.subplots()
wide5.plot(ax=ax, color='white', edgecolor='black')
plt.show()

