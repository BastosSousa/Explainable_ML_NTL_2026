# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 14:45:43 2025

@author: Natalia
"""

import folium
import matplotlib.backends.backend_pdf
from scipy.spatial import distance_matrix
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import geopandas as gpd

import matplotlib.pyplot as plt

from datetime import datetime

import numpy as np


import os
import platform

from scipy.interpolate import griddata


import warnings
# %%
import pysal.lib
from pysal.lib import weights
import pysal.model
from esda.moran import Moran, Moran_Local
import esda.moran
import splot.esda
from splot.esda import moran_scatterplot
from shapely.geometry import Point
# %%
import pyproj
from geopandas import GeoDataFrame

# %%
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

# %%

caminho_anhalt = diretorio_atual+barra+volta_nivel+barra + \
    'DVG_ALKIS'+barra+'DVG_ALKIS.shp'


caminho_shp = diretorio_atual+barra+volta_nivel+barra + \
    'vg-hist.utm32s.shape'+barra+'daten'+barra+'utm32s' + \
    barra+'shape'+barra+'VG-Hist_1990-10-03_KRS.shp'


# %%

crs = {'init': 'epsg:4326'}

dados_anhalt = gpd.read_file(caminho_anhalt)
type(dados_anhalt)
dados_anhalt.crs


dados_shp = gpd.read_file(caminho_shp)
type(dados_shp)
dados_shp.crs


# %%

fig = plt.figure(figsize=(10, 5))  # vulner_dia
ax = fig.add_subplot()
dados_anhalt.boundary.plot(ax=ax, color='red', label='Regions sachsen anhalt')
dados_shp.plot(ax=ax)
plt.legend()
plt.show()


# %%
caminho_dados = diretorio_atual+barra + \
    volta_nivel+barra+'inkar_2024'+barra+'inkar_2024.csv'
    
# %%
caminho_dados_2 = diretorio_atual+barra + \
    volta_nivel+barra+'Zensus2022_Energietraeger_10km-Gitter.csv'

#%%

dados = pd.read_csv(caminho_dados, sep=';', decimal=",")

#%%
dados_Zensus2022 = pd.read_csv(caminho_dados_2, sep=',')

#%%
dados_Zensus2022['geometry'] = [Point(xy) for xy in zip(dados_Zensus2022.x_mp_10km, dados_Zensus2022.y_mp_10km)] 

#%%
gdf = GeoDataFrame(dados_Zensus2022, crs="EPSG:3035", geometry=dados_Zensus2022.geometry)

#%%
gdf = gdf.to_crs(epsg=25832)

# %%

dados_2 = dados[dados['Zeitbezug'] == 2020]

dados_head = dados_2.head(200)
# %%

dados_3 = dados_2[['Kuerzel', 'Name', 'Wert']].reset_index(drop=True)

# %%
contagem_valores = dados_3['Kuerzel'].value_counts()

# %%
dados_head = dados_3.head(200)

# %%
df_pivot = dados_3.pivot_table(
    index="Name", columns="Kuerzel", values="Wert", aggfunc="first").reset_index()

# %%
# df_grouped = dados_3.groupby(["Name", "Kuerzel"])["Wert"].apply(list).unstack()

# %%
dados_head = df_pivot.head(100)

#%%

df_pivot = df_pivot.rename(columns={'Name': 'GEN'})

#%%

df_merged = df_pivot.merge(dados_shp, on='GEN', how='left').drop_duplicates()

#%%
df_limpo = df_merged.dropna(subset=["geometry"])

#%%
dados_head = df_limpo.head(100)

#%% 
df_limpo = df_limpo.set_geometry('geometry')

#%%
fig = plt.figure(figsize=(10, 10))  
col = "a_hybrid"

ax = fig.add_subplot()
df_limpo.plot(ax=ax,
    column=col,     # coluna usada no choropleth
    cmap="OrRd",            # colormap (ou 'viridis', 'Blues', etc.)
    legend=True,            # mostra legenda
    figsize=(8, 6),         # tamanho do gráfico
    edgecolor="black",      # contorno dos polígonos
    linewidth=0.5
)
dados_anhalt.boundary.plot(ax=ax, color='red', label='Regions sachsen anhalt')
dados_shp.boundary.plot(ax=ax, color='blue')
gdf.plot(ax=ax, color='purple',marker='o', markersize=9)
plt.title(f"Choropleth map of '{col}' value", fontsize=14)
plt.legend()
plt.show()
#%%







