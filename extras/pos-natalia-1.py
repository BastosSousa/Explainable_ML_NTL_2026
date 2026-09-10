# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 15:36:32 2024

@author: Natalia Bostos
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import geopandas as gpd
#import rasterio
#from rasterio.plot import show
import matplotlib.pyplot as plt
import seaborn as sns
#import pyproj
#from shapely.geometry import box
#import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
#from sklearn import metrics
#from scipy.spatial.distance import cdist
import os
import platform
#from scipy.interpolate import interp2d
from scipy.interpolate import griddata
#from sklearn.cluster import KMeans
#import shapely.geometry
from shapely.geometry import Point

#%%
diretorio_atual=os.getcwd()
#barra="\\"
nome_sistema_operacional = platform.system()

if nome_sistema_operacional == 'Linux':
  barra = "/"
  volta_nivel = "../"
else:
  barra = "\\"
  volta_nivel = "..\\"
print (nome_sistema_operacional)
print (volta_nivel)
print (barra)

#%%

caminho_dados = diretorio_atual+barra+volta_nivel+barra+'entradas'+barra+'Fiscalizacoes.xlsx'
caminho_dados_eletricos = diretorio_atual+barra+volta_nivel+barra+'entradas'+barra+'nova_tabela.xlsx'
caminho_perdas = diretorio_atual+barra+volta_nivel+barra+'entradas'+barra+'entrada_MI.xlsx'

caminho_variaveis = diretorio_atual+barra+volta_nivel+barra+'entradas'+barra+'Mappe1.xlsx'

caminho_shape_FLO = diretorio_atual+barra+volta_nivel+barra+'1717075318251_exportacao'+barra+'cad_edificacao.shp'
caminho_shape_FLO_2 = diretorio_atual+barra+volta_nivel+barra+'1717075318251_exportacao'+barra+'gvw_territoriais_visualizacao.shp'
caminho_shape_FLO_3 = diretorio_atual+barra+volta_nivel+barra+'1717075318251_exportacao'+barra+'patrimonio_territorial.shp'

caminho_shape_FLO_IBGE = diretorio_atual+barra+volta_nivel+barra+'1717075279063_exportacao'+barra+'censo2022_cnefe.shp'

#%%

dados = pd.read_excel(caminho_dados)
dados_eletricos = pd.read_excel(caminho_dados_eletricos)
dados_perdas = pd.read_excel(caminho_perdas)
dados_variaveis = pd.read_excel(caminho_variaveis)

#%%

crs = {'init': 'epsg:4326'}

dados_shape_FLO = gpd.read_file(caminho_shape_FLO)
type(dados_shape_FLO)
dados_shape_FLO.crs

dados_shape_FLO_2 = gpd.read_file(caminho_shape_FLO_2)
type(dados_shape_FLO_2)
dados_shape_FLO_2.crs

dado_shape_FLO_3 = gpd.read_file(caminho_shape_FLO_3)
type(dado_shape_FLO_3)
dado_shape_FLO_3.crs

#%%
dados_shape_FLO['geometry'] = dados_shape_FLO.geometry.to_crs(4326)
dados_shape_FLO_2['geometry'] = dados_shape_FLO_2.geometry.to_crs(4326)
dado_shape_FLO_3['geometry'] = dado_shape_FLO_3.geometry.to_crs(4326)

#%%
fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot()
dados_shape_FLO.boundary.plot(ax=ax,color = 'black',label = 'cad_edificacao')
dados_shape_FLO_2.boundary.plot(ax=ax,color = 'yellow',label = 'gvw_territoriais_visualizacao')
#dado_shape_FLO_3.boundary.plot(ax=ax,color = 'yellow',label = 'patrimonio_territorial')

plt.show()



#%%
dados = dados.drop(columns=['ref', 'ref2'])

#%%
dados = dados[dados['PARECER2'] != 0]
dados = dados[dados['PARECER2'] != 3]

#%%
dados_2 = pd.merge(dados,dados_variaveis , how='left', on='UC').fillna(0)

#%%
# dados_2.groupby('Tipo')['Consumo'].plot(linewidth=0.5,legend='True')
# plt.show()

#%%

#%%

def normalize(dataset):
    dataNorm=((dataset-dataset.min())/(dataset.max()-dataset.min()))*20
    dataNorm["UC"]=dataset["UC"]
    dataNorm["LATITUDE"]=dataset["LATITUDE"]
    dataNorm["LONGITUDE"]=dataset["LONGITUDE"]
    dataNorm["UNI_TR_MT"]=dataset["UNI_TR_MT"]
    dataNorm["Cod_setor"]=dataset["Cod_setor"]
    dataNorm["Porc"]=dataset["Porc"] 
    dataNorm["y_coords"]=dataset["y_coords"]
    dataNorm["x_coords"]=dataset["x_coords"]
    dataNorm["PARECER2_TOTAL"]=dataset["PARECER2_TOTAL"]
    return dataNorm

#%%
data=normalize(dados_variaveis)

#%%
data["Porc"]=data["Porc"]/100

#%%
# box_plot = sns.boxplot(x = dados['Hora'],y = dados['Consumo'], hue = dados['Tipo'],
#             palette = 'husl')

#%%
# sns.set(style='whitegrid')
# facecolor = '#eaeaf2'
# fig, ax = plt.subplots(figsize=(10, 6), facecolor=facecolor)
# ax = sns.boxplot(x = dados['Hora'],y = dados['Consumo'], hue = dados['Tipo'],
#             palette = 'husl')

# font_color = '#525252'
# csfont = {'fontname':'Georgia'}
# hfont = {'fontname':'Calibri'}

# ax.set_ylabel('Consumo', fontsize=16, color=font_color, **hfont)
# for label in (ax.get_xticklabels() + ax.get_yticklabels()):
#     label.set(fontsize=16, color=font_color, **hfont)
    
# title = 'Average Consumption'
# fig.suptitle(title, y=.97, fontsize=22, color=font_color, **csfont)
# plt.subplots_adjust(top=0.85)
        
# for i, box in enumerate(ax.artists):
#     col = box.get_facecolor()
#     plt.setp(ax.lines[i*6+5], mfc=col, mec=col)

# lines = ax.get_lines()
# categories = ax.get_xticks()

# # for cat in categories:
# #     y = round(lines[4+cat*6].get_ydata()[0],1) 
# #     ax.text(
# #         cat, 
# #         y, 
# #         f'{y}', 
# #         ha='center', 
# #         va='center', 
# #         fontweight='semibold', 
# #         size=10,
# #         color='white',
# #         bbox=dict(facecolor='#828282', edgecolor='#828282')
# #     )    
  
# plt.show()    
# filename = 'sns-boxplot'
# plt.savefig(filename+'.png', facecolor=facecolor)
    