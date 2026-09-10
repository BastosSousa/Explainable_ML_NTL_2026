# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 13:56:03 2026

@author: Natalia
"""


import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt


from geopandas import sjoin
import os
import warnings
import platform
from shapely.geometry import Point, box


from shapely.ops import unary_union



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

#%%
# ===============================
# PATHS
# ===============================
out_dir = diretorio_atual + barra + volta_nivel + barra + 'saidas'
os.makedirs(out_dir, exist_ok=True)

# ===============================
# 1. CARREGAR DADOS
# ===============================
# dados DSO
caminho_dados = diretorio_atual + barra + volta_nivel + barra + 'entradas' + barra + 'inkar_2024' + barra + 'inkar_2024.csv'
# caminho_dados_DSO = diretorio_atual + barra + volta_nivel + barra + 'entradas' + barra + 'DSO-Sachsen-Anhalt-file.xlsx'
# caminho_dados_DSO_L = diretorio_atual + barra + volta_nivel + barra + 'entradas' + barra + 'DSO-LowerSaxony.xlsx'
caminho_dados_2 = diretorio_atual + barra + volta_nivel + barra + 'entradas' + barra + 'Zensus2022_Energietraeger_10km-Gitter.csv'
# dso_sa = pd.read_excel(caminho_dados_DSO)
# dso_ls = pd.read_excel(caminho_dados_DSO_L)

# dso_sa["state"] = "Sachsen-Anhalt"
# dso_ls["state"] = "Lower Saxony"

# dados_combinados = pd.concat([dso_sa, dso_ls], ignore_index=True)

#%%
caminho_shp = (diretorio_atual + barra + volta_nivel + barra +'vg-hist.utm32s.shape' + barra + 'daten' + barra + 'utm32s' + barra + 'shape' 
               + barra + 'VG-Hist_1990-10-03_LAN.shp')

# carregar shapefile real (polígonos)
dados_shp = gpd.read_file(caminho_shp)

dados_shp = dados_shp.to_crs(epsg=4326)
#%%

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# fig = plt.figure(figsize=(10, 10))  

# ax = fig.add_subplot()
# #dados_anhalt.boundary.plot(ax=ax, color='red', label='Regions sachsen anhalt')
# dados_shp.boundary.plot(ax=ax, color='gray')
# plt.legend()
# plt.show()


def add_north_arrow(ax, x=0.92, y=0.92, size=0.08):
    ax.annotate(
        'N',
        xy=(x, y),
        xytext=(x, y - size),
        xycoords=ax.transAxes,
        arrowprops=dict(
            facecolor='black',
            width=2,
            headwidth=8
        ),
        ha='center',
        va='center',
        fontsize=12,
        fontweight='bold'
    )


# deslocamentos (em graus)
offsets = {
    "04": (0.0, 0.15),  # (dx, dy)
    "02": (0.0, 0.15)
}

plt.rcParams.update({

    "font.serif": ["Times New Roman"],
    "font.size": 15
})

fig, ax = plt.subplots(figsize=(10, 10))

dados_shp.plot(ax=ax, color="lightgray", edgecolor="red")

# destacar AGS específicos
ags_destacados = ["03", "04", "15"]

dados_shp["AGS"] = dados_shp["AGS"].astype(str).str.zfill(2)

dados_highlight = dados_shp[
    dados_shp["AGS"].isin(ags_destacados)
]

dados_highlight.plot(
    ax=ax,
    color="orange",
    edgecolor="black",
    linewidth=2
)

dados_shp["centroid"] = dados_shp.geometry.representative_point()

for _, row in dados_shp.iterrows():
    x = row["centroid"].x
    y = row["centroid"].y
    ags = str(row["AGS"]).zfill(2)

    dx, dy = offsets.get(ags, (0, 0))

    # texto
    ax.text(
        x + dx,
        y + dy,
        ags,
        fontsize=16,
        ha="center"
    )

    # linha (apenas se deslocado)
    if (dx, dy) != (0, 0):
        ax.plot([x, x + dx], [y, y + dy], color="black", linewidth=2.5)

# labels
ax.set_xlabel("Longitude", fontsize=21)
ax.set_ylabel("Latitude", fontsize=21)

# diminuir apenas os números dos eixos
ax.tick_params(axis="both", labelsize=16)

# ticks (menos poluição)
ax.xaxis.set_major_locator(mticker.MaxNLocator(4))
ax.yaxis.set_major_locator(mticker.MaxNLocator(4))

# formato em graus
# ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}°E"))
# ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.2f}°N"))


# grid cartográfico
ax.grid(True, linestyle="--", linewidth=2.0, alpha=0.6)

add_north_arrow(ax)

plt.tight_layout()
plt.show()



#%%
# # plot do shapefile
# fig, ax = plt.subplots(figsize=(10, 10))
# #dados_shp.plot(ax=ax, edgecolor="black", facecolor="none")
# dados_shp.plot(ax=ax, color="lightgray", edgecolor="red")


# # calcular centróides
# dados_shp["centroid"] = dados_shp.geometry.representative_point()

# # adicionar labels (AGS)
# for idx, row in dados_shp.iterrows():
#     x = row["centroid"].x
#     y = row["centroid"].y
    
#     ax.text(
#         x, y,
#         str(row["AGS"]),
#         fontsize=12,
#         ha="center",
#         va="center"
#     )

# # grid
# ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

# # labels dos eixos
# ax.set_xlabel("Longitude")
# ax.set_ylabel("Latitude")

# # controlar número de linhas
# ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
# ax.yaxis.set_major_locator(mticker.MaxNLocator(5))


# # ax.set_title("AGS Codes per Polygon")
# # ax.axis("off")
# plt.tight_layout()
# plt.show()