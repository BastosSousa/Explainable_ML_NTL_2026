# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 14:32:23 2026

@author: Natalia
"""

import geopandas as gpd
import pandas as pd
import folium
import os
import warnings
import platform

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
caminho_dados_DSO = diretorio_atual + barra + volta_nivel + barra + 'entradas' + barra + 'DSO-Sachsen-Anhalt-file.xlsx'
caminho_dados_DSO_L = diretorio_atual + barra + volta_nivel + barra + 'entradas' + barra + 'DSO-LowerSaxony.xlsx'
caminho_shp = (diretorio_atual + barra + volta_nivel + barra +'vg-hist.utm32s.shape' + barra + 'daten' + barra + 'utm32s' + barra + 'shape' 
               + barra + 'VG-Hist_1990-10-03_KRS.shp')

# carregar shapefile real (polígonos)
dados_shp = gpd.read_file(caminho_shp)

dados_shp = dados_shp.to_crs(epsg=4326)

# carregar resultado interpolado
gdf_interp = gpd.read_file(f"{out_dir}/resultado_interpolado.geojson")

# carregar DSO (para filtrar cidades válidas)
#%%%%
dso_sa = pd.read_excel(caminho_dados_DSO)
dso_ls = pd.read_excel(caminho_dados_DSO_L)

dso_sa["state"] = "Sachsen-Anhalt"
dso_ls["state"] = "Lower Saxony"

dso = pd.concat([dso_sa, dso_ls])

# cidades válidas
cidades_validas = dso["Ort"].dropna().unique()

#%%%
# filtrar shapefile
col_nome = "GEN"  # ajuste se necessário
gdf_poligonos = dados_shp[dados_shp[col_nome].isin(cidades_validas)].copy()

# agregar interpolação por cidade
df_regiao = gdf_interp.groupby("GEN").mean(numeric_only=True).reset_index()

# merge
gdf_final = gdf_poligonos.merge(dso, left_on=col_nome, right_on="GEN", how="left")

#%%%
# mapa
mapa = folium.Map(location=[52.0, 10.5], zoom_start=6)

# função de estilo
def estilo(feature):
    estado = feature["properties"].get("state")
    cor = "blue" if estado == "Lower Saxony" else "green"
    return {
        "fillColor": cor,
        "color": "black",
        "weight": 1,
        "fillOpacity": 0.6
    }

# popup em blocos
def popup_html(props):
    html = f"<h4>{props.get('GEN')}</h4>"
    for k,v in props.items():
        if isinstance(v,(int,float)):
            html += f"<b>{k}:</b> {round(v,2)}<br>"
    return html

folium.GeoJson(
    gdf_final,
    style_function=estilo,
    tooltip=folium.GeoJsonTooltip(fields=["GEN"]),
    popup=folium.GeoJsonPopup(fields=list(df_regiao.columns))
).add_to(mapa)

mapa.save(f"{out_dir}/mapa_final.html")

print("HTML MAP it's done")