# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 14:54:05 2026

@author: Natalia
"""

# -*- coding: utf-8 -*-
"""
Relevância espacial de features:
Score = R² (Netzverluste) + Moran I

Output:
- saidas/top10_features_relevancia.csv
"""
import time
import requests

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from pysal.lib import weights
from esda.moran import Moran
from geopandas import sjoin
import os
import warnings
import platform
from shapely.geometry import Point, box

from scipy.spatial import distance_matrix

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import libpysal
from esda.moran import Moran

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

# dados interpolados (script 02)
gdf_interp = gpd.read_file(f"{out_dir}/resultado_interpolado.geojson")

# dados DSO
caminho_dados = diretorio_atual + barra + volta_nivel + barra + 'entradas' + barra + 'inkar_2024' + barra + 'inkar_2024.csv'
caminho_dados_DSO = diretorio_atual + barra + volta_nivel + barra + 'entradas' + barra + 'DSO-Sachsen-Anhalt-file.xlsx'
caminho_dados_DSO_L = diretorio_atual + barra + volta_nivel + barra + 'entradas' + barra + 'DSO-LowerSaxony.xlsx'
caminho_dados_2 = diretorio_atual + barra + volta_nivel + barra + 'entradas' + barra + 'Zensus2022_Energietraeger_10km-Gitter.csv'
dso_sa = pd.read_excel(caminho_dados_DSO)
dso_ls = pd.read_excel(caminho_dados_DSO_L)

dso_sa["state"] = "Sachsen-Anhalt"
dso_ls["state"] = "Lower Saxony"

dados_combinados = pd.concat([dso_sa, dso_ls], ignore_index=True)

#%%
caminho_shp = (diretorio_atual + barra + volta_nivel + barra +'vg-hist.utm32s.shape' + barra + 'daten' + barra + 'utm32s' + barra + 'shape' 
               + barra + 'VG-Hist_1990-10-03_KRS.shp')

# carregar shapefile real (polígonos)
dados_shp = gpd.read_file(caminho_shp)

dados_shp = dados_shp.to_crs(epsg=4326)

#%%
# ===============================
# 2. IDENTIFICAR TARGET
# ===============================
target_cols = [c for c in dados_combinados.columns 
               if any(k in c.lower() for k in ['verlust','loss'])]

if len(target_cols) == 0:
    raise ValueError("none column Netzverluste")

target_col = target_cols[0]
print("Target usado:", target_col)

#%%
# ===============================
# 3. GEOREFERENCIAR DSO
# ===============================

# %%
def geocode_city(city, country="Germany"):
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": f"{city}, {country}", "format": "json", "limit": 1}
    headers = {"User-Agent": "seu_app_nome (seu_email@exemplo.com)"}

    response = requests.get(url, params=params, headers=headers)
    if response.status_code == 200:
        data = response.json()
        if len(data) > 0:
            return float(data[0]["lat"]), float(data[0]["lon"])
    return None, None

# %%
# Use este bloco apenas se ainda não tiver latitude/longitude
lats = []
long = []
for city in dados_combinados["Ort"]:
    lat, lon = geocode_city(city)
    lats.append(lat)
    long.append(lon)
    time.sleep(1)
dados_combinados["latitude"] = lats
dados_combinados["longitude"] = long

#%%
gdf_dso = gpd.GeoDataFrame(
    dados_combinados,
    geometry=gpd.points_from_xy(dados_combinados.longitude, dados_combinados.latitude),
    crs="EPSG:4326"
)

#%%
gdf_dso_joined = sjoin(
    gdf_dso,
    dados_shp,
    how="left",
    predicate="within"
)

#%%
col_regiao = "GEN" 
gdf_dso_joined = gdf_dso_joined.merge(
    dados_shp[[col_regiao, "geometry"]],
    on=col_regiao,
    how="left"
)

#%%
gdf_dso_joined = gdf_dso_joined.drop(columns=["geometry_y"])

#%%

gdf_dso_joined = gdf_dso_joined.rename(columns={"geometry_x": "geometry"})

#%%
gdf_dso_joined = gpd.GeoDataFrame(gdf_dso_joined, geometry="geometry", crs=dados_shp.crs)

#%%%

dados_Zensus2022 = pd.read_csv(caminho_dados_2, sep=',')
# %%

dados_Zensus2022['geometry'] = [
    Point(xy) for xy in zip(dados_Zensus2022.x_mp_10km, dados_Zensus2022.y_mp_10km)
]
#%%
gdf_Zensus2022 =  gpd.GeoDataFrame(dados_Zensus2022, crs="EPSG:3035", geometry=dados_Zensus2022.geometry)

#%%
gdf_Zensus2022 = gdf_Zensus2022.to_crs(epsg=4326)

#%%
dados = pd.read_csv(caminho_dados, sep=';', decimal=",")

# %%
dados_2 = dados[dados['Zeitbezug'] == 2020]
dados_3 = dados_2[['Kuerzel', 'Name', 'Wert']].reset_index(drop=True)

df_pivot = dados_3.pivot_table(
    index="Name", columns="Kuerzel", values="Wert", aggfunc="first"
).reset_index()

df_pivot = df_pivot.rename(columns={'Name': 'GEN'})

df_pivot.to_csv(out_dir + barra + 'df_pivot.csv', sep=';', index=False, encoding='utf-8-sig')

#%%

df_pivot = df_pivot.rename(columns={'GEN': 'Name'})

#%%
# lats = []
# long = []
# for city in df_pivot["Name"]:
#     lat, lon = geocode_city(city)
#     lats.append(lat)
#     long.append(lon)
#     time.sleep(1)
# df_pivot["latitude"] = lats
# df_pivot["longitude"] = long

# #%%

# df_pivot.to_csv(out_dir + barra + 'df_pivot_2.csv', sep=';', index=False, encoding='utf-8-sig')

#%%

# df_pivot_2 = pd.read_csv(out_dir + barra + 'df_pivot_2.csv', sep=';', encoding='utf-8')

# #%%
# gdf_df_pivot_2 = gpd.GeoDataFrame(
#     df_pivot_2,
#     geometry=gpd.points_from_xy(df_pivot_2.longitude, df_pivot_2.latitude),
#     crs="EPSG:4326"
# )

# #%%
# gdf_df_pivot_2_joined = sjoin(
#     gdf_df_pivot_2,
#     dados_shp,
#     how="left",
#     predicate="within"
# )

# #%%
# col_regiao = "GEN" 
# gdf_df_pivot_2_joined = gdf_df_pivot_2_joined.merge(
#     dados_shp[[col_regiao, "geometry"]],
#     on=col_regiao,
#     how="left"
# )

# #%%
# gdf_df_pivot_2_joined = gdf_df_pivot_2_joined.drop(columns=["geometry_x"])

# #%%

# gdf_df_pivot_2_joined = gdf_df_pivot_2_joined.rename(columns={"geometry_y": "geometry"})

# #%%
# gdf_df_pivot_2_joined = gpd.GeoDataFrame(gdf_df_pivot_2_joined, geometry="geometry", crs=dados_shp.crs)

# #%%
# # associar região
# gdf_final = gdf_df_pivot_2_joined.merge(
#     gdf_dso_joined,
#     on="GEN",
#     how="right"
# )

# #%%
# gdf_final = gdf_final.drop(columns=["geometry_x"])

# #%%

# gdf_final = gdf_final.rename(columns={"geometry_y": "geometry"})

# #%%
# gdf_final = gpd.GeoDataFrame(gdf_final, geometry="geometry", crs=dados_shp.crs)

# #%%

# gdf_final_2 = sjoin(
#     gdf_Zensus2022,
#     gdf_final,
#     how="right",
#     predicate="within"
# )

# #%%
# gdf_final_3 = gdf_final_2.dropna(axis=1, how="all")

# #%%
# gdf_final_3.to_csv(out_dir + barra + 'gdf_final_3.csv', sep=';', index=False, encoding='utf-8-sig')

#%%%%%%%%%%%%%%%%%%

# gdf_final_4 = pd.read_csv(out_dir + barra + 'gdf_final_3.csv', sep=';', encoding='utf-8')

# #%%
# gdf_final_4_2 = gdf_final_4.copy()

# #%%
# gdf_final_4_2 = gdf_final_4_2[['Ort', 'Straße', 'Hausnummer', 'Land', 'Registrierungsdatum', 'Datum der letzten Aktualisierung', 
# 'ACER-Code', 'Geschlossenes Verteilernetz', 'Tätigkeitsstatus', 'Tätigkeitsbeginn', 'Tätigkeitsende', 
# 'Summe der Netzverluste [kWh]', 'Preis [Ct/kWh]', 'Anzahl der Entnahmestellen jeweils für alle Netz', 
# 'Einwohnerzahl', 'state','latitude_y','longitude_y']]

# #%%
# gdf_final_4 = gdf_final_4.drop(columns=['latitude_y', 'longitude_y', 'index_right_y', 'OBJECTID_y', 'STG_y', 'ADE_y', 
#                                         'AGS_y', 'TGS_y', 'IBZ_y', 'BEZ_y', 'NBD_y', 'TYP_y', 'NLS_y', 'SDV_y', 'SDV_AGS_y', 
#                                         'SDV_TGS_y', 'EWZ_y', 'KFL_y', 'AGS_ST_y', 'SHAPE_Leng_y', 'SHAPE_Area_y','MaStR-Nr.', 
#                                         'Name des Marktakteurs', 'Marktfunktion', 'Marktrollen', 'Bundesland', 'Postleitzahl', 
#                                         'Ort', 'Straße', 'Hausnummer', 'Land', 'Registrierungsdatum', 'Datum der letzten Aktualisierung', 
#                                         'ACER-Code', 'Geschlossenes Verteilernetz', 'Tätigkeitsstatus', 'Tätigkeitsbeginn', 'Tätigkeitsende', 
#                                         'Summe der Netzverluste [kWh]', 'Preis [Ct/kWh]', 'Anzahl der Entnahmestellen jeweils für alle Netz', 
#                                         'Einwohnerzahl', 'state', 'index_right_x','index_left'])

# #%%

# gdf_final_4 = gdf_final_4.rename(columns={'latitude_x': 'latitude','longitude_x':'longitude','OBJECTID_x':'OBJECTID', 
#                                           'STG_x':'STG', 'ADE_x':'ADE', 'AGS_x':'AGS', 'TGS_x':'TGS', 'IBZ_x':'IBZ', 'BEZ_x':'BEZ', 
#                                           'NBD_x':'NBD', 'TYP_x':'TYP', 'NLS_x':'NLS', 'SDV_x':'SDV', 'SDV_AGS_x':'SDV_AGS', 'SDV_TGS_x':'SDV_TGS', 
#                                           'EWZ_x':'EWZ', 'KFL_x':'KFL', 'AGS_ST_x':'AGS_ST', 'SHAPE_Leng_x':'SHAPE_Leng', 'SHAPE_Area_x':'SHAPE_Area'})

# #%%
# gdf_final_4["geometry"] = gdf_final_4["geometry"].apply(wkt.loads)

# #%%
# gdf_final_4 = gpd.GeoDataFrame(gdf_final_4, geometry="geometry", crs=dados_shp.crs)

# #%%

# # res = list(df_numeric.columns)
# # print(res)

# # #%%
# # df_numeric = gdf_final_4.select_dtypes(include=[np.number]).copy()

# # #%%

# # df_numeric = df_numeric.drop(columns=['x_mp_10km', 'y_mp_10km','latitude', 'longitude', 'OBJECTID', 'ADE', 'AGS', 'IBZ', 'SDV_AGS', 
# #                                       'EWZ', 'KFL', 'AGS_ST', 'SHAPE_Leng', 'SHAPE_Area'])

# #%%

# gdf_final_4["x"] = gdf_final_4.latitude
# gdf_final_4["y"] = gdf_final_4.longitude

# #%%
# # df_numeric = df_numeric.select_dtypes(include=[np.number]).copy()

# # #%%
# # from scipy.spatial import cKDTree

# # def idw_per_column(gdf, df_numeric, x_col="x", y_col="y", k=8, power=2):
# #     coords = gdf[[x_col, y_col]].values.copy()
# #     result = df_numeric.copy()

# #     for col in df_numeric.columns:
# #         print(f"Interpolando: {col}")

# #         values = df_numeric[col].values

# #         # máscara
# #         mask_valid = ~np.isnan(values)
# #         mask_nan = np.isnan(values)

# #         # se não há NaN → pula
# #         if mask_nan.sum() == 0:
# #             continue

# #         # se poucos pontos válidos → pula
# #         if mask_valid.sum() < 5:
# #             print(f" poucos dados válidos, pulando")
# #             continue

# #         sample_points = coords[mask_valid]
# #         unknown_points = coords[mask_nan]
# #         sample_values = values[mask_valid]

# #         # KDTree
# #         tree = cKDTree(sample_points)

# #         dist, idx = tree.query(unknown_points, k=min(k, len(sample_points)))

# #         dist[dist == 0] = 1e-10

# #         weights = 1 / dist**power
# #         weights = weights / weights.sum(axis=1, keepdims=True)

# #         interpolated = np.sum(sample_values[idx] * weights, axis=1)

# #         # preencher
# #         result.loc[mask_nan, col] = interpolated

# #     return result

# # #%%
# # df_interp = idw_per_column(
# #     gdf_final_4,
# #     df_numeric,
# #     k=8,
# #     power=2
# # )

# # #%%
# # gdf_final_5_interp = gdf_final_4.copy()

# # #%%
# # gdf_final_5_interp[df_interp.columns] = df_interp

# # #%%
# # gdf_final_5_interp.to_csv(out_dir + barra + 'gdf_final_5_interp.csv', sep=';', index=False, encoding='utf-8-sig')

# #%%
# gdf_final_5_interp = pd.read_csv(out_dir + barra + 'gdf_final_5_interp.csv', sep=';', encoding='utf-8')

# #%%
# gdf_final_5_interp["geometry"] = gdf_final_5_interp["geometry"].apply(wkt.loads)

# #%%
# gdf_final_5_interp = gpd.GeoDataFrame(gdf_final_5_interp, geometry="geometry", crs=dados_shp.crs)

# #%%
# look = gdf_final_5_interp.head(20)
# #%%
# res = list(look.columns)
# print(res)

# #%%
# coords = np.array(list(zip(gdf_final_4.x, gdf_final_4.y)))

# #%%

# w = libpysal.weights.KNN.from_array(coords, k=8)
# w.transform = "r"

# #%%

# gdf_final_4_3 =  gdf_final_4_2[gdf_final_4_2['state'].isin(['Lower Saxony', 'Sachsen-Anhalt'])]

# gdf_final_4_3 = gdf_final_4_3.drop_duplicates()

# gdf_final_4_3 = gdf_final_4_3.dropna(subset=['Summe der Netzverluste [kWh]'])

# #%%

# gdf_final_4_3 = gpd.GeoDataFrame(gdf_final_4_3, geometry=gpd.points_from_xy(gdf_final_4_3.longitude_y, gdf_final_4_3.latitude_y))

# #%%
# gdf_final_4_3 = gpd.GeoDataFrame(gdf_final_4_3, geometry="geometry", crs=gdf_final_5_interp.crs)
                  
# #%%

# A = gdf_final_4_3.reset_index(drop=True)
# B = gdf_final_5_interp.reset_index(drop=True)

# #%%
# joined = gpd.sjoin(A, B, how="left", predicate="within")

# #%%
# # Keep only one row per point
# result = (
#     joined
#     .groupby(level=0)
#     .first()
#     .reset_index(drop=True)
#     .drop(columns=["index_right"], errors="ignore")
# )

# #%%
# # Drop helper column
# result = result.drop(columns=["index_right"], errors="ignore")

# #%%
# import matplotlib.pyplot as plt

# fig = plt.figure(figsize=(10, 10))  

# ax = fig.add_subplot()
# # df_limpo.plot(ax=ax,
# #     column=col,     # coluna usada no choropleth
# #     cmap="OrRd",            # colormap (ou 'viridis', 'Blues', etc.)
# #     legend=True,            # mostra legenda
# #     figsize=(8, 6),         # tamanho do gráfico
# #     edgecolor="black",      # contorno dos polígonos
# #     linewidth=0.5
# # )
# #dados_anhalt.boundary.plot(ax=ax, color='red', label='Regions sachsen anhalt')
# dados_shp.boundary.plot(ax=ax, color='blue')
# A.plot(ax=ax, color='purple',marker='o', markersize=9)
# plt.title("kjkertyjener", fontsize=14)
# plt.legend()
# plt.show()


# #%%

# target_col = 'Summe der Netzverluste [kWh]'

# if target_col not in gdf_interp.columns:
#     raise ValueError(f"{target_col} não encontrada!")

# #%%
# # ===============================
# # 2. FILTRAR COLUNAS NUMÉRICAS
# # ===============================
# df_numeric = gdf_interp.select_dtypes(include=[np.number]).copy()

# # remover colunas que NÃO devem entrar
# cols_to_drop = [
#     target_col,
#     'latitude', 'longitude','Anzahl der Entnahmestellen jeweils für alle Netz',
#     'latitude_y', 'longitude_y',
#     'x', 'y','x_mp_10km',
#     'y_mp_10km','ADE', 'AGS', 'IBZ', 'SDV_AGS', 'EWZ', 'KFL', 'AGS_ST',
#     'OBJECTID',	'H_SPNV_ChestP','KH_SPNV_Stroke','lg_nst'

# ]

# df_numeric = df_numeric.drop(
#     columns=[c for c in cols_to_drop if c in df_numeric.columns],
#     errors='ignore'
# )

# print(f"Total features numéricas: {len(df_numeric.columns)}")

# #%%

# # ===============================
# # 3. MATRIZ ESPACIAL (KNN)
# # ===============================
# coords = np.array(list(zip(
#     gdf_interp.geometry.x,
#     gdf_interp.geometry.y
# )))

# w = libpysal.weights.KNN.from_array(coords, k=4)
# w.transform = "r"

# #%%

# target_col = 'Summe der Netzverluste [kWh]'

# # converter para string primeiro (segurança)
# gdf_interp[target_col] = gdf_interp[target_col].astype(str)

# # remover separador de milhar (.)
# gdf_interp[target_col] = gdf_interp[target_col].str.replace('.', '', regex=False)

# # trocar vírgula por ponto
# gdf_interp[target_col] = gdf_interp[target_col].str.replace(',', '.', regex=False)

# # converter para float
# gdf_interp[target_col] = pd.to_numeric(gdf_interp[target_col], errors='coerce')

# #%%
# df_numeric = df_numeric.drop(index=4).reset_index(drop=True)

# #%%
# # remover NaNs do target
# y = gdf_interp[target_col].values

# mask_target = ~np.isnan(y)

# #%%
# # 5. LOOP FEATURES
# # ===============================
# results = []

# for col in df_numeric.columns:

#     x = gdf_interp[col].values

#     # máscara válida (sem NaN)
#     mask = ~np.isnan(x) & mask_target

#     if mask.sum() < 20:
#         continue

#     x_valid = x[mask].reshape(-1, 1)
#     y_valid = y[mask]

#     # -----------------------
#     # R²
#     # -----------------------
#     try:
#         model = LinearRegression().fit(x_valid, y_valid)
#         y_pred = model.predict(x_valid)
#         r2 = r2_score(y_valid, y_pred)
#     except:
#         r2 = np.nan

#     # -----------------------
#     # Moran I
#     # -----------------------
#     try:
#         moran = Moran(x, w)
#         moran_I = moran.I
#     except:
#         moran_I = np.nan

#     # -----------------------
#     # Score final
#     # -----------------------
#     score = r2 + moran_I

#     results.append([col, r2, moran_I, score])


# #%%

# # ===============================
# # 6. DATAFRAME RESULTADOS
# # ===============================
# df_results = pd.DataFrame(
#     results,
#     columns=["feature", "r2", "moran_I", "score_final"]
# )

# #%%


