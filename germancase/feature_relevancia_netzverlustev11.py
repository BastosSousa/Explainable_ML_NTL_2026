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
from shapely.ops import unary_union

from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW

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

#dados_shp.plot()
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

dados_head = dados.head(1000)

dados_informe = dados['Wert'].value_counts()

# %%
dados_2 = dados[dados['Zeitbezug'] == 2020]

INKAR_Bereich = dados_2[['Bereich','Kuerzel']]

#dados_3 = dados_2[['Kuerzel', 'Name', 'Wert']].reset_index(drop=True)

# df_pivot = dados_3.pivot_table(
#     index="Name", columns="Kuerzel", values="Wert", aggfunc="first"
# ).reset_index()

# df_pivot = df_pivot.rename(columns={'Name': 'GEN'})

# df_pivot.to_csv(out_dir + barra + 'df_pivot.csv', sep=';', index=False, encoding='utf-8-sig')

#%%

# df_pivot = df_pivot.rename(columns={'GEN': 'Name'})

# #%%
# lats = []
# long = []
# for city in df_pivot["Name"]:
#     lat, lon = geocode_city(city)
#     lats.append(lat)
#     long.append(lon)
#     time.sleep(1)
# df_pivot["latitude"] = lats
# df_pivot["longitude"] = long

# #%%hredjzdtr

# df_pivot.to_csv(out_dir + barra + 'df_pivot_2.csv', sep=';', index=False, encoding='utf-8-sig')

#%%

df_pivot_2 = pd.read_csv(out_dir + barra + 'df_pivot_2.csv', sep=';', encoding='utf-8')

df_pivot_2_head = df_pivot_2.head(100)


#%%
# gdf_df_pivot_2 = gpd.GeoDataFrame(
#     df_pivot_2,
#     geometry=gpd.points_from_xy(df_pivot_2.longitude, df_pivot_2.latitude),
#     crs="EPSG:4326"
# )

#%%
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

#%%
# associar região
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

#%%

# gdf_final_2 = sjoin(
#     gdf_Zensus2022,
#     gdf_final,
#     how="right",
#     predicate="within"
# )

#%%
# gdf_final_3 = gdf_final_2.dropna(axis=1, how="all")

# #%%
# gdf_final_3.to_csv(out_dir + barra + 'gdf_final_3.csv', sep=';', index=False, encoding='utf-8-sig')

#%%%%%%%%%%%%%%%%%%

gdf_final_4 = pd.read_csv(out_dir + barra + 'gdf_final_3.csv', sep=';', encoding='utf-8')

#%%
gdf_final_4_2 = gdf_final_4.copy()

#%%
gdf_final_4_2 = gdf_final_4_2[['Ort', 'Straße', 'Hausnummer', 'Land', 'Registrierungsdatum', 'Datum der letzten Aktualisierung', 
'ACER-Code', 'Geschlossenes Verteilernetz', 'Tätigkeitsstatus', 'Tätigkeitsbeginn', 'Tätigkeitsende', 
'Summe der Netzverluste [kWh]', 'Preis [Ct/kWh]', 'Anzahl der Entnahmestellen jeweils für alle Netz', 
'Einwohnerzahl', 'state','latitude_y','longitude_y']]

#%%
gdf_final_4 = gdf_final_4.drop(columns=['latitude_y', 'longitude_y', 'index_right_y', 'OBJECTID_y', 'STG_y', 'ADE_y', 
                                        'AGS_y', 'TGS_y', 'IBZ_y', 'BEZ_y', 'NBD_y', 'TYP_y', 'NLS_y', 'SDV_y', 'SDV_AGS_y', 
                                        'SDV_TGS_y', 'EWZ_y', 'KFL_y', 'AGS_ST_y', 'SHAPE_Leng_y', 'SHAPE_Area_y','MaStR-Nr.', 
                                        'Name des Marktakteurs', 'Marktfunktion', 'Marktrollen', 'Bundesland', 'Postleitzahl', 
                                        'Ort', 'Straße', 'Hausnummer', 'Land', 'Registrierungsdatum', 'Datum der letzten Aktualisierung', 
                                        'ACER-Code', 'Geschlossenes Verteilernetz', 'Tätigkeitsstatus', 'Tätigkeitsbeginn', 'Tätigkeitsende', 
                                        'Summe der Netzverluste [kWh]', 'Preis [Ct/kWh]', 'Anzahl der Entnahmestellen jeweils für alle Netz', 
                                        'Einwohnerzahl', 'state', 'index_right_x'])

#%%

gdf_final_4 = gdf_final_4.rename(columns={'latitude_x': 'latitude','longitude_x':'longitude','OBJECTID_x':'OBJECTID', 
                                          'STG_x':'STG', 'ADE_x':'ADE', 'AGS_x':'AGS', 'TGS_x':'TGS', 'IBZ_x':'IBZ', 'BEZ_x':'BEZ', 
                                          'NBD_x':'NBD', 'TYP_x':'TYP', 'NLS_x':'NLS', 'SDV_x':'SDV', 'SDV_AGS_x':'SDV_AGS', 'SDV_TGS_x':'SDV_TGS', 
                                          'EWZ_x':'EWZ', 'KFL_x':'KFL', 'AGS_ST_x':'AGS_ST', 'SHAPE_Leng_x':'SHAPE_Leng', 'SHAPE_Area_x':'SHAPE_Area'})

#%%
gdf_final_4["geometry"] = gdf_final_4["geometry"].apply(wkt.loads)

#%%
gdf_final_4 = gpd.GeoDataFrame(gdf_final_4, geometry="geometry", crs=dados_shp.crs)

#%%

res = list(gdf_final_4.columns)
print(res)

#%%
df_numeric = gdf_final_4.select_dtypes(include=[np.number]).copy()

#%%

df_numeric = df_numeric.drop(columns=['latitude', 'longitude', 'OBJECTID', 'ADE', 'AGS', 'IBZ', 'SDV_AGS', 
                                      'EWZ', 'KFL', 'AGS_ST', 'SHAPE_Leng', 'SHAPE_Area'])

#%%

gdf_final_4["x"] = gdf_final_4.latitude
gdf_final_4["y"] = gdf_final_4.longitude

#%%
df_numeric = df_numeric.select_dtypes(include=[np.number]).copy()

#%%
from scipy.spatial import cKDTree

#%%
def idw_per_column(gdf, df_numeric, x_col="x", y_col="y", k=8, power=2):
    coords = gdf[[x_col, y_col]].values.copy()
    result = df_numeric.copy()

    for col in df_numeric.columns:
        print(f"Interpolando: {col}")

        values = df_numeric[col].values

        # máscara
        mask_valid = ~np.isnan(values)
        mask_nan = np.isnan(values)

        # se não há NaN → pula
        if mask_nan.sum() == 0:
            continue

        # se poucos pontos válidos → pula
        if mask_valid.sum() < 5:
            print("poucos dados válidos, pulando")
            continue

        sample_points = coords[mask_valid]
        unknown_points = coords[mask_nan]
        sample_values = values[mask_valid]

        # KDTree
        tree = cKDTree(sample_points)

        dist, idx = tree.query(unknown_points, k=min(k, len(sample_points)))

        dist[dist == 0] = 1e-10

        weights = 1 / dist**power
        weights = weights / weights.sum(axis=1, keepdims=True)

        interpolated = np.sum(sample_values[idx] * weights, axis=1)

        # preencher
        result.loc[mask_nan, col] = interpolated

    return result


#%%
# gdf_final_5_interp = gdf_final_4.copy()

# #%%
# gdf_final_5_interp[df_interp.columns] = df_interp

# #%%
# gdf_final_5_interp.to_csv(out_dir + barra + 'gdf_final_5_interp.csv', sep=';', index=False, encoding='utf-8-sig')

#%%
gdf_final_5_interp = pd.read_csv(out_dir + barra + 'gdf_final_5_interp.csv', sep=';', encoding='utf-8')

#%%
gdf_final_5_interp["geometry"] = gdf_final_5_interp["geometry"].apply(wkt.loads)

#%%
gdf_final_5_interp = gpd.GeoDataFrame(gdf_final_5_interp, geometry="geometry", crs=dados_shp.crs)

#%%
look = gdf_final_5_interp.head(20)
#%%
res = list(look.columns)
print(res)

#%%

gdf_final_4_3 =  gdf_final_4_2[gdf_final_4_2['state'].isin(['Lower Saxony', 'Sachsen-Anhalt'])]

gdf_final_4_3 = gdf_final_4_3.drop_duplicates()

gdf_final_4_3 = gdf_final_4_3.dropna(subset=['Summe der Netzverluste [kWh]'])

#%%

gdf_final_4_3 = gpd.GeoDataFrame(gdf_final_4_3, geometry=gpd.points_from_xy(gdf_final_4_3.longitude_y, gdf_final_4_3.latitude_y))

#%%
gdf_final_4_3 = gpd.GeoDataFrame(gdf_final_4_3, geometry="geometry", crs=gdf_final_5_interp.crs)
                  
#%%

A = gdf_final_4_3.reset_index(drop=True)
B = gdf_final_5_interp.reset_index(drop=True)

#%%
joined = gpd.sjoin(A, B, how="left", predicate="within")

#%%
# Keep only one row per point
result = (
    joined
    .groupby(level=0)
    .first()
    .reset_index(drop=True)
    .drop(columns=["index_right"], errors="ignore")
)

#%%
# Drop helper column
result = result.drop(columns=["index_right"], errors="ignore")

#%%
import matplotlib.pyplot as plt

# fig = plt.figure(figsize=(10, 10))  

# ax = fig.add_subplot()
# #dados_anhalt.boundary.plot(ax=ax, color='red', label='Regions sachsen anhalt')
# dados_shp.boundary.plot(ax=ax, color='blue')
# A.plot(ax=ax, color='purple',marker='o', markersize=9)
# plt.title("kjkertyjener", fontsize=14)
# plt.legend()
# plt.show()


#%% ERRO @@@@@@@@@@@@@@@@@@@@@@@@@@@@

target_col = 'Summe der Netzverluste [kWh]'

if target_col not in result.columns:
    raise ValueError(f"{target_col} não encontrada!")

#%%
# ===============================
# 2. FILTRAR COLUNAS NUMÉRICAS
# ===============================
df_numeric_2 = result.select_dtypes(include=[np.number]).copy()

#%%
# remover colunas que NÃO devem entrar
cols_to_drop = [
    target_col,
    'latitude', 'longitude','Anzahl der Entnahmestellen jeweils für alle Netz',
    'latitude_y', 'longitude_y',
    'x', 'y','x_mp_10km',
    'y_mp_10km','ADE', 'AGS', 'IBZ', 'SDV_AGS', 'EWZ', 'KFL', 'AGS_ST',
    'OBJECTID',	'H_SPNV_ChestP','KH_SPNV_Stroke','lg_nst'

]

df_numeric_2 = df_numeric_2.drop(
    columns=[c for c in cols_to_drop if c in df_numeric_2.columns],
    errors='ignore'
)

print(f"Total features numéricas: {len(df_numeric.columns)}")

#%%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

df_scaled_data = df_numeric_2.copy()

scaled_data = scaler.fit_transform(df_scaled_data)

# Convert back to DataFrame
standardized_df = pd.DataFrame(scaled_data, columns=df_scaled_data.columns)

#%%
# ===============================
# 3. MATRIZ ESPACIAL (KNN)
# ===============================
coords = np.array(list(zip(
    result.geometry.x,
    result.geometry.y
)))

w = libpysal.weights.KNN.from_array(coords, k=4)
w.transform = "r"

#%%

target_col = 'Summe der Netzverluste [kWh]'

# converter para string primeiro (segurança)
result[target_col] = result[target_col].astype(str)

# remover separador de milhar (.)
result[target_col] = result[target_col].str.replace('.', '', regex=False)

# trocar vírgula por ponto
result[target_col] = result[target_col].str.replace(',', '.', regex=False)

# converter para float
result[target_col] = pd.to_numeric(result[target_col], errors='coerce')

#%%
# remover NaNs do target
y = result[target_col].values

mask_target = ~np.isnan(y)

#%%
# 5. LOOP FEATURES
# ===============================
results = []

for col in standardized_df.columns:

    x = result[col].values

    # máscara válida (sem NaN)
    mask = ~np.isnan(x) & mask_target

    if mask.sum() < 20:
        continue

    x_valid = x[mask].reshape(-1, 1)
    y_valid = y[mask]
    

    # -----------------------
    # R²
    # -----------------------
    try:
        model = LinearRegression().fit(x_valid, y_valid)
        y_pred = model.predict(x_valid)
        r2 = r2_score(y_valid, y_pred)
    except:
        r2 = np.nan

    # -----------------------
    # Moran I
    # -----------------------
    try:
        moran = Moran(x, w)
        moran_I = moran.I
    except:
        moran_I = np.nan

    # -----------------------
    # Score final
    # -----------------------
    score = r2 + moran_I

    results.append([col, r2, moran_I, score])


#%%

# ===============================
# 6. DATAFRAME RESULTADOS
# ===============================
df_results = pd.DataFrame(
    results,
    columns=["feature", "r2", "moran_I", "score_final"]
)

#%%
df_results["r2_norm"] = (
    (df_results["r2"] - df_results["r2"].min()) /
    (df_results["r2"].max() - df_results["r2"].min())
)

#%%
df_results["moran_norm"] = (
    (df_results["moran_I"] - df_results["moran_I"].min()) /
    (df_results["moran_I"].max() - df_results["moran_I"].min())
)

#%%
df_results["score_final"] = (
    df_results["r2_norm"] + df_results["moran_norm"]
)

#%%
top6 = df_results.sort_values(by="r2_norm", ascending=False).head(6)

print(top6[["feature", "r2_norm"]])

top6_feature =  top6["feature"].tolist()

#%%

# pegar as geometrias
geom_27 = dados_shp.loc[27, "geometry"]
geom_30 = dados_shp.loc[30, "geometry"]

#%%
# unir geometrias
geom_unida = unary_union([geom_27, geom_30])

#%%
# copiar dados do index 30
attrs = dados_shp.loc[30].drop("geometry")

# criar novo GeoDataFrame
novo = gpd.GeoDataFrame([attrs], geometry=[geom_unida], crs=dados_shp.crs)

#%%
# atualizar diretamente a geometria do index 30
dados_shp.at[30, "geometry"] = geom_unida

# (opcional) remover o 27
dados_shp = dados_shp.drop(index=27)

dados_shp =dados_shp.reset_index(drop=True)

#%%
fig, ax = plt.subplots()

dados_shp.plot(ax=ax, color="lightgray", edgecolor="black")

dados_shp.loc[[30]].plot(ax=ax, color="red")

plt.show()

#%%
dadod_shp_filt = dados_shp[['AGS_ST','geometry']]
dadod_shp_filt["AGS_ST"] = dadod_shp_filt["AGS_ST"].astype("int64")


#%%
gdf_result = result.merge(
    dadod_shp_filt,
    on="AGS_ST",
    how="left"
)
#%%

gdf_result = gdf_result.drop(columns=["geometry_x"], errors="ignore")

#%%
gdf_result = gpd.GeoDataFrame(gdf_result, geometry="geometry_y", crs=dados_shp.crs)

#%%
#gdf_result["geometry_y"] = gdf_result["geometry_y"].apply(wkt.loads)

#%%
gdf_final_5_interp = gpd.GeoDataFrame(gdf_final_5_interp, geometry="geometry", crs=dados_shp.crs)

#%%

mask = []

for i, geom_i in enumerate(dados_shp.geometry):
    dentro = False
    for j, geom_j in enumerate(dados_shp.geometry):
        if i != j and geom_i.within(geom_j):
            dentro = True
            break
    mask.append(not dentro)

dados_shp_clean = dados_shp[mask]

#%%
# fig = plt.figure(figsize=(10, 10))  

# ax = fig.add_subplot()
# gdf_result.plot(
#     column="q_rwm_dpf_10_bev",
#     cmap="RdYlBu",
#     linewidth=0.2,
#     edgecolor="black",
#     legend=True,
#     ax=ax
# )
# A.plot(ax=ax, color='purple', marker='o', markersize=9)
# #dados_anhalt.boundary.plot(ax=ax, color='red', label='Regions sachsen anhalt')
# dados_shp.boundary.plot(ax=ax, color='gray')
# A.plot(ax=ax, color='purple',marker='o', markersize=9)

# for idx, row in A.iterrows():
#     x = row.geometry.x
#     y = row.geometry.y
#     label = row["Ort"]
    
#     ax.annotate(
#         text=label,
#         xy=(x, y),
#         xytext=(3, 3),  # deslocamento do texto
#         textcoords="offset points",
#         fontsize=12,
#         color="black"
#     )

# ax.set_title("Heatmap - q_rwm_dpf_10_bev",fontsize=14)
# plt.tight_layout()
# plt.savefig("heatmap_q_rwm.png", dpi=300)
# plt.show()

#%%
for feature in top6_feature:
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()

    gdf_result.plot(
        column=feature,
        cmap="RdYlBu",
        linewidth=0.2,
        edgecolor="black",
        legend=True,
        ax=ax
    )

    dados_shp.boundary.plot(ax=ax, color='gray')

    A.plot(ax=ax, color='purple', marker='o', markersize=9)

    # labels
    for idx, row in A.iterrows():
        x = row.geometry.x
        y = row.geometry.y
        label = row["Ort"]

        ax.annotate(
            text=label,
            xy=(x, y),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=12,
            color="black"
        )

    ax.set_title(f"Heatmap - {feature}", fontsize=14)

    plt.tight_layout()
    plt.savefig(f"heatmap_{feature}.png", dpi=300)
    plt.show()

#%%%

y = result[target_col].values

X = standardized_df.values
coords = np.array(coords)

#%%
from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05
)

model.fit(X, y)

#%%
import shap

#%%
explainer = shap.TreeExplainer(model)

#%%
shap_values = explainer.shap_values(X)

#%%

shap_abs = np.abs(shap_values)

global_importance = shap_abs.mean(axis=0)

#%%

gdf_GW_SHAP = gpd.GeoDataFrame(
    shap_values,
    geometry=gpd.points_from_xy(coords[:,0], coords[:,1])
)

#%%
global_importance = np.abs(shap_values).mean(axis=0)

importance_df = pd.DataFrame({
    "feature": standardized_df.columns,
    "importance": global_importance
}).sort_values("importance", ascending=False)

#%%

from sklearn.cluster import KMeans

clusters = KMeans(4).fit_predict(shap_values)
gdf_GW_SHAP["cluster"] = clusters

#%%
import re

def build_inkar_groups(columns):
    """
    Cria grupos temáticos INKAR com base em padrões nos nomes das variáveis.
    """

    inkar_groups = {
        "arbeitslosigkeit_allgemein": [],
        "arbeitslosigkeit_struktur": [],
        "wirtschaft": [],
        "demografie": [],
        "bildung": [],
        "infrastruktur": [],
        "sonstige": []
    }

    for col in columns:
        col_lower = col.lower()

        if re.search(r"(sus|arbeitslos|alo|alg|sbg)", col_lower):
            inkar_groups["arbeitslosigkeit_allgemein"].append(col)

   
        elif re.search(r"(svb|beschäft|erwerb|arbeitnehmer)", col_lower):
            inkar_groups["arbeitslosigkeit_struktur"].append(col)

   
        elif re.search(r"(bev|einwohner|alter|geburten|sterbe|dichte|rwm)", col_lower):
            inkar_groups["demografie"].append(col)

     
        elif re.search(r"(bip|wirtschaft|umsatz|unternehmen|betriebe|svw)", col_lower):
            inkar_groups["wirtschaft"].append(col)

  
        elif re.search(r"(bildung|schule|abitur|akadem|studium)", col_lower):
            inkar_groups["bildung"].append(col)

 
        elif re.search(r"(verkehr|straße|netz|energie|versorgung|infrastruktur)", col_lower):
            inkar_groups["infrastruktur"].append(col)

        else:
            inkar_groups["sonstige"].append(col)

    # remover grupos vazios
    inkar_groups = {k: v for k, v in inkar_groups.items() if len(v) > 0}

    return inkar_groups
#%%
inkar_groups = build_inkar_groups(standardized_df.columns)

for group, cols in inkar_groups.items():
    print(f"{group}: {len(cols)} variáveis")


#%%
group_importance = {}

for group, cols in inkar_groups.items():
    idx = [standardized_df.columns.get_loc(c) for c in cols]
    group_importance[group] = np.abs(shap_values[:, idx]).mean()


#%% group importances

df_imp = pd.Series(group_importance)
df_norm = df_imp / df_imp.sum()
df_norm.sort_values(ascending=False)

#%%

for group, cols in inkar_groups.items():
    idx = [standardized_df.columns.get_loc(c) for c in cols]
    gdf_GW_SHAP[group] = abs(shap_values[:, idx]).mean(axis=1)
    
#%%
#gdf_GW_SHAP.plot(column="demografie", legend=True)

#%%
# converter para DataFrame
df_imp = pd.Series(group_importance).sort_values(ascending=False)

# normalizar (%)
df_imp_norm = df_imp / df_imp.sum()

# # plot
# plt.figure(figsize=(10,6))
# df_imp_norm.plot(kind="bar")

# plt.ylabel("Relative Importance (%)")
# plt.title("GW-SHAP Importance by INKAR Group")
# plt.xticks(rotation=45, ha="right")
# plt.tight_layout()

# plt.show()
#%%
plt.figure(figsize=(10,6))
bars = plt.bar(df_imp_norm.index, df_imp_norm.values)

plt.ylabel("Relative Importance")
plt.title("Relative Contribution of INKAR Groups")

# adicionar valores no topo
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.2f}", 
             ha='center', va='bottom')

plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

#%%

for group, cols in inkar_groups.items():

    idx = [
        standardized_df.columns.get_loc(c)
        for c in cols
    ]

    gdf_GW_SHAP[group] = (
        np.abs(shap_values[:, idx])
        .mean(axis=1)
    )

gdf_GW_SHAP.plot(column="arbeitslosigkeit_allgemein", legend=True)
plt.title("Spatial Importance – arbeitslosigkeit_allgemein")
plt.show()

#%%
for group in inkar_groups.keys():
    gdf_GW_SHAP[group + "_norm"] = gdf_GW_SHAP[group] / gdf_GW_SHAP[group].max()

#%%
groups_to_plot = ["demografie_norm", "arbeitslosigkeit_allgemein_norm", "wirtschaft_norm"]

fig, axes = plt.subplots(1, len(groups_to_plot), figsize=(15,5))

for i, group in enumerate(groups_to_plot):
    gdf_GW_SHAP.plot(
        column=group,
        cmap="viridis",
        legend=True,
        ax=axes[i]
    )
    axes[i].set_title(group)
    axes[i].axis("off")

plt.tight_layout()
plt.show()

#%%

inkar_translation = {
    "arbeitslosigkeit_allgemein": "Unemployment Rate",
    "arbeitslosigkeit_struktur": "Employment Composition",
    "wirtschaft": "Economic Indicators",
    "demografie": "Demographic Structure",
    "bildung": "Educational Attainment",
    "infrastruktur": "Infrastructure",
    "sonstige": "Other Factors"
}


#%%
# import matplotlib.ticker as mticker

# # =========================
# # 1. preparar dados globais
# # =========================
# df_imp = pd.Series(group_importance).sort_values(ascending=False)
# df_imp_norm = df_imp / df_imp.sum()

# # =========================
# # 2. preparar dados espaciais
# # =========================
# for group, cols in inkar_groups.items():
#     idx = [standardized_df.columns.get_loc(c) for c in cols]
#     gdf_GW_SHAP[group] = np.abs(shap_values[:, idx]).mean(axis=1)

# # escolher grupos principais
# groups_to_plot = df_imp_norm.index[:3].tolist()

# #%%
# # =========================
# # 3. criar figura estilo paper
# # =========================
# fig = plt.figure(figsize=(12, 10))

# # grid layout
# gs = fig.add_gridspec(2, 3, height_ratios=[1, 2])

# # =========================
# # 4. BARPLOT (topo)
# # =========================
# ax_bar = fig.add_subplot(gs[0, :])

# bars = ax_bar.bar(df_imp_norm.index, df_imp_norm.values)

# ax_bar.set_ylabel("Relative Importance")
# #ax_bar.set_title("Global Importance of INKAR Groups (Geographically Weighted -SHAP)")

# # valores nos bars
# for bar in bars:
#     yval = bar.get_height()
#     ax_bar.text(
#         bar.get_x() + bar.get_width()/2,
#         yval,
#         f"{yval:.2f}",
#         ha='center',
#         va='bottom',
#         fontsize=18
#     )

# df_imp_norm.index = [
#     inkar_translation.get(i, i) for i in df_imp_norm.index
# ]
# ax_bar.set_xticklabels(df_imp_norm.index, rotation=45, ha="right")

# # =========================
# # 5. MAPAS (embaixo)
# # =========================
# for i, group in enumerate(groups_to_plot):
#     ax = fig.add_subplot(gs[1, i])

#     gdf_GW_SHAP.plot(
#         column=group + "_norm",
#         cmap="viridis",
#         legend=True,
#         ax=ax,
#         legend_kwds={
#         "shrink": 0.6
#         }
#     )

#     title = inkar_translation.get(group, group).replace(" ", "\n")
#     ax.set_title(title)
#     ax.set_xlabel("Longitude")
#     ax.set_ylabel("Latitude")
#     ax.xaxis.set_major_locator(mticker.MaxNLocator(4))
#     ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
  
#     ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
#     ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
#     #ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

# # =========================
# # 6. estilo final
# # =========================
# plt.tight_layout()
# plt.show()
#%%

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 24,
    "axes.labelsize": 24,
    "axes.titlesize": 24,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "axes.linewidth": 1.0
})

# =========================
# 1. PREPARAR DADOS GLOBAIS
# =========================
df_imp = pd.Series(group_importance).sort_values(ascending=False)
df_imp_norm = df_imp / df_imp.sum()

# =========================
# 2. PREPARAR DADOS ESPACIAIS
# =========================
for group, cols in inkar_groups.items():
    idx = [standardized_df.columns.get_loc(c) for c in cols]
    gdf_GW_SHAP[group] = np.abs(shap_values[:, idx]).mean(axis=1)

# Escolher grupos principais usando os nomes originais
groups_to_plot = df_imp.index[:3].tolist()

# Traduzir rótulos para o Barplot
df_imp_norm_translated = df_imp_norm.copy()
df_imp_norm_translated.index = [
    inkar_translation.get(i, i) for i in df_imp_norm_translated.index
]


# =========================================================
# FIGURA 1: IMPORTÂNCIA GLOBAL (BARPLOT)
# =========================================================
fig1, ax_bar = plt.subplots(figsize=(10, 5))

bars = ax_bar.bar(
    df_imp_norm_translated.index, df_imp_norm_translated.values
)

ax_bar.set_ylabel("Relative Importance")
ax_bar.set_xticks(range(len(df_imp_norm_translated.index)))
ax_bar.set_xticklabels(
    df_imp_norm_translated.index, rotation=45, ha="right", fontsize=21
)

# Adicionar valores no topo das barras
for bar in bars:
    yval = bar.get_height()
    ax_bar.text(
        bar.get_x() + bar.get_width() / 2,
        yval,
        f"{yval:.2f}",
        ha="center",
        va="bottom",
        fontsize=21,
    )

ax_bar.grid(True, linestyle="--", linewidth=0.5, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("global_importance_barplot.png", dpi=300, bbox_inches="tight")
plt.show()


# =========================================================
# FIGURA 2: MAPAS ESPACIAIS (GDF GW-SHAP)
# =========================================================
fig2, axes = plt.subplots(1, len(groups_to_plot), figsize=(15, 5))

# Garantir indexação de array caso haja apenas 1 mapa
if len(groups_to_plot) == 1:
    axes = [axes]

for i, group in enumerate(groups_to_plot):
    ax = axes[i]

    gdf_GW_SHAP.plot(
        column=group + "_norm",
        cmap="viridis",
        legend=True,
        ax=ax,
        legend_kwds={"shrink": 0.6},
    )

    title = inkar_translation.get(group, group).replace(" ", "\n")
    ax.set_title(title, fontsize=21)

    # Rótulos dos eixos X e Y
    ax.set_xlabel("Longitude", fontsize=21)
    ax.set_ylabel("Latitude", fontsize=21)

    # Formatação das marcas dos eixos
    ax.xaxis.set_major_locator(mticker.MaxNLocator(4))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    # Grid ativado nos mapas
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.savefig("gw_shap_spatial_maps.png", dpi=300, bbox_inches="tight")
plt.show()



#%% GW - SHAP

from scipy.spatial.distance import cdist

def gaussian_kernel(distances, bandwidth):
    return np.exp(-(distances**2) / (2 * bandwidth**2))

#%%
coords_array = np.array(coords)

dist_matrix = cdist(coords_array, coords_array)

bandwidth = np.percentile(dist_matrix, 10)  # or tune this

weights = gaussian_kernel(dist_matrix, bandwidth)

# normalize rows
weights = weights / weights.sum(axis=1, keepdims=True)

shap_abs = np.abs(shap_values)

gw_shap = weights @ shap_abs   # matrix multiplication

gdf_GW_SHAP_features = gpd.GeoDataFrame(
    gw_shap,
    geometry=gpd.points_from_xy(coords[:,0], coords[:,1])
)

#%%
gw_group_importance = {}

for group, cols in inkar_groups.items():
    idx = [standardized_df.columns.get_loc(c) for c in cols]
    
    # group SHAP per observation
    group_shap = shap_abs[:, idx].mean(axis=1)
    
    # apply spatial weighting
    gw_group = weights @ group_shap
    
    gdf_GW_SHAP[group + "_gw"] = gw_group
    
    # global importance (mean of local GW values)
    gw_group_importance[group] = gw_group.mean()

#%%
from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8
)

model.fit(X, y)

#%%
y_pred = model.predict(X)

print("R²:", r2_score(y, y_pred))

explainer = shap.Explainer(model, X)
shap_values = explainer(X)

importances = abs(shap_values.values).mean(axis=0)

#%%
imp = np.array(importances) 
imp_sorted = np.sort(imp)[::-1]

#%%
importances = np.abs(shap_values.values).mean(axis=0)

importances_pct = (
    importances /
    importances.sum()
) * 100

#%%
top_k = 10
indices = np.argsort(importances)[::-1][:top_k]

feature_names = standardized_df.columns[indices]

plt.figure(figsize=(8,5))

plt.barh(
    feature_names,
    importances_pct[indices]
)

plt.gca().invert_yaxis()

plt.xlabel("Relative Importance (%)")
plt.ylabel("Feature")

plt.title("Top XGBoost Feature Importances")

plt.tight_layout()
plt.show()

#%%

# # SPATIAL RELEVANCE ASSESSMENT PIPELINE
# # VISUAL RESULTS

# plt.rcParams.update({
#     "font.family": "serif",
#     "font.serif": ["Times New Roman"],
#     "font.size": 24
# })
# # =========================================================
# # 1. SCATTERPLOT
# # Moran's I vs R²
# # =========================================================

# fig, ax = plt.subplots(figsize=(8,10))

# scatter = ax.scatter(
#     df_results["r2_norm"],
#     df_results["moran_norm"],
#     s=60,
#     alpha=0.7
# )

# # labels top features
# top = df_results.nlargest(6, "score_final")

# for _, row in top.iterrows():

#     ax.text(
#         row["r2_norm"] + 0.008,
#         row["moran_norm"] - 0.020,
#         row["feature"],
#         fontsize=24,
        
#     )

# ax.set_xlabel(r"Normalized $R^2$")
# ax.set_ylabel("Normalized Moran's I")

# # ax.set_title(
# #     "Spatial Relevance Assessment",
# #     fontsize=14
# # )

# ax.grid(
#     True,
#     linestyle="--",
#     alpha=0.4
# )

# plt.tight_layout()

# plt.savefig(
#     "scatter_spatial_relevance.png",
#     dpi=300,
#     bbox_inches="tight"
# )

# plt.show()
#%%

# =========================================================
# SPATIAL RELEVANCE ASSESSMENT PIPELINE - VISUAL RESULTS
# =========================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 21,
    "axes.labelsize": 21,
    "xtick.labelsize": 21,
    "ytick.labelsize": 21,
})

fig, ax = plt.subplots(figsize=(8, 4))

# 1. Background Shading (Green upper triangle, Pink lower triangle)
# Upper zone (y > x): Pale Green
ax.fill_between([0, 1], [0, 1], [1, 1], color="#e2f0d9", zorder=1, alpha=0.8)
# Lower zone (y < x): Pale Pink/Red
ax.fill_between([0, 1], [0, 0], [0, 1], color="#fce4d6", zorder=1, alpha=0.8)

# 2. Diagonal Reference Line (y = x)
ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1, alpha=0.7, zorder=2)

# 3. Scatter Plot
ax.scatter(
    df_results["r2_norm"],
    df_results["moran_norm"],
    s=35,
    color="#5b9bd5",
    alpha=1.0,
    edgecolors="none",
    zorder=3
)

# 4. Feature Annotations
top = df_results.nlargest(6, "score_final")

for _, row in top.iterrows():
    ax.text(
        row["r2_norm"] + 0.015,
        row["moran_norm"] - 0.010,
        row["feature"],
        fontsize=10,
        zorder=4
    )

# 5. Axes, Grid, and Limits
ax.set_xlabel(r"Normalized $R^2$")
ax.set_ylabel("Normalized Moran's I")

ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)

ax.grid(True, linestyle=":", linewidth=0.5, color="gray", alpha=0.4, zorder=1)

plt.tight_layout()

# Save Figure
plt.savefig(
    "scatter_spatial_relevance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


#%% ================================
# PERFORMANCE METRICS
# =================================
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# metrics
r2 = r2_score(y, y_pred)

rmse = np.sqrt(mean_squared_error(y, y_pred))

mae = mean_absolute_error(y, y_pred)

# normalized metrics
nrmse_range = rmse / (y.max() - y.min())

nrmse_std = rmse / np.std(y)

relative_mae = mae / np.mean(y)

print(f"R²: {r2:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")
print(f"NRMSE (range): {nrmse_range:.3f}")
print(f"NRMSE (std): {nrmse_std:.3f}")
print(f"Relative MAE: {relative_mae:.3f}")


#%%
# plt.rcParams.update({
#     "font.family": "serif",
#     "font.serif": ["Times New Roman"],
#     "font.size": 24
# })


# #%% ================================
# # OBSERVED VS PREDICTED
# # =================================
# import matplotlib.pyplot as plt

# plt.rcParams.update({
#     "font.family": "serif",
#     "font.serif": ["Times New Roman"],
#     "font.size": 24,
#     "axes.labelsize": 24,
#     "axes.titlesize": 24,
#     "xtick.labelsize": 24,
#     "ytick.labelsize": 24,
#     "axes.linewidth": 1.0
# })

# fig, ax = plt.subplots(figsize=(5, 5))

# # scatter
# ax.scatter(
#     y,
#     y_pred,
#     alpha=1.0,
#     s=35,
#     edgecolors="none"
# )

# # diagonal reference line
# lims = [
#     min(y.min(), y_pred.min()),
#     max(y.max(), y_pred.max())
# ]

# ax.plot(lims, lims, linestyle="--", linewidth=1, color="black", alpha=0.8)

# # labels (com unidade física)
# ax.set_xlabel(r"Observed Energy Loss in kWh")
# ax.set_ylabel(r"Predicted Energy Loss in kWh")

# # grid leve (estilo paper)
# ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)

# # limites iguais (importante para parity plot)
# ax.set_xlim(lims)
# ax.set_ylim(lims)

# # métricas (caixa discreta)
# metrics_text = (
#     f"$R^2$ = {r2:.3f}\n"
#     f"NRMSE = {nrmse_range:.3f}\n"
#     f"MAE rel. = {relative_mae:.3f}"
# )

# ax.text(
#     0.05, 0.95,
#     metrics_text,
#     transform=ax.transAxes,
    
#     verticalalignment="top",
#     bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="gray")
# )

# plt.tight_layout()

# # export em alta qualidade (ESSENCIAL para paper)
# plt.savefig(
#     "observed_vs_predicted_kwh.pdf",
#     format="pdf",
#     bbox_inches="tight"
# )

# plt.savefig(
#     "observed_vs_predicted_kwh.png",
#     dpi=300,
#     bbox_inches="tight"
# )

# plt.show()


#%% ================================
# OBSERVED VS PREDICTED (LOG-LOG)
# =================================

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 24,
    "axes.labelsize": 24,
    "axes.titlesize": 24,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "axes.linewidth": 1.0
})

fig, ax = plt.subplots(figsize=(5, 5))

# Set log-log scale
ax.set_xscale("log")
ax.set_yscale("log")

# Scatter plot
ax.scatter(
    y,
    y_pred,
    alpha=1.0,
    s=45,
    edgecolors="none"
)

# Diagonal reference line (ensure strictly positive limits for log scale)
min_val = min(y[y > 0].min(), y_pred[y_pred > 0].min())
max_val = max(y.max(), y_pred.max())
lims = [min_val, max_val]

ax.plot(lims, lims, linestyle="--", linewidth=1, color="black", alpha=0.8)

# Labels
ax.set_xlabel(r"Observed Energy Loss in kWh")
ax.set_ylabel(r"Predicted Energy Loss in kWh")

# Grid (minor grid included for better log-scale readability)
ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.3)

# Equal limits for parity plot
ax.set_xlim(lims)
ax.set_ylim(lims)

# Metrics text box
metrics_text = (
    f"$R^2$ = {r2:.3f}\n"
    f"NRMSE = {nrmse_range:.3f}\n"
    f"MAE rel. = {relative_mae:.3f}"
)

ax.text(
    0.55, 0.10,
    metrics_text,
    transform=ax.transAxes,
    verticalalignment="bottom",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="gray")
)

plt.tight_layout()

# Export high quality images
plt.savefig(
    "observed_vs_predicted_kwh_loglog.pdf",
    format="pdf",
    bbox_inches="tight"
)

plt.savefig(
    "observed_vs_predicted_kwh_loglog.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


#%% ================================
# # RESIDUAL DISTRIBUTION
# # =================================
# residuals = y - y_pred

# # normalized residuals
# residuals_norm = residuals / (y.max() - y.min())

# plt.figure(figsize=(7,5))

# plt.hist(
#     residuals_norm,
#     bins=30
# )

# plt.xlabel("Normalized Residuals")
# plt.ylabel("Frequency")

# plt.title("Normalized Residual Distribution")

# plt.tight_layout()
# plt.show()

# #%% ================================
# # SHAP FEATURE IMPORTANCE
# # =================================
# importances = abs(shap_values.values).mean(axis=0)

# top_k = 10

# indices = np.argsort(importances)[::-1][:top_k]

# # normalize importances
# importances_norm = importances / importances.sum()

# plt.figure(figsize=(10,6))

# plt.barh(
#     standardized_df.columns[indices],
#     importances_norm[indices]
# )

# plt.gca().invert_yaxis()

# plt.xlabel("Normalized Mean |SHAP value|")

# plt.title("Top XGBoost Feature Importances")

# plt.tight_layout()
# plt.show()










