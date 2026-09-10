# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 14:45:43 2025

@author: Natalia
Versão otimizada com:
- VIF iterativo (VIF < 5)
- remoção de colunas ruins/constantes/muito vazias
- eliminação de listas duplicadas de colunas
- uso direto das colunas finais no scaler e na interpolação
"""

import os
import platform
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
from geopandas import sjoin
from shapely.geometry import Point, box
from shapely import wkt

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

from scipy.spatial import distance_matrix
from scipy.interpolate import griddata

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor

import folium
import pyproj
import pysal.lib
from pysal.lib import weights
import pysal.model
from esda.moran import Moran, Moran_Local
import esda.moran
import splot.esda
from splot.esda import moran_scatterplot
from pyproj import Transformer
from geopy.geocoders import Nominatim
from time import sleep
import requests

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
caminho_dados_DSO = diretorio_atual + barra + volta_nivel + barra + 'entradas' + barra + 'DSO-Sachsen-Anhalt-file.xlsx'
caminho_dados_DSO_L = diretorio_atual + barra + volta_nivel + barra + 'entradas' + barra + 'DSO-LowerSaxony.xlsx'

caminho_anhalt = diretorio_atual + barra + volta_nivel + barra + 'DVG_ALKIS' + barra + 'DVG_ALKIS.shp'
caminho_shp = (
    diretorio_atual + barra + volta_nivel + barra +
    'vg-hist.utm32s.shape' + barra + 'daten' + barra + 'utm32s' +
    barra + 'shape' + barra + 'VG-Hist_1990-10-03_KRS.shp'
)
caminho_dados = diretorio_atual + barra + volta_nivel + barra + 'entradas' + barra + 'inkar_2024' + barra + 'inkar_2024.csv'
caminho_dados_2 = diretorio_atual + barra + volta_nivel + barra + 'entradas' + barra + 'Zensus2022_Energietraeger_10km-Gitter.csv'
caminho_dados_gdf_df_pivot = diretorio_atual + barra + volta_nivel + barra + 'entradas' + barra + 'gdf_df_pivot.csv'

# %%
out_dir = diretorio_atual + barra + volta_nivel + barra + 'saidas'
fig_dir = diretorio_atual + barra + volta_nivel + barra + 'Figures' + barra + 'heatmaps-new'
os.makedirs(out_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)

# %%
crs = {'init': 'epsg:4326'}

dados_anhalt = gpd.read_file(caminho_anhalt)
dados_shp = gpd.read_file(caminho_shp)
dados_todos = pd.read_excel(caminho_dados_DSO)
dados_todos_L = pd.read_excel(caminho_dados_DSO_L)

#%%

dados_combinados = pd.concat([dados_todos, dados_todos_L], ignore_index=True)

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

# %%
gdf_dados_todos = gpd.GeoDataFrame(
    dados_combinados,
    geometry=gpd.points_from_xy(dados_combinados.longitude, dados_combinados.latitude),
    crs="EPSG:4326"
)
gdf_dados_todos = gdf_dados_todos.to_crs(25832)

# %%
gdf_joined = sjoin(
    gdf_dados_todos,
    dados_shp,
    how="left",
    predicate="within"
)

# %%
dados = pd.read_csv(caminho_dados, sep=';', decimal=",")
dados_Zensus2022 = pd.read_csv(caminho_dados_2, sep=',')

# %%
dados_Zensus2022['geometry'] = [
    Point(xy) for xy in zip(dados_Zensus2022.x_mp_10km, dados_Zensus2022.y_mp_10km)
]
gdf = GeoDataFrame(dados_Zensus2022, crs="EPSG:3035", geometry=dados_Zensus2022.geometry)
gdf = gdf.to_crs(epsg=25832)

# %%
dados_2 = dados[dados['Zeitbezug'] == 2020]
dados_3 = dados_2[['Kuerzel', 'Name', 'Wert']].reset_index(drop=True)

df_pivot = dados_3.pivot_table(
    index="Name", columns="Kuerzel", values="Wert", aggfunc="first"
).reset_index()

df_pivot = df_pivot.rename(columns={'Name': 'GEN'})

# %%
gdf_df_pivot = pd.read_csv(caminho_dados_gdf_df_pivot, sep=',', encoding='utf-8')
gdf_df_pivot["geometry"] = gdf_df_pivot["geometry"].apply(wkt.loads)

gdf_df_pivot = gpd.GeoDataFrame(
    gdf_df_pivot,
    geometry="geometry",
    crs="EPSG:25832"
)
gdf_df_pivot = gdf_df_pivot.to_crs(epsg=4326)
gdf_df_pivot = gdf_df_pivot[~gdf_df_pivot.geometry.isna()].copy()

# %%
df_merged = gdf_df_pivot.drop_duplicates()
df_limpo = df_merged.dropna(subset=["geometry"]).copy()
df_limpo = df_limpo.set_geometry('geometry')
df_limpo = df_limpo[df_limpo["latitude"].notna() & df_limpo["longitude"].notna()].copy()

# %%
def make_grid_fixed_area(gdf, area_km2=10, crs=None):
    xmin, ymin, xmax, ymax = gdf.total_bounds
    cell_size = (area_km2 * 1e6) ** 0.5

    if crs is None:
        crs = gdf.crs

    grid_cells = []
    for x0 in np.arange(xmin, xmax, cell_size):
        for y0 in np.arange(ymin, ymax, cell_size):
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            grid_cells.append(box(x0, y0, x1, y1))

    grid = gpd.GeoDataFrame(geometry=grid_cells, crs=crs)
    return grid

# %%
grid = make_grid_fixed_area(gdf, area_km2=10, crs="EPSG:25832")

# %%
dados_anhalt['AGS'] = dados_anhalt['AGS'].astype(str)
dados_anhalt_2 = dados_anhalt[dados_anhalt['AGS'].str.startswith('15')]
gdados_anhalt_2 = GeoDataFrame(dados_anhalt_2, crs="EPSG:25832", geometry=dados_anhalt_2.geometry)

# %% 17
df_nan = df_limpo[df_limpo.isna().any(axis=1)].copy()
df_no_nan = df_limpo.dropna().copy()

df_no_nan["y"] = df_no_nan.centroid.map(lambda p: p.y)
df_no_nan["x"] = df_no_nan.centroid.map(lambda p: p.x)
sample_points = df_no_nan[['x', 'y']].values

df_nan['y_coords'] = df_nan.centroid.map(lambda p: p.y)
df_nan['x_coords'] = df_nan.centroid.map(lambda p: p.x)
unknown_points = df_nan[['x_coords', 'y_coords']].values

# %% ============================================================
# PRÉ-PROCESSAMENTO OTIMIZADO + VIF
# ============================================================
norma_teste_10 = df_no_nan.copy()

cols_drop_base = ["geometry", "y", "x", "GEN", "latitude", "longitude"]
cols_drop_base = [c for c in cols_drop_base if c in norma_teste_10.columns]
norma_teste_10 = norma_teste_10.drop(columns=cols_drop_base, errors='ignore')

norma_teste_10 = norma_teste_10.apply(pd.to_numeric, errors='coerce')
norma_teste_10 = norma_teste_10.dropna(axis=1, how='all')

nunique = norma_teste_10.nunique(dropna=True)
cols_constantes = nunique[nunique <= 1].index.tolist()
norma_teste_10 = norma_teste_10.drop(columns=cols_constantes, errors='ignore')

limiar_na = 0.4
frac_na = norma_teste_10.isna().mean()
cols_muito_nan = frac_na[frac_na > limiar_na].index.tolist()
norma_teste_10 = norma_teste_10.drop(columns=cols_muito_nan, errors='ignore')

norma_teste_10 = norma_teste_10.fillna(norma_teste_10.median(numeric_only=True))

print("Número inicial de variáveis numéricas:", norma_teste_10.shape[1])

# %%
def remover_correlacao_alta(df, limiar=0.95):
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > limiar)]
    df_red = df.drop(columns=to_drop, errors='ignore')
    return df_red, to_drop

norma_teste_corr, variaveis_removidas_corr = remover_correlacao_alta(norma_teste_10, limiar=0.95)

pd.DataFrame({"variavel_removida_corr": variaveis_removidas_corr}).to_csv(
    out_dir + barra + 'correlacao_variaveis_removidas.csv',
    sep=';', index=False, encoding='utf-8-sig'
)

print("Variáveis após filtro de correlação:", norma_teste_corr.shape[1])

# %%
def calcular_vif_iterativo(df, thresh=5.0, verbose=True):
    df_vif = df.copy()
    df_vif = df_vif.apply(pd.to_numeric, errors='coerce')
    df_vif = df_vif.dropna(axis=1, how='all')
    df_vif = df_vif.fillna(df_vif.median(numeric_only=True))

    nunique = df_vif.nunique(dropna=True)
    cols_const = nunique[nunique <= 1].index.tolist()
    if cols_const:
        df_vif = df_vif.drop(columns=cols_const, errors='ignore')
        if verbose:
            print("Colunas constantes removidas:", cols_const)

    removidas = []

    while df_vif.shape[1] > 1:
        vif_data = pd.DataFrame({
            "variavel": df_vif.columns,
            "VIF": [variance_inflation_factor(df_vif.values, i) for i in range(df_vif.shape[1])]
        })

        max_vif = vif_data["VIF"].max()

        if max_vif < thresh:
            break

        var_remover = vif_data.sort_values("VIF", ascending=False).iloc[0]["variavel"]
        removidas.append(var_remover)

        if verbose:
            print(f"Removendo '{var_remover}' com VIF = {max_vif:.2f}")

        df_vif = df_vif.drop(columns=[var_remover])

    vif_final = pd.DataFrame({
        "variavel": df_vif.columns,
        "VIF": [variance_inflation_factor(df_vif.values, i) for i in range(df_vif.shape[1])]
    }).sort_values("VIF", ascending=False)

    return df_vif, vif_final, removidas

# %% 21 -----------------
norma_teste_vif, tabela_vif_final, variaveis_removidas_vif = calcular_vif_iterativo(
    norma_teste_corr,
    thresh=5.0,
    verbose=True
)

print("\nQuantidade inicial de variáveis:", norma_teste_10.shape[1])
print("Após filtro de correlação:", norma_teste_corr.shape[1])
print("Quantidade final após VIF:", norma_teste_vif.shape[1])
print("Quantidade removida por VIF:", len(variaveis_removidas_vif))

tabela_vif_final.to_csv(
    out_dir + barra + 'vif_final.csv',
    sep=';', index=False, encoding='utf-8-sig'
)

pd.DataFrame({"variavel_removida_vif": variaveis_removidas_vif}).to_csv(
    out_dir + barra + 'vif_variaveis_removidas.csv',
    sep=';', index=False, encoding='utf-8-sig'
)

# %%
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(norma_teste_vif.values)

norma_teste_1 = pd.DataFrame(
    x_scaled,
    columns=norma_teste_vif.columns,
    index=norma_teste_vif.index
)

values = norma_teste_vif.values
new_names = list(norma_teste_vif.columns)

print("Dimensão final da matriz para interpolação:", values.shape)

# %% ============================================================
# INTERPOLAÇÃO IDW
# ============================================================
def idw_interpolation(sample_points, unknown_points, values, power=2):
    sample_points = np.asarray(sample_points)
    unknown_points = np.asarray(unknown_points)
    values = np.asarray(values)

    mask = ~np.isnan(values)
    sample_points = sample_points[mask]
    values = values[mask]

    distances = distance_matrix(sample_points, unknown_points)
    distances[distances == 0] = 1e-10

    weights = 1 / np.power(distances, power)
    interpolated_values = np.sum(weights * values[:, np.newaxis], axis=0) / np.sum(weights, axis=0)
    return interpolated_values

# %%
chunk_size = 1400
chunk_unknown_points = np.split(unknown_points, range(chunk_size, unknown_points.shape[0], chunk_size))
n_cols_interp = values.shape[1]

def process_chunk(chunk, sample_points, values, power=2):
    chunk_result = pd.DataFrame(index=range(len(chunk)))
    for i in range(n_cols_interp):
        interpolated_values = idw_interpolation(sample_points, chunk, values[:, i], power=power)
        chunk_result[i] = interpolated_values
    return chunk_result

# %%
resultado_list = []
for i, chunk in enumerate(chunk_unknown_points):
    print(f'Processing Chunk {i+1} de {len(chunk_unknown_points)}')
    resultado_list.append(process_chunk(chunk, sample_points, values, power=2))

resultado = pd.concat(resultado_list, ignore_index=True)

# %%
resultado_2 = resultado.copy().reset_index(drop=True)
old_names = list(range(len(new_names)))
resultado_3 = resultado_2.rename(columns=dict(zip(old_names, new_names)))

teste_4 = df_nan[['geometry', 'GEN', 'latitude', 'longitude']].copy().reset_index(drop=True)
resultado_4 = pd.merge(teste_4, resultado_3, left_index=True, right_index=True)
resultado_4 = gpd.GeoDataFrame(resultado_4, geometry='geometry', crs=df_nan.crs)
resultado_4 = resultado_4.to_crs(epsg=25832)

# %%
fig = plt.figure(figsize=(10, 10))
col = new_names[0] if len(new_names) > 0 else None

if col is not None:
    ax = fig.add_subplot()
    resultado_4.plot(
        ax=ax,
        column=col,
        cmap="OrRd",
        legend=True,
        figsize=(8, 6),
        edgecolor="black",
        linewidth=0.5
    )
    dados_shp.boundary.plot(ax=ax)
    plt.title(f"Choropleth map of '{col}'", fontsize=14)
    plt.savefig(out_dir + barra + f"choropleth_{col}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

# %%
resultado_5 = resultado_4.to_crs(epsg=4326)
dados_shp_2 = dados_shp.to_crs(epsg=4326)

num_pts = 1800
x_grid = np.linspace(min(resultado_5['longitude']), max(resultado_5['longitude']), num_pts)
y_grid = np.linspace(min(resultado_5['latitude']), max(resultado_5['latitude']), num_pts)
X, Y = np.meshgrid(x_grid, y_grid)

matplotlib.use('Agg')
cols_interp = resultado_5.columns.drop(['latitude', 'longitude', 'geometry', 'GEN'])

for col in cols_interp:
    Z = griddata(
        list(zip(resultado_5['longitude'], resultado_5['latitude'])),
        resultado_5[col],
        (X, Y),
        method='linear'
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(
        Z,
        extent=(
            min(resultado_5['longitude']) - 0.01,
            max(resultado_5['longitude']) + 0.01,
            min(resultado_5['latitude']) - 0.01,
            max(resultado_5['latitude']) + 0.01
        ),
        origin='lower'
    )
    dados_shp_2.boundary.plot(ax=ax, color='red', linewidth=0.8)
    ax.set_title(f'Interpolação – {col}')
    fig.colorbar(im, ax=ax)

    out_file = os.path.join(fig_dir, f'interpolation_{col}.png')
    fig.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

# %%
# resultado_5.to_csv(out_dir + barra + 'IDW-Result.csv', index=False, sep=';')

print("Processamento concluído.")
print("Saídas salvas em:", out_dir)
print("Figuras de interpolação salvas em:", fig_dir)
