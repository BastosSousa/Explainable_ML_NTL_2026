# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 08:03:31 2025

@author: Natalia
"""

# from statsmodels.formula.api import logit

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
#%%
import pysal.lib
from pysal.lib import weights
import pysal.model
from esda.moran import Moran, Moran_Local
import esda.moran
import splot.esda
from splot.esda import moran_scatterplot


#%%
import pyproj

#%%
warnings.filterwarnings('ignore')

#%%
start = datetime.now()
start_time = start.strftime('%H:%M:%S')

# start = time.time()
# fmt = time.gmtime(start)
# strf = time.strftime("%D %T", fmt)

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

# %%

caminho_ISL07_INSPECOES = diretorio_atual+barra + \
    volta_nivel+barra+'saidas'+barra+'ISL07-INSPECOES.csv'

caminho_UCS_VARIAVEIS = diretorio_atual+barra+volta_nivel + \
    barra+'saidas'+barra+'ISL07-UCS-VARIAVEIS.csv'

caminho_x = diretorio_atual+barra+volta_nivel + \
    barra+'saidas'+barra+'Pontos-interpolados-idw-2.csv'
    
caminho_dados_artigo = diretorio_atual+barra + \
    volta_nivel+barra+'saidas'+barra+'dados_papaer.csv'
    
#%%

caminho_12 = diretorio_atual+barra + \
    volta_nivel+barra+'entradas'+barra+'gdf_dados_todos_12.xlsx'


caminho_ibge4 = diretorio_atual+barra+volta_nivel+barra + \
    'entradas'+barra+'dados_ibge_4.shp'
    
#%%

crs = {'init': 'epsg:4326'}

dados_ibge_4 = gpd.read_file(caminho_ibge4)
type(dados_ibge_4)
dados_ibge_4.crs  
    

# %%

teste_3 = pd.read_csv(caminho_x, sep=';')

dados_todos_12 = pd.read_csv(caminho_ISL07_INSPECOES, sep=';')

dados_todos_15 = pd.read_csv(caminho_UCS_VARIAVEIS, sep=';')

gdf_dados_todos_12 = pd.read_excel(caminho_12)


#%%
dados_todos_10_2 = teste_3.copy()

#%%

dados_todos_10_2 = dados_todos_10_2[dados_todos_10_2['LONGITUDE'] != 0]

#%%

def mad_based_outlier(points, threshold=3):
  # The heart of our map: the median.
  median = np.median(points)
  # How far off the path each point is.
  deviation = np.abs(points - median)
  # The 'average' deviation in our landscape.
  mad = np.median(deviation)
  # A score to identify those who wander too far.
  modified_z_score = 0.6745 * deviation / mad
  return modified_z_score > threshold

#%%

# Implement the mad_based_outlier function to check for outliers in both longitude and latitude
outliers_longitude = mad_based_outlier(gdf_dados_todos_12['LONGITUDE'])
outliers_latitude = mad_based_outlier(gdf_dados_todos_12['LATITUDE'])

outliers = outliers_longitude | outliers_latitude

#%%
gdf_dados_todos_12_2 = gdf_dados_todos_12.loc[~outliers]

#%%

###Plot geopoints before fixing outliers

plt.figure(figsize=(10, 6))
plt.scatter(gdf_dados_todos_12[~outliers]['LONGITUDE'], gdf_dados_todos_12[~outliers]['LATITUDE'], c='blue', label='Inliers')
plt.scatter(gdf_dados_todos_12[outliers]['LONGITUDE'], gdf_dados_todos_12[outliers]['LATITUDE'], c='red', label='Outliers')
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.title('MAD Outlier Detection Before Fixing')
plt.legend()
plt.show()

#%%% Inicio da Analise geo espacial dos dados de fiscalizacao
from shapely.ops import nearest_points
import shapely.geometry


gs = gpd.GeoSeries.from_wkt(gdf_dados_todos_12_2['geometry'])

gdf = gpd.GeoDataFrame(gdf_dados_todos_12_2, geometry=gs, crs="EPSG:4326")


bbox = gdf.total_bounds
polygon = shapely.geometry.box(*bbox, ccw=True)
area = polygon.area


#%%

def get_nearest_values(row, other_gdf, point_column='geometry', value_column="geometry"):
    """
    Find the nearest point and return the corresponding value from specified value column.
    """

    # Create an union of the other GeoDataFrame's geometries:
    other_points = other_gdf["geometry"].unary_union
    other_points = other_points.difference(row[point_column])

    # Find the nearest points
    nearest_geoms = nearest_points(row[point_column], other_points)

    # Get corresponding values from the other df
    nearest_data = other_gdf.loc[other_gdf["geometry"] == nearest_geoms[1]]

    nearest_value = nearest_data[value_column].values[0]

    return nearest_value

#%%
gdf['Nearest'] = gdf.apply(lambda row: get_nearest_values(row, gdf), axis=1)

#%%
gdf['Distance'] = gdf.apply(lambda row: row.geometry.distance(row['Nearest']), axis=1)

#%%
sumDist = gdf['Distance'].sum()

#%%
count = len(gdf)
do = float(sumDist) / count
de = float(0.55397 /(count / area)**(1/3))
d = float(do / de)
SE = float(0.20136 /(count / area)**(1/3))
zscore = float((do - de) / SE)

#%%
print(f'Observed mean distance: {do}')
print(f'Expected mean distance: {de}')
print(f'Nearest neighbour index: {d}')
print(f'Number of points: {count}')
print(f'NNI : {SE}')
print(f'Z-Score: {zscore}')

#%% Creates nearest neighbor weights matrix based on k nearest neighbors.

k=3
#%%%
gdf["ID"] = range(gdf.shape[0])

state_weights = pysal.lib.weights.KNN.from_dataframe(gdf, ids="ID", k=k)
state_weights.plot(gdf=gdf, indexed_on="ID")
plt.show()

#%%

matriz = pd.DataFrame(*state_weights.full()).astype(int)
#state_weights.nonzero
#wq.s0
#wq.pct_nonzero
#density (compliment of sparsity): 
#wq.pct_nonzero which is equal to 100*(w.s0/w.n**2)

#%%

# w_kernel = weights.distance.Kernel.from_dataframe(gdf_dados_todos_12)

w_adaptive  = weights.distance.Kernel.from_dataframe(gdf,fixed=False, k=k)

#w_kernel.function
#w_kernel.bandwidth[0:5]
#w_kernel.pct_nonzero
full_matrix, ids = w_adaptive.full()

#%%
mx_knn3 = weights.KNN.from_dataframe(gdf, k=k)

#%%
# fig = plt.figure(figsize=(10, 5)) ###vulner_dia
# ax = fig.add_subplot()
# dados_ibge.boundary.plot(ax=ax,color = 'grey',label = 'Census Sectors IBGE')
# #dados_ibge.centroid.plot(ax=ax,color = 'blue')
# shape_bairros_3.plot(column="cd_uep", categorical=True, cmap="Pastel2", ax=ax)
# dados_ivs.boundary.plot(ax=ax,color = 'red',label = 'Census Sectors HDU')
# mx_knn3.plot(gdf_dados_todos_12,edge_kws=dict(linewidth=1, color="black"),node_kws=dict(marker="*",label = 'CUs'),ax=ax)
# #ax.set_axis_off()
# #ax.set_title("Weights by $K$-NN 3")
# plt.xlabel("Longitude",fontsize=12)
# plt.ylabel("Latitude",fontsize=12)
# ax.legend(loc='lower center',facecolor='white', framealpha=1)
# plt.show()

#%%
w_rook = weights.contiguity.Rook.from_dataframe(gdf)
#print(w_rook.pct_nonzero)
s = pd.Series(w_rook.cardinalities)
#s.plot.hist(bins=s.unique().shape[0]);
# plt.show()

#%%
neighbors = w_rook.neighbors.copy()
adjlist = w_rook.to_adjlist()
#adjlist.head()

#%%
adjlist_fisca = adjlist.merge(
    gdf[["PARECER2_TOTAL"]],
    how="left",
    left_on="focal",
    right_index=True,
).merge(
    gdf[["PARECER2_TOTAL"]],
    how="left",
    left_on="neighbor",
    right_index=True,
    suffixes=("_focal", "_neighbor"),
)
        
#adjlist_fisca.info()

#%%

adjlist_fisca["diff"] = (adjlist_fisca["PARECER2_TOTAL_focal"]- adjlist_fisca["PARECER2_TOTAL_neighbor"])

all_pairs = np.subtract.outer(gdf["PARECER2_TOTAL"].values, gdf["PARECER2_TOTAL"].values)

complement_wr = 1 - w_rook.sparse.toarray()

non_neighboring_diffs = (complement_wr * all_pairs).flatten()

#%%

# f = plt.figure(figsize=(12, 3))
# plt.hist(
#     non_neighboring_diffs,
#     color="lightgrey",
#     edgecolor="k",
#     density=True,
#     bins=10,
#     label="Nonneighbors",
# )
# plt.hist(
#     adjlist_fisca["diff"],
#     color="salmon",
#     edgecolor="orangered",
#     linewidth=3,
#     density=True,
#     histtype="step",
#     bins=10,
#     label="Neighbors",
# )
# sns.despine()
# plt.ylabel("Density")
# plt.xlabel("Fiscalization Differences")
# plt.legend()
# plt.show()

#%%
extremes = adjlist_fisca.sort_values("diff", ascending=False).head()
#extremes

# %%
dados_ibge_4['centroid'] = dados_ibge_4['geometry'].centroid

#%%
dados_ibge_4['lat'] = dados_ibge_4.centroid.map(lambda p: p.y)
dados_ibge_4['lon'] = dados_ibge_4.centroid.map(lambda p: p.x)

#%%
# fig = plt.figure(figsize=(10, 5)) ###vulner_dia
# ax = fig.add_subplot()
# dados_ibge_3.boundary.plot(ax=ax,color = 'blue',label = 'Regions IBGE')

# #dados_ibge.centroid.plot(ax=ax,color = 'blue')
# shape_UCs_merge.plot(column="cd_bairro", categorical=True, cmap="Pastel2", ax=ax)
# plt.scatter(teste_3.LONGITUDE, teste_3.LATITUDE, c='red', s=0.5)
# plt.scatter(dados_ibge_3.lon, dados_ibge_3.lat, c='black', marker='x')

# #dados_ivs.boundary.plot(ax=ax,color = 'red',label = 'Regions IVS')
# # axis = sns.kdeplot(x = gdf_dados_todos_12.centroid.x, y = gdf_dados_todos_12.centroid.y,
# #                    weights = gdf_dados_todos_12["PARECER2"],
# #                    fill=True, gridsize=500, bw_adjust=0.25, cmap="coolwarm")
# # mx_knn3.plot(gdf_dados_todos_12,edge_kws=dict(linewidth=1, color="black"),node_kws=dict(marker="*"),ax=ax)
# # first_focus = gdf_dados_todos_12.iloc[[0, 49,64,83]]
# # second_focus = gdf_dados_todos_12.iloc[[12, 151]]
# # first_focus.plot(color="red", ax=ax)
# # second_focus.plot(color="red", ax=ax)
# #dados_ivs.apply(lambda x: ax.annotate(text=x.CD_GEOCODM, xy=x.geometry.centroid.coords[0], xytext=(1.5, 2.5), textcoords='offset points', fontsize=12),axis=1)
# plt.show()

#%% https://michaelminn.net/tutorials/python-points/index.html
# https://pysal.org/notebooks/viz/splot/esda_morans_viz.html
# https://www.itl.nist.gov/div898/handbook/index.htm


moran = esda.moran.Moran(gdf["PARECER2_TOTAL"], state_weights)

splot.esda.plot_moran(moran, zstandard=True, figsize=(10,4))
plt.savefig(r'C:\Pos\figuras\moran.svg')
plt.show()

# w.transform = "B"
# w.transform = "B"
# state_weights.weights

#%%Spatial Weights and Spatial Lag

gdf['w_PARECER2_TOTAL'] = weights.lag_spatial(state_weights, gdf['PARECER2_TOTAL'])

# census_tract_sp['w_total_monthly_income'] = weights.lag_spatial(w, census_tract_sp['total_monthly_income'])

#%%Global Spatial Autocorrelation

y_pop_count = gdf["PARECER2_TOTAL"]

moran = Moran(y_pop_count,state_weights)

#moran.I

# p-value to test if the result of Moran’s statistic is significant or not

moran.p_sim

# With a p-value of 0.001 we can reject the null hypothesis, therefore we can assume that the variable is not randomly distributed 
# in space and with the value of the Moran’s I statistic we can tell that it have positive correlation.
# We can use the Moran’s I scatterplot to help visualize the correlation.
#%%
# moran_scatterplot(moran)
# plt.show()


#The values in the x and y axis are in standard deviation. The x axis (Attribute) is referenced to the pop_count variable 
# (pop count of each tract) and the y axis (Spatial Lag) is referenced to the w_pop_count variable (pop count of the neighbors for each tract). 
# Looking at the plot we can analyse that as the x axis gets higher, so does the y axis in general. In other words, when pop_count is high in a 
# tract, its neighbors also tend to have high values.

#%% Local Spatial Autocorrelation

# local spatial autocorrelation helps to identify where clusters are in a map

# Local Moran's I
pop_count_local_moran = Moran_Local(y_pop_count, state_weights)

# Plotting Local Moran's I scatterplot of pop_count
fig, ax = moran_scatterplot(pop_count_local_moran, p=0.05);
plt.xlabel('Attribute', size=20)
plt.ylabel('Spatial Lag', size=20)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.text(1.96, 0.51, 'HH', fontsize=25)
plt.text(1.95, -1.0, 'HL', fontsize=25)
plt.text(-0.55, 0.6, 'LH', fontsize=25)
plt.text(-0.55, -1, 'LL', fontsize=25)
plt.savefig(r'C:\Pos\Run-py-Figures\SpatialLag.svg')
plt.show()

# grey dots are observations that doesn’t have statistical significance in the relationship of values of the observed tract value 
# and the neighbohood weighted value.

#%%

# creating column with local_moran classification
gdf['pop_count_local_moran'] = pop_count_local_moran.q

# Dict to map local moran's classification codes
local_moran_classification = {1: 'HH', 2: 'LH', 3: 'LL', 4: 'HL'}

# Mapping local moran's classification codes
gdf['pop_count_local_moran'] = gdf['pop_count_local_moran'].map(local_moran_classification)

# p-value for each observation/neighbor pair
gdf['pop_count_local_moran_p_sim'] = pop_count_local_moran.p_sim

# If p-value > 0.05 it is not statistical significant
gdf['pop_count_local_moran'] = np.where(gdf['pop_count_local_moran_p_sim'] > 0.05, 'ns', gdf['pop_count_local_moran'])

#%%
gdf_dados_todos_13 = gdf.copy()
gdf_dados_todos_13 =gdf_dados_todos_13[['Cod_setor','UC','w_PARECER2_TOTAL','pop_count_local_moran','pop_count_local_moran_p_sim','geometry','LATITUDE','LONGITUDE']]

#gdf_dados_todos_13.crs = "EPSG:4326"

# Plotting Local Moran's I classification map of pop_count column

#%%

m  = folium.Map(
    location=[-27.678173, -48.545003],
    tiles='cartodbpositron',
    zoom_start=13,
    min_zoom=6,
    )

gdf.explore(m=m,
    column='pop_count_local_moran',
    height='90%',
    width='90%',
    
    cmap=[
    '#D7191C', # Red
    '#FDAE61', # Orange
    '#e696c3', # Very soft pink
    '#9058d7', # Moderate violet
    '#119290'  # Grey
    ],
    style_kwds={
    'stroke': True,
    'edgecolor': 'k',
    'linewidth': 5.5,
    'fillOpacity': 5}
)

from folium.plugins import MarkerCluster

c=folium.FeatureGroup(name="UNI_TR_MT",overlay=True)
cf_cluster = MarkerCluster(name="UNI_TR_MT").add_to(m)


for i,row in gdf.iterrows():
    lat = gdf.at[i, 'LATITUDE']  #latitude
    lng = gdf.at[i, 'LONGITUDE']  #longitude
    popup = '<br>'+'<a href="https://www.google.com/maps?layer=c&cbll=' + str(gdf.at[i, 'LATITUDE']) + ',' + str(gdf.at[i, 'LONGITUDE']) + '" target="blank">GOOGLE STREET VIEW</a>'
    cf_marker = folium.Marker(location=[lat,lng], popup=popup, icon = folium.Icon(color="blue", icon="remove-sign"))
    cf_cluster.add_child(cf_marker).add_to(m) 



#map.save(outfile='BLANK_map.html')
m.save(r'C:\Pos\Run-py-Figures\Map1.html')

#%%

# world = gpd.read_file(geodatasets.get_path("naturalearth.land"))
# fig, ax = plt.subplots(figsize=(24, 18))
# world.plot(ax=ax, color="grey")
# gdf_dados_todos_12.plot(column="pop_count_local_moran", ax=ax, legend=True)
# plt.title("pop_count_local_moran")
# plt.show()

#%%
tiff_crs = 'EPSG:32722'  # Exemplo: UTM Zone 22S

google_earth_points = list(zip(gdf_dados_todos_13['LATITUDE'], gdf_dados_todos_13['LONGITUDE'], gdf_dados_todos_13['pop_count_local_moran']))

#Filtrar as coordenadas por grupos de NR_LOCZ_EQPTO_RD repetidos
grouped_coords = {}
for lat, lon, nr_locz in google_earth_points:
    if nr_locz in grouped_coords:
        grouped_coords[nr_locz].append((lat, lon))
    else:
        grouped_coords[nr_locz] = [(lat, lon)]
        

transformer = pyproj.Transformer.from_crs(pyproj.CRS('EPSG:4326'), tiff_crs, always_xy=True)

#%%  
# import seaborn as sns
# import rasterio
# from rasterio.plot import show

# with rasterio.open('C:/Pos/entradas/tapera1.tif') as src:
#     # Carregar os dados da imagem
#     image_data = src.read()
#     extent = rasterio.plot.plotting_extent(src)
#     # Plotar a imagem TIFF do bairro
#     plt.imshow(image_data.transpose([1, 2, 0]), extent=extent)
#     color_map = {}
#     next_color = 0
#     #dados_ibge_2.plot(ax=plt.gca(), edgecolor = 'red', facecolor='none')
#     for nr_locz, coords_list in grouped_coords.items():
#         if nr_locz not in color_map:
#             color_map[nr_locz] = sns.color_palette("tab10", len(grouped_coords))[next_color]
#             next_color += 1
           
#         coords_x, coords_y = zip(*[transformer.transform(lon, lat) for lat, lon in coords_list])
#         plt.scatter(coords_x, coords_y, color=color_map[nr_locz], s=12,label=nr_locz)
#         plt.legend(title = "LISA quadrant")

# plt.savefig('map.png', dpi=fig.dpi)

# plt.show()

#%%
moran_local = esda.moran.Moran_Local(gdf["PARECER2_TOTAL"], state_weights)

splot.esda.lisa_cluster(moran_local, gdf[["PARECER2_TOTAL", "geometry"]])

plt.show()

#%%
base_perdas = pd.read_excel(r'C:\Pos\entradas\entrada_MI.xlsx', sheet_name='Planilha1')

base_perdas_1 = base_perdas[['Cod_setor', 'Perdas']]

#%%
dados_todos_15['Cod_setor'] = dados_todos_15['Cod_setor'].astype('int64')

df_merged = dados_todos_15.merge(base_perdas_1, on='Cod_setor', how='left').dropna()

#%%
gdf_dados_todos_13['Cod_setor'] = gdf_dados_todos_13['Cod_setor'].astype(np.int64)


# chunks = np.array_split(df_merged, 500)

#%%

df_merged_2 = df_merged.merge(gdf_dados_todos_13, on=['Cod_setor','UC'], how='left').dropna()


#%%
df_merged_2.to_csv(caminho_dados_artigo, index=False, sep=';')



