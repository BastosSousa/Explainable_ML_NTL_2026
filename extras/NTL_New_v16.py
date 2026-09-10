# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 15:32:07 2024

@author: Natalia Bastos
"""

import folium

import numpy as np


import os
import platform
import pandas as pd

from folium.plugins import MarkerCluster

#%%
import geopandas as gpd

import h3
import seaborn as sns
from shapely.geometry import Polygon

import matplotlib.pyplot as plt

from sklearn.preprocessing import Normalizer

from sklearn.metrics import davies_bouldin_score

from sklearn.cluster import KMeans
#%%

#from sklearn_extra.cluster import KMedoids
#import kmedoids

#%%
from pylab import bone, pcolor, colorbar

from sklearn.preprocessing import LabelEncoder

import umap.umap_ as umap

from kneed import KneeLocator

from minisom import MiniSom

from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import MinMaxScaler


from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import shap
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


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
caminho_dados2 = diretorio_atual+barra + \
    volta_nivel+barra+'saidas'+barra+'dados_papaer2.csv'
#%%

caminho_dados_final1 = diretorio_atual+barra + \
    volta_nivel+barra+'saidas'+barra+'data_final1.csv'
   
caminho_dados_final3 = diretorio_atual+barra + \
    volta_nivel+barra+'saidas'+barra+'data_toFuzzy.csv'
  
    
caminho_dados_final2 = diretorio_atual+barra + \
    volta_nivel+barra+'saidas'+barra+'data_finalFuzzy.csv'

#%%
dados2 = pd.read_csv(caminho_dados2,sep=';')


#df = dados2.head(200)
df = dados2.copy()
#%%


df = df.rename(
    columns={'LATITUDE_x': 'LATITUDE', 'LONGITUDE_x':'LONGITUDE'})

#%%

#df = df.rename(columns={'x_coords': 'LONGITUDE','y_coords':'LATITUDE'})

#%%
map = folium.Map(location=[-27.687077,-48.553949],
                 tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                 attr = 'Esri',
                 name = 'Esri Satellite',
                 zoom_start=19,
                 min_zoom=11,
                
                 max_bounds=True,
                 control_scale=True,
                 prefer_canvas=True
                 )

c=folium.FeatureGroup(name="UNI_TR_MT",overlay=True)
cf_cluster = MarkerCluster(name="UNI_TR_MT").add_to(map)


for i,row in df.iterrows():
    lat = df.at[i, 'LATITUDE']  #latitude
    lng = df.at[i, 'LONGITUDE']  #longitude
    popup = '<br>'+'<a href="https://www.google.com/maps?layer=c&cbll=' + str(df.at[i, 'LATITUDE']) + ',' + str(df.at[i, 'LONGITUDE']) + '" target="blank">GOOGLE STREET VIEW</a>'
    cf_marker = folium.Marker(location=[lat,lng], popup=popup, icon = folium.Icon(color="blue", icon="remove-sign"))
    cf_cluster.add_child(cf_marker)


map.save(r'C:\Pos\Run-py-Figures\BLANK_map.html')

#%%
#contagem_SETORES = df['Cod_setor'].value_counts()


#%%

gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.LONGITUDE, df.LATITUDE), crs='EPSG:4326')

#gdf = gdf[['w_PARECER2_TOTAL','LATITUDE','LONGITUDE']]
#%%

res = 10

def geo_to_h3(row):
  return h3.latlng_to_cell(lat=row.LATITUDE,lng=row.LONGITUDE,res = res)


gdf['h3_cell'] = gdf.apply(geo_to_h3,axis=1)

#%%

final_table_columns = ['Cod_setor', 'UC', 'UNI_TR_MT','PARECER2',
'Nearest', 'Distance', 'ID','y_coords', 'x_coords','LATITUDE_y', 'LONGITUDE_y']
gdf = gdf.drop(columns=[col for col in gdf if col in final_table_columns])

#%%

# gdf = gdf.drop(columns=['Cod_setor', 'UC', 'Bairro', 'CD_ALIMENTADOR','PARECER2',
# 'Nearest', 'Distance', 'ID','y_coords', 'x_coords'])

#%%

h3_df = gdf.groupby('h3_cell')['w_PARECER2_TOTAL'].describe().reset_index()

#%%

def cell_to_shapely(cell):
    coords = h3.cell_to_boundary(cell)
    flipped = tuple(coord[::-1] for coord in coords)
    return Polygon(flipped)

#%%
h3_geoms = h3_df['h3_cell'].apply(lambda x: cell_to_shapely(x))
h3_gdf = gpd.GeoDataFrame(data=h3_df, geometry=h3_geoms, crs=4326)

#%%

# fig, ax = plt.subplots()
# h3_gdf.plot(ax=ax, color='white', edgecolor='black')
# plt.show()

#%%

parecer = (gdf.groupby('h3_cell')
                          .w_PARECER2_TOTAL
                          .agg(list)
                          .to_frame("ids")
                          .reset_index())


# Let's count each points inside the hexagon
parecer['count'] =(parecer['ids']
                      .apply(lambda w_PARECER2_TOTAL:len(w_PARECER2_TOTAL)))
#%%

def add_geometry(row):
  points = h3.cell_to_boundary(row['h3_cell'])
  return Polygon(points)
#Apply function into our dataframe
parecer['geometry'] = (parecer.apply(add_geometry,axis=1))


#%%
geojson_obj = h3_gdf.to_json()


#%%

# ax = h3_gdf.plot(figsize=(15, 15), column='count', cmap='RdBu',legend = True)

# ax.axis('on')

#%%
map = folium.Map(location=[ -27.5969, -48.5495], tiles="CartoDB Positron", zoom_start=13)

# for _, r in h3_gdf.iterrows():
#     # Without simplifying the representation of each borough,
#     # the map might not be displayed
#     sim_geo = gpd.GeoSeries(r["geometry"])
#     geo_j = sim_geo.to_json()
#     geo_j = folium.GeoJson(data=geo_j, style_function=lambda x: {"fillColor": "orange"})
#     folium.Popup(r["count"]).add_to(geo_j)
#     geo_j.add_to(map)

folium.Choropleth(
    geo_data=geojson_obj,
    name="choropleth",
    data= h3_gdf,
    columns=["h3_cell", "count"],
    key_on='feature.properties.h3_cell',
    fill_color="YlGn",

    legend_name="Energy Theft cases sum",
    highlight = True
).add_to(map)

highlights = folium.features.GeoJson(
    geojson_obj,style_function= lambda x: {'color':'transparent', 'fillColor':'transparent', 'weight':0},
    highlight_function = lambda x: {'fillColor': '#000000', 'color':'#000000', 'fillOpacity': 0.50, 'weight': 0.1},
    tooltip=folium.features.GeoJsonTooltip(fields=[ 'h3_cell','count' ], 
                                           aliases = [ 'h3_cell id: ', 'Sum of cases: '], 
                                           labels = True,sticky = False))
map.add_child(highlights)
map.keep_in_front(highlights)

for i,row in df.iterrows():
    lat = df.at[i, 'LATITUDE']  #latitude
    lng = df.at[i, 'LONGITUDE']  #longitude
    popup = '<br>'+'<a href="https://www.google.com/maps?layer=c&cbll=' + str(df.at[i, 'LATITUDE']) + ',' + str(df.at[i, 'LONGITUDE']) + '" target="blank">GOOGLE STREET VIEW</a>'
    cf_marker = folium.Marker(location=[lat,lng], popup=popup, icon = folium.Icon(color="blue", icon="remove-sign"))
    map.add_child(cf_marker)

folium.LayerControl().add_to(map)
map.save(r'C:\Pos\Run-py-Figures\hexmap.html')

#%%

gdf_to_normalize = gdf.copy()
gdf_to_normalize.drop(columns = ['LATITUDE', 'LONGITUDE','geometry', 'pop_count_local_moran','pop_count_local_moran_p_sim','h3_cell', 'w_PARECER2_TOTAL','PARECER2_TOTAL','Perdas'], inplace = True)

#%% Normalizar

normal_values = Normalizer().fit_transform(gdf_to_normalize.values)

#%% Aplicar UMAP

reducer = umap.UMAP()
embedding = reducer.fit_transform(normal_values)

#%%

csfont = {'fontname':'Times New Roman'}
plt.rc('font',family='Times New Roman')
#plt.title('UMAP embedding');
plt.scatter(embedding[:, 0], embedding[:, 1], cmap='Spectral', s=5)
plt.ylabel("UMAP component 2",size=15)
plt.xlabel("UMAP component 1",size=15)
plt.show()

#%%
WCSS = []

for i in range(1,10):
  umap_pca = KMeans(n_clusters = i, init = "k-means++", random_state = 0)
  umap_pca.fit(embedding)
  WCSS.append(umap_pca.inertia_)


#%%
numb_k = range(1,10)

kn = KneeLocator(numb_k, WCSS, curve='convex', direction='decreasing')
k = kn.knee
#print(kn.knee)

#%%
# plt.figure(figsize = (10,6))
# plt.plot(range(1,10), WCSS, marker = "o", linestyle = "--")
# plt.grid()
# plt.title("Cluster using Umap Scores")
# plt.ylabel("WCSS")
# plt.xlabel("Number of Clusters")
# plt.show()

#%%
kmeans_umap = KMeans(n_clusters = k, init = "k-means++", random_state = 0)
kmeans_umap.fit(embedding)

#%%

# Concatening the original df with the components informations present in scores_pca
df_umap_kmeans = pd.concat([gdf, pd.DataFrame(embedding)], axis = 1)

#%%
# Renaming the column label from each component
df_umap_kmeans.columns.values[-2:] = ["comp1", "comp2"]

df_umap_kmeans["segment_kmeans_umap"] = kmeans_umap.labels_

#%%
# Mapping each cluster segmentation and renaming their labels 
df_umap_kmeans["segment"] = df_umap_kmeans["segment_kmeans_umap"].map({0:"Cluster 1", 1:"Cluster 2", 2:"Cluster 3",3:"Cluster 4"})

#%%

# df1 = df_umap_kmeans.iloc[ :, 99:104]
#%%
# df1['segment'] = df_umap_kmeans['segment']
sns.scatterplot(data=df_umap_kmeans, x="comp1", y="comp2", hue='segment')

#%%

df_umap_kmeans_2 = df_umap_kmeans[['w_PARECER2_TOTAL', 'pop_count_local_moran',
                                   'pop_count_local_moran_p_sim','h3_cell','segment_kmeans_umap','PARECER2_TOTAL','Perdas']]


#%%

df_umap_kmeans_2 = df_umap_kmeans_2.drop_duplicates(["h3_cell", "pop_count_local_moran_p_sim"])

df_umap_kmeans_2 = df_umap_kmeans_2.loc[df_umap_kmeans_2['pop_count_local_moran'].isin(['HH','LH'])]

df_umap_kmeans_2['geometry'] = df_umap_kmeans_2['h3_cell'].apply(lambda x: cell_to_shapely(x))
df_umap_kmeans_2 = gpd.GeoDataFrame(data=df_umap_kmeans_2, geometry='geometry', crs=4326)

#%%
df2 = df_umap_kmeans.copy()

df2["target"] = df2["segment_kmeans_umap"]

contagem_valores = df2['target'].value_counts()

#%%

le = LabelEncoder()
label = le.fit_transform(df2['target'])
df2.loc[:,'target'] = label

#%%
df2_data = df2[['comp1','comp2']].values

#%%
df2_label = df2.target

contagem_valores = df2_label.value_counts()

#%%
df3_data = df2.copy()

df3_data.drop(columns = ['LATITUDE', 'LONGITUDE','geometry','comp1', 'comp2','pop_count_local_moran',
                         'segment_kmeans_umap', 'segment', 'target','som_cluster', 'Clusters'], inplace = True)

#%%

# SOM_X_AXIS_NODES  = 19
# SOM_Y_AXIS_NODES  = 19
# SOM_N_VARIABLES  = df2_data.shape[1]
# NEIGHBORHOOD_FUNC = 'gaussian'
# DISTANCE_FUNC = 'euclidean'


# RANDOM_SEED = 0

# #%%

# som = MiniSom(
#         SOM_X_AXIS_NODES,
#         SOM_Y_AXIS_NODES,
#         SOM_N_VARIABLES,
#         sigma=2,
#         learning_rate=0.5,
#         neighborhood_function=NEIGHBORHOOD_FUNC,

#         random_seed=RANDOM_SEED
#         )

# #%%
# som.pca_weights_init(df2_data)
# N_ITERATIONS = 500*SOM_X_AXIS_NODES*SOM_Y_AXIS_NODES
# som.train_random(df2_data, N_ITERATIONS)  

# #%%

# # Obtendo a matriz de distâncias do SOM
# distance_map = som.distance_map()
# # Plotando o mapa de calor
# plt.figure(figsize=(10, 10))
# plt.pcolor(distance_map, cmap='bone')
# plt.colorbar()
# #plt.title('SOM heatmap')
# plt.ylabel("SOM_Y_AXIS",**csfont,fontsize=15)
# plt.xlabel("SOM_X_AXIS",**csfont,fontsize=15)
# plt.show()

# #%%
# y = df2_label
# #%%
# # Visualizing the results

# # fig = plt.figure(figsize = (15,10))
# # ax = fig.add_subplot(111)
# # bone()
# # pcolor(som.distance_map().T)
# # colorbar()

# # for i,x in enumerate(df2_data): 
# #     w = som.winner(x)
# #     word = y[i]
# #     ax.plot((w[0],),(w[1],), 'x', c='r') 
# #     ax.annotate(word, (w[0]+0.2, w[1]+0.2), ha='center', va='center',size = 12,color='white')
# # plt.show()

# weights = som.get_weights()  
# # shape: (som_x, som_y, input_len)
# weights_flat = weights.reshape(-1, weights.shape[-1])
# # shape: (som_x * som_y, input_len)

# n_clusters = k  # choose desired number of final clusters

# agg = AgglomerativeClustering(
#     n_clusters=n_clusters,
#     linkage="ward"
# )

# neuron_labels = agg.fit_predict(weights_flat)

# #%%

# Z = linkage(weights_flat, method="ward")

# plt.figure(figsize=(10, 5))
# dendrogram(Z, no_labels=True)
# #plt.title("Hierarchical clustering of SOM neurons (Ward)")
# plt.xlabel("Neuron index",**csfont,fontsize=15)
# plt.ylabel("Distance",**csfont,fontsize=15)
# plt.show()

 
# #%%
# som_shape = (SOM_X_AXIS_NODES, SOM_Y_AXIS_NODES)
# # each neuron represents a cluster
# winner_coordinates = np.array([som.winner(x) for x in df2_data]).T
# # with np.ravel_multi_index we convert the bidimensional
# # coordinates to a monodimensional index
# cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)
# #%%

# df2["som_cluster"] = cluster_index

# weights = som.get_weights().reshape(-1, 2)

# Z = linkage(weights, method="ward")
# neuron_clusters = fcluster(Z, t=4, criterion="maxclust")

# df2["som_cluster"] = neuron_clusters[
#     np.ravel_multi_index(winner_coordinates, som_shape)
# ]

# #%%
# contagem_valores = df2["som_cluster"].value_counts()

# #%%
# # plt.figure(figsize=(8, 6))
# # sc = plt.scatter(
# #     df2["comp1"],
# #     df2["comp2"],
# #     c=df2["som_cluster"],
# #     cmap="tab20",
# #     s=10
# # )
# # plt.colorbar(sc, label="SOM cluster index")
# # #plt.title("UMAP projection colored by SOM clusters")
# # plt.xlabel("comp1",**csfont,fontsize=15)
# # plt.ylabel("comp2",**csfont,fontsize=15)
# # plt.show()

# #%%

# # Dict to map local moran's classification codes
# df2_clusters = {1: 'Cluster 1', 2: 'Cluster 2', 3: 'Cluster 3', 4: 'Cluster 4'}

# # Mapping local moran's classification codes
# df2['Clusters'] = df2['som_cluster'].map(df2_clusters)


# #%%
# ax = plt.figure(figsize=(11,9))
# ax = sns.scatterplot(data=df2, x = 'comp1', y ='comp2',
#                      hue='Clusters',
#     s=100
# )

# plt.legend(loc='upper right')
# plt.xlabel("comp1",**csfont,fontsize=15)
# plt.ylabel("comp2",**csfont,fontsize=15)
# plt.show()

# # n_kmc = KMeans(n_clusters = len(df2_label.value_counts()))

# # n_kmc.fit(df2_data)

# # cluster_vec = n_kmc.predict(df2_data)

# #%%

# cluster_df = pd.DataFrame(df2_data,columns =df2[['comp1','comp2']].columns)

# #%%
# cluster_df['clusters'] = df2["som_cluster"].values

# #%%

# # plt.scatter(df2_data[cluster_vec==0, 0],df2_data[cluster_vec==0, 1], s=100, c='red', label ='Cluster 1')
# # plt.scatter(df2_data[cluster_vec==1, 0], df2_data[cluster_vec==1, 1], s=100, c='blue', label ='Cluster 2')
# # plt.scatter(df2_data[cluster_vec==2, 0], df2_data[cluster_vec==2, 1], s=100, c='green', label ='Cluster 3')
# # plt.scatter(df2_data[cluster_vec==3, 0], df2_data[cluster_vec==3, 1], s=100, c='orange', label ='Cluster 4')
# # plt.legend()
# # plt.show()

# #%%

# y = cluster_df.clusters

# #%%

# davies_bouldin_score(df2_data, y)

# #%%
# csfont = {'fontname':'Times New Roman'}
# plt.rc('font',family='Times New Roman')
# # Visualizing the results

# fig = plt.figure(figsize = (10,10))
# ax = fig.add_subplot(111)
# bone()
# pcolor(som.distance_map().T)

# # Colorbar with custom font size
# cbar = colorbar()
# cbar.ax.tick_params(labelsize=15)  # set tick label size

# # Axis ticks
# ax.tick_params(axis='both', which='major', labelsize=15)

# for i,x in enumerate(df2_data): 
#     w = som.winner(x)
#     word = y[i]
#     ax.plot((w[0],),(w[1],), 'x', c='r') 
#     ax.annotate(word, (w[0]+0.2, w[1]+0.2), ha='center', va='center',**csfont,size = 12,color='white')
#     ax.set_ylabel("SOM_Y_AXIS",**csfont,fontsize=15)
#     ax.set_xlabel("SOM_X_AXIS",**csfont,fontsize=15)
# plt.show()

#%%
# df4_data["cluster_index_som"] = cluster_index

# contagem_valores = df4_data["cluster_index_som"].value_counts()

# df4_data = df4_data.reset_index()

#%%

# df4_comp = df4_data[["cluster_index_som","PARECER2_TOTAL"]]

#%%
df3_data['geometry'] = df3_data['h3_cell'].apply(lambda x: cell_to_shapely(x))

#%%

df3_data = gpd.GeoDataFrame(data=df3_data, geometry='geometry', crs=4326)

#df3_data["clusters"] = cluster_df.clusters

#df3_data_2 = df3_data[['PARECER2_TOTAL','h3_cell', 'ivs','idhm' , 'Perdas','geometry','Porc','PO_NOMINAL_TRAFO','clusters']]

#%%
from scipy.stats import gaussian_kde

h3_counts = df3_data.groupby("h3_cell").size()

plt.figure(figsize=(10,5))

plt.hist(h3_counts, bins=50, density=True, alpha=0.5)

kde = gaussian_kde(h3_counts)
x_range = np.linspace(h3_counts.min(), h3_counts.max(), 500)
plt.plot(x_range, kde(x_range), linewidth=2)

plt.xscale("log")

plt.title("Distribution of samples by H3 (with KDE, log scale)")
plt.xlabel("Number of samples per H3 (log)")
plt.ylabel("Density")

plt.show()

#%%
# # 1. contagem por H3
# h3_counts = df3_data_2.groupby("h3_cell").size().reset_index(name="n_samples")

# # 2. juntar com geometria (assumindo que df3_data_2 já tem geometry por linha)
# h3_geom = df3_data_2[["h3_cell", "geometry"]].drop_duplicates()

# h3_map = h3_geom.merge(h3_counts, on="h3_cell", how="left")

# # 3. converter para GeoDataFrame
# h3_gdf = gpd.GeoDataFrame(h3_map, geometry="geometry", crs=4326)

# # 4. plot
# fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# h3_gdf.plot(
#     column="n_samples",
#     cmap="viridis",
#     legend=True,
#     ax=ax
# )

# ax.set_title("Densidade de amostras por célula H3")
# ax.set_axis_off()

# plt.show()

#%%
# plt.figure(figsize=(10,5))

# plt.hist(h3_counts, bins=50, density=True, alpha=0.5)

# kde = gaussian_kde(h3_counts)
# x_range = np.linspace(h3_counts.min(), h3_counts.max(), 500)
# plt.plot(x_range, kde(x_range), linewidth=2)

# plt.xscale("log")

# plt.title("Distribuição de amostras por H3 (com KDE, escala log)")
# plt.xlabel("Número de amostras por H3 (log)")
# plt.ylabel("Densidade")

# plt.show()
#%%

base_features = [
    c for c in df3_data.columns
    if c not in ["h3_cell", "geometry", "Perdas","clusters"]
]

agg_dict = {}

for col in base_features:
    agg_dict[col] = ["mean", "std"]

agg_dict["Perdas"] = ["mean", "std", "count"]

h3_agg = df3_data.groupby("h3_cell").agg(agg_dict)

h3_agg.columns = [
    f"{col}_{stat}" for col, stat in h3_agg.columns
]

h3_agg = h3_agg.reset_index()
#%%
from sklearn.feature_selection import VarianceThreshold

X_temp = h3_agg.drop(columns=["h3_cell", "Perdas_mean"])

selector = VarianceThreshold(threshold=0.0)
X_filtered = selector.fit_transform(X_temp)

kept_cols = X_temp.columns[selector.get_support()]

X = pd.DataFrame(X_filtered, columns=kept_cols)


#%% XGBoost
y = h3_agg["Perdas_mean"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R²:", r2_score(y_test, y_pred))

#%%

importance = pd.DataFrame({
    "feature": X.columns,
    "gain": model.feature_importances_
}).sort_values("gain", ascending=False)


#%%
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

shap_df = pd.DataFrame(shap_values, columns=X.columns)
shap_df["h3_cell"] = h3_agg["h3_cell"].values

#%%

top_features = (
    shap_df.drop(columns=["h3_cell"])
    .abs()
    .mean()
    .sort_values(ascending=False)
    .head(20)
    .index
)

print(top_features)


#%%
train_pred = model .predict(X_train)

print(
    "Train R²:",
    r2_score(y_train, train_pred)
)

print(
    "Test R²:",
    r2_score(y_test, y_pred)
)

#%%
h3_agg["pred"] = model.predict(X)
h3_agg["residual"] = y - h3_agg["pred"]


#%%

shap_df["h3_cell"] = h3_agg["h3_cell"].values

shap_h3 = shap_df.groupby("h3_cell").mean().reset_index()

#%%

geo_h3 = df3_data[["h3_cell", "geometry"]].drop_duplicates()

map_df = (
    h3_agg[["h3_cell", "pred", "residual"]]
    .merge(shap_h3, on="h3_cell")
    .merge(geo_h3, on="h3_cell")
)

gdf = gpd.GeoDataFrame(map_df, geometry="geometry", crs=4326)
#%%
top_shap_feature = (
    shap_df.drop(columns=["h3_cell"])
    .abs()
    .mean()
    .sort_values(ascending=False)
    .index[0]
)

#%%

fig, axes = plt.subplots(1, 3, figsize=(20, 7))

# -------------------------
# 1. RISCO (predição)
# -------------------------
gdf.plot(
    column="pred",
    cmap="viridis",
    legend=True,
    ax=axes[0]
)
axes[0].set_title("Risco previsto (Perdas)")
axes[0].set_axis_off()

# -------------------------
# 2. ERRO (residual)
# -------------------------
gdf.plot(
    column="residual",
    cmap="coolwarm",
    legend=True,
    ax=axes[1]
)
axes[1].set_title("Erro do modelo (Residual)")
axes[1].set_axis_off()

# -------------------------
# 3. SHAP (impacto da variável)
# -------------------------
gdf.plot(
    column=top_shap_feature,
    cmap="plasma",
    legend=True,
    ax=axes[2]
)
axes[2].set_title(f"Impacto SHAP: {top_shap_feature}")
axes[2].set_axis_off()

plt.tight_layout()
plt.show()

#%%

shap_importance = (
    shap_df.drop(columns=["h3_cell"])
    .abs()
    .mean()
    .sort_values(ascending=False)
)

top_cols = shap_importance.head(15).index.tolist()

#%%

som_df = shap_h3[top_cols].copy()

scaler = StandardScaler()

X_som = scaler.fit_transform(som_df[top_cols])


#%%

SOM_X_AXIS_NODES = 16
SOM_Y_AXIS_NODES = 16

som = MiniSom(
    x=SOM_X_AXIS_NODES,
    y=SOM_Y_AXIS_NODES,
    input_len=X_som.shape[1],
    sigma=2,
    learning_rate=0.5,
    random_seed=42
)

som.pca_weights_init(X_som)
N_ITERATIONS = 500*SOM_X_AXIS_NODES*SOM_Y_AXIS_NODES
som.train_random(X_som, N_ITERATIONS)
#%%

# Obtendo a matriz de distâncias do SOM
distance_map = som.distance_map()
# Plotando o mapa de calor
plt.figure(figsize=(10, 10))
plt.pcolor(distance_map, cmap='bone')
plt.colorbar()
#plt.title('SOM heatmap')
plt.ylabel("SOM_Y_AXIS",**csfont,fontsize=15)
plt.xlabel("SOM_X_AXIS",**csfont,fontsize=15)
plt.show()

#%%

weights = som.get_weights()
weights_flat = weights.reshape(-1, weights.shape[-1])

#%%
# scores = []

# for k in range(2,11):

#     ward = AgglomerativeClustering(
#         n_clusters=k,
#         linkage="ward"
#     )

#     labels = ward.fit_predict(weights_flat)

#     score = silhouette_score(
#         weights_flat,
#         labels
#     )

#     scores.append(score)

# best_k = np.argmax(scores) + 2

#%%
Z = linkage(weights_flat, method="ward")

plt.figure(figsize=(12,6))

dendrogram(
    Z,
    truncate_mode="lastp",
    p=30,              # mostra últimos agrupamentos
    leaf_rotation=90,
    leaf_font_size=10,
    show_contracted=True
)

plt.xlabel("Clusters")
plt.ylabel("Ward Distance")
plt.title("Dendrograma - SOM + Ward")

plt.show()

#%%
ward = AgglomerativeClustering(
    n_clusters=3,
    linkage="ward"
)

neuron_clusters = ward.fit_predict(
    weights_flat
)

#%%

winner = np.array(
    [som.winner(x) for x in X_som]
)

winner_idx = np.ravel_multi_index(
    winner.T,
    (SOM_X_AXIS_NODES,
     SOM_Y_AXIS_NODES)
)

som_cluster = neuron_clusters[winner_idx]

#%%

cluster_df = pd.DataFrame({

    "h3_cell": shap_h3["h3_cell"],

    "som_cluster": som_cluster

})

#%%
risk_df = h3_agg.merge(
    cluster_df,
    on="h3_cell"
)
#%%

plt.rc('font', family='Times New Roman')

fig, ax = plt.subplots(figsize=(10,10))

bone()

# U-Matrix
pcolor(som.distance_map().T)

cbar = colorbar()
cbar.ax.tick_params(labelsize=15)

# neurônios SOM
for i in range(SOM_X_AXIS_NODES):
    for j in range(SOM_Y_AXIS_NODES):

        idx = np.ravel_multi_index(
            ([i], [j]),
            (SOM_X_AXIS_NODES, SOM_Y_AXIS_NODES)
        )[0]

        cluster = neuron_clusters[idx]

        ax.text(
            i + 0.5,
            j + 0.5,
            str(cluster),
            ha='center',
            va='center',
            fontsize=10,
            color='white'
        )

ax.set_xlabel("SOM X", fontsize=15)
ax.set_ylabel("SOM Y", fontsize=15)

plt.title("U-Matrix com clusters WARD")
plt.show()

##CONTINUAR


#%%%
# plt.figure(figsize=(10,10))

# bone()
# pcolor(som.distance_map().T)

# for i, w in enumerate(winner):

#     plt.scatter(
#         w[0] + 0.5,
#         w[1] + 0.5,
#         c=risk_df["pred"].iloc[i],
#         cmap="viridis",
#         s=30
#     )

# plt.colorbar(label="Perdas previstas")

# plt.title("SOM colorido pelo risco previsto")

# plt.show()

 
#%%

import seaborn as sns

plt.figure(figsize=(10,6))

sns.boxplot(
    data=risk_df,
    x="som_cluster",
    y="pred"
)

plt.xlabel("Cluster SOM-WARD")
plt.ylabel("Perdas previstas")

plt.title("Distribuição do risco por cluster")

plt.show()


#%%

risk_df.groupby("som_cluster")["pred"].agg([
    "count",
    "mean",
    "median",
    "std"
])

#%%
risk_df["som_cluster"].value_counts()

#%% verifica qual melhor escolha de k para ward

# scores = []

# for k in range(2, 11):

#     ward = AgglomerativeClustering(
#         n_clusters=k,
#         linkage="ward"
#     )

#     labels = ward.fit_predict(weights_flat)

#     score = silhouette_score(
#         weights_flat,
#         labels
#     )

#     scores.append(score)

#     print(f"K = {k} | Silhouette = {score:.4f}")

# #%%
# plt.figure(figsize=(8,5))

# plt.plot(
#     range(2,11),
#     scores,
#     marker='o'
# )

# plt.xlabel("Número de clusters (K)")
# plt.ylabel("Silhouette Score")
# plt.title("Silhouette Score - SOM + Ward")

# plt.grid(True)

# plt.show()


#%%

cluster_df = pd.DataFrame({
    "h3_cell": shap_h3["h3_cell"],
    "som_cluster": som_cluster
})

h3_final = h3_agg.merge(
    cluster_df,
    on="h3_cell"
)


#%%

perfil = (
    h3_final
    .groupby("som_cluster")[top_cols]
    .mean()
)

#%% padronizar para comparar

perfil_z = pd.DataFrame(
    StandardScaler().fit_transform(perfil),
    columns=perfil.columns,
    index=perfil.index
)

#%% Graficos
plt.figure(figsize=(12,6))

sns.heatmap(
    perfil_z,
    cmap="RdBu_r",
    center=0
)

plt.title("Perfis territoriais SOM-WARD")
plt.show()

#%%
h3_final.groupby("som_cluster")["pred"].agg([
    "mean",
    "median",
    "std"
])

#%%
h3_final["geometry"] = h3_final["h3_cell"].apply(cell_to_shapely)

gdf_clusters = gpd.GeoDataFrame(
    h3_final,
    geometry="geometry",
    crs="EPSG:4326"
)
#%%
cluster_colors = {
    0: "#1f78b4",  # azul
    1: "#e31a1c",  # vermelho
    2: "#33a02c"   # verde
}

import matplotlib.colors as mcolors

clusters = sorted(gdf_clusters["som_cluster"].unique())

cmap = plt.cm.Set1

cluster_colors = {
    c: mcolors.to_hex(cmap(i))
    for i, c in enumerate(clusters)
}

#%%
from folium import GeoJsonPopup
from folium.plugins import MousePosition

#%%
perfil_nome = {
    0: "Medium Risk Territory",
    1: "High Risk Territory",
    2: "Low Risk Territory"
}

gdf_clusters["perfil"] = (
    gdf_clusters["som_cluster"]
    .map(perfil_nome)
)

perfil_colors = {
    "High Risk Territory": "#d73027",
    "Medium Risk Territory": "#fc8d59",
    "Low Risk Territory": "#1a9850"
}

center = [
    gdf_clusters.geometry.centroid.y.mean(),
    gdf_clusters.geometry.centroid.x.mean()
]

m = folium.Map(
    location=center,
    zoom_start=13,
    tiles=None
)

folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Esri',
    name='Satellite',
    overlay=False,
    control=True
).add_to(m)

folium.TileLayer(
    'CartoDB positron',
    name='Street Map'
).add_to(m)

popup = GeoJsonPopup(
    fields=[
        "perfil",
        "pred",
        "Perdas_mean",
        "Perdas_std",
        "Perdas_count"
    ],
    aliases=[
        "Territorial Profile",
        "Predicted Losses",
        "Average Losses",
        "Std Dev",
        "Samples"
    ],
    localize=True
)

folium.GeoJson(
    gdf_clusters,
    name="Territorial Profiles",

    style_function=lambda feat: {

        "fillColor":
            perfil_colors[
                feat["properties"]["perfil"]
            ],

        "color":"white",

        "weight":0.3,

        "fillOpacity":0.65

    },

    popup=popup

).add_to(m)
    
legend_html = """
<div style="
position: fixed;
top: 180px;
right: 20px;
width: 270px;
background-color: white;
border-radius:10px;
box-shadow: 0 0 15px rgba(0,0,0,0.3);
padding:15px;
z-index:9999;
font-size:14px;
font-family:Arial;
">

<h4 style="margin-top:0;">
DSO Risk Profiles
</h4>

<p>
<b>Territorial Classification</b>
</p>

<div>
<span style="
background:#d73027;
width:18px;
height:18px;
display:inline-block;
margin-right:10px;">
</span>
High Risk Territory
</div>

<div>
<span style="
background:#fc8d59;
width:18px;
height:18px;
display:inline-block;
margin-right:10px;">
</span>
Medium Risk Territory
</div>

<div>
<span style="
background:#1a9850;
width:18px;
height:18px;
display:inline-block;
margin-right:10px;">
</span>
Low Risk Territory
</div>

<hr>

<p>
<b>Source:</b><br>
XGBoost + SHAP + SOM-WARD
</p>

</div>
"""

m.get_root().html.add_child(
    folium.Element(legend_html)
)    

title_html = """
<div style="
position: fixed;
top: 20px;
left: 50%;
transform: translateX(-50%);
background-color:white;
padding:10px 20px;
border-radius:8px;
box-shadow:0 0 15px rgba(0,0,0,0.2);
z-index:9999;
font-size:18px;
font-weight:bold;
font-family:Arial;
">

Electric Loss Risk Assessment
(H3 Spatial Profiles)

</div>
"""

m.get_root().html.add_child(
    folium.Element(title_html)
)
    
MousePosition().add_to(m)
folium.LayerControl().add_to(m)
m.save("DSO_Risk_Profiles.html")