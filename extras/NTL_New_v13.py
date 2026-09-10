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

#%%
# from sklearn.decomposition import PCA
# pca = PCA()
# pca.fit(normal_values)
# per_var = np.round(pca.explained_variance_ratio_*100, decimals = 1)

# #%%
# plt.figure(figsize = (10,6))
# plt.plot(range(1, len(per_var)+1), per_var.cumsum(), marker = "o", linestyle = "--")
# plt.grid()
# plt.ylabel("Percentage Cumulative of Explained Variance")
# plt.xlabel("Number of Components")
# plt.title("Explained Variance by Component")
# plt.show()

#%%

# pca = PCA(n_components = 5)
# pca.fit(normal_values)

# scores_pca = pca.transform(normal_values)

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
# mew = df_umap_kmeans_2.explore(
#     column='pop_count_local_moran',
  
#     cmap='autumn',
#     width='90%',
#     height='90%',
#     categorical = True,


#     style_kwds={'fillOpacity': 1.0, 'opacity': 1.0}
#     )

# tile = folium.TileLayer(
#       tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
#       attr = 'Esri',
#       name = 'Esri Satellite',
#       overlay = False,
#       control = True
#       ).add_to(mew)

# for _, r in h3_gdf.iterrows():
#     # Without simplifying the representation of each borough,
#     # the map might not be displayed
#     sim_geo = gpd.GeoSeries(r["geometry"])
#     geo_j = sim_geo.to_json()
#     geo_j = folium.GeoJson(data=geo_j, style_function=lambda x: {"fillColor": "orange"},control=False)

#     geo_j.add_to(mew)
    
# # for i,row in df.iterrows():
# #     lat = df.at[i, 'LATITUDE']  #latitude
# #     lng = df.at[i, 'LONGITUDE']  #longitude
# #     popup = '<br>'+'<a href="https://www.google.com/maps?layer=c&cbll=' + str(df.at[i, 'LATITUDE']) + ',' + str(df.at[i, 'LONGITUDE']) + '" target="blank">GOOGLE STREET VIEW</a>'
# #     cf_marker = folium.Marker(location=[lat,lng], popup=popup, icon = folium.Icon(color="blue", icon="remove-sign"))
# #     mew.add_child(cf_marker)
# folium.LayerControl().add_to(mew)

# mew.save(r'C:\Pos\Run-py-Figures\outfp.html')

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

df3_data.drop(columns = ['LATITUDE', 'LONGITUDE','geometry','pop_count_local_moran','comp1', 'comp2',
                         'segment_kmeans_umap', 'segment', 'target','pop_count_local_moran_p_sim'], inplace = True)

#%%

SOM_X_AXIS_NODES  = 19
SOM_Y_AXIS_NODES  = 19
SOM_N_VARIABLES  = df2_data.shape[1]
NEIGHBORHOOD_FUNC = 'gaussian'
DISTANCE_FUNC = 'euclidean'


RANDOM_SEED = 0

#%%

som = MiniSom(
        SOM_X_AXIS_NODES,
        SOM_Y_AXIS_NODES,
        SOM_N_VARIABLES,
        sigma=2,
        learning_rate=0.5,
        neighborhood_function=NEIGHBORHOOD_FUNC,

        random_seed=RANDOM_SEED
        )

#%%
som.pca_weights_init(df2_data)
N_ITERATIONS = 500*SOM_X_AXIS_NODES*SOM_Y_AXIS_NODES
som.train_random(df2_data, N_ITERATIONS)  

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
y = df2_label
#%%
# Visualizing the results

# fig = plt.figure(figsize = (15,10))
# ax = fig.add_subplot(111)
# bone()
# pcolor(som.distance_map().T)
# colorbar()

# for i,x in enumerate(df2_data): 
#     w = som.winner(x)
#     word = y[i]
#     ax.plot((w[0],),(w[1],), 'x', c='r') 
#     ax.annotate(word, (w[0]+0.2, w[1]+0.2), ha='center', va='center',size = 12,color='white')
# plt.show()

weights = som.get_weights()  
# shape: (som_x, som_y, input_len)
weights_flat = weights.reshape(-1, weights.shape[-1])
# shape: (som_x * som_y, input_len)

n_clusters = k  # choose desired number of final clusters

agg = AgglomerativeClustering(
    n_clusters=n_clusters,
    linkage="ward"
)

neuron_labels = agg.fit_predict(weights_flat)

#%%

Z = linkage(weights_flat, method="ward")

plt.figure(figsize=(10, 5))
dendrogram(Z, no_labels=True)
#plt.title("Hierarchical clustering of SOM neurons (Ward)")
plt.xlabel("Neuron index",**csfont,fontsize=15)
plt.ylabel("Distance",**csfont,fontsize=15)
plt.show()

 
#%%
som_shape = (SOM_X_AXIS_NODES, SOM_Y_AXIS_NODES)
# each neuron represents a cluster
winner_coordinates = np.array([som.winner(x) for x in df2_data]).T
# with np.ravel_multi_index we convert the bidimensional
# coordinates to a monodimensional index
cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)
#%%

df2["som_cluster"] = cluster_index

weights = som.get_weights().reshape(-1, 2)

Z = linkage(weights, method="ward")
neuron_clusters = fcluster(Z, t=4, criterion="maxclust")

df2["som_cluster"] = neuron_clusters[
    np.ravel_multi_index(winner_coordinates, som_shape)
]

#%%
contagem_valores = df2["som_cluster"].value_counts()

#%%
# plt.figure(figsize=(8, 6))
# sc = plt.scatter(
#     df2["comp1"],
#     df2["comp2"],
#     c=df2["som_cluster"],
#     cmap="tab20",
#     s=10
# )
# plt.colorbar(sc, label="SOM cluster index")
# #plt.title("UMAP projection colored by SOM clusters")
# plt.xlabel("comp1",**csfont,fontsize=15)
# plt.ylabel("comp2",**csfont,fontsize=15)
# plt.show()

#%%

# Dict to map local moran's classification codes
df2_clusters = {1: 'Cluster 1', 2: 'Cluster 2', 3: 'Cluster 3', 4: 'Cluster 4'}

# Mapping local moran's classification codes
df2['Clusters'] = df2['som_cluster'].map(df2_clusters)


#%%
ax = plt.figure(figsize=(11,9))
ax = sns.scatterplot(data=df2, x = 'comp1', y ='comp2',
                     hue='Clusters',
    s=100
)

plt.legend(loc='upper right')
plt.xlabel("comp1",**csfont,fontsize=15)
plt.ylabel("comp2",**csfont,fontsize=15)
plt.show()

# n_kmc = KMeans(n_clusters = len(df2_label.value_counts()))

# n_kmc.fit(df2_data)

# cluster_vec = n_kmc.predict(df2_data)

#%%

cluster_df = pd.DataFrame(df2_data,columns =df2[['comp1','comp2']].columns)

#%%
cluster_df['clusters'] = df2["som_cluster"].values

#%%

# plt.scatter(df2_data[cluster_vec==0, 0],df2_data[cluster_vec==0, 1], s=100, c='red', label ='Cluster 1')
# plt.scatter(df2_data[cluster_vec==1, 0], df2_data[cluster_vec==1, 1], s=100, c='blue', label ='Cluster 2')
# plt.scatter(df2_data[cluster_vec==2, 0], df2_data[cluster_vec==2, 1], s=100, c='green', label ='Cluster 3')
# plt.scatter(df2_data[cluster_vec==3, 0], df2_data[cluster_vec==3, 1], s=100, c='orange', label ='Cluster 4')
# plt.legend()
# plt.show()

#%%

y = cluster_df.clusters

#%%

davies_bouldin_score(df2_data, y)

#%%
csfont = {'fontname':'Times New Roman'}
plt.rc('font',family='Times New Roman')
# Visualizing the results

fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot(111)
bone()
pcolor(som.distance_map().T)

# Colorbar with custom font size
cbar = colorbar()
cbar.ax.tick_params(labelsize=15)  # set tick label size

# Axis ticks
ax.tick_params(axis='both', which='major', labelsize=15)

for i,x in enumerate(df2_data): 
    w = som.winner(x)
    word = y[i]
    ax.plot((w[0],),(w[1],), 'x', c='r') 
    ax.annotate(word, (w[0]+0.2, w[1]+0.2), ha='center', va='center',**csfont,size = 12,color='white')
    ax.set_ylabel("SOM_Y_AXIS",**csfont,fontsize=15)
    ax.set_xlabel("SOM_X_AXIS",**csfont,fontsize=15)
plt.show()

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

df3_data["clusters"] = cluster_df.clusters

df3_data_2 = df3_data[['PARECER2_TOTAL','h3_cell', 'ivs','idhm' , 'Perdas','geometry','Porc','PO_NOMINAL_TRAFO','clusters']]


#%% XGBoost

model_df = df3_data.copy()

target = "Perdas"

drop_cols = [
    "geometry",
    "h3_cell",
    'som_cluster',
    'clusters',
    "Clusters"  # NÃO usar cluster como feature (leakage)
]

model_df = model_df.drop(columns=[c for c in drop_cols if c in model_df.columns])

model_df = model_df.dropna(subset=[target])

X = model_df.drop(columns=[target])
y = model_df[target]

# manter apenas numérico
X = X.select_dtypes(include=[np.number])

h3_index = df3_data_2.loc[X.index, "h3_cell"]
#%% SPLIT CORRETO (SEM LEAKAGE ESPACIAL)

gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)

groups = df3_data_2.loc[X.index, "h3_cell"]

train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

h3_train = h3_index.iloc[train_idx]
h3_test = h3_index.iloc[test_idx]

#%%  MODELO XGBOOST

xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R2:", r2)
#%% Selecao de variaveis

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

shap_df = pd.DataFrame(shap_values, columns=X_test.columns)

# importância global
importance = shap_df.abs().mean().sort_values(ascending=False)

top_features = importance.head(15).index.tolist()

print("Top features:", top_features)

#%% reducao de dados  (UMAP + SOM)

X_selected_train = X_train[top_features]
X_selected_test = X_test[top_features]

#%%

reducer = umap.UMAP(random_state=42)

embedding_train = reducer.fit_transform(X_selected_train)
embedding_test = reducer.transform(X_selected_test)

#%%

scaler = MinMaxScaler()
X_som = scaler.fit_transform(X_selected_train)
X_som_teste = scaler.fit_transform(X_selected_test)


SOM_X_AXIS_NODES = 15
SOM_Y_AXIS_NODES = 15

som = MiniSom(
    x=15,
    y=15,
    input_len=X_som.shape[1],
    sigma=2,
    learning_rate=0.5,
    random_seed=42
)

som.pca_weights_init(X_som)
som.train_random(X_som, 10000)

#%%

weights = som.get_weights()
weights_flat = weights.reshape(-1, weights.shape[-1])

Z = linkage(weights_flat, method="ward")

ward = AgglomerativeClustering(
    n_clusters=5,
    linkage="ward"
)

neuron_clusters = ward.fit_predict(weights_flat)
#%%%
winner_train = np.array([som.winner(x) for x in X_som])

winner_idx_train = np.ravel_multi_index(
    winner_train.T,
    (SOM_X_AXIS_NODES, SOM_Y_AXIS_NODES)
)

som_train_clusters = neuron_clusters[winner_idx_train]

#%%

winner_test = np.array([som.winner(x) for x in X_som_teste])

winner_idx_test = np.ravel_multi_index(
    winner_test.T,
    (SOM_X_AXIS_NODES, SOM_Y_AXIS_NODES)
)

som_test_clusters = neuron_clusters[winner_idx_test]

#%%
som_train_df = pd.DataFrame({
    "h3_cell": h3_train.values,
    "Perdas": y_train.values,
    "som_cluster": som_train_clusters
})

#%%

som_test_df = pd.DataFrame({
    "h3_cell": h3_test.values,
    "Perdas": y_test.values,
    "som_cluster": som_test_clusters
})

#%%

som_full_df = pd.concat([som_train_df, som_test_df], axis=0)

som_full_df = som_full_df.groupby("h3_cell").agg({
    "som_cluster": "first"
}).reset_index()

#%%
df_h3 = df3_data_2.groupby("h3_cell").mean(numeric_only=True).reset_index()
#%%

df3_data_3 = df_h3.join(
    som_full_df.set_index("h3_cell"),
    on="h3_cell"
)

##CONTINUAR


#%%%




 
#%%

# # previsão do modelo
# df3_data_3["perdas_pred"] = xgb_model.predict(X_train)

# # risco médio por célula H3
# h3_risk = df3_data_3.groupby("h3_cell").agg({
#     "perdas_pred": "mean",
#     "som_cluster": lambda x: x.mode()[0]
# }).reset_index()

# #%%

# fuzzy_df = h3_risk.copy()

# scaler = MinMaxScaler()

# fuzzy_df[["risk_norm"]] = scaler.fit_transform(fuzzy_df[["perdas_pred"]])

# #%%%

# fig, ax = plt.subplots(figsize=(8,6))

# sc = ax.scatter(
#     embedding_train[:,0],
#     embedding_train[:,1],
#     c=y_train,
#     cmap="RdYlBu"
# )

# # legenda contínua
# cbar = plt.colorbar(sc, ax=ax)
# cbar.set_label("Losses", fontsize=12)

# ax.set_title("UMAP - Risk Structure (Exploratory)")
# ax.set_xlabel("UMAP 1")
# ax.set_ylabel("UMAP 2")

# plt.tight_layout()
# plt.show()

# #%%
# # dataset final para fuzzy system
# fuzzy_input = fuzzy_df[[
#     "h3_cell",
#     "risk_norm",
#     "som_cluster"
# ]]

# print(fuzzy_input.head())

# #%% SHAP + TRUE GEOGRAPHICALLY WEIGHTED SHAP (GW-SHAP)

# explainer = shap.TreeExplainer(xgb_model)
# shap_values = explainer.shap_values(X_test)

# shap_df = pd.DataFrame(shap_values, columns=X_test.columns).reset_index(drop=True)

# # adicionar h3 correto do TESTE
# shap_df["h3_cell"] = df3_data_2.loc[X_test.index, "h3_cell"].values
# #%%%

# m_som = df3_data.explore(
#     column="clusters",
#     categorical=True,
#     cmap="Set1",
#     legend=True,
#     tooltip=[
#         "clusters",
#         "Perdas",
#         "ivs",
#         "idhm"
#     ],
#     popup=True,
#     highlight=True,
#     style_kwds={
#         "color": "black",
#         "weight": 0.5
#     }
# )

# folium.TileLayer(
#     "CartoDB positron",
#     name="Base Map"
# ).add_to(m_som)

# folium.LayerControl().add_to(m_som)

# m_som.save(
#     r"C:\Pos\Run-py-Figures\SOM_clusters.html"
# )

# m_som

#%%


#%%












