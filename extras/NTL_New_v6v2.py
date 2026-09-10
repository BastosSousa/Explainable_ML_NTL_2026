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
plt.title('UMAP embedding of random colours');
plt.scatter(embedding[:, 0], embedding[:, 1], cmap='Spectral', s=5)
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

# df1['segment'] = df_umap_kmeans['segment']
# sns.pairplot(df1[0:], hue='segment')

#%%

df_umap_kmeans_2 = df_umap_kmeans[['w_PARECER2_TOTAL', 'pop_count_local_moran',
                                   'pop_count_local_moran_p_sim','h3_cell','segment_kmeans_umap','PARECER2_TOTAL','Perdas']]


#%%

df_umap_kmeans_2 = df_umap_kmeans_2.drop_duplicates(["h3_cell", "pop_count_local_moran_p_sim"])

df_umap_kmeans_2 = df_umap_kmeans_2.loc[df_umap_kmeans_2['pop_count_local_moran'].isin(['HH','LH'])]

df_umap_kmeans_2['geometry'] = df_umap_kmeans_2['h3_cell'].apply(lambda x: cell_to_shapely(x))
df_umap_kmeans_2 = gpd.GeoDataFrame(data=df_umap_kmeans_2, geometry='geometry', crs=4326)

#%%



mew = df_umap_kmeans_2.explore(
    column='pop_count_local_moran',
  
    cmap='autumn',
    width='90%',
    height='90%',
    categorical = True,


    style_kwds={'fillOpacity': 1.0, 'opacity': 1.0}
    )

tile = folium.TileLayer(
      tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      attr = 'Esri',
      name = 'Esri Satellite',
      overlay = False,
      control = True
      ).add_to(mew)

for _, r in h3_gdf.iterrows():
    # Without simplifying the representation of each borough,
    # the map might not be displayed
    sim_geo = gpd.GeoSeries(r["geometry"])
    geo_j = sim_geo.to_json()
    geo_j = folium.GeoJson(data=geo_j, style_function=lambda x: {"fillColor": "orange"},control=False)

    geo_j.add_to(mew)
    
# for i,row in df.iterrows():
#     lat = df.at[i, 'LATITUDE']  #latitude
#     lng = df.at[i, 'LONGITUDE']  #longitude
#     popup = '<br>'+'<a href="https://www.google.com/maps?layer=c&cbll=' + str(df.at[i, 'LATITUDE']) + ',' + str(df.at[i, 'LONGITUDE']) + '" target="blank">GOOGLE STREET VIEW</a>'
#     cf_marker = folium.Marker(location=[lat,lng], popup=popup, icon = folium.Icon(color="blue", icon="remove-sign"))
#     mew.add_child(cf_marker)
folium.LayerControl().add_to(mew)

mew.save(r'C:\Pos\Run-py-Figures\outfp.html')

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

SOM_X_AXIS_NODES  = 10
SOM_Y_AXIS_NODES  = 10
SOM_N_VARIABLES  = df2_data.shape[1]
NEIGHBORHOOD_FUNC = 'gaussian'
DISTANCE_FUNC = 'euclidean'

RANDOM_SEED = 0

#%%

som = MiniSom(
        SOM_X_AXIS_NODES,
        SOM_Y_AXIS_NODES,
        SOM_N_VARIABLES,
        sigma=1.5,
        learning_rate=0.5,
        neighborhood_function=NEIGHBORHOOD_FUNC,
        random_seed=RANDOM_SEED
        )

#%%
som.pca_weights_init(df2_data)
N_ITERATIONS = 5000
som.train_random(df2_data, N_ITERATIONS)  

#%%     Number of neurons sqrt(number-of-samples)*5 = M
#       X_axis = Y_axis = sqrt(M)

#%%

# Obtendo a matriz de distâncias do SOM
distance_map = som.distance_map()
# Plotando o mapa de calor
plt.figure(figsize=(10, 10))
plt.pcolor(distance_map, cmap='hot', alpha=0.8)
plt.colorbar()
plt.title('SOM heatmap')
plt.show()

#%%
plt.figure(figsize=(10,10))
plt.pcolor(som.distance_map().T, cmap= 'hot' )
plt.colorbar()


#%%
y = df2_label
#%%
# Visualizing the results

fig = plt.figure(figsize = (15,10))
ax = fig.add_subplot(111)
bone()
pcolor(som.distance_map().T)
colorbar()

for i,x in enumerate(df2_data): 
    w = som.winner(x)
    word = y[i]
    ax.plot((w[0],),(w[1],), 'x', c='r') 
    ax.annotate(word, (w[0]+0.2, w[1]+0.2), ha='center', va='center',size = 12,color='white')
plt.show()

 
#%%
som_shape = (SOM_X_AXIS_NODES, SOM_Y_AXIS_NODES)
# each neuron represents a cluster
winner_coordinates = np.array([som.winner(x) for x in df2_data]).T
# with np.ravel_multi_index we convert the bidimensional
# coordinates to a monodimensional index
cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)

#%%
# Plotting the clusters 
# plt.figure(figsize=(10,8))

# for c in np.unique(cluster_index):
#     plt.scatter(df2_data[cluster_index == c, 1],
#                 df2_data[cluster_index == c, 0], label='cluster='+str(c), alpha=.7)

#%%

n_kmc = KMeans(n_clusters = len(df2_label.value_counts()))

n_kmc.fit(df2_data)

cluster_vec = n_kmc.predict(df2_data)

#%%

cluster_df = pd.DataFrame(df2_data,columns =df2[['comp1','comp2']].columns)

#%%
cluster_df['clusters'] = cluster_vec

#%%
cluster_df['w_PARECER2_TOTAL'] = df2['w_PARECER2_TOTAL']
cluster_df['Perdas'] = df2['Perdas']

#%%

# plt.scatter(df2_data[cluster_vec==0, 0],df2_data[cluster_vec==0, 1], s=100, c='red', label ='Cluster 1')
# plt.scatter(df2_data[cluster_vec==1, 0], df2_data[cluster_vec==1, 1], s=100, c='blue', label ='Cluster 2')
# plt.scatter(df2_data[cluster_vec==2, 0], df2_data[cluster_vec==2, 1], s=100, c='green', label ='Cluster 3')
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
cbar.ax.tick_params(labelsize=18)  # set tick label size

# Axis ticks
ax.tick_params(axis='both', which='major', labelsize=18)

for i,x in enumerate(df2_data): 
    w = som.winner(x)
    word = y[i]
    ax.plot((w[0],),(w[1],), 'x', c='r') 
    ax.annotate(word, (w[0]+0.2, w[1]+0.2), ha='center', va='center',**csfont,size = 18,color='white')
    ax.set_ylabel("SOM_Y_AXIS",**csfont,fontsize=18)
    ax.set_xlabel("SOM_X_AXIS",**csfont,fontsize=18)
plt.show()

#%%
contagem_valores = cluster_df["clusters"].value_counts()

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

#df3_data_2["clusters"] = cluster_df.clusters

#df3_data_2['UNI_TR_MT'] = df2['UNI_TR_MT']

#%%
# contagem_trafos = df3_data_2["UNI_TR_MT"].value_counts()

# df3_trafos = df3_data_2.groupby(['h3_cell', 'UNI_TR_MT']).size()


#%%

#df4_data.boundary.plot(linewidth=1, edgecolor="black")

# df4_data.plot(column='Porc', legend=True, cmap='OrRd')

m = df3_data_2.explore(
    column="PARECER2_TOTAL",  # make choropleth based on "BoroName" column
    tooltip="clusters",  # show "BoroName" value in tooltip (on hover)
    popup=True,  # show all values in popup (on click)
    highlight=True,
    legend=True,
    cmap="hot",  

    style_kwds=dict(color="black"),
    legend_kwds=dict(colorbar=True) # use black outline
)

tile = folium.TileLayer(
      tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      attr = 'Esri',
      name = 'Esri Satellite',
      overlay = True,
      control = True
      ).add_to(m)

m.save(r'C:\Pos\Run-py-Figures\m.html')


#%%

# ax = df4_data.plot(figsize=(15, 15), column='cluster_index_som', cmap='Accent',legend = True)
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# ax.axis('on')

#%%
# contagem_valores = df4_data['cluster_index_som'].value_counts()

# # colors = {0: 'Bisque',11: 'Red',12: 'Orange',13: 'Yellow',4: 'Green',5: 'Blue',6: 'Indigo',7: 'Violet',8: 'Pink',9: 'Brown',10: 'Purple'}

# # color_list = [colors[group] for group in df_clust_pca_kmeans['cluster_index_som']]

# #%%
# df_clust_pca_kmeans['color'] = color_list

# df_clust_pca_kmeans["cluster_index_som"] = df_clust_pca_kmeans["cluster_index_som"].apply(
#     str)

#%%
h3_geoms2 = df_umap_kmeans['h3_cell'].apply(lambda x: cell_to_shapely(x))
h3_gdf2 = gpd.GeoDataFrame(data=df2,
                            geometry=h3_geoms2, crs=4326)

#%%

df3_data.to_csv(caminho_dados_final1, index=False, sep=';')


#%%
from sklearn.preprocessing import MinMaxScaler
#%%

scaler = MinMaxScaler()

cluster_df[["umap_1_scaled", "umap_2_scaled","Perdas_norm", "Parecer_norm"]] = scaler.fit_transform(cluster_df[['comp1','comp2',"Perdas", "w_PARECER2_TOTAL"]])

cluster_df['h3_cell'] = df2['h3_cell']
#%%

cluster_df.to_csv(caminho_dados_final3, index=False, sep=';')

#%%
h3_geoms = cluster_df['h3_cell'].apply(lambda x: cell_to_shapely(x))
cluster_df_2 = gpd.GeoDataFrame(data=cluster_df, geometry=h3_geoms, crs=4326)

cluster_df_2.to_csv(caminho_dados_final2, index=False, sep=';')

#%%

dadosUCs = pd.read_csv(caminho_dados2,sep=';')

#%%
dadosUCs2 = dadosUCs.rename(
    columns={'LATITUDE_x': 'LATITUDE', 'LONGITUDE_x':'LONGITUDE'})
#%%

dadosUCs2 = dadosUCs2[['Cod_setor', 'UNI_TR_MT', 'UC','Porc',
       'LATITUDE', 'LONGITUDE',]]

#%%
gdf_dadosUCs = gpd.GeoDataFrame(dadosUCs2, geometry=gpd.points_from_xy(dadosUCs2.LONGITUDE, dadosUCs2.LATITUDE), crs='EPSG:4326')

#%%
import osmnx as ox

from shapely.geometry import Point
from shapely.ops import unary_union

#%%

coords = dadosUCs2[['LATITUDE', 'LONGITUDE']].values
    
#%%
# Compute a center point and buffer distance (meters)
center = coords.mean(axis=0)
buffer = 7000  # meters

# Create the street network for driving
#G = ox.graph_from_point(center, dist=buffer, network_type='drive')

#%%
# Convert DataFrame points to list of (lat, lon)

points = [Point(xy) for xy in zip(dadosUCs2["LONGITUDE"], dadosUCs2["LATITUDE"])]

# Merge into a single MultiPoint geometry
multi_point = unary_union(points)

# Create a buffered polygon (in degrees, so small value)
polygon = multi_point.buffer(0.01)  # ≈ ~1 km buffer

G = ox.graph_from_polygon(polygon, network_type="walk")

#%%
# --- Plot network ---
fig, ax = ox.plot_graph(G, show=False, close=False)

# --- Plot your custom points on top ---
ax.scatter(
    coords[:,1],  # longitudes (x-axis)
    coords[:,0],  # latitudes (y-axis)
    color="red",
    s=50,
    zorder=5,
    label="Input points"
)

ax.legend()
plt.show()
#%%

#     # Create Folium map centered on points
# m = folium.Map(location=center.tolist(), zoom_start=13)

# # --- Compute and plot routes between consecutive points ---
# for i in range(len(df) - 1):
#     orig = (dadosUCs2.loc[i, "LATITUDE"], dadosUCs2.loc[i, "LONGITUDE"])
#     dest = (dadosUCs2.loc[i + 1, "LATITUDE"], dadosUCs2.loc[i + 1, "LONGITUDE"])

#     # Find nearest graph nodes
#     orig_node = ox.distance.nearest_nodes(G, orig[1], orig[0])
#     dest_node = ox.distance.nearest_nodes(G, dest[1], dest[0])

#     # Compute shortest path
#     route = ox.shortest_path(G, orig_node, dest_node, weight="length")

#     # Extract coordinates
#     route_coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in route]

#     # Add route line
#     folium.PolyLine(
#         route_coords,
#         color="blue",
#         weight=5,
#         opacity=0.8
       
#     ).add_to(m)

    
# m.save(r'C:\Pos\Run-py-Figures\mm.html')
