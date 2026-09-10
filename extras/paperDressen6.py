# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 15:32:07 2024

@author: Natalia Bostos
"""

import folium
import jinja2
import branca

from branca.element import MacroElement
from jinja2 import Template
import os
import platform
import pandas as pd
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
    volta_nivel+barra+'saidas'+barra+'dados_papaer.csv'


dados2 = pd.read_csv(caminho_dados2,sep=';')


#df = dados2.head(200)
df = dados2.copy()

#%%

df = df.rename(columns={'LONGITUDE_x': 'LONGITUDE','LATITUDE_x':'LATITUDE'})

df = df.drop(['LONGITUDE_y', 'LATITUDE_y'], axis=1)

#%%
map = folium.Map(location=[-27.687077,-48.553949],
                 tiles = 'OpenStreetMap',
                 zoom_start=11,
                 min_zoom=6,
                
                 max_bounds=True,
                 control_scale=True,
                 prefer_canvas=True
                 )

#%%
from folium.plugins import MarkerCluster

c=folium.FeatureGroup(name="UNI_TR_MT",overlay=True)
cf_cluster = MarkerCluster(name="UNI_TR_MT").add_to(map)


for i,row in df.iterrows():
    lat = df.at[i, 'LATITUDE']  #latitude
    lng = df.at[i, 'LONGITUDE']  #longitude
    popup = '<br>'+'<a href="https://www.google.com/maps?layer=c&cbll=' + str(df.at[i, 'LATITUDE']) + ',' + str(df.at[i, 'LONGITUDE']) + '" target="blank">GOOGLE STREET VIEW</a>'
    cf_marker = folium.Marker(location=[lat,lng], popup=popup, icon = folium.Icon(color="blue", icon="remove-sign"))
    cf_cluster.add_child(cf_marker)


map.save(outfile='BLANK_map.html')

#%%
import geopandas as gpd
from shapely.geometry import Point
import h3

#%%

gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.LONGITUDE, df.LATITUDE), crs='EPSG:4326')

#gdf = gdf[['w_PARECER2_TOTAL','LATITUDE','LONGITUDE']]
#%%

res = 10

def geo_to_h3(row):
  return h3.latlng_to_cell(lat=row.LATITUDE,lng=row.LONGITUDE,res = res)


gdf['h3_cell'] = gdf.apply(geo_to_h3,axis=1)

#%%

# gdf = gdf.drop(columns=['Cod_setor', 'UNI_TR_MT', 'UC', 'Bairro', 'CD_ALIMENTADOR','PARECER2',
# 'Nearest', 'Distance', 'ID','y_coords', 'x_coords',])

#%%

h3_df = gdf.groupby('h3_cell')['w_PARECER2_TOTAL'].describe().reset_index()

#%%

from shapely.geometry import Polygon

def cell_to_shapely(cell):
    coords = h3.cell_to_boundary(cell)
    flipped = tuple(coord[::-1] for coord in coords)
    return Polygon(flipped)

#%%
h3_geoms = h3_df['h3_cell'].apply(lambda x: cell_to_shapely(x))
h3_gdf = gpd.GeoDataFrame(data=h3_df, geometry=h3_geoms, crs=4326)

#%%
import matplotlib.pyplot as plt
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
from shapely.geometry import Polygon


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
import geopandas as gpd
import geodatasets
import folium
import matplotlib.pyplot as plt

#%%
# map = folium.Map(location=[ -27.5969, -48.5495], tiles="CartoDB Positron", zoom_start=9)

# # for _, r in h3_gdf.iterrows():
# #     # Without simplifying the representation of each borough,
# #     # the map might not be displayed
# #     sim_geo = gpd.GeoSeries(r["geometry"])
# #     geo_j = sim_geo.to_json()
# #     geo_j = folium.GeoJson(data=geo_j, style_function=lambda x: {"fillColor": "orange"})
# #     folium.Popup(r["count"]).add_to(geo_j)
# #     geo_j.add_to(map)

# folium.Choropleth(
#     geo_data=geojson_obj,
#     name="choropleth",
#     data= h3_gdf,
#     columns=["h3_cell", "count"],
#     key_on='feature.properties.h3_cell',
#     fill_color="YlGn",

#     legend_name="Energy Theft cases sum",
#     highlight = True
# ).add_to(map)

# highlights = folium.features.GeoJson(
#     geojson_obj,style_function= lambda x: {'color':'transparent', 'fillColor':'transparent', 'weight':0},
#     highlight_function = lambda x: {'fillColor': '#000000', 'color':'#000000', 'fillOpacity': 0.50, 'weight': 0.1},
#     tooltip=folium.features.GeoJsonTooltip(fields=[ 'h3_cell','count' ], 
#                                            aliases = [ 'h3_cell id: ', 'Sum of cases: '], 
#                                            labels = True,sticky = False))
# map.add_child(highlights)
# map.keep_in_front(highlights)

# for i,row in df.iterrows():
#     lat = df.at[i, 'LATITUDE']  #latitude
#     lng = df.at[i, 'LONGITUDE']  #longitude
#     popup = '<br>'+'<a href="https://www.google.com/maps?layer=c&cbll=' + str(df.at[i, 'LATITUDE']) + ',' + str(df.at[i, 'LONGITUDE']) + '" target="blank">GOOGLE STREET VIEW</a>'
#     cf_marker = folium.Marker(location=[lat,lng], popup=popup, icon = folium.Icon(color="blue", icon="remove-sign"))
#     map.add_child(cf_marker)

# folium.LayerControl().add_to(map)
# map.save("hexmap.html")

#%%

gdf_to_normalize = gdf.copy()
gdf_to_normalize.drop(columns = ['LATITUDE', 'LONGITUDE','geometry', 'pop_count_local_moran','h3_cell'], inplace = True)

#%%
from sklearn.preprocessing import Normalizer
normal_values = Normalizer().fit_transform(gdf_to_normalize.values)

#%%
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns

pca = PCA()
pca.fit(normal_values)
per_var = np.round(pca.explained_variance_ratio_*100, decimals = 1)

#%%
# plt.figure(figsize = (10,6))
# plt.plot(range(1, len(per_var)+1), per_var.cumsum(), marker = "o", linestyle = "--")
# plt.grid()
# plt.ylabel("Percentage Cumulative of Explained Variance")
# plt.xlabel("Number of Components")
# plt.title("Explained Variance by Component")
# plt.show()

#%%

pca = PCA(n_components = 5)
pca.fit(normal_values)

scores_pca = pca.transform(normal_values)

#%%
from sklearn.cluster import KMeans

WCSS = []

for i in range(1,30):
  kmeans_pca = KMeans(n_clusters = i, init = "k-means++", random_state = 0)
  kmeans_pca.fit(scores_pca)
  WCSS.append(kmeans_pca.inertia_)


#%%

plt.figure(figsize = (10,6))
plt.plot(range(1,30), WCSS, marker = "o", linestyle = "--")
plt.grid()
plt.title("Cluster using PCA Scores")
plt.ylabel("WCSS")
plt.xlabel("Number of Clusters")
plt.show()

#%%
kmeans_pca = KMeans(n_clusters = 3, init = "k-means++", random_state = 0)
kmeans_pca.fit(scores_pca)

#%%

# Concatening the original df with the components informations present in scores_pca
df_clust_pca_kmeans = pd.concat([gdf, pd.DataFrame(scores_pca)], axis = 1)

# Renaming the column label from each component
df_clust_pca_kmeans.columns.values[-5:] = ["comp1", "comp2", "comp3", "comp4", "comp5"]

# Seting the cluster label to each observation, using the atribute .labels_ 
df_clust_pca_kmeans["segment_kmeans_pca"] = kmeans_pca.labels_

# Mapping each cluster segmentation and renaming their labels 
df_clust_pca_kmeans["segment"] = df_clust_pca_kmeans["segment_kmeans_pca"].map({0:"Cluster 1", 1:"Cluster 2", 2:"Cluster 3"})

#%%
df1 = df_clust_pca_kmeans.iloc[ :, 98:103]

df1['segment'] = df_clust_pca_kmeans['segment']
sns.pairplot(df1[0:], hue='segment')

#%%

mew = df_clust_pca_kmeans.explore(
    column='segment_kmeans_pca',
    tiles="CartoDB Positron",
    cmap='tab10',
    width='90%',
    height='90%',
    categorical = True,


    style_kwds={'fillOpacity': 1.0, 'opacity': 0.0}
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
    geo_j = folium.GeoJson(data=geo_j, style_function=lambda x: {"fillColor": "orange"})

    geo_j.add_to(mew)
    
# for i,row in df.iterrows():
#     lat = df.at[i, 'LATITUDE']  #latitude
#     lng = df.at[i, 'LONGITUDE']  #longitude
#     popup = '<br>'+'<a href="https://www.google.com/maps?layer=c&cbll=' + str(df.at[i, 'LATITUDE']) + ',' + str(df.at[i, 'LONGITUDE']) + '" target="blank">GOOGLE STREET VIEW</a>'
#     cf_marker = folium.Marker(location=[lat,lng], popup=popup, icon = folium.Icon(color="blue", icon="remove-sign"))
#     mew.add_child(cf_marker)

mew.save(r'C:\Pos\Run-py-Figures\outfp.html')

#%%
df2 = df_clust_pca_kmeans.copy()

df2["target"] = df["pop_count_local_moran"]

contagem_valores = df2['target'].value_counts()

#%%

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
label = le.fit_transform(df2['target'])
df2.loc[:,'target'] = label

#%%
from minisom import MiniSom
#%%
df2_data = df2.iloc[:,103:108].values
df2_label = df2.target



contagem_valores = df2_label.value_counts()

#%%

SOM_X_AXIS_NODES  = 10
SOM_Y_AXIS_NODES  = 10
SOM_N_VARIABLES  = df2_data.shape[1]
ALPHA = 0.05
DECAY_FUNC = 'linear_decay_to_zero'
SIGMA_DECAY_FUNC = 'linear_decay_to_one'
NEIGHBORHOOD_FUNC = 'gaussian'
DISTANCE_FUNC = 'euclidean'
TOPOLOGY = 'hexagonal'
RANDOM_SEED = 0

#%%

som = MiniSom(
        SOM_X_AXIS_NODES,
        SOM_Y_AXIS_NODES,
        SOM_N_VARIABLES,

        learning_rate=ALPHA,
        neighborhood_function=NEIGHBORHOOD_FUNC,
        activation_distance=DISTANCE_FUNC,
        topology=TOPOLOGY,
        sigma_decay_function = SIGMA_DECAY_FUNC,
        decay_function = DECAY_FUNC,
        random_seed=RANDOM_SEED,
        )

#%%
som.pca_weights_init(df2_data)
N_ITERATIONS = 50000
som.train_random(df2_data, N_ITERATIONS)  

#%%
# import SimpSOM as sps

# #Build a network 20x20 with a weights format taken from the raw_data.
# net = sps.somNet(5, 5, df2_data)

# #Train the network for 10000 epochs and with initial learning rate of 0.1.
# net.train(5000, 0.01)

# #Save the weights to file
# net.save('filename_weights')

# #Print a map of the network nodes and colour them according to the first feature (column number 0) of the dataset
# #and then according to the distance between each node and its neighbours.
# net.nodes_graph(colnum=0)


# #Project the datapoints on the new 2D network map.
# #Cluster the datapoints according to the Mean Shift algorithm from sklearn.

# net.diff_graph()
# Obtendo a matriz de distâncias do SOM
distance_map = som.distance_map()
# Plotando o mapa de calor
plt.figure(figsize=(10, 10))
plt.pcolor(distance_map, cmap='hot', alpha=0.8)
plt.colorbar()
plt.title('Mapa de Calor do SOM')
plt.show()

#%%
plt.figure(figsize=(10,10))
plt.pcolor(som.distance_map().T, cmap= 'Blues' )
plt.colorbar()

#%%
y = df2_label

from pylab import bone, pcolor, colorbar, plot, show
# Visualizing the results
#%%
plt.figure(figsize = (15,10))
bone()
pcolor(som.distance_map().T)
colorbar()
# check if customers got approval or not, green - yes, red - no
markers = ['o','s'] # o/circle for red - no approval, s/square for green - approval
colors = ['r','g']
for i,x in enumerate(df2_data): # i becomes index, and x is a vector containing other attributes
    w = som.winner(x)
    plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]], markeredgecolor = colors[y[i]], markerfacecolor = 'None'
        , markersize = 10, markeredgewidth = 2)

#%%
som_shape = (SOM_X_AXIS_NODES, SOM_Y_AXIS_NODES)
# each neuron represents a cluster
winner_coordinates = np.array([som.winner(x) for x in df2_data]).T
# with np.ravel_multi_index we convert the bidimensional
# coordinates to a monodimensional index
cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)

#%%
df_clust_pca_kmeans["cluster_index_som"] = cluster_index

#%%
# plotting the clusters using the first 2 dimentions of the data
for c in np.unique(cluster_index):
    plt.scatter(df2_data[cluster_index == c, 0],
                df2_data[cluster_index == c, 1], label='cluster='+str(c), alpha=.7)
#%%

ax = df_clust_pca_kmeans.plot(figsize=(15, 15), column='cluster_index_som', cmap='Accent',legend = True)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
ax.axis('on')

#%%
contagem_valores = df_clust_pca_kmeans['cluster_index_som'].value_counts()

colors = {0: 'Bisque',11: 'Red',12: 'Orange',13: 'Yellow',4: 'Green',5: 'Blue',6: 'Indigo',7: 'Violet',8: 'Pink',9: 'Brown',10: 'Purple'}

color_list = [colors[group] for group in df_clust_pca_kmeans['cluster_index_som']]

#%%
df_clust_pca_kmeans['color'] = color_list

df_clust_pca_kmeans["cluster_index_som"]= df_clust_pca_kmeans["cluster_index_som"].apply(str)

#%%
h3_geoms2 = df_clust_pca_kmeans['h3_cell'].apply(lambda x: cell_to_shapely(x))
h3_gdf2 = gpd.GeoDataFrame(data=df_clust_pca_kmeans, geometry=h3_geoms2, crs=4326)

#%%

# cores = (h3_gdf2.groupby('h3_cell')
#                           .color
#                           .agg(list)
#                           .to_frame("cores")
#                           .reset_index())


#%%
grouped = h3_gdf2.groupby('h3_cell')['Porc'].mean()


grouped_list = [grouped[group] for group in h3_gdf2['h3_cell']]

h3_gdf2['AVG Porc'] = grouped_list

#%%
# df_clust_pca_kmeans.plot(color=df_clust_pca_kmeans['color'])  

# mew = df_clust_pca_kmeans.explore(
#     column='cluster_index_som',
#     tiles="CartoDB Positron",
#     cmap='tab10',
#     width='90%',
#     height='90%',
#     categorical = True,
#     #legend = 'segment',

#     style_kwds={'fillOpacity': 1.0, 'opacity': 0.4}
#     )




mew = folium.Map(location=[ -27.5969, -48.5495], tiles="CartoDB Positron", zoom_start=9)



for _, r in h3_gdf2.iterrows():
    # Without simplifying the representation of each borough,
    # the map might not be displayed
    sim_geo = gpd.GeoSeries(r["geometry"])
    geo_j = sim_geo.to_json()
    geo_j = folium.GeoJson(h3_gdf2,style_function=lambda feature: {
        "fillColor": "green"

        if "11" in feature["properties"]["cluster_index_som"].lower()
        else 'blue'
        if "12" in feature["properties"]["cluster_index_som"].lower()
        else "#ffff00",
        'fillOpacity': .5,
        "color": "black",
        "weight": 2,
        "dashArray": "5, 5"
        },
    popup = folium.GeoJsonPopup(
        fields=["h3_cell", "AVG Porc"],
        aliases=["h3_cell", "% Porc"],
        localize=True,
        labels=True,
        style="background-color: yellow;",
    ))

    geo_j.add_to(mew)


mew.save(r'C:\Pos\Run-py-Figures\outSOM2.html')

