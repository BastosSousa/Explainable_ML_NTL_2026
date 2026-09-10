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
caminho_dados = diretorio_atual+barra + \
    volta_nivel+barra+'entradas'+barra+'Mappe1.xlsx'

caminho_dados2 = diretorio_atual+barra + \
    volta_nivel+barra+'saidas'+barra+'dados_paper.csv'

dados = pd.read_excel(caminho_dados)
dados2 = pd.read_csv(caminho_dados2,sep=';')


#df = dados2.head(200)
df = dados2.copy()

#%%

df = df.rename(columns={'x_coords': 'LONGITUDE','y_coords':'LATITUDE'})

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

gdf = gdf[['w_PARECER2_TOTAL','LATITUDE','LONGITUDE']]
#%%

res = 10

def geo_to_h3(row):
  return h3.latlng_to_cell(lat=row.LATITUDE,lng=row.LONGITUDE,res = res)


gdf['h3_cell'] = gdf.apply(geo_to_h3,axis=1)



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

map = folium.Map(location=[ -27.5969, -48.5495], tiles="CartoDB Positron", zoom_start=9)

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



folium.LayerControl().add_to(map)
map.save("hexmap.html")

#%%









