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
  return h3.geo_to_h3(lat=row.LATITUDE,lng=row.LONGITUDE,resolution = res)


gdf['h3_cell'] = gdf.apply(geo_to_h3,axis=1)

#%%

h3_df = gdf.groupby('h3_cell')['w_PARECER2_TOTAL'].describe().reset_index()

#%%

# from shapely.geometry import Polygon

# def cell_to_shapely(cell):
#     coords = h3.h3_to_geo_boundary(cell)
#     flipped = tuple(coord[::-1] for coord in coords)
#     return Polygon(flipped)

#%%
# h3_geoms = h3_df['h3_cell'].apply(lambda x: cell_to_shapely(x))
# h3_gdf = gpd.GeoDataFrame(data=h3_df, geometry=h3_geoms, crs=4326)

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
  points = h3.h3_to_geo_boundary(row['h3_cell'], True)
  return Polygon(points)
#Apply function into our dataframe
parecer['geometry'] = (parecer
                                .apply(add_geometry,axis=1))


#%%
import folium
import matplotlib
from geojson import Feature
from geojson import Feature, FeatureCollection
import simplejson as json

#%%
def hexagons_dataframe_to_geojson(df_hex,file_output=None,column_name="count"):
    
    list_features = []
    for i,row in df_hex.iterrows():
  
            geometry_for_row = { "type" : "Polygon", "coordinates": [h3.h3_to_geo_boundary(h=row["h3_cell"],geo_json=True)]}
            feature = Feature(geometry = geometry_for_row , id=row["h3_cell"], properties = {column_name : row[column_name]})
            list_features.append(feature)

    feat_collection = FeatureCollection(list_features)


    geojson_result = json.dumps(feat_collection)
    
    return geojson_result

def get_color(custom_cm, val, vmin, vmax):
    return matplotlib.colors.to_hex(custom_cm((val-vmin)/(vmax-vmin)))

def choropleth_map(df_aggreg, column_name = "value", border_color = 'black', fill_opacity = 0.7, color_map_name = "Blues", initial_map = None):
    """
    Creates choropleth maps given the aggregated data. initial_map can be an existing map to draw on top of.
    """    
    #colormap
    min_value = df_aggreg[column_name].min()
    max_value = df_aggreg[column_name].max()
    mean_value = df_aggreg[column_name].mean()
    print(f"Colour column min value {min_value}, max value {max_value}, mean value {mean_value}")
    print(f"Hexagon cell count: {df_aggreg['h3_cell'].nunique()}")
    
    # the name of the layer just needs to be unique, put something silly there for now:
    name_layer = "Choropleth " + str(df_aggreg)
    
    if initial_map is None:
        initial_map = folium.Map(location= [47, 4], zoom_start=5.5, tiles="cartodbpositron")

    #create geojson data from dataframe
    geojson_data = hexagons_dataframe_to_geojson(df_hex = df_aggreg, column_name = column_name)

    # color_map_name 'Blues' for now, many more at https://matplotlib.org/stable/tutorials/colors/colormaps.html to choose from!
    custom_cm = matplotlib.cm.get_cmap(color_map_name)

    folium.GeoJson(
            geojson_data,
            style_function=lambda feature: {
                'fillColor': get_color(custom_cm, feature['properties'][column_name], vmin=min_value, vmax=max_value),
                'color': border_color,
                'weight': 1,
                'fillOpacity': fill_opacity 
            }, 
            name = name_layer
        ).add_to(initial_map)

    return initial_map
#%%
hexmap = choropleth_map(df_aggreg = parecer, color_map_name = "Oranges", column_name = "count")
hexmap.save("hexmap.html")

#%%
import plotly.express as px
import matplotlib.pyplot as plt
#%%

gdf_parecer = gpd.GeoDataFrame(parecer, geometry='geometry', crs='EPSG:4326')

geojson_obj = (hexagons_dataframe_to_geojson
                (gdf_parecer,
                 hex_id_field='h3_cell',
                 value_field='count',
                 geometry_field='geometry'))


#%%
fig = px.choropleth_mapbox(gdf_parecer, geojson=geojson_obj, locations='h3_cell', color='count',
                           color_continuous_scale="Viridis",
                           range_color=(0, 12),
                           mapbox_style="open-street-map",
                           zoom=3, 
                           center = {"lat": 37.0902, "lon": -95.7129},
                           opacity=0.5,
                           labels={'unemp':'unemployment rate'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()
#%%
parecer.plot(column='count', figsize=(10, 10))


#%%

# class ClickForOneMarker(folium.ClickForMarker):

#     _template = Template(u"""
#     {% macro script(this, kwargs) %}
#     var new_mark = L.marker();
#     function newMarker(e){
#     new_mark.setLatLng(e.latlng).addTo({{this._parent.get_name()}});
#     new_mark.dragging.enable();
#     new_mark.on('dblclick', function(e){ {{this._parent.get_name()}}.removeLayer(e.target)})
#     var lat = e.latlng.lat.toFixed(4),
#     lng = e.latlng.lng.toFixed(4);
#     new_mark.bindPopup("<a href="https://www.google.com/maps?layer=c&cbll=' + str(df.at[i, 'lat']) + ',' + str(df.at[i, 'lng']) + '" target="blank">GOOGLE STREET VIEW</a>");
#     parent.document.getElementById("latitude").value = lat;
#     parent.document.getElementById("longitude").value =lng;
#     };
#     {{this._parent.get_name()}}.on('click', newMarker);
#     {% endmacro %}
                            
# """)
    
#     def __init__(self, popup=None):
#         super(ClickForOneMarker, self).__init__(popup)
#         self._name = 'Google Street View'

# gsv = ClickForOneMarker()

# map.add_child(gsv)

# map.save(outfile='BLANK_map.html')