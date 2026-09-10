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
    
dados = pd.read_excel(caminho_dados)

df = dados.head(20)

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
    cf_marker = folium.Marker(location=[lat,lng], popup=popup, icon = folium.Icon(color="red", icon="remove-sign"))
    cf_cluster.add_child(cf_marker)


map.save(outfile='BLANK_map.html')

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