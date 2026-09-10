# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 15:27:25 2024

@author: Natalia Bostos
"""

from mpl_toolkits.mplot3d import Axes3D

# from statsmodels.formula.api import logit
from scipy import stats
from folium.plugins import HeatMap
import folium
import matplotlib.backends.backend_pdf
from scipy.spatial import distance_matrix
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import geopandas as gpd
#import rasterio
#from rasterio.plot import show
import matplotlib.pyplot as plt
import time
from datetime import datetime
#import pyproj
#from shapely.geometry import box
#import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
#from sklearn import metrics
#from scipy.spatial.distance import cdist
import os
import platform
#from scipy.interpolate import interp2d
from scipy.interpolate import griddata
#from sklearn.cluster import KMeans
#import shapely.geometry
from shapely.geometry import Point
import seaborn as sns
import warnings
#%%
import pysal.lib
from pysal.lib import weights
import pysal.model
from esda.moran import Moran, Moran_Local
import esda.moran
import splot.esda
from splot.esda import moran_scatterplot, plot_local_autocorrelation, lisa_cluster
from libpysal.weights import Rook

#%%
import geodatasets
import rasterio
import pyproj
from rasterio.plot import show

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

# caminho_dados_CELESC = diretorio_atual+barra+volta_nivel + \
#     barra+'entradas'+barra+'DADOSCELESC.xlsx'

caminho_dados = diretorio_atual+barra+volta_nivel + \
    barra+'entradas'+barra+'DADOSTAPERA.xlsx'

caminho_dados_todos = diretorio_atual+barra+volta_nivel+barra+'entradas'+barra+'DADOSCELESC.xlsx'

caminho_dados_inspe = diretorio_atual+barra+volta_nivel+barra + \
    'entradas'+barra+'Fiscalizações Florianópolis 01012019 - 21062023.xlsx'

caminho_gdf_quadras = diretorio_atual+barra+volta_nivel+barra + \
    'entradas'+barra+'shapefileFLOR'+barra+'patrimonio_territorial.shp'

caminho_dados_ivsUDH = diretorio_atual+barra+volta_nivel+barra + \
    'entradas'+barra+'atlasivs_dadosbrutos_Florianopolis.xlsx'

caminho_dados_ivs = diretorio_atual+barra+volta_nivel+barra+'entradas' + \
    barra+'Shapes_RM_Florianópolis'+barra+'UDHs_RM_Florianópolis.shp'

caminho_dados_ibge = diretorio_atual+barra+volta_nivel+barra + \
    'entradas'+barra+'Mapas_IBGE'+barra+'42SEE250GC_SIR.shp'

caminho_dados_ibge_dom1 = diretorio_atual+barra + \
    volta_nivel+barra+'entradas'+barra+'dados_comp.xlsm'

caminho_dados_ibge_esco = diretorio_atual+barra + \
    volta_nivel+barra+'entradas'+barra+'dados_escolhidos.xlsx'

caminho_dados_eletricos = diretorio_atual+barra + \
    volta_nivel+barra+'entradas'+barra+'nova_tabela.xlsx'

caminho_fisca = diretorio_atual+barra+volta_nivel + \
    barra+'entradas'+barra+'fiscalizacoes.xlsx'

# %% caminho saidas

caminho_ISL07_INSPECOES = diretorio_atual+barra + \
    volta_nivel+barra+'saidas'+barra+'ISL07-INSPECOES.csv'

caminho_UCS_VARIAVEIS = diretorio_atual+barra+volta_nivel + \
    barra+'saidas'+barra+'ISL07-UCS-VARIAVEIS.csv'

caminho_x = diretorio_atual+barra+volta_nivel + \
    barra+'saidas'+barra+'Pontos-interpolados-idw.csv'
    
caminho_dados_artigo = diretorio_atual+barra + \
    volta_nivel+barra+'saidas'+barra+'dados_papaer.csv'

# %%

dados = pd.read_excel(caminho_dados)

# dados_celesc = pd.read_excel(caminho_dados_CELESC)

dados_todos = pd.read_excel(caminho_dados_todos)

dados_inspe = pd.read_excel(caminho_dados_inspe)

dados_escolhidos = pd.read_excel(caminho_dados_ibge_esco)

dados_ivsUDH = pd.read_excel(caminho_dados_ivsUDH, sheet_name='UDH')

# tabela_final = pd.read_csv(caminho_tabela_final,sep=',')

dados_ibge_dom1 = pd.read_excel(caminho_dados_ibge_dom1, sheet_name='dom1')

dados_eletricos = pd.read_excel(caminho_dados_eletricos)

dados_fisca = pd.read_excel(caminho_fisca)

#%%
crs = {'init': 'epsg:4326'}

gdf_quadras = gpd.read_file(caminho_gdf_quadras)
type(gdf_quadras)
gdf_quadras.crs

dados_ivs = gpd.read_file(caminho_dados_ivs)
type(dados_ivs)
dados_ivs.crs

dados_ibge = gpd.read_file(caminho_dados_ibge)
type(dados_ibge)
dados_ibge.crs

# %%
dados_eletricos = dados_eletricos.rename(
    columns={'Transformador': 'UNI_TR_MT'})

# %%
# Nome da coluna que você deseja analisar
nome_coluna = 'UNI_TR_MT' 

cod_alimentador = ['ISL07']
nm_BAIRRO = ['CANASVIEIRAS - FNS']

#dados_todos_2 = dados_todos[dados_todos['CD_ALIMENTADOR'].isin(cod_alimentador)]

dados_todos_2 = dados_todos[dados_todos['BAIRRO'].isin(nm_BAIRRO)]

#dados_todos_2 = dados_todos.copy()

BAIRROs = dados_todos_2.BAIRRO.drop_duplicates()

#%%
dados_todos_2 = dados_todos_2[['UC','TIPO','MEDIDOR','MODELO','ANO', 'ENDERECO',	'COMPLEMENTO','BAIRRO',	'CIDADE',	'REGIONAL',	'GR_TENS_FAT',	
                               'GR_TENS_FORN',	'TIPO_FASE',	'SITUACAO',	'DISJUNTOR','UNI_TR_MT',	'POT_NOMINAL',	'CD_ALIMENTADOR',	'LATITUDE',	'LONGITUDE']]

#%%
dados_todos_2 = pd.merge(dados_todos_2, dados_eletricos, how='left', on='UNI_TR_MT')

#%%
le = LabelEncoder()
label2 = le.fit_transform(dados_todos_2['ENDERECO'])
#dados_todos_2.drop('Endereço', axis=1, inplace=True)
dados_todos_2.loc[:,'ENDERECO'] = label2

#%%

dados_todos_3 = dados_todos_2.reset_index()

contagem_valores2 = dados_todos_3['ENDERECO'].value_counts().reset_index()

#contagem_valores2 = contagem_valores2.rename(columns={'Endereco':'UCs'})

#%%
soma = contagem_valores2.head(79)
# somaUcs = soma.UCs.sum()
#%%
dados_todos_4 = dados_todos_3.copy()

#%%

dados_ivsUDH = dados_ivsUDH[dados_ivsUDH['ano']==2010]


dados_ivsUDH = dados_ivsUDH.rename(columns={'UDH':'UDH_ATLAS'})


dados_ivsUDH_2 = dados_ivsUDH[[ 'UDH_ATLAS','ano','ivs','ivs_infraestrutura_urbana','ivs_capital_humano','ivs_renda_e_trabalho', 'idhm','vulner_dia','t_eletrica','t_empregador_18m']]

dados_ivsUDH_2.loc[:,'UDH_ATLAS']=dados_ivsUDH_2['UDH_ATLAS'].astype(str)

#%%
dados_ibge_3 = dados_ibge.copy()


dados_ibge_3['geometry'] = dados_ibge_3.geometry.to_crs(4326)

#%%
dados_ibge_3['centroid'] =  dados_ibge_3['geometry'].centroid

#%%

gdf_dados_todos_4 = gpd.GeoDataFrame(dados_todos_4,geometry=gpd.points_from_xy(dados_todos_4.LONGITUDE, dados_todos_4.LATITUDE))

dados_ibge_3.crs = gdf_dados_todos_4.crs

join_left_df = gdf_dados_todos_4.sjoin(dados_ibge_3, how="left")


#%%
dados_ivs.crs = dados_ibge.crs

pointdf = dados_ibge[['CD_GEOCODI','geometry']]

pointdf['centroid'] = pointdf['geometry'].centroid

join_ivs_ibge = pointdf.sjoin(dados_ivs, how="left")

join_ivs_ibge = join_ivs_ibge.rename(columns={'CD_GEOCODI':'Cod_setor'})

join_ivs_ibge.dropna(inplace=True)

#%%

join_ivs = pd.merge(join_ivs_ibge, dados_ivsUDH, how='left', on='UDH_ATLAS')

#%%
dados_ibge_dom1['Cod_setor']=dados_ibge_dom1['Cod_setor'].astype(str)
#%%

join_ibge = pd.merge(join_ivs_ibge, dados_ibge_dom1, how='left', on='Cod_setor')
#%%
#print(join_ibge.columns.tolist())

join_ivs_2 = join_ivs[['Cod_setor','UDH_ATLAS','ivs', 'ivs_infraestrutura_urbana', 'ivs_capital_humano', 'ivs_renda_e_trabalho', 'idhm', 
                       'idhm_long', 'idhm_educ', 'idhm_renda', 'idhm_educ_sub_esc', 'idhm_educ_sub_freq', 
                       't_sem_agua_esgoto', 't_sem_lixo', 't_vulner_mais1h', 't_mort1', 't_c0a5_fora', 't_c6a14_fora', 
                       't_m10a17_filho', 't_mchefe_fundin_fmenor', 't_analf_15m', 't_cdom_fundin', 't_p15a24_nada', 't_vulner', 
                       't_desocup18m', 't_p18m_fundin_informal', 't_vulner_depende_idosos', 't_atividade10a14', 'espvida', 
                       't_pop18m_fundc', 't_pop5a6_escola', 't_pop11a13_ffun', 't_pop15a17_fundc', 't_pop18a20_medioc', 
                       'renda_per_capita', 'populacao', 't_fmor5', 't_razdep', 't_fectot', 't_env', 'mchefe_fmenor', 
                       'pop0a1', 'pop1a3', 'pop4', 'pop5', 'pop6', 'pop6a10', 'pop6a17', 'pop11a13', 
                       'pop11a14', 'pop12a14', 'pop15m', 'pop15a17', 'pop15a24', 'pop16a18', 'pop18m', 'pop18a20', 'pop18a24', 
                       'pop19a21', 'pop25m', 'pop65m', 'pea10a14', 'pea15a17', 'pea18m', 't_eletrica', 't_densidadem2', 
                       't_analf_18m', 't_analf_25m', 'rdpc_def_vulner', 't_renda_trab', 'i_gini', 't_carteira_18m', 't_scarteira_18m', 
                       't_setorpublico_18m', 't_contapropria_18m', 't_empregador_18m', 't_formal_18m', 't_fundc_ocup18m', 
                       't_medioc_ocup18m', 't_supec_ocup18m', 't_renda_todos_trabalhos', 't_nremunerado_18m']]


#%%
join_ibge = join_ibge[['Cod_setor','UDH_ATLAS','aguageral', 'aguadepoço', 'aguaoutras', 'banheiroexclusivo', 'ban_esgoto', 
                       'ban_fossa', 'ban_fossarudi', 'ban_vala', 'ban_riomar', 'ban_outros', 'semban', 'coletalixo', 
                       'comEE', 'semEE', 'comEE_semmedidor']]

#%%
join_ibge = join_ibge.replace('X', np.nan)

join_ibge.dropna(inplace=True)

#%%
join_ivs_2.dropna(inplace=True)

#%%
# join_ivs_2.to_excel(r'C:/celesc/Inputs/join_ivs_2.xlsx', sheet_name='Planilha1', index=False)

# join_ibge.to_excel(r'C:/celesc/Inputs/join_ibge.xlsx', sheet_name='Planilha1', index=False)

#%%
dados_todos_5 = join_left_df.copy()

#%%
dados_todos_6 = dados_todos_5[[ 'CD_GEOCODI','UC','MEDIDOR','BAIRRO','CD_ALIMENTADOR', 'LATITUDE', 
                               'LONGITUDE','centroid','UNI_TR_MT','PO_NOMINAL_TRAFO', 'Dif', 'Porc']]

dados_todos_6 = dados_todos_6.rename(columns={'CD_GEOCODI':'Cod_setor'})

dados_todos_6['Cod_setor']=dados_todos_6['Cod_setor'].astype(str)

#%%
dados_escolhidos['Cod_setor']=dados_escolhidos['Cod_setor'].astype(str)

#%%
dados_todos_7 = pd.merge(dados_todos_6, dados_escolhidos, how='left', on='Cod_setor')

#%%

dados_ivs['centroid'] = dados_ivs['geometry'].centroid

#%%
gdf_dados_todos_7 = gpd.GeoDataFrame(dados_todos_7,geometry=gpd.points_from_xy(dados_todos_7.LONGITUDE, dados_todos_7.LATITUDE))

gdf_dados_todos_7 = gdf_dados_todos_7.drop(['centroid'], axis=1)

#%%
dados_ivs.crs = gdf_dados_todos_7.crs

join_left_df = gdf_dados_todos_7.sjoin(dados_ivs, how="left")

#%%
join_left_df = join_left_df.drop(['index_right'], axis=1)

#%%
join_left_df_novo = join_left_df.sjoin(dados_ivs,how="left")

#%%
join_left_df_novo = join_left_df_novo[['Cod_setor','UDH_ATLAS_left','UNI_TR_MT','CD_ALIMENTADOR','UC', 'MEDIDOR', 'BAIRRO','Situacao_setor', 
                      'domicilio','medio_pessoas', 'rend_responsavel', 'rend_medio', 'saneamento',
                      '%alfabetizados', '%ate5salarios', '%5a10', 'geometry', 'centroid_right','LATITUDE','LONGITUDE','PO_NOMINAL_TRAFO', 'Dif',
                      'Porc']]

#%%
join_left_df_novo = join_left_df_novo.rename(columns={'UDH_ATLAS_left':'UDH_ATLAS','centroid_right':'centroid','PO_NOMINAL_TRAFO':'PO_NOMINAL_TRAFO' })

#%%
join_left_df_novo = pd.merge(join_left_df_novo, dados_ivsUDH, how='left', on='UDH_ATLAS')

#%%
join_left_df_novo = join_left_df_novo[['Cod_setor', 'UDH_ATLAS', 'UNI_TR_MT', 'CD_ALIMENTADOR','UC',
       'MEDIDOR', 'BAIRRO','Situacao_setor','PO_NOMINAL_TRAFO', 'Dif',
       'Porc','domicilio', 'medio_pessoas', 'rend_responsavel', 'rend_medio',
       'saneamento', '%alfabetizados', '%ate5salarios', '%5a10','ivs', 'ivs_infraestrutura_urbana','ivs_capital_humano', 'ivs_renda_e_trabalho', 'idhm', 'idhm_long',
       'idhm_educ', 'idhm_renda', 'idhm_educ_sub_esc','idhm_educ_sub_freq', 'prosp_soc', 't_sem_agua_esgoto',
       't_sem_lixo', 't_vulner_mais1h', 't_mort1', 't_c0a5_fora','t_c6a14_fora', 't_m10a17_filho', 't_mchefe_fundin_fmenor',
       't_analf_15m', 't_cdom_fundin', 't_p15a24_nada', 't_vulner','t_desocup18m', 't_p18m_fundin_informal',
       't_vulner_depende_idosos', 't_atividade10a14', 'espvida','t_pop18m_fundc', 't_pop5a6_escola', 't_pop11a13_ffun',
       't_pop15a17_fundc', 't_pop18a20_medioc', 'renda_per_capita','populacao', 't_fmor5', 't_razdep', 't_fectot', 't_env',
       'vulner15a24', 'mchefe_fmenor', 'vulner_dia', 'dom_vulner_idoso','pop0a1', 'pop1a3', 'pop4', 'pop5', 'pop6', 'pop6a10', 'pop6a17',
       'pop11a13', 'pop11a14', 'pop12a14', 'pop15m', 'pop15a17','pop15a24', 'pop16a18', 'pop18m', 'pop18a20', 'pop18a24',
       'pop19a21', 'pop25m', 'pop65m', 'pea10m', 'pea10a14', 'pea15a17','pea18m', 't_eletrica', 't_densidadem2', 't_analf_18m',
       't_analf_25m', 'rdpc_def_vulner', 't_renda_trab', 'i_gini','t_carteira_18m', 't_scarteira_18m', 't_setorpublico_18m',
       't_contapropria_18m', 't_empregador_18m', 't_formal_18m','t_fundc_ocup18m', 't_medioc_ocup18m', 't_supec_ocup18m',
       't_renda_todos_trabalhos', 't_nremunerado_18m','geometry', 'centroid','LATITUDE','LONGITUDE']] 

#%%
join_left_df_novo = join_left_df_novo.replace('X', np.nan)

join_left_df_novo = join_left_df_novo.dropna(axis = 1, how = 'all')

#%%
dados_todos_8 = join_left_df_novo.copy()

#%%
# dados_todos_9 = dados_todos_8[['Cod_setor','UDH_ATLAS','UC', 'TIPO', 'MEDIDOR', 'BAIRRO', 'CD_ALIMENTADOR',
#        'LATITUDE', 'LONGITUDE','centroid','UNI_TR_MT', 'domicilio', 'domicilio', 'medio_pessoas', 'rend_responsavel', 'rend_medio',
#        'saneamento', '%alfabetizados', '%ate5salarios', '%5a10']]
dados_todos_9 = dados_todos_8.copy()

dados_todos_9['UDH_ATLAS']  = dados_todos_9['UDH_ATLAS'].astype(str)
dados_todos_9.UDH_ATLAS = dados_todos_9.UDH_ATLAS.str.replace('.', '')

#%%

for ind in dados_todos_9.index:
    dados_todos_9.UDH_ATLAS.values[ind] = dados_todos_9.UDH_ATLAS.values[ind][:-1]
    

#%%
#dados_todos_10 = pd.merge(dados_todos_9, dados_ivsUDH_2, how='left', on='UDH_ATLAS')
dados_todos_10 = dados_todos_9.copy()

#%%
cod_setor = dados_todos_10['Cod_setor'].value_counts().reset_index()

#%%
#dados_todos_10 = dados_todos_10.drop(['vulner_dia'],axis=1)

dados_todos_10 = dados_todos_10.replace(np.nan, None)

#%%

# X_Y = []

# for i in range(len(dados_todos_10)):
#     X_Y.append([dados_todos_10['LONGITUDE'][i], dados_todos_10['LATITUDE'][i]])

    
    #%%

# le = LabelEncoder()
# label = le.fit_transform(dados_todos_10['Cod_setor'])
# dados_todos_10.drop('Cod_setor', axis=1, inplace=True)
# dados_todos_10['Cod_setor'] = label

#%%
le = LabelEncoder()
label = le.fit_transform(dados_todos_10['UDH_ATLAS'])
dados_todos_10.drop('UDH_ATLAS', axis=1, inplace=True)
dados_todos_10['UDH_ATLAS'] = label

#%%

dados_todos_10['MEDIDOR']=dados_todos_10['MEDIDOR'].astype(str)

le = LabelEncoder()
label = le.fit_transform(dados_todos_10['MEDIDOR'])
dados_todos_10.drop('MEDIDOR', axis=1, inplace=True)
dados_todos_10['MEDIDOR'] = label

#%%

# dados_todos_10['TIPO']=dados_todos_10['TIPO'].astype(str)

# le = LabelEncoder()
# label = le.fit_transform(dados_todos_10['TIPO'])
# dados_todos_10.drop('TIPO', axis=1, inplace=True)
# dados_todos_10['TIPO'] = label
                                      
#%%
dados_todos_10_count = dados_todos_10['UNI_TR_MT'].value_counts().reset_index()

#%%
dados_inspe_2 = dados_inspe[['UC', 'SS', 'SEQ', 'DATA DA FISCALIZAÇÃO', 'UNIDADE', 'MUNICIPIO',
        'BAIRRO','PARECER2']]

dados_inspe_3 = dados_inspe_2[dados_inspe_2['PARECER2'] != 0]
dados_inspe_3 = dados_inspe_3[dados_inspe_3['PARECER2'] != 3]

dados_inspe_32 = dados_inspe_3[dados_inspe_3['BAIRRO'].isin(nm_BAIRRO)]


dados_inspe_4 = dados_inspe_32[['UC','PARECER2','DATA DA FISCALIZAÇÃO']]

#dados_todos_2 = dados_todos[dados_todos['CD_ALIMENTADOR'].isin(cod_alimentador)]

dados_todos_UC = dados_todos_10[['Cod_setor','UNI_TR_MT','UC','BAIRRO','CD_ALIMENTADOR','LATITUDE','LONGITUDE']]

dados_todos_11 = pd.merge(dados_todos_UC,dados_inspe_4 , how='left', on='UC').fillna(0)

#%%

gdf = gpd.GeoDataFrame(dados_todos_11, geometry=gpd.points_from_xy(dados_todos_11.LONGITUDE, dados_todos_11.LATITUDE))

#%%
# Convert the time to a datetime object
gdf['DATE'] = pd.to_datetime(gdf['DATA DA FISCALIZAÇÃO'])

gdf = gdf[gdf['PARECER2'] != 0]

gdf['year'] = gdf['DATE'].dt.year

#%%

# fig, ax = plt.subplots(figsize=(10,8))
# # Create a 3D axis
# ax = fig.add_subplot(111, projection='3d')
# # Plot the data
# sc = ax.scatter(gdf.geometry.x, gdf.geometry.y, gdf['year'], c=gdf['PARECER2'], cmap='Reds')
# # Add labels
# ax.set_xlabel('LONGITUDE')
# ax.set_ylabel('LATITUDE')
# ax.set_zlabel('TIME')
# plt.colorbar(sc, label='PARECER2')
# # Display the plot
# plt.show()

#%%

fig, ax = plt.subplots(figsize=(6,4))
dados_ibge.boundary.plot(ax=ax,color = 'grey',label = 'regiões IBGE')
gdf.plot(column="PARECER2", scheme='NaturalBreaks', k=4, cmap='coolwarm', legend=True, legend_kwds={'bbox_to_anchor':(1.12, 0.55)}, ax=ax)
plt.title('Relative Inspections per year')
plt.tight_layout()
ax.axis("off")
#plt.savefig('map1969',dpi=300, bbox_inches='tight')
plt.show()



#%%


reshaped_df = gdf[['year','Cod_setor','PARECER2']]


reshaped_df = reshaped_df.groupby(['Cod_setor','year']).sum()
#%%
reshaped_df = reshaped_df.reset_index()

#%%
# reshaped_df.groupby(['Cod_setor','year']).sum().unstack().plot()
# plt.show()

#%%

reshaped_df_count = gdf['Cod_setor'].value_counts().reset_index()

#%%

from shapely.ops import nearest_points
import shapely.geometry
 
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
print('\n')
print(f'Observed mean distance: {do}')
print(f'Expected mean distance: {de}')
print(f'Nearest neighbour index: {d}')
print(f'Number of points: {count}')
print(f'NNI : {SE}')
print(f'Z-Score: {zscore}')
print('\n')
print('######### Hypothesis result:\n')

if SE < 1:
    print('R < 1 , the distribution is clustered')
if SE > 1:
    print('R > 1 , the distribution is uniform')
if SE == 1:
    print('R = 1 , the distribution is random')
if zscore < -1.96:
    print('zscore < -1.96 , the distribution is clustered')
if zscore > 1.96:
    print('zscore > 1.96 , the distribution is clustered')
if 1.96 < zscore > -1.96:
    print('1.96 < zscore > -1.96 , the distribution is random')  


#%% Creates nearest neighbor weights matrix based on k nearest neighbors.

#from splot.esda import moran_scatterplot, plot_local_autocorrelation, lisa_cluster

#%%
gdf["ID"] = range(gdf.shape[0])

gdf = gdf[gdf['ID'] != 553]
gdf = gdf[gdf['ID'] != 445]

#%%
state_weights = pysal.lib.weights.KNN.from_dataframe(gdf, ids="ID", k=3)
state_weights.plot(gdf=gdf, indexed_on="ID")
plt.show()

#%%
matriz = pd.DataFrame(*state_weights.full()).astype(int)

#%%
w_adaptive  = weights.distance.Kernel.from_dataframe(gdf,fixed=False, k=3)
full_matrix, ids = w_adaptive.full()
#%%
mx_knn3 = weights.KNN.from_dataframe(gdf, k=3)

#%%
w_rook = weights.contiguity.Rook.from_dataframe(gdf)
s = pd.Series(w_rook.cardinalities)
# s.plot.hist(bins=s.unique().shape[0])
# plt.show()

#%%
neighbors = w_rook.neighbors.copy()
adjlist = w_rook.to_adjlist()

#%%
adjlist_fisca = adjlist.merge(
    gdf[["PARECER2"]],
    how="left",
    left_on="focal",
    right_index=True,
).merge(
    gdf[["PARECER2"]],
    how="left",
    left_on="neighbor",
    right_index=True,
    suffixes=("_focal", "_neighbor"),
)

#%%

adjlist_fisca["diff"] = (adjlist_fisca["PARECER2_focal"]- adjlist_fisca["PARECER2_neighbor"])

all_pairs = np.subtract.outer(gdf["PARECER2"].values, gdf["PARECER2"].values)

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

moran = esda.moran.Moran(gdf["PARECER2"], state_weights)

splot.esda.plot_moran(moran, zstandard=True, figsize=(10,4))

plt.show()  
        
#%%Spatial Weights and Spatial Lag

gdf['w_PARECER2'] = weights.lag_spatial(state_weights, gdf['PARECER2'])

# census_tract_sp['w_total_monthly_income'] = weights.lag_spatial(w, census_tract_sp['total_monthly_income'])

#%%Global Spatial Autocorrelation

y_pop_count = gdf["PARECER2"]

moran = Moran(y_pop_count,state_weights)

#moran.I

# p-value to test if the result of Moran’s statistic is significant or not

moran.p_sim

# With a p-value of 0.001 we can reject the null hypothesis, therefore we can assume that the variable is not randomly distributed 
# in space and with the value of the Moran’s I statistic we can tell that it have positive correlation.
# We can use the Moran’s I scatterplot to help visualize the correlation.  

#%%
moran_scatterplot(moran)
plt.show()     

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
moran_local = esda.moran.Moran_Local(gdf["PARECER2"], state_weights)

splot.esda.lisa_cluster(moran_local, gdf[["PARECER2", "geometry"]])

plt.show()

#%%

# Visualize tree equity as histogram
plt.hist(gdf['PARECER2'], bins=5, edgecolor='black')
plt.xlabel("Fiscalization by Unity")
plt.ylabel("Frequency")
plt.show()

#%%
import scipy
norm_cdf = scipy.stats.norm.cdf(gdf.w_PARECER2)

#%%
#plt.plot(gdf.w_PARECER2,norm_cdf,c='red',zorder=1)
sns.lineplot(x=gdf.w_PARECER2, y=norm_cdf)
plt.xlabel('Feature Value')
plt.scatter(gdf.w_PARECER2,norm_cdf,edgecolor='black',c='red',zorder = 2)
plt.ylabel('Cumulative Probability')
plt.title('Feature CDF'); 
plt.grid()
plt.show()

#%%
from pykrige.ok import OrdinaryKriging

lon = gdf.LONGITUDE
lat = gdf.LATITUDE
z = gdf.w_PARECER2

# Generate a regular grid with 60° longitude and 30° latitude steps:
grid_lon = np.arange(min(lon), max(lon), 0.002)
grid_lat = np.arange(min(lat), max(lat), 0.002)

# Create ordinary kriging object:
OK = OrdinaryKriging(
    lon,
    lat,
    z,
    variogram_model="linear",
    verbose=False,
    enable_plotting=False,
    coordinates_type="geographic",
)

# Execute on grid:
z1, ss1 = OK.execute("grid", grid_lon, grid_lat)

# Create ordinary kriging object ignoring curvature:
OK = OrdinaryKriging(
    lon, lat, z, variogram_model="linear", verbose=False, enable_plotting=False
)

# Execute on grid:
z2, ss2 = OK.execute("grid", grid_lon, grid_lat)

#%%

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot()
ax.imshow(z1, extent=[min(lon), max(lon), min(lat), max(lat)], origin="lower")
ax.set_title("geo-coordinates")
plt.show()

#%%
import statsmodels.api as sm

gdf_2 = gdf.copy()

gdf_2['Date'] = pd.to_datetime(gdf_2['DATE'])

gdf_2.set_index('Date', inplace=True)


#%%

s=sm.tsa.seasonal_decompose(gdf_2.w_PARECER2,period=4)
# fig = s.plot()
# fig.set_size_inches((12, 9))
# fig.tight_layout()
# plt.show()

#%%

gdf_3 = gdf_2.groupby([gdf_2['DATE'].dt.year,gdf_2['DATE'].dt.month])['PARECER2'].mean()

gdf_4 = gdf_2.groupby([gdf_2['DATE'].dt.year,gdf_2['DATE'].dt.month])['PARECER2'].sum()

#%%
decomposition = sm.tsa.seasonal_decompose(gdf_3,period=6)

decomposition_sum = sm.tsa.seasonal_decompose(gdf_4,period=6)

# decomposition.observed.plot()
# decomposition.seasonal.plot()
# decomposition.resid.plot()

# plt.show()

#%%
decomposition.trend.plot(figsize = (10,4))
plt.xlabel('Time')
plt.ylabel('Fraud Cases Total')
plt.title('Trend in monthly fraud detection')
plt.show()
#%%
import matplotlib.gridspec as gridspec

# Create 2x2 sub plots
gs = gridspec.GridSpec(2, 2)

plt.figure()
ax = plt.subplot(gs[1, 0]) # row 0, col 0
decomposition.observed.plot()
plt.title('Observed values in monthly fraud detection')
plt.xlabel('Time')
plt.ylabel('Fraud Cases mean')
plt.grid()

ax = plt.subplot(gs[0, 1]) # row 0, col 1
decomposition_sum.trend.plot()
plt.title('Trend curve in monthly fraud detection')
plt.xlabel('Time')
plt.ylabel('Fraud Cases Total')
plt.grid()

ax = plt.subplot(gs[0,0]) # row 1, span all columns
decomposition_sum.observed.plot()
plt.title('Observed values in monthly fraud detection')
plt.xlabel('Time')
plt.ylabel('Fraud Cases total')
plt.grid()

ax = plt.subplot(gs[1, 1]) # row 1, span all columns
decomposition.trend.plot()
plt.title('Trend curve in monthly fraud detection')
plt.xlabel('Time')
plt.ylabel('Fraud Cases mean')
plt.grid()
plt.show()

#%%


fig, ax = plt.subplots()
# Create a 3D axis
ax = fig.add_subplot(111, projection='3d')
# Plot the data
sc = ax.scatter(gdf_2.geometry.x, gdf_2.geometry.y, gdf_2['year'], c=gdf_2['pop_count_local_moran_p_sim'], cmap='Blues')
# Add labels
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_zlabel('Time')
plt.colorbar(sc, label='local moran')
# Display the plot
plt.show()

#%%
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px

gdf_5 = gdf_2[gdf_2['pop_count_local_moran'] != 'ns']


pio.renderers.default='browser'


fig  = px.scatter_3d(gdf_5, x=gdf_5.geometry.x, y=gdf_5.geometry.y, z='DATE',
              color='pop_count_local_moran')

fig.show()




















