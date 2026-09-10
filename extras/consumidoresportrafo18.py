from statsmodels.formula.api import logit
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

caminho_dados_todos = diretorio_atual+barra + \
    volta_nivel+barra+'entradas'+barra+'Consumidores.xlsx'

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

dados_todos_2 = dados_todos[dados_todos['CD_ALIMENTADOR'].isin(
    cod_alimentador)]

## Dados celesc : falta "['UNI_TR_MT', 'NR_LOCZ_EQPTO_RD'] not in index"

bairros = dados_todos_2.Bairro.drop_duplicates()


# %%
dados_todos_2 = dados_todos_2[['UC', 'Tipo', 'Medidor', 'Marca e Modelo', 'Ano','Complemento', 'Bairro',	'Cidade',	'Regional',	
                               'Tensão Fat.','Tensão Forn.',	'Tipo Fase',	'Situação UC',	'Disjuntor', 'UNI_TR_MT', 'NR_LOCZ_EQPTO_RD',	
                               'PO_NOMINAL_TRAFO',	'CD_ALIMENTADOR',	'LATITUDE',	'LONGITUDE']]

# %%
dados_todos_2 = pd.merge(dados_todos_2, dados_eletricos,how='left', on='UNI_TR_MT')

# %%
dados_todos_3 = dados_todos_2.reset_index()

# %%
dados_todos_4 = dados_todos_3.copy()

# %%
dados_ivsUDH = dados_ivsUDH[dados_ivsUDH['ano'] == 2010]

dados_ivsUDH = dados_ivsUDH.rename(columns={'UDH': 'UDH_ATLAS'})

# %%
dados_ibge_3 = dados_ibge.copy()

dados_ibge_3['geometry'] = dados_ibge_3.geometry.to_crs(4326)

# %%
dados_ibge_3['centroid'] = dados_ibge_3['geometry'].centroid

# %%
gdf_dados_todos_4 = gpd.GeoDataFrame(dados_todos_4, geometry=gpd.points_from_xy(
    dados_todos_4.LONGITUDE, dados_todos_4.LATITUDE))

gdf_dados_todos_4 = gdf_dados_todos_4.set_crs(4326, allow_override=True)

dados_ibge_3.crs = gdf_dados_todos_4.crs

join_left_df = gdf_dados_todos_4.sjoin(dados_ibge_3, how="left")


# %%
dados_ivs.crs = dados_ibge.crs

pointdf = dados_ibge[['CD_GEOCODI', 'geometry']]

pointdf['centroid'] = pointdf['geometry'].centroid

join_ivs_ibge = pointdf.sjoin(dados_ivs, how="left")

join_ivs_ibge = join_ivs_ibge.rename(columns={'CD_GEOCODI': 'Cod_setor'})

join_ivs_ibge.dropna(inplace=True)

# %%
join_ivs = pd.merge(join_ivs_ibge, dados_ivsUDH, how='left', on='UDH_ATLAS')

# %%
dados_ibge_dom1['Cod_setor'] = dados_ibge_dom1['Cod_setor'].astype(str)

# %%
join_ibge = pd.merge(join_ivs_ibge, dados_ibge_dom1,how='left', on='Cod_setor')

# %%
# print(join_ibge.columns.tolist())

join_ivs_2 = join_ivs[['Cod_setor', 'UDH_ATLAS', 'ivs', 'ivs_infraestrutura_urbana', 'ivs_capital_humano', 'ivs_renda_e_trabalho', 'idhm',
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


# %%
join_ibge = join_ibge[['Cod_setor', 'UDH_ATLAS', 'aguageral', 'aguadepoço', 'aguaoutras', 'banheiroexclusivo', 'ban_esgoto',
                       'ban_fossa', 'ban_fossarudi', 'ban_vala', 'ban_riomar', 'ban_outros', 'semban', 'coletalixo',
                       'comEE', 'semEE', 'comEE_semmedidor']]

# %%
join_ibge = join_ibge.replace('X', np.nan)

join_ibge.dropna(inplace=True)

# %%
join_ivs_2.dropna(inplace=True)

# %%
dados_todos_5 = join_left_df.copy()

# %%
dados_todos_6 = dados_todos_5[['CD_GEOCODI', 'UC', 'Tipo', 'Medidor', 'Bairro', 'CD_ALIMENTADOR', 'LATITUDE',
                               'LONGITUDE', 'centroid', 'UNI_TR_MT', 'PO_NOMINAL_TRAFO_y', 'Dif', 'Porc']]

dados_todos_6 = dados_todos_6.rename(columns={'CD_GEOCODI': 'Cod_setor'})

dados_todos_6['Cod_setor'] = dados_todos_6['Cod_setor'].astype(str)

# %%
dados_escolhidos['Cod_setor'] = dados_escolhidos['Cod_setor'].astype(str)

# %%
dados_todos_7 = pd.merge(dados_todos_6, dados_escolhidos,how='left', on='Cod_setor')

# %%
dados_ivs['centroid'] = dados_ivs['geometry'].centroid

# %%
gdf_dados_todos_7 = gpd.GeoDataFrame(dados_todos_7, geometry=gpd.points_from_xy(dados_todos_7.LONGITUDE, dados_todos_7.LATITUDE))

gdf_dados_todos_7 = gdf_dados_todos_7.drop(['centroid'], axis=1)

# %%
dados_ivs.crs = gdf_dados_todos_7.crs

join_left_df = gdf_dados_todos_7.sjoin(dados_ivs, how="left")

# %%
join_left_df = join_left_df.drop(['index_right'], axis=1)

# %%
join_left_df_novo = join_left_df.sjoin(dados_ivs, how="left")

# %%
join_left_df_novo = join_left_df_novo[['Cod_setor', 'UDH_ATLAS_left', 'UNI_TR_MT', 'CD_ALIMENTADOR', 'UC', 'Tipo', 'Medidor', 'Bairro', 'Situacao_setor',
                                       'Tipo_setor', 'domicilio', 'medio_pessoas', 'rend_responsavel', 'rend_medio', 'saneamento',
                                       '%alfabetizados', '%ate5salarios', '%5a10', 'geometry', 'centroid_right', 'LATITUDE', 'LONGITUDE', 'PO_NOMINAL_TRAFO_y', 'Dif',
                                       'Porc']]

# %%
join_left_df_novo = join_left_df_novo.rename(
    columns={'UDH_ATLAS_left': 'UDH_ATLAS', 'centroid_right': 'centroid', 'PO_NOMINAL_TRAFO_y': 'PO_NOMINAL_TRAFO'})

# %%
join_left_df_novo = pd.merge(join_left_df_novo, dados_ivsUDH, how='left', on='UDH_ATLAS')

# %%
join_left_df_novo = join_left_df_novo[['Cod_setor', 'UDH_ATLAS', 'UNI_TR_MT', 'CD_ALIMENTADOR', 'UC',
                                       'Tipo', 'Medidor', 'Bairro', 'Situacao_setor', 'Tipo_setor', 'PO_NOMINAL_TRAFO', 'Dif',
                                       'Porc', 'domicilio', 'medio_pessoas', 'rend_responsavel', 'rend_medio',
                                       'saneamento', '%alfabetizados', '%ate5salarios', '%5a10', 'ivs', 'ivs_infraestrutura_urbana', 'ivs_capital_humano', 'ivs_renda_e_trabalho', 'idhm', 'idhm_long',
                                       'idhm_educ', 'idhm_renda', 'idhm_educ_sub_esc', 'idhm_educ_sub_freq', 'prosp_soc', 't_sem_agua_esgoto',
                                       't_sem_lixo', 't_vulner_mais1h', 't_mort1', 't_c0a5_fora', 't_c6a14_fora', 't_m10a17_filho', 't_mchefe_fundin_fmenor',
                                       't_analf_15m', 't_cdom_fundin', 't_p15a24_nada', 't_vulner', 't_desocup18m', 't_p18m_fundin_informal',
                                       't_vulner_depende_idosos', 't_atividade10a14', 'espvida', 't_pop18m_fundc', 't_pop5a6_escola', 't_pop11a13_ffun',
                                       't_pop15a17_fundc', 't_pop18a20_medioc', 'renda_per_capita', 'populacao', 't_fmor5', 't_razdep', 't_fectot', 't_env',
                                       'vulner15a24', 'mchefe_fmenor', 'vulner_dia', 'dom_vulner_idoso', 'pop0a1', 'pop1a3', 'pop4', 'pop5', 'pop6', 'pop6a10', 'pop6a17',
                                       'pop11a13', 'pop11a14', 'pop12a14', 'pop15m', 'pop15a17', 'pop15a24', 'pop16a18', 'pop18m', 'pop18a20', 'pop18a24',
                                       'pop19a21', 'pop25m', 'pop65m', 'pea10m', 'pea10a14', 'pea15a17', 'pea18m', 't_eletrica', 't_densidadem2', 't_analf_18m',
                                       't_analf_25m', 'rdpc_def_vulner', 't_renda_trab', 'i_gini', 't_carteira_18m', 't_scarteira_18m', 't_setorpublico_18m',
                                       't_contapropria_18m', 't_empregador_18m', 't_formal_18m', 't_fundc_ocup18m', 't_medioc_ocup18m', 't_supec_ocup18m',
                                       't_renda_todos_trabalhos', 't_nremunerado_18m', 'geometry', 'centroid', 'LATITUDE', 'LONGITUDE']]

# %%
join_left_df_novo = join_left_df_novo.replace('X', np.nan)

join_left_df_novo = join_left_df_novo.dropna(axis=1, how='all')

# %%
dados_todos_8 = join_left_df_novo.copy()

# %%
dados_todos_9 = dados_todos_8.copy()

dados_todos_9['UDH_ATLAS'] = dados_todos_9['UDH_ATLAS'].astype(str)

dados_todos_9.UDH_ATLAS = dados_todos_9.UDH_ATLAS.str.replace('.', '')

# %%
for ind in dados_todos_9.index:
    dados_todos_9.UDH_ATLAS.values[ind] = dados_todos_9.UDH_ATLAS.values[ind][:-1]

# %%
dados_todos_10 = dados_todos_9.copy()

# %%
cod_setor = dados_todos_10['Cod_setor'].value_counts().reset_index()

# %%
dados_todos_10 = dados_todos_10.replace(np.nan, None)

# %%

# X_Y = []

# for i in range(len(dados_todos_10)):
#     X_Y.append([dados_todos_10['LONGITUDE'][i], dados_todos_10['LATITUDE'][i]])

# %%

# le = LabelEncoder()
# label = le.fit_transform(dados_todos_10['Cod_setor'])
# dados_todos_10.drop('Cod_setor', axis=1, inplace=True)
# dados_todos_10['Cod_setor'] = label

# %%
le = LabelEncoder()
label = le.fit_transform(dados_todos_10['UDH_ATLAS'])
dados_todos_10.drop('UDH_ATLAS', axis=1, inplace=True)
dados_todos_10['UDH_ATLAS'] = label

# %%

dados_todos_10['Medidor'] = dados_todos_10['Medidor'].astype(str)

le = LabelEncoder()
label = le.fit_transform(dados_todos_10['Medidor'])
dados_todos_10.drop('Medidor', axis=1, inplace=True)
dados_todos_10['Medidor'] = label

# %%

dados_todos_10['Tipo'] = dados_todos_10['Tipo'].astype(str)

le = LabelEncoder()
label = le.fit_transform(dados_todos_10['Tipo'])
dados_todos_10.drop('Tipo', axis=1, inplace=True)
dados_todos_10['Tipo'] = label

# %%
dados_todos_10_count = dados_todos_10['UNI_TR_MT'].value_counts().reset_index()

#%% Leitura dos dados de Inspecao

#%%
# 1) Filtrar as inspeções que resultaram em parecer diferente de "normal".
# 2) Fazer um laço while para percorrer todas as inspeções positivas, e
# quando ter sido realizada em algum consumidor de ISL07, atribuir ao transformador essa informação.

dados_inspe_2 = dados_inspe[['UC', 'SS', 'SEQ', 'DATA DA FISCALIZAÇÃO', 'UNIDADE', 'MUNICIPIO',
                             'BAIRRO', 'PARECER2']]

dados_inspe_3 = dados_inspe_2[dados_inspe_2['PARECER2'] != 0]
dados_inspe_3 = dados_inspe_3[dados_inspe_3['PARECER2'] != 3]

dados_inspe_4 = dados_inspe_3[['UC', 'PARECER2']]

dados_todos_UC = dados_todos_10[['Cod_setor', 'UNI_TR_MT', 'UC', 'Bairro', 'CD_ALIMENTADOR', 'LATITUDE', 'LONGITUDE']]

dados_todos_11 = pd.merge(dados_todos_UC, dados_inspe_4,how='left', on='UC').fillna(0)

dados_todos_11_count = dados_todos_11['PARECER2'].value_counts().reset_index()

dados_todos_11_count_2 = dados_todos_11.groupby(['Bairro'])['PARECER2'].count().reset_index()

dados_todos_11_count_3 = dados_todos_11.groupby(['UNI_TR_MT'])['PARECER2'].count().reset_index()
#%%
dados_todos_11_sum = dados_todos_11.groupby("UNI_TR_MT")["PARECER2"].agg(PARECER2_TOTAL=lambda x: x[x != 0].count()).reset_index()

dados_todos_11 = pd.merge(dados_todos_11, dados_todos_11_sum, how='left', on='UNI_TR_MT').fillna(0)

dados_todos_12 = dados_todos_11[['UNI_TR_MT', 'PARECER2_TOTAL']].drop_duplicates()

#%% 
dados_todos_10 = dados_todos_10.replace('X', np.nan)

dados_todos_13 = dados_todos_10[['UNI_TR_MT', 'UDH_ATLAS', 'domicilio', 'medio_pessoas', 'rend_responsavel', 'rend_medio', 'saneamento',
                                 '%alfabetizados', '%ate5salarios', '%5a10', 'ivs', 'ivs_infraestrutura_urbana', 'ivs_capital_humano',
                                 'ivs_renda_e_trabalho', 'idhm', 't_eletrica', 't_empregador_18m']].drop_duplicates()

dados_todos_14 = pd.merge(dados_todos_12, dados_todos_13, how='left', on='UNI_TR_MT')

#%%
gdf_dados_todos_11 = gpd.GeoDataFrame(dados_todos_11, geometry=gpd.points_from_xy(dados_todos_11.LONGITUDE, dados_todos_11.LATITUDE))

gdf_dados_todos_11['PARECER2'] = gdf_dados_todos_11['PARECER2'].astype('Int64')

gdf_dados_todos_12 = gdf_dados_todos_11[gdf_dados_todos_11['PARECER2'] != 0]

#%%  Preparar para Rotina de Interpolacao
           
# %%
y = dados_todos_10.copy()
y = y.pop('UNI_TR_MT')

# %%
x_1 = dados_todos_10.copy()

# %%
x_1 = x_1.replace('X', np.nan)

# %%
x_2 = x_1[x_1['saneamento'].isnull()].reset_index(drop=True)

# %%
x_1 = x_1.dropna(subset=['saneamento']).reset_index(drop=True)

# %%
teste_1 = x_1[['UDH_ATLAS', 'centroid', 'domicilio', 'medio_pessoas', 'rend_responsavel', 'rend_medio', 'saneamento', '%alfabetizados',
               '%ate5salarios', '%5a10', 'ivs', 'ivs_infraestrutura_urbana', 'ivs_capital_humano', 'ivs_renda_e_trabalho', 'idhm', 'idhm_long',
               'idhm_educ', 'idhm_renda', 'idhm_educ_sub_esc', 'idhm_educ_sub_freq', 't_sem_agua_esgoto', 't_sem_lixo',
               't_vulner_mais1h', 't_mort1', 't_c0a5_fora', 't_c6a14_fora', 't_m10a17_filho', 't_mchefe_fundin_fmenor', 't_analf_15m',
               't_cdom_fundin', 't_p15a24_nada', 't_vulner', 't_desocup18m', 't_p18m_fundin_informal', 't_vulner_depende_idosos',
               't_atividade10a14', 'espvida', 't_pop18m_fundc', 't_pop5a6_escola', 't_pop11a13_ffun', 't_pop15a17_fundc', 't_pop18a20_medioc',
               'renda_per_capita', 'populacao', 't_fmor5', 't_razdep', 't_fectot', 't_env', 'mchefe_fmenor', 'pop0a1', 'pop1a3', 'pop4', 'pop5',
               'pop6', 'pop6a10', 'pop6a17', 'pop11a13', 'pop11a14', 'pop12a14', 'pop15m', 'pop15a17', 'pop15a24', 'pop16a18', 'pop18m', 'pop18a20',
               'pop18a24', 'pop19a21', 'pop25m', 'pop65m', 'pea10a14', 'pea15a17', 'pea18m', 't_eletrica', 't_densidadem2', 't_analf_18m',
               't_analf_25m', 'rdpc_def_vulner', 't_renda_trab', 'i_gini', 't_carteira_18m', 't_scarteira_18m', 't_setorpublico_18m',
               't_contapropria_18m', 't_empregador_18m', 't_formal_18m', 't_fundc_ocup18m', 't_medioc_ocup18m', 't_supec_ocup18m',
               't_renda_todos_trabalhos', 't_nremunerado_18m']]

teste_2 = x_1[['UC', 'LATITUDE', 'LONGITUDE', 'UNI_TR_MT',
               'Cod_setor', 'PO_NOMINAL_TRAFO', 'Dif', 'Porc']]

teste_1["y"] = teste_1.centroid.map(lambda p: p.y)

teste_1["x"] = teste_1.centroid.map(lambda p: p.x)

# %%

sample_points = teste_1[['x', 'y']].values

#values = teste_1['ivs'].values

values = teste_1[['medio_pessoas', 'rend_responsavel', 'rend_medio', 'saneamento', '%alfabetizados',
                  '%ate5salarios', '%5a10', 'ivs', 'ivs_infraestrutura_urbana', 'ivs_capital_humano', 'ivs_renda_e_trabalho', 'idhm', 'idhm_long',
                  'idhm_educ', 'idhm_renda', 'idhm_educ_sub_esc', 'idhm_educ_sub_freq', 't_sem_agua_esgoto', 't_sem_lixo',
                  't_vulner_mais1h', 't_mort1', 't_c0a5_fora', 't_c6a14_fora', 't_m10a17_filho', 't_mchefe_fundin_fmenor', 't_analf_15m',
                  't_cdom_fundin', 't_p15a24_nada', 't_vulner', 't_desocup18m', 't_p18m_fundin_informal', 't_vulner_depende_idosos',
                  't_atividade10a14', 'espvida', 't_pop18m_fundc', 't_pop5a6_escola', 't_pop11a13_ffun', 't_pop15a17_fundc', 't_pop18a20_medioc',
                  'renda_per_capita', 'populacao', 't_fmor5', 't_razdep', 't_fectot', 't_env', 'mchefe_fmenor', 'pop0a1', 'pop1a3', 'pop4', 'pop5',
                  'pop6', 'pop6a10', 'pop6a17', 'pop11a13', 'pop11a14', 'pop12a14', 'pop15m', 'pop15a17', 'pop15a24', 'pop16a18', 'pop18m', 'pop18a20',
                  'pop18a24', 'pop19a21', 'pop25m', 'pop65m', 'pea10a14', 'pea15a17', 'pea18m', 't_eletrica', 't_densidadem2', 't_analf_18m',
                  't_analf_25m', 'rdpc_def_vulner', 't_renda_trab', 'i_gini', 't_carteira_18m', 't_scarteira_18m', 't_setorpublico_18m',
                  't_contapropria_18m', 't_empregador_18m', 't_formal_18m', 't_fundc_ocup18m', 't_medioc_ocup18m', 't_supec_ocup18m',
                  't_renda_todos_trabalhos', 't_nremunerado_18m']].values

# %%
# Define the unknown point coordinates (e.g., a 2D grid)
teste_2['y_coords'] = teste_2['LATITUDE']
teste_2['x_coords'] = teste_2['LONGITUDE']

unknown_points = teste_2[['x_coords', 'y_coords']].values

# %%


def idw_interpolation(sample_points, unknown_points, values, power=2):
    """
    Perform IDW interpolation.

    Parameters:
    sample_points (array-like): Known point coordinates (2D array: n x 2).
    unknown_points (array-like): Unknown point coordinates to interpolate (2D array: m x 2).
    values (array-like): Known point values (1D array: n).
    power (int, optional): Power parameter for IDW. Default is 2.

    Returns:
    interpolated_values (array-like): Interpolated values at the unknown_points (1D array: m).
    """
    # Calculate the distance matrix between known and unknown points
    distances = distance_matrix(sample_points, unknown_points)

    # Avoid division by zero
    distances[distances == 0] = 1e-10

    # Calculate weights using the inverse distance raised to the power
    weights = 1 / np.power(distances, power)

    # Calculate the interpolated values
    interpolated_values = np.sum(
        weights * values[:, np.newaxis], axis=0) / np.sum(weights, axis=0)

    return interpolated_values

# %%
# Perform IDW interpolation

columns = len(values[0])

for i in range(columns):
    interpolated_values = idw_interpolation(
        sample_points, unknown_points, values[:, i], power=2)
    teste_2[i] = interpolated_values

# interpolated_values = idw_interpolation(sample_points, unknown_points, values, power=2)

# teste_2['ivs_idw'] = interpolated_values
# %%
teste_4 = x_2[['UC', 'LATITUDE', 'LONGITUDE', 'UNI_TR_MT','Cod_setor', 'PO_NOMINAL_TRAFO', 'Dif', 'Porc']]
teste_4['y_coords'] = teste_4['LATITUDE']
teste_4['x_coords'] = teste_4['LONGITUDE']

unknown_points_2 = teste_4[['x_coords', 'y_coords']].values

# %%
columns = len(values[0])

for i in range(columns):
    interpolated_values = idw_interpolation(
        sample_points, unknown_points_2, values[:, i], power=2)
    teste_4[i] = interpolated_values
# %%
teste_2 = pd.concat([teste_2, teste_4])

teste_2_count = teste_2['UNI_TR_MT'].value_counts().reset_index()

# %%
# Create your linearly-spaced 2D grid, the higher num_pts the higher the resolution
num_pts = 100
x_grid = np.linspace(min(teste_2['x_coords']),
                     max(teste_2['x_coords']), num_pts)
y_grid = np.linspace(min(teste_2['y_coords']),
                     max(teste_2['y_coords']), num_pts)

X, Y = np.meshgrid(x_grid, y_grid)

# %%

out_pdf = r'C:\Pos\figuras\interpolation.pdf'

pdf = matplotlib.backends.backend_pdf.PdfPages(out_pdf)

figs = plt.figure()
names = ['medio_pessoas', 'rend_responsavel', 'rend_medio', 'saneamento', '%alfabetizados',
         '%ate5salarios', '%5a10', 'ivs', 'ivs_infraestrutura_urbana', 'ivs_capital_humano', 'ivs_renda_e_trabalho', 'idhm', 'idhm_long',
         'idhm_educ', 'idhm_renda', 'idhm_educ_sub_esc', 'idhm_educ_sub_freq', 't_sem_agua_esgoto', 't_sem_lixo',
         't_vulner_mais1h', 't_mort1', 't_c0a5_fora', 't_c6a14_fora', 't_m10a17_filho', 't_mchefe_fundin_fmenor', 't_analf_15m',
         't_cdom_fundin', 't_p15a24_nada', 't_vulner', 't_desocup18m', 't_p18m_fundin_informal', 't_vulner_depende_idosos',
         't_atividade10a14', 'espvida', 't_pop18m_fundc', 't_pop5a6_escola', 't_pop11a13_ffun', 't_pop15a17_fundc', 't_pop18a20_medioc',
         'renda_per_capita', 'populacao', 't_fmor5', 't_razdep', 't_fectot', 't_env', 'mchefe_fmenor', 'pop0a1', 'pop1a3', 'pop4', 'pop5',
         'pop6', 'pop6a10', 'pop6a17', 'pop11a13', 'pop11a14', 'pop12a14', 'pop15m', 'pop15a17', 'pop15a24', 'pop16a18', 'pop18m', 'pop18a20',
         'pop18a24', 'pop19a21', 'pop25m', 'pop65m', 'pea10a14', 'pea15a17', 'pea18m', 't_eletrica', 't_densidadem2', 't_analf_18m',
         't_analf_25m', 'rdpc_def_vulner', 't_renda_trab', 'i_gini', 't_carteira_18m', 't_scarteira_18m', 't_setorpublico_18m',
         't_contapropria_18m', 't_empregador_18m', 't_formal_18m', 't_fundc_ocup18m', 't_medioc_ocup18m', 't_supec_ocup18m',
         't_renda_todos_trabalhos', 't_nremunerado_18m']


cnt = 0

for i in range(columns):

    Z = griddata([(x, y) for x, y in zip(teste_2['x_coords'],
                 teste_2['y_coords'])], teste_2[i], (X, Y), method='linear')
    plt.ioff()
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()

    dados_ibge.boundary.plot(ax=ax, color='grey', label='regiões')
    plt.scatter(teste_4['x_coords'], teste_4['y_coords'],
                10, 'r', edgecolor='w', label='UC')
    plt.imshow(Z, extent=(min(teste_2['x_coords']-0.1), max(teste_2['x_coords']+0.1), min(
        teste_2['y_coords']-0.1), max(teste_2['y_coords']+0.1)), origin='lower')
    plt.legend(loc="upper left")
    plt.colorbar()
    # plt.show()
    plt.title('Interpolação {0}'.format(names[cnt]))
    pdf.savefig(fig)
    cnt += 1
    plt.close(fig)

pdf.close()

#%%
teste_2_2 = teste_2.copy()

# %%
old_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
             30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
             47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
             64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
             81, 82, 83, 84, 85, 86]

new_names = ['medio_pessoas', 'rend_responsavel', 'rend_medio', 'saneamento', '%alfabetizados',
             '%ate5salarios', '%5a10', 'ivs', 'ivs_infraestrutura_urbana', 'ivs_capital_humano', 'ivs_renda_e_trabalho', 'idhm', 'idhm_long',
             'idhm_educ', 'idhm_renda', 'idhm_educ_sub_esc', 'idhm_educ_sub_freq', 't_sem_agua_esgoto', 't_sem_lixo',
             't_vulner_mais1h', 't_mort1', 't_c0a5_fora', 't_c6a14_fora', 't_m10a17_filho', 't_mchefe_fundin_fmenor', 't_analf_15m',
             't_cdom_fundin', 't_p15a24_nada', 't_vulner', 't_desocup18m', 't_p18m_fundin_informal', 't_vulner_depende_idosos',
             't_atividade10a14', 'espvida', 't_pop18m_fundc', 't_pop5a6_escola', 't_pop11a13_ffun', 't_pop15a17_fundc', 't_pop18a20_medioc',
             'renda_per_capita', 'populacao', 't_fmor5', 't_razdep', 't_fectot', 't_env', 'mchefe_fmenor', 'pop0a1', 'pop1a3', 'pop4', 'pop5',
             'pop6', 'pop6a10', 'pop6a17', 'pop11a13', 'pop11a14', 'pop12a14', 'pop15m', 'pop15a17', 'pop15a24', 'pop16a18', 'pop18m', 'pop18a20',
             'pop18a24', 'pop19a21', 'pop25m', 'pop65m', 'pea10a14', 'pea15a17', 'pea18m', 't_eletrica', 't_densidadem2', 't_analf_18m',
             't_analf_25m', 'rdpc_def_vulner', 't_renda_trab', 'i_gini', 't_carteira_18m', 't_scarteira_18m', 't_setorpublico_18m',
             't_contapropria_18m', 't_empregador_18m', 't_formal_18m', 't_fundc_ocup18m', 't_medioc_ocup18m', 't_supec_ocup18m',
             't_renda_todos_trabalhos', 't_nremunerado_18m']

# %%
teste_2_3 = teste_2_2.rename(columns=dict(zip(old_names, new_names)))

# %%
teste_3 = teste_2_3.copy()

teste_3.dropna(inplace=True)

#%% Fim da rotina de Interpolacao

#%%
dados_todos_15 = pd.merge(teste_3, dados_todos_12, how='left', on='UNI_TR_MT')

#%%
dados_todos_15_count = dados_todos_15['UNI_TR_MT'].value_counts().reset_index()

dados_parecer_count = dados_todos_15['PARECER2_TOTAL'].value_counts().reset_index()

#%% Exportar saidas 

dados_todos_12.to_csv(caminho_ISL07_INSPECOES, index=False, sep=';')

dados_todos_15.to_csv(caminho_UCS_VARIAVEIS, index=False, sep=';')

teste_3.to_csv(caminho_x, index=False, sep=';')

# %%

# dados_todos_10.plot(kind="scatter", x="LONGITUDE", y="LATITUDE", grid=True,s=dados_todos_10["comEE_semmedidor"]
#                     ,label="DP permanentes com energia elétrica e sem medidor", c="ivs", cmap="jet", colorbar=True,legend=True,
#                     sharex=False, figsize=(10, 7))

# dados_todos_10.plot(kind="scatter", x="LONGITUDE", y="LATITUDE", grid=True,s=dados_todos_10["domicilio"]/10
#                     ,label="N° de domicilios", c="ivs", cmap="jet", colorbar=True,legend=True,
#                     sharex=False, figsize=(10, 7))


# plt.show()
# %%

#AssertionErrordados_todos_10_2 = dados_todos_10.copy()

dados_todos_10_2 = teste_3.copy()

# %%
# dados_todos_10_2_count = dados_todos_10_2['Clusters'].value_counts().reset_index()

# #%%

# dados_todos_10_2_count_2 = dados_todos_10_2_count.loc[dados_todos_10_2_count['count'] < 2].reset_index(drop=True)

# #%%
# # C:\Users\Natalia Bostos\anaconda3\Lib\site-packages\sklearn\base.py:1151: ConvergenceWarning: Number of distinct clusters (1) found smaller than n_clusters (2). Possibly due to duplicate points in X.
# #   return fit_method(estimator, *args, **kwargs)

# ################ atualizar!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# for value in range(len(dados_todos_10_2_count_2)):
#     drop = dados_todos_10_2_count_2['Clusters'][value]
#     #print(drop)
#     dados_todos_10_2 = dados_todos_10_2[dados_todos_10_2['Clusters'] != drop]

# #%%
# def get_cluster(df_1):
#     m = df_1[['Cod_setor']]
#     km = KMeans(n_clusters=2, init='random', max_iter=100, n_init=1, verbose=0).fit(m)
#     kmeans_predict = km.predict(m)
#     return pd.Series(kmeans_predict, index=df_1.index)

# dados_todos_10_2['SubCluster'] = dados_todos_10_2.groupby('Clusters').apply(get_cluster).droplevel(0)

# #%%
# dados_todos_10_3 = dados_todos_10_2.copy()

# #%%

# def prefixAdd(dataframe):
#   global df1
#   df1 = dataframe.copy()
#   df1 = df1.reset_index(drop=True)
#   df1['SubCluster'] = df1['SubCluster'].astype('str')
#   df1['Clusters'] = df1['Clusters'].astype('str')
#   for i in range(len(df1)):
#       # print(df1['CULTURA'][i])
#       if df1['SubCluster'][i] == '0':
#           df1['SubCluster'][i] = 'A-' + df1['Clusters'][i]

#       if df1['SubCluster'][i] == '1':
#           df1['SubCluster'][i] = 'B-' + df1['Clusters'][i]

#   return df1

# #%%
# dados_todos_10_4 = prefixAdd(dados_todos_10_3)

# #%%

# dados_todos_10_4_count_1 = dados_todos_10_4.groupby(['Clusters','SubCluster']).size().reset_index(name='N° UCs')

# #dados_todos_10_4_count_1.to_csv(diretorio_atual+barra+volta_nivel+barra+'saidas'+barra+'SubCluster.csv',index=False)

# #%%
# dados_todos_10_4['Clusters'] = dados_todos_10_4['Clusters'].astype(float)
# #%%
# dados_todos_10_4['Clusters'] = dados_todos_10_4['Clusters'].astype(int)

#%%
# fig = plt.figure(figsize=(10, 5)) ###vulner_dia
# ax = fig.add_subplot()
# dados_ibge.boundary.plot(ax=ax,color = 'grey',label = 'regiões IBGE')
# #dados_ibge.centroid.plot(ax=ax,color = 'blue')
# dados_ivs.boundary.plot(ax=ax,color = 'red',label = 'regiões IVS')
# #dados_ivs.boundary.plot(ax=ax,color = 'grey',label = 'regiões')
# #dados_ibge.apply(lambda x: ax.annotate(text=x.ID, xy=x.geometry.centroid.coords[0], ha='center', fontsize=9),axis=1);

# axis = sns.kdeplot(x = gdf_dados_todos_12.centroid.x, y = gdf_dados_todos_12.centroid.y,
#                    weights = gdf_dados_todos_12["PARECER2"],
#                    fill=True, gridsize=500, bw_adjust=0.25, cmap="coolwarm")

# gdf_dados_todos_12.plot(facecolor="none", edgecolor="gray", ax=axis)

# axis.set_axis_off()

# plt.show()

#%%% Inicio da Analise geo espacial dos dados de fiscalizacao
from shapely.ops import nearest_points
import shapely.geometry
 
bbox = gdf_dados_todos_12.total_bounds
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
gdf_dados_todos_12['Nearest'] = gdf_dados_todos_12.apply(lambda row: get_nearest_values(row, gdf_dados_todos_12), axis=1)

#%%
gdf_dados_todos_12['Distance'] = gdf_dados_todos_12.apply(lambda row: row.geometry.distance(row['Nearest']), axis=1)
#%%
sumDist = gdf_dados_todos_12['Distance'].sum()

#%%
count = len(gdf_dados_todos_12)
do = float(sumDist) / count
de = float(0.55397/(count / area) ** (1. / 3))
d = float(do / de)
SE = float(0.20136 / (count / area) ** (1. / 3))
zscore = float((do - de) / SE)

#%%
print(f'Observed mean distance: {do}')
print(f'Expected mean distance: {de}')
print(f'Nearest neighbour index: {d}')
print(f'Number of points: {count}')
print(f'Z-Score: {zscore}')

#If the index (average nearest neighbor ratio) is less than 1, the pattern exhibits clustering. 
#If the index is greater than 1, the trend is toward dispersion.

#%%Creates nearest neighbor weights matrix based on k nearest neighbors.

#from splot.esda import moran_scatterplot, plot_local_autocorrelation, lisa_cluster

gdf_dados_todos_12["ID"] = range(gdf_dados_todos_12.shape[0])

state_weights = pysal.lib.weights.KNN.from_dataframe(gdf_dados_todos_12, ids="ID", k=3)
state_weights.plot(gdf=gdf_dados_todos_12, indexed_on="ID")
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

w_adaptive  = weights.distance.Kernel.from_dataframe(gdf_dados_todos_12,fixed=False, k=3)

#w_kernel.function
#w_kernel.bandwidth[0:5]
#w_kernel.pct_nonzero
full_matrix, ids = w_adaptive.full()

#%%
mx_knn3 = weights.KNN.from_dataframe(gdf_dados_todos_12, k=3)

#%%
# fig = plt.figure(figsize=(10, 5)) ###vulner_dia
# ax = fig.add_subplot()
# dados_ibge.boundary.plot(ax=ax,color = 'grey',label = 'Regions IBGE')
# #dados_ibge.centroid.plot(ax=ax,color = 'blue')
# dados_ivs.plot(column="UDH_ATLAS", categorical=True, cmap="Pastel2", ax=ax)
# dados_ivs.boundary.plot(ax=ax,color = 'red',label = 'Regions IVS')
# mx_knn3.plot(gdf_dados_todos_12,edge_kws=dict(linewidth=1, color="black"),node_kws=dict(marker="*"),ax=ax)
# ax.set_axis_off()
# ax.set_title("Weights by $K$-NN 3")
# ax.legend(bbox_to_anchor=(1, 1), bbox_transform=fig.transFigure)
# plt.show()

#%%
w_rook = weights.contiguity.Rook.from_dataframe(gdf_dados_todos_12)
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
    gdf_dados_todos_12[["PARECER2_TOTAL"]],
    how="left",
    left_on="focal",
    right_index=True,
).merge(
    gdf_dados_todos_12[["PARECER2_TOTAL"]],
    how="left",
    left_on="neighbor",
    right_index=True,
    suffixes=("_focal", "_neighbor"),
)
        
#adjlist_fisca.info()

#%%

adjlist_fisca["diff"] = (adjlist_fisca["PARECER2_TOTAL_focal"]- adjlist_fisca["PARECER2_TOTAL_neighbor"])

all_pairs = np.subtract.outer(gdf_dados_todos_12["PARECER2_TOTAL"].values, gdf_dados_todos_12["PARECER2_TOTAL"].values)

complement_wr = 1 - w_rook.sparse.toarray()

non_neighboring_diffs = (complement_wr * all_pairs).flatten()

#%%

f = plt.figure(figsize=(12, 3))
plt.hist(
    non_neighboring_diffs,
    color="lightgrey",
    edgecolor="k",
    density=True,
    bins=10,
    label="Nonneighbors",
)
plt.hist(
    adjlist_fisca["diff"],
    color="salmon",
    edgecolor="orangered",
    linewidth=3,
    density=True,
    histtype="step",
    bins=10,
    label="Neighbors",
)
sns.despine()
plt.ylabel("Density")
plt.xlabel("Fiscalization Differences")
plt.legend()
plt.show()

#%%
extremes = adjlist_fisca.sort_values("diff", ascending=False).head()
#extremes

#%%

fig = plt.figure(figsize=(10, 5)) ###vulner_dia
ax = fig.add_subplot()
dados_ibge.boundary.plot(ax=ax,color = 'grey',label = 'Regions IBGE')
#dados_ibge.centroid.plot(ax=ax,color = 'blue')
dados_ivs.plot(column="UDH_ATLAS", categorical=True, cmap="Pastel2", ax=ax)
dados_ivs.boundary.plot(ax=ax,color = 'red',label = 'Regions IVS')
# axis = sns.kdeplot(x = gdf_dados_todos_12.centroid.x, y = gdf_dados_todos_12.centroid.y,
#                    weights = gdf_dados_todos_12["PARECER2"],
#                    fill=True, gridsize=500, bw_adjust=0.25, cmap="coolwarm")
# mx_knn3.plot(gdf_dados_todos_12,edge_kws=dict(linewidth=1, color="black"),node_kws=dict(marker="*"),ax=ax)
# first_focus = gdf_dados_todos_12.iloc[[0, 49,64,83]]
# second_focus = gdf_dados_todos_12.iloc[[12, 151]]
# first_focus.plot(color="red", ax=ax)
# second_focus.plot(color="red", ax=ax)
plt.show()

#%% https://michaelminn.net/tutorials/python-points/index.html
# https://pysal.org/notebooks/viz/splot/esda_morans_viz.html
# https://www.itl.nist.gov/div898/handbook/index.htm


moran = esda.moran.Moran(gdf_dados_todos_12["PARECER2_TOTAL"], state_weights)

splot.esda.plot_moran(moran, zstandard=True, figsize=(10,4))

plt.show()

# w.transform = "B"
# w.transform = "B"
# state_weights.weights

#%%Spatial Weights and Spatial Lag

gdf_dados_todos_12['w_PARECER2_TOTAL'] = weights.lag_spatial(state_weights, gdf_dados_todos_12['PARECER2_TOTAL'])

# census_tract_sp['w_total_monthly_income'] = weights.lag_spatial(w, census_tract_sp['total_monthly_income'])

#%%Global Spatial Autocorrelation

y_pop_count = gdf_dados_todos_12["PARECER2_TOTAL"]

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

plt.text(1.96, 0.51, 'HH', fontsize=25)
plt.text(1.95, -1.0, 'HL', fontsize=25)
plt.text(-0.55, 0.6, 'LH', fontsize=25)
plt.text(-0.55, -1, 'LL', fontsize=25)
plt.show()

# grey dots are observations that doesn’t have statistical significance in the relationship of values of the observed tract value 
# and the neighbohood weighted value.

#%%

# creating column with local_moran classification
gdf_dados_todos_12['pop_count_local_moran'] = pop_count_local_moran.q

# Dict to map local moran's classification codes
local_moran_classification = {1: 'HH', 2: 'LH', 3: 'LL', 4: 'HL'}

# Mapping local moran's classification codes
gdf_dados_todos_12['pop_count_local_moran'] = gdf_dados_todos_12['pop_count_local_moran'].map(local_moran_classification)

# p-value for each observation/neighbor pair
gdf_dados_todos_12['pop_count_local_moran_p_sim'] = pop_count_local_moran.p_sim

# If p-value > 0.05 it is not statistical significant
gdf_dados_todos_12['pop_count_local_moran'] = np.where(gdf_dados_todos_12['pop_count_local_moran_p_sim'] > 0.05, 'ns', gdf_dados_todos_12['pop_count_local_moran'])

#%%
gdf_dados_todos_13 = gdf_dados_todos_12.copy()
gdf_dados_todos_13 =gdf_dados_todos_13[['Cod_setor','w_PARECER2_TOTAL','pop_count_local_moran','pop_count_local_moran_p_sim','geometry','LATITUDE','LONGITUDE']]
#gdf_dados_todos_13.crs = "EPSG:4326"

# Plotting Local Moran's I classification map of pop_count column

#%%

m  = folium.Map(
    location=[-27.6886, -48.5622],
    tiles='cartodbpositron',
    zoom_start=12,
    min_zoom=6,
    )

gdf_dados_todos_12.explore(m=m,
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

m.save('teste.html')


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
#        if nr_locz not in color_map:
#             color_map[nr_locz] = sns.color_palette("tab10", len(grouped_coords))[next_color]
#             next_color += 1
           
#        coords_x, coords_y = zip(*[transformer.transform(lon, lat) for lat, lon in coords_list])
#        plt.scatter(coords_x, coords_y, color=color_map[nr_locz], s=12,label=nr_locz)
#        plt.legend(title = "LISA quadrant")

# plt.savefig('map.png', dpi=fig.dpi)

# plt.show()

#%%
moran_local = esda.moran.Moran_Local(gdf_dados_todos_12["PARECER2_TOTAL"], state_weights)

splot.esda.lisa_cluster(moran_local, gdf_dados_todos_12[["PARECER2_TOTAL", "geometry"]])

plt.show()

#%%
local = gdf_dados_todos_11[["LATITUDE", "LONGITUDE", "PARECER2_TOTAL"]]

lat_longs = list(map(list, zip(local["LATITUDE"], local["LONGITUDE"], local["PARECER2_TOTAL"])))

#%%
# map_h = folium.Map(
#     location=[-27.6886, -48.5622],
#     tiles="CartoDB positron",
#     zoom_start=12,
#     min_zoom=6,
#     max_zoom=18)

# HeatMap(lat_longs, min_opacity=0.2,

#         radius=50, blur=50,
#         max_zoom=1).add_to(map_h)

# hm_name = "Heat_Map_FISCALI.html"
# map_h.save(hm_name)

#%%
base_perdas = pd.read_excel(r'C:\Pos\entradas\entrada_MI.xlsx', sheet_name='Planilha1')

base_perdas_1 = base_perdas[['Cod_setor', 'Perdas']]

#%%
dados_todos_15['Cod_setor'] = dados_todos_15['Cod_setor'].astype('int64')

df_merged = dados_todos_15.merge(base_perdas_1, on='Cod_setor', how='left').dropna()

#%%
local_perdas = df_merged[["LATITUDE", "LONGITUDE", "Perdas"]]

lat_longs_perdas = list(map(list, zip(local_perdas["LATITUDE"], local_perdas["LONGITUDE"], local_perdas["Perdas"])))

#%%
# map_h_perdas = folium.Map(
#     location=[-27.6886, -48.5622],
#     tiles="CartoDB positron",
#     zoom_start=12,
#     min_zoom=6,
#     max_zoom=18)

# HeatMap(lat_longs_perdas, min_opacity=0.2,

#         radius=50, blur=50,
#         max_zoom=1).add_to(map_h_perdas)

# hm_name_perdas = "Heat_Map_Perdas.html"
# map_h_perdas.save(hm_name_perdas)

#%%
# df_merged["Porc"] = df_merged["Porc"]/100

# #%%
# df_merged["PARECER2_TOTAL"] = df_merged["PARECER2_TOTAL"].astype(float)

# #%%
# df_merged['Choice'] = np.where((df_merged['PARECER2_TOTAL'] > 0), 1, 0)

#%%

# out_pdf = r'C:\Pos\figuras\kdeplot.pdf'

# pdf = matplotlib.backends.backend_pdf.PdfPages(out_pdf)

# figs = plt.figure()
# names = ['medio_pessoas', 'rend_responsavel', 'rend_medio', 'saneamento', '%alfabetizados',
#           '%ate5salarios', '%5a10', 'ivs', 'ivs_infraestrutura_urbana', 'ivs_capital_humano', 'ivs_renda_e_trabalho', 'idhm', 'idhm_long',
#           'idhm_educ', 'idhm_renda', 'idhm_educ_sub_esc', 'idhm_educ_sub_freq', 't_sem_agua_esgoto', 't_sem_lixo',
#           't_vulner_mais1h', 't_mort1', 't_c0a5_fora', 't_c6a14_fora', 't_m10a17_filho', 't_mchefe_fundin_fmenor', 't_analf_15m',
#           't_cdom_fundin', 't_p15a24_nada', 't_vulner', 't_desocup18m', 't_p18m_fundin_informal', 't_vulner_depende_idosos',
#           't_atividade10a14', 'espvida', 't_pop18m_fundc', 't_pop5a6_escola', 't_pop11a13_ffun', 't_pop15a17_fundc', 't_pop18a20_medioc',
#           'renda_per_capita', 'populacao', 't_fmor5', 't_razdep', 't_fectot', 't_env', 'mchefe_fmenor', 'pop0a1', 'pop1a3', 'pop4', 'pop5',
#           'pop6', 'pop6a10', 'pop6a17', 'pop11a13', 'pop11a14', 'pop12a14', 'pop15m', 'pop15a17', 'pop15a24', 'pop16a18', 'pop18m', 'pop18a20',
#           'pop18a24', 'pop19a21', 'pop25m', 'pop65m', 'pea10a14', 'pea15a17', 'pea18m', 't_eletrica', 't_densidadem2', 't_analf_18m',
#           't_analf_25m', 'rdpc_def_vulner', 't_renda_trab', 'i_gini', 't_carteira_18m', 't_scarteira_18m', 't_setorpublico_18m',
#           't_contapropria_18m', 't_empregador_18m', 't_formal_18m', 't_fundc_ocup18m', 't_medioc_ocup18m', 't_supec_ocup18m',
#           't_renda_todos_trabalhos', 't_nremunerado_18m']



# for elem in names:
#     if elem in df_merged.keys():
#         # print(f"{elem}")
#         a = f"{elem}"
#         plt.figure()
#         sns.set(rc={"figure.figsize":(8, 5)})
        
#         sns_plot  = sns.kdeplot(df_merged.loc[(df_merged['Choice']==1),a], color='r', fill=True, label='1')
#         sns_plot =  sns.kdeplot(df_merged.loc[(df_merged['Choice']==0),a], color='b', fill=True, label='0')
#         fig = sns_plot.get_figure()
#         #plt.show()
#         pdf.savefig(fig)

   
# pdf.close()

#%%
# from esda.moran import Moran
# from libpysal.weights.contiguity import Queen
# moran = []

# w = Queen.from_dataframe(df_merged)
# w.transform = 'r'


# for elem in names:
#     if elem in df_merged.keys():
#         a = f"{elem}"
#         w = Queen.from_dataframe(df_merged)
#         m = Moran(df_merged['Choice'].values, w)
#         moran.append(m)
          
        
#%%
# import statsmodels.api as sm
# from statsmodels.formula.api import logit


# affair_mod = logit(
#     "Choice ~ rend_responsavel + saneamento"
#     "+ ivs + ivs_infraestrutura_urbana + ivs_renda_e_trabalho + idhm"
#     " + renda_per_capita",
#     df_merged,
# ).fit()

# %%
# teste_5 = dados_fisca_2[['UC','LATITUDE','LONGITUDE','UNI_TR_MT','Cod_setor','PO_NOMINAL_TRAFO', 'Dif','Porc']]
# teste_5['y_coords'] = teste_5['LATITUDE']
# teste_5['x_coords'] = teste_5['LONGITUDE']

# unknown_points_3 = teste_5[['x_coords', 'y_coords']].values

# #%%
# # fig = plt.figure(figsize=(10,10))
# # ax = fig.add_subplot()
# # dados_ibge.boundary.plot(ax=ax,color = 'grey',label = 'regiões')
# # plt.scatter(teste_4['x_coords'], teste_4['y_coords'], 10, 'r', edgecolor='w', label='UC')
# # plt.show()

# #%%
# columns = len(values[0])

# for i in range(columns):
#     interpolated_values = idw_interpolation(sample_points, unknown_points_2, values[:,i], power=2)
#     teste_4[i] = interpolated_values


# %%

# fig = plt.figure(figsize=(10, 5)) ###vulner_dia
# ax = fig.add_subplot()
# dados_ibge.boundary.plot(ax=ax,color = 'grey',label = 'regiões IBGE')
# #dados_ibge.centroid.plot(ax=ax,color = 'blue')
# #dados_ivs.boundary.plot(ax=ax,color = 'red',label = 'regiões IVS')
# #dados_ivs.boundary.plot(ax=ax,color = 'grey',label = 'regiões')
# #dados_ibge.apply(lambda x: ax.annotate(text=x.ID, xy=x.geometry.centroid.coords[0], ha='center', fontsize=9),axis=1);
# plt.plot(dados_todos_4.LONGITUDE, dados_todos_4.LATITUDE, 'o',color='blue', label='UCs com dados IBGE e IVS')
# #plt.plot(x_2.LONGITUDE, x_2.LATITUDE, 'x',color='magenta', label='UCs sem dados no IBGE')
# #gdf_dados_todos_12.centroid.plot(ax=ax,color='blue', label='UCs',markersize=4)
# #gdf_dados_todos_12.apply(lambda x: ax.annotate(text=x.PARECER2, xy=x.geometry.centroid.coords[0], xytext=(1.5, 2.5), textcoords='offset points', fontsize=12),axis=1)
# #ax.text(0.3,0.05,'1-IRREGULAR ADULTERACAO NA MEDICAO\n2-IRREGULAR CONSUMO FORA MEDIDO\n4-AVARIA NO EQUI.MEDICAO\n5-LIGACAO CLANDESTINA AUTO-LIGADO',transform=ax.transAxes, bbox=dict(facecolor='grey',edgecolor='black',boxstyle='square'))
# #plt.plot(dados_todos_11.LONGITUDE, dados_todos_11.LATITUDE, 'x',color='magenta', label='UCs Inspeções')
# plt.axis('equal')
# #ax.legend(bbox_to_anchor=(1, 1), bbox_transform=fig.transFigure)
# plt.show()

#%%
# end = datetime.now()
# end_time = end.strftime('%H:%M:%S')
# #%%
# delta = end - start

# #%%
# print("The time of execution start is :", start_time)
# print("The time of execution end is :", end_time)

# print("The time of execution is :", delta.total_seconds(), "seconds")


#%% ivs ivs urban infrastructure ivs human capital ivs income e work idhm vulner day t electrical  t employer 18m

# dados_ivsUDH2_artigo = dados_ivsUDH_2.rename(columns={'ivs_infraestrutura_urbana':'ivs_urban_infrastructure','ivs_capital_humano':'ivs_human_capital',
#                                                       'ivs_renda_e_trabalho':'ivs_income_e_work','vulner_dia':'vulner_day','t_eletrica': 't_electrical',
#                                                       't_empregador_18m':'t_employer_18m'})
# #%%
# dados_ivsUDH2_artigo = dados_ivsUDH2_artigo.drop('ano', axis=1)

# #%%
# dados_ivsUDH2_artigo = dados_ivsUDH2_artigo.drop('vulner_day', axis=1)
# dados_ivsUDH2_artigo =dados_ivsUDH2_artigo.drop('UDH_ATLAS', axis=1)

# #%%
# artigo_describe = dados_ivsUDH2_artigo.describe()

# #%%

# dados_ibge_artigo = dados_ibge_dom1[['domicilio','aguageral', 'aguadepoço','banheiroexclusivo','ban_esgoto','ban_outros']]

# dados_ibge_artigo = dados_ibge_artigo.replace('X', np.nan)

# artigo2_describe = dados_ibge_artigo.describe()

# #boxplot = dados_ibge_artigo.boxplot()

# boxplot =dados_ibge_artigo.boxplot(whis=[5, 95])
# plt.show()
# #%%

# dados_ibge_artigo_2 = dados_ibge_artigo.copy()

# dados_ibge_artigo_2['aguageral'] = np.arcsinh(dados_ibge_artigo_2['aguageral'].values) 

# dados_ibge_artigo_2['aguadepoço'] = np.arcsinh(dados_ibge_artigo_2['aguadepoço'].values) 

# dados_ibge_artigo_2['banheiroexclusivo'] = np.arcsinh(dados_ibge_artigo_2['banheiroexclusivo'].values) 

# dados_ibge_artigo_2['domicilio'] = np.arcsinh(dados_ibge_artigo_2['domicilio'].values)
 
# dados_ibge_artigo_2['ban_esgoto'] = np.arcsinh(dados_ibge_artigo_2['ban_esgoto'].values) 

# dados_ibge_artigo_2['ban_outros'] = np.arcsinh(dados_ibge_artigo_2['ban_outros'].values) 


#%%
# fig, ax = plt.subplots()
# #boxplot  = dados_ibge_artigo_2[['aguageral', 'aguadepoço','banheiroexclusivo']].boxplot(whis=[5, 95])

# boxplot  = dados_ibge_artigo.boxplot(whis=[5, 95])


# plt.show()


# %%

# plt.subplot(2, 1, 1)
# #pylab.hist(dados_ibge_artigo_2['aguageral'][~np.isnan(dados_ibge_artigo_2['aguageral'])])
# ax1= sns.distplot(dados_ibge_artigo_2['aguadepoço'][~np.isnan(dados_ibge_artigo_2['aguadepoço'])])
# plt.title("aguageral arcsinh ", y=-0.20)

# plt.subplot(2, 1, 2)
# ax= sns.distplot(dados_ibge_artigo['aguadepoço'][~np.isnan(dados_ibge_artigo['aguadepoço'])])
# plt.title("aguageral raw", y=-0.30)
# plt.show()

# %%

# join_ivs_2.to_excel(r'C:/celesc/entradas/join_ivs_2.xlsx',sheet_name='Planilha1', index=False)

# join_ibge.to_excel(r'C:/celesc/entradas/join_ibge.xlsx', sheet_name='Planilha1', index=False)

#%%

caminho_shape_FLO = diretorio_atual+barra+volta_nivel+barra+'1717075318251_exportacao'+barra+'cad_edificacao.shp'
caminho_shape_FLO_2 = diretorio_atual+barra+volta_nivel+barra+'1717075318251_exportacao'+barra+'gvw_territoriais_visualizacao.shp'
caminho_shape_FLO_3 = diretorio_atual+barra+volta_nivel+barra+'1717075318251_exportacao'+barra+'patrimonio_territorial.shp'
caminho_shape_FLO_4 = diretorio_atual+barra+volta_nivel+barra+'1717504933056_exportacao'+barra+'plan_uep.shp'

caminho_shape_FLO_IBGE = diretorio_atual+barra+volta_nivel+barra+'1717075279063_exportacao'+barra+'censo2022_cnefe.shp'

#%%

crs = {'init': 'epsg:4326'}

dados_shape_FLO = gpd.read_file(caminho_shape_FLO)
type(dados_shape_FLO)
dados_shape_FLO.crs

dados_shape_FLO_2 = gpd.read_file(caminho_shape_FLO_2)
type(dados_shape_FLO_2)
dados_shape_FLO_2.crs

dado_shape_FLO_3 = gpd.read_file(caminho_shape_FLO_3)
type(dado_shape_FLO_3)
dado_shape_FLO_3.crs

dado_shape_FLO_4 = gpd.read_file(caminho_shape_FLO_4)
type(dado_shape_FLO_4)
dado_shape_FLO_4.crs

#%%
dados_shape_FLO['geometry'] = dados_shape_FLO.geometry.to_crs(4326)
dados_shape_FLO_2['geometry'] = dados_shape_FLO_2.geometry.to_crs(4326)
dado_shape_FLO_3['geometry'] = dado_shape_FLO_3.geometry.to_crs(4326)

#%%
# fig = plt.figure(figsize=(20,20))
# ax = fig.add_subplot()
# dados_shape_FLO.boundary.plot(ax=ax,color = 'black',label = 'cad_edificacao')
# dados_shape_FLO_2.boundary.plot(ax=ax,color = 'yellow',label = 'gvw_territoriais_visualizacao')
# #dado_shape_FLO_3.boundary.plot(ax=ax,color = 'yellow',label = 'patrimonio_territorial')
# dado_shape_FLO_4.plot(column='nm_uep', categorical=True, ax=ax)

# plt.show()

#%%

filtros_bairros = pd.read_csv(r'C:/Pos/entradas/Filtro_Bairros.csv',delimiter = ',')

lista = list(filtros_bairros.cd_uep.values)


#%%

dado_shape_FLO_4['cd_uep'] = pd.to_numeric(dado_shape_FLO_4['cd_uep'])
#%%

shape_bairros = dado_shape_FLO_4[dado_shape_FLO_4['cd_uep'].isin(lista)]

shape_bairros_2 = shape_bairros[['cd_uep','nm_uep','cd_bairro', 'cd_regiao','geometry']]

#%%
#gdf_dados_todos_4['geometry'] = gdf_dados_todos_4.geometry.to_crs(4326)

gdf_dados_todos_4_2 = gdf_dados_todos_4.to_crs('EPSG:4326')

#%%
shape_bairros_2 = shape_bairros_2.to_crs('EPSG:4326')

shape_UCs = gdf_dados_todos_4_2.sjoin(shape_bairros_2, how="left")

shape_UCs = shape_UCs[shape_UCs['cd_uep'].notna()]

bairros_UCS = gdf_dados_todos_4_2.Bairro.drop_duplicates()

bairros_UCS_2 = shape_UCs.cd_uep.drop_duplicates()

lista_2 = list(bairros_UCS_2)

shape_bairros_3 = shape_bairros_2[shape_bairros_2['cd_uep'].isin(lista_2)]


#%%

nbh_count_df = shape_UCs.groupby('cd_uep')['UC'].nunique().reset_index()

nbh_count_df.rename(columns={'UC':'nb'}, inplace=True)

shape_UCs_merge = pd.merge(shape_bairros_2, nbh_count_df, on='cd_uep')


#%%
from folium.features import DivIcon

shape_UCs_merge['lat'] = shape_UCs_merge.centroid.map(lambda p: p.y)
shape_UCs_merge['lon'] = shape_UCs_merge.centroid.map(lambda p: p.x)

#%%
# import branca
# legend_html = """
# {% macro html(this, kwargs) %}
# <div style="
#     position: fixed;
#     bottom: 150px;
#     right: 650px;
#     width: 250px;
#     height: 80px;
#     z-index:9999;
#     font-size:14px;
#     ">
#     <p><a style="color:#e6194b;font-size:150%;margin-left:20px;">◼</a>&emsp;Consonants</p>
#     <p><a style="color:#19e6b4;font-size:150%;margin-left:20px;">◼</a>&emsp;Vowels</p>
# </div>
# <div style="
#     position: fixed;
#     bottom: 150px;
#     right: 650px;
#     width: 250px;
#     height: 80px;
#     z-index:9998;
#     font-size:14px;
#     background-color: #ffffff;
#     filter: blur(8px);
#     -webkit-filter: blur(8px);
#     opacity: 0.7;
#     ">
# </div>
# {% endmacro %}
# """
# legend = branca.element.MacroElement()
# legend._template = branca.element.Template(legend_html)

#%%
map_h = folium.Map(
    location=[-27.6886, -48.5622],
    tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr = 'Esri',
    name = 'Esri Satellite',
    overlay = False,

    control = True)

style_function = lambda x: {

    'color': 'orange',
    'weight': 3,
    'fillOpacity': 0.3
}

folium.GeoJson(shape_bairros_3,
                style_function=style_function,
                tooltip=folium.GeoJsonTooltip(
                    fields=['cd_uep'],
                    aliases=['cd_uep'],
                    localize=True
                )
                
).add_to(map_h) 

# folium.Choropleth(
#     geo_data = shape_bairros_3,
#     data = shape_UCs_merge,
#     columns = ['cd_uep', 'nb'],
#     key_on = 'feature.properties.cd_uep',
#     fill_color = 'YlOrRd',
#     legend_name="QTD UCs",
#     line_color = "black",
#     highlight=True
    
# ).add_to(map_h) 

for i, row in shape_UCs_merge.iterrows():
  folium.map.Marker(
      [row['lat'],row['lon']],
      icon=DivIcon(
          icon_size=(100,24),
          icon_anchor=(+60,0),
          html=f'<div style="font-size:40px; color:white;">{row["nb"]}</div>',
          
      )
  ).add_to(map_h)

shape_UCs_merge.explore(
    m=map_h,
    column="nb",
    scheme="naturalbreaks",  
    legend=True, 
    k=10,
    cmap='gist_heat_r',
    legend_kwds=dict(), 
    name="UCs",
    tooltip='nb'
)

folium.LayerControl().add_to(map_h)

#map_h.get_root().add_child(legend)
#folium.map.CustomPane('labels').add_to(map_h)
#folium.TileLayer('CartoDBPositronOnlyLabels',pane='labels').add_to(map_h)
hm_name = "MapaSep.html"
map_h.save(hm_name)

#%%



