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

import seaborn as sns
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

# caminho_dados_todos = diretorio_atual+barra+volta_nivel + \
#     barra+'entradas'+barra+'DADOSCELESC.xlsx'

caminho_dados_todos = diretorio_atual+barra+volta_nivel + \
    barra+'entradas'+barra+'ConsumidoresTodos.xlsx'


# caminho_dados = diretorio_atual+barra+volta_nivel + \
#     barra+'entradas'+barra+'DADOSTAPERA.xlsx'

# caminho_dados_todos = diretorio_atual+barra + \
#     volta_nivel+barra+'entradas'+barra+'Consumidores.xlsx'

# caminho_dados_inspe = diretorio_atual+barra+volta_nivel+barra + \
#     'entradas'+barra+'Fiscalizações Florianópolis 01012019 - 21062023.xlsx'

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
    volta_nivel+barra+'entradas'+barra+'carregamentoFLORIPA.xlsx'

caminho_dados_inspe = diretorio_atual+barra+volta_nivel + \
    barra+'entradas'+barra+'Fiscalizacoes.xlsx'

# %% caminho saidas

caminho_ISL07_INSPECOES = diretorio_atual+barra + \
    volta_nivel+barra+'saidas'+barra+'ISL07-INSPECOES-2.csv'

caminho_UCS_VARIAVEIS = diretorio_atual+barra+volta_nivel + \
    barra+'saidas'+barra+'ISL07-UCS-VARIAVEIS-2.csv'

caminho_x = diretorio_atual+barra+volta_nivel + \
    barra+'saidas'+barra+'Pontos-interpolados-idw-2.csv'
    
# caminho_dados_artigo = diretorio_atual+barra + \
#     volta_nivel+barra+'saidas'+barra+'dados_papaer-2.csv'

# %%

#dados = pd.read_excel(caminho_dados)

# dados_celesc = pd.read_excel(caminho_dados_CELESC)

dados_todos = pd.read_excel(caminho_dados_todos)

dados_inspe = pd.read_excel(caminho_dados_inspe)

dados_escolhidos = pd.read_excel(caminho_dados_ibge_esco)

dados_ivsUDH = pd.read_excel(caminho_dados_ivsUDH, sheet_name='UDH')

# tabela_final = pd.read_csv(caminho_tabela_final,sep=',')

dados_ibge_dom1 = pd.read_excel(caminho_dados_ibge_dom1, sheet_name='dom1')

dados_eletricos = pd.read_excel(caminho_dados_eletricos)

#dados_fisca = pd.read_excel(caminho_fisca)

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

# dados_todos_2 = dados_todos[dados_todos['CD_ALIMENTADOR'].isin(
#     cod_alimentador)]


dados_todos_2 = dados_todos.copy()

## Dados celesc : falta "['UNI_TR_MT', 'NR_LOCZ_EQPTO_RD'] not in index"

#bairros = dados_todos_2.BAIRRO.drop_duplicates()


# %%
dados_todos_2 = pd.merge(dados_todos_2, dados_eletricos,how='left', on='UNI_TR_MT').dropna()

# %%
dados_todos_3 = dados_todos_2.reset_index(drop=True)

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

#%%
dados_todos_4['LATITUDE'] = dados_todos_4['LATITUDE']/100000000
dados_todos_4['LONGITUDE'] = dados_todos_4['LONGITUDE']/100000000


# %%
gdf_dados_todos_4 = gpd.GeoDataFrame(dados_todos_4, geometry=gpd.points_from_xy(
    dados_todos_4.LONGITUDE, dados_todos_4.LATITUDE))

#%%
gdf_dados_todos_4 = gdf_dados_todos_4.set_crs(4326, allow_override=True)

#%%
#dados_ibge_3.crs = gdf_dados_todos_4.crs

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

#%%

dados_todos_5 = dados_todos_5.drop(['CD_GEOCODI_left'], axis=1)
dados_todos_5 = dados_todos_5.rename(columns={'CD_GEOCODI_right': 'CD_GEOCODI'})

# %%
dados_todos_6 = dados_todos_5[['CD_GEOCODI', 'UC', 'LATITUDE','LONGITUDE', 'UNI_TR_MT','Consumo','PO_NOMINAL_TRAFO', 'Porc']]

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

#%%
#gdf_dados_todos_7 = gdf_dados_todos_7.drop(['centroid'], axis=1)

# %%
dados_ivs.crs = gdf_dados_todos_7.crs

join_left_df = gdf_dados_todos_7.sjoin(dados_ivs, how="left")

# %%
join_left_df = join_left_df.drop(['index_right'], axis=1)

# %%
join_left_df_novo = join_left_df.sjoin(dados_ivs, how="left")

# %%
join_left_df_novo = join_left_df_novo[['Cod_setor', 'UDH_ATLAS_left', 'UNI_TR_MT','UC','Situacao_setor',
                                       'Tipo_setor', 'domicilio', 'medio_pessoas', 'rend_responsavel', 'rend_medio', 'saneamento',
                                       '%alfabetizados', '%ate5salarios', '%5a10', 'geometry', 'centroid_right', 'LATITUDE', 'LONGITUDE', 'PO_NOMINAL_TRAFO',
                                       'Porc']]

# %%
join_left_df_novo = join_left_df_novo.rename(
    columns={'UDH_ATLAS_left': 'UDH_ATLAS', 'centroid_right': 'centroid', 'PO_NOMINAL_TRAFO_y': 'PO_NOMINAL_TRAFO', 'TIPO_left':'Tipo', 
             'MEDIDOR':'Medidor', 'BAIRRO':'Bairro'})

# %%
join_left_df_novo = pd.merge(join_left_df_novo, dados_ivsUDH, how='left', on='UDH_ATLAS')

# %%
join_left_df_novo = join_left_df_novo[['Cod_setor', 'UDH_ATLAS', 'UNI_TR_MT', 'UC',
                                       'Situacao_setor', 'Tipo_setor', 'PO_NOMINAL_TRAFO',
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
dados_todos_10_count = dados_todos_10['UNI_TR_MT'].value_counts().reset_index()

#%% Leitura dos dados de Inspecao

#%%
# 1) Filtrar as inspeções que resultaram em parecer diferente de "normal".
# 2) Fazer um laço while para percorrer todas as inspeções positivas, e
# quando ter sido realizada em algum consumidor de ISL07, atribuir ao transformador essa informação.

dados_inspe_2 = dados_inspe[['UC', 'SS', 'SEQ', 'DATA DA FISCALIZAÇÃO', 'UNIDADE', 'MUNICIPIO',
                             'BAIRRO', 'PARECER2']]

dados_inspe_3 = dados_inspe_2[dados_inspe_2['PARECER2'] != 0] # normal
dados_inspe_3 = dados_inspe_3[dados_inspe_3['PARECER2'] != 3] # ressarcimento

dados_inspe_4 = dados_inspe_3[['UC', 'PARECER2']]

#%%
dados_todos_UC = dados_todos_10[['Cod_setor', 'UNI_TR_MT', 'UC', 'LATITUDE', 'LONGITUDE']]

dados_todos_UC['UC'] = dados_todos_UC['UC'].astype('Int64')

#%%
dados_todos_11 = pd.merge(dados_todos_UC, dados_inspe_4,how='left', on='UC')

dados_todos_11_count = dados_todos_11['PARECER2'].value_counts().reset_index()

#dados_todos_11_count_2 = dados_todos_11.groupby(['Bairro'])['PARECER2'].count().reset_index()

dados_todos_11_count_3 = dados_todos_11.groupby(['UNI_TR_MT'])['PARECER2'].count().reset_index()

#%%
dados_todos_11_sum = dados_todos_11.groupby("UNI_TR_MT")["PARECER2"].agg(PARECER2_TOTAL=lambda x: x[x != 0].count()).reset_index()

dados_todos_11 = pd.merge(dados_todos_11, dados_todos_11_sum, how='left', on='UNI_TR_MT').fillna(0)

dados_todos_12 = dados_todos_11[['UNI_TR_MT', 'PARECER2_TOTAL']].drop_duplicates()

dados_todos_12['UNI_TR_MT'] = dados_todos_12['UNI_TR_MT'].astype('Int64')

#%% 

dados_todos_10 = dados_todos_10.replace('X', np.nan)

dados_todos_13 = dados_todos_10[['UNI_TR_MT', 'UDH_ATLAS', 'domicilio', 'medio_pessoas', 'rend_responsavel', 'rend_medio', 'saneamento',
                                 '%alfabetizados', '%ate5salarios', '%5a10', 'ivs', 'ivs_infraestrutura_urbana', 'ivs_capital_humano',
                                 'ivs_renda_e_trabalho', 'idhm', 't_eletrica', 't_empregador_18m']].drop_duplicates()


#%%
dados_todos_13['UNI_TR_MT'] = dados_todos_13['UNI_TR_MT'].astype('Int64')

#%%
#dados_todos_14 = pd.merge(dados_todos_12, dados_todos_13, how='left', on='UNI_TR_MT')

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

#%%
teste_1 = x_1[['UDH_ATLAS','centroid','domicilio', 'medio_pessoas','rend_responsavel', 'rend_medio', 'saneamento', '%alfabetizados',
               '%ate5salarios', '%5a10', 'ivs', 'ivs_infraestrutura_urbana','ivs_capital_humano', 'ivs_renda_e_trabalho', 'idhm', 'idhm_long',
               'idhm_educ', 'idhm_renda', 'idhm_educ_sub_esc','idhm_educ_sub_freq', 't_sem_agua_esgoto', 't_sem_lixo',
               't_vulner_mais1h', 't_mort1', 't_c0a5_fora', 't_c6a14_fora','t_m10a17_filho', 't_mchefe_fundin_fmenor', 't_analf_15m',
               't_cdom_fundin', 't_p15a24_nada', 't_vulner', 't_desocup18m','t_p18m_fundin_informal', 't_vulner_depende_idosos',
               't_atividade10a14', 'espvida', 't_pop18m_fundc', 't_pop5a6_escola','t_pop11a13_ffun', 't_pop15a17_fundc', 't_pop18a20_medioc',
               'renda_per_capita', 'populacao', 't_fmor5', 't_razdep', 't_fectot','t_env', 'mchefe_fmenor', 'pop0a1', 'pop1a3', 'pop4', 'pop5',
               'pop6', 'pop6a10', 'pop6a17', 'pop11a13', 'pop11a14', 'pop12a14','pop15m', 'pop15a17', 'pop15a24', 'pop16a18', 'pop18m', 'pop18a20',
               'pop18a24', 'pop19a21', 'pop25m', 'pop65m', 'pea10a14', 'pea15a17','pea18m', 't_eletrica', 't_densidadem2', 't_analf_18m',
               't_analf_25m', 'rdpc_def_vulner', 't_renda_trab', 'i_gini','t_carteira_18m', 't_scarteira_18m', 't_setorpublico_18m',
               't_contapropria_18m', 't_empregador_18m', 't_formal_18m','t_fundc_ocup18m', 't_medioc_ocup18m', 't_supec_ocup18m',
               't_renda_todos_trabalhos', 't_nremunerado_18m']].drop_duplicates()


#%%

selec =['UC','LATITUDE','LONGITUDE','UNI_TR_MT','Cod_setor','PO_NOMINAL_TRAFO','Porc']

teste_2 = x_1[x_1.columns.intersection(selec)] 

teste_1["y"] = teste_1.centroid.map(lambda p: p.y)

teste_1["x"] = teste_1.centroid.map(lambda p: p.x)

#%%
from scipy.spatial import distance_matrix

sample_points = teste_1[['x', 'y']].values

#%% pre processing para a etapa de interpolacao

from sklearn import preprocessing

norma_teste_10= teste_1[['medio_pessoas','rend_responsavel', 'rend_medio', 'saneamento', '%alfabetizados',
               '%ate5salarios', '%5a10', 'ivs', 'ivs_infraestrutura_urbana','ivs_capital_humano', 'ivs_renda_e_trabalho', 'idhm', 'idhm_long',
               'idhm_educ', 'idhm_renda', 'idhm_educ_sub_esc','idhm_educ_sub_freq', 't_sem_agua_esgoto', 't_sem_lixo',
               't_vulner_mais1h', 't_mort1', 't_c0a5_fora', 't_c6a14_fora','t_m10a17_filho', 't_mchefe_fundin_fmenor', 't_analf_15m',
               't_cdom_fundin', 't_p15a24_nada', 't_vulner', 't_desocup18m','t_p18m_fundin_informal', 't_vulner_depende_idosos',
               't_atividade10a14', 'espvida', 't_pop18m_fundc', 't_pop5a6_escola','t_pop11a13_ffun', 't_pop15a17_fundc', 't_pop18a20_medioc',
               'renda_per_capita', 'populacao', 't_fmor5', 't_razdep', 't_fectot','t_env', 'mchefe_fmenor', 'pop0a1', 'pop1a3', 'pop4', 'pop5',
               'pop6', 'pop6a10', 'pop6a17', 'pop11a13', 'pop11a14', 'pop12a14','pop15m', 'pop15a17', 'pop15a24', 'pop16a18', 'pop18m', 'pop18a20',
               'pop18a24', 'pop19a21', 'pop25m', 'pop65m', 'pea10a14', 'pea15a17','pea18m', 't_eletrica', 't_densidadem2', 't_analf_18m',
               't_analf_25m', 'rdpc_def_vulner', 't_renda_trab', 'i_gini','t_carteira_18m', 't_scarteira_18m', 't_setorpublico_18m',
               't_contapropria_18m', 't_empregador_18m', 't_formal_18m','t_fundc_ocup18m', 't_medioc_ocup18m', 't_supec_ocup18m',
               't_renda_todos_trabalhos', 't_nremunerado_18m']]

x = norma_teste_10.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
norma_teste_1 = pd.DataFrame(x_scaled, columns= norma_teste_10.columns)


#%% pre processing para a etapa de interpolacao

values = norma_teste_1[['medio_pessoas','rend_responsavel', 'rend_medio', 'saneamento', '%alfabetizados',
               '%ate5salarios', '%5a10', 'ivs', 'ivs_infraestrutura_urbana','ivs_capital_humano', 'ivs_renda_e_trabalho', 'idhm', 'idhm_long',
               'idhm_educ', 'idhm_renda', 'idhm_educ_sub_esc','idhm_educ_sub_freq', 't_sem_agua_esgoto', 't_sem_lixo',
               't_vulner_mais1h', 't_mort1', 't_c0a5_fora', 't_c6a14_fora','t_m10a17_filho', 't_mchefe_fundin_fmenor', 't_analf_15m',
               't_cdom_fundin', 't_p15a24_nada', 't_vulner', 't_desocup18m','t_p18m_fundin_informal', 't_vulner_depende_idosos',
               't_atividade10a14', 'espvida', 't_pop18m_fundc', 't_pop5a6_escola','t_pop11a13_ffun', 't_pop15a17_fundc', 't_pop18a20_medioc',
               'renda_per_capita', 'populacao', 't_fmor5', 't_razdep', 't_fectot','t_env', 'mchefe_fmenor', 'pop0a1', 'pop1a3', 'pop4', 'pop5',
               'pop6', 'pop6a10', 'pop6a17', 'pop11a13', 'pop11a14', 'pop12a14','pop15m', 'pop15a17', 'pop15a24', 'pop16a18', 'pop18m', 'pop18a20',
               'pop18a24', 'pop19a21', 'pop25m', 'pop65m', 'pea10a14', 'pea15a17','pea18m', 't_eletrica', 't_densidadem2', 't_analf_18m',
               't_analf_25m', 'rdpc_def_vulner', 't_renda_trab', 'i_gini','t_carteira_18m', 't_scarteira_18m', 't_setorpublico_18m',
               't_contapropria_18m', 't_empregador_18m', 't_formal_18m','t_fundc_ocup18m', 't_medioc_ocup18m', 't_supec_ocup18m',
               't_renda_todos_trabalhos', 't_nremunerado_18m']].values

#%%
# Define the unknown point coordinates (e.g., a 2D grid)
teste_2['y_coords'] = teste_2['LATITUDE']
teste_2['x_coords'] = teste_2['LONGITUDE']

unknown_points = teste_2[['x_coords', 'y_coords']].values

#%% Funcao de interpolacao

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
    interpolated_values = np.sum(weights * values[:, np.newaxis], axis=0) / np.sum(weights, axis=0)

    return interpolated_values


#%% Dividir dados em pacotes menores de numero 'chunk_size' de linhas

chunk_size = 1500

chunk_unknown_points = np.split(unknown_points, range(chunk_size, unknown_points.shape[0], chunk_size))


#%% preparar dataframe que ira receber os dados interpolados
chunk_result = pd.DataFrame()
result =  pd.DataFrame()
resultado = pd.DataFrame()

#%%
# for chunk in chunk_unknown_points: 
#     columns = len(values[0])
#     for i in range(columns):
#         interpolated_values = idw_interpolation(sample_points, chunk, values[:,i], power=2)
#         chunk_result[i] =  interpolated_values
#     # if result is None:
#         result = chunk_result
#     # else:
#     #    result = result.add(chunk_result, fill_value=0)
#     resultado= pd.concat([resultado, result])

#%% funcao para aplicar a funcao de interpolacao para os pacotes
columns = len(values[0])

def process_chunk(chunk):
    global resultado
    global result
    global chunk_result
    #for chunk in chunk_unknown_points:
    

    for i in range(columns):
        interpolated_values = idw_interpolation(sample_points, chunk, values[:,i], power=2)
        chunk_result[i] =  pd.Series(interpolated_values)
            # if result is None:
        result = chunk_result
            # else:
            #    result = result.add(chunk_result, fill_value=0)
    resultado = pd.concat([resultado, result])

#%% chamar a funcao de aplicar interpolacao para todos os pacotes criados

for i, chunk in enumerate(chunk_unknown_points):
    print(f'Processing Chunk {i+1}')
    process_chunk(chunk)


#%%
teste_4 = x_2[['UC','LATITUDE','LONGITUDE','UNI_TR_MT','Cod_setor','PO_NOMINAL_TRAFO','Porc']]
#teste_4 = x_2[['UC','LATITUDE','LONGITUDE','UNI_TR_MT','Cod_setor']].dropna(how='all')

teste_4 = teste_4.mask(teste_4.eq('None')).dropna()

teste_4['y_coords'] = teste_4['LATITUDE']
teste_4['x_coords'] = teste_4['LONGITUDE']

unknown_points_2 = teste_4[['x_coords', 'y_coords']].values

chunk_unknown_points_2 = np.split(unknown_points_2, range(chunk_size, unknown_points_2.shape[0], chunk_size))

#%%

for i, chunk in enumerate(chunk_unknown_points_2):
    print(f'Processing Chunk {i+1}')
    process_chunk(chunk)

#%%
teste_2 = pd.concat([teste_2,teste_4])

#%%
teste_2 = teste_2.reset_index(drop=True)

#%%

resultado_2 = resultado.dropna().reset_index(drop=True)

#%%
teste_2 = pd.concat([teste_2, resultado_2], axis=1)

#%%
#teste_2_2 = teste_2.drop('index', axis=1)

teste_2_2 = teste_2.copy()

#%%
teste_2_2 = teste_2_2.dropna().reset_index(drop=True)

#%%
teste_2_count = teste_2_2['UNI_TR_MT'].value_counts().reset_index()

#%%
# Create your linearly-spaced 2D grid, the higher num_pts the higher the resolution 
num_pts = 500
x_grid = np.linspace(min(teste_2_2['x_coords']), max(teste_2_2['x_coords']), num_pts)
y_grid = np.linspace(min(teste_2_2['y_coords']), max(teste_2_2['y_coords']), num_pts)

X,Y = np.meshgrid(x_grid, y_grid)

#%%
import matplotlib.backends.backend_pdf

columns = len(values[0])
out_pdf = r'C:\Pos\Run-py-Figures\interpolation.pdf'

pdf = matplotlib.backends.backend_pdf.PdfPages(out_pdf)

figs = plt.figure()
names = ['medio_pessoas','rend_responsavel', 'rend_medio', 'saneamento', '%alfabetizados',
          '%ate5salarios', '%5a10', 'ivs', 'ivs_infraestrutura_urbana','ivs_capital_humano', 'ivs_renda_e_trabalho', 'idhm', 'idhm_long',
          'idhm_educ', 'idhm_renda', 'idhm_educ_sub_esc','idhm_educ_sub_freq', 't_sem_agua_esgoto', 't_sem_lixo',
          't_vulner_mais1h', 't_mort1', 't_c0a5_fora', 't_c6a14_fora','t_m10a17_filho', 't_mchefe_fundin_fmenor', 't_analf_15m',
          't_cdom_fundin', 't_p15a24_nada', 't_vulner', 't_desocup18m','t_p18m_fundin_informal', 't_vulner_depende_idosos',
          't_atividade10a14', 'espvida', 't_pop18m_fundc', 't_pop5a6_escola','t_pop11a13_ffun', 't_pop15a17_fundc', 't_pop18a20_medioc',
          'renda_per_capita', 'populacao', 't_fmor5', 't_razdep', 't_fectot','t_env', 'mchefe_fmenor', 'pop0a1', 'pop1a3', 'pop4', 'pop5',
          'pop6', 'pop6a10', 'pop6a17', 'pop11a13', 'pop11a14', 'pop12a14','pop15m', 'pop15a17', 'pop15a24', 'pop16a18', 'pop18m', 'pop18a20',
          'pop18a24', 'pop19a21', 'pop25m', 'pop65m', 'pea10a14', 'pea15a17','pea18m', 't_eletrica', 't_densidadem2', 't_analf_18m',
          't_analf_25m', 'rdpc_def_vulner', 't_renda_trab', 'i_gini','t_carteira_18m', 't_scarteira_18m', 't_setorpublico_18m',
          't_contapropria_18m', 't_empregador_18m', 't_formal_18m','t_fundc_ocup18m', 't_medioc_ocup18m', 't_supec_ocup18m',
          't_renda_todos_trabalhos', 't_nremunerado_18m']


cnt = 0

for i in range(columns):
    
    Z = griddata([(x,y) for x,y in zip(teste_2_2['x_coords'],teste_2_2['y_coords'])], teste_2_2[i], (X, Y), method='linear')
    plt.ioff()
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot()
    
    dados_ibge.boundary.plot(ax=ax,color = 'grey',label = 'regiões')
    plt.scatter(teste_2_2['x_coords'], teste_2_2['y_coords'], 10, 'r', edgecolor='w', label='UC')
    plt.imshow(Z, extent=(min(teste_2['x_coords']-0.01), max(teste_2['x_coords']+0.01), min(
         teste_2['y_coords']-0.01), max(teste_2['y_coords']+0.01)), origin='lower')
    plt.legend(loc="upper left")
    plt.colorbar()
    #plt.show()
    plt.title('Interpolação {0}'.format(names[cnt]))

    cnt += 1
    plt.savefig(r'C:\Pos\Run-py-Figures\myfilename%03d.svg'%(cnt))
    

#%%
#teste_2_2 = teste_2.copy()
#%%
#teste_2_2 = teste_2.copy()

#%%
old_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
              30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
              47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
              64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
              81, 82, 83, 84, 85, 86] 

new_names = ['medio_pessoas','rend_responsavel', 'rend_medio', 'saneamento', '%alfabetizados',
          '%ate5salarios', '%5a10', 'ivs', 'ivs_infraestrutura_urbana','ivs_capital_humano', 'ivs_renda_e_trabalho', 'idhm', 'idhm_long',
          'idhm_educ', 'idhm_renda', 'idhm_educ_sub_esc','idhm_educ_sub_freq', 't_sem_agua_esgoto', 't_sem_lixo',
          't_vulner_mais1h', 't_mort1', 't_c0a5_fora', 't_c6a14_fora','t_m10a17_filho', 't_mchefe_fundin_fmenor', 't_analf_15m',
          't_cdom_fundin', 't_p15a24_nada', 't_vulner', 't_desocup18m','t_p18m_fundin_informal', 't_vulner_depende_idosos',
          't_atividade10a14', 'espvida', 't_pop18m_fundc', 't_pop5a6_escola','t_pop11a13_ffun', 't_pop15a17_fundc', 't_pop18a20_medioc',
          'renda_per_capita', 'populacao', 't_fmor5', 't_razdep', 't_fectot','t_env', 'mchefe_fmenor', 'pop0a1', 'pop1a3', 'pop4', 'pop5',
          'pop6', 'pop6a10', 'pop6a17', 'pop11a13', 'pop11a14', 'pop12a14','pop15m', 'pop15a17', 'pop15a24', 'pop16a18', 'pop18m', 'pop18a20',
          'pop18a24', 'pop19a21', 'pop25m', 'pop65m', 'pea10a14', 'pea15a17','pea18m', 't_eletrica', 't_densidadem2', 't_analf_18m',
          't_analf_25m', 'rdpc_def_vulner', 't_renda_trab', 'i_gini','t_carteira_18m', 't_scarteira_18m', 't_setorpublico_18m',
          't_contapropria_18m', 't_empregador_18m', 't_formal_18m','t_fundc_ocup18m', 't_medioc_ocup18m', 't_supec_ocup18m',
          't_renda_todos_trabalhos', 't_nremunerado_18m']

#%%
# teste_2 = teste_2.rename(columns={0: 'medio_pessoas', 1:'rend_responsavel', 2:'rend_medio',3:'saneamento',4:'%alfabetizados'
#                                   , 5:'%ate5salarios', 6:'%5a10', 7:'ivs', 8:'ivs_infraestrutura_urbana', 9:'ivs_capital_humano'
#                                   , 10:'ivs_renda_e_trabalho', 11: 'idhm', 12:'t_eletrica', 13:'t_empregador_18m'})


teste_2_3 = teste_2_2.rename(columns=dict(zip(old_names, new_names)))

#%%
teste_3 = teste_2_3.copy()

teste_3.dropna(inplace=True)

#%% Fim da rotina de Interpolacao

#%%
dados_todos_15 = pd.merge(teste_3, dados_todos_12, how='left', on='UNI_TR_MT')

#%%
dados_todos_15_count = dados_todos_15['UNI_TR_MT'].value_counts().reset_index()

dados_parecer_count = dados_todos_15['PARECER2_TOTAL'].value_counts().reset_index()

dados_todos_15_count2 = dados_todos_15[['UC','UNI_TR_MT','PARECER2_TOTAL']]

#%% Exportar saidas 

dados_todos_12.to_csv(caminho_ISL07_INSPECOES, index=False, sep=';')

dados_todos_15.to_csv(caminho_UCS_VARIAVEIS, index=False, sep=';')

teste_3.to_csv(caminho_x, index=False, sep=';')

# %%
#AssertionErrordados_todos_10_2 = dados_todos_10.copy()

dados_todos_10_2 = teste_3.copy()

#%%

gdf_dados_todos_12.to_excel('C:\Pos\entradas\gdf_dados_todos_12-2.xlsx')  

#%%
dados_ibge_4 = dados_ibge_3.copy()

dados_ibge_4 = dados_ibge_4.drop('centroid', axis=1)


#%%
dados_ibge_4.to_file('C:\Pos\entradas\dados_ibge_4.shp', driver='ESRI Shapefile')


