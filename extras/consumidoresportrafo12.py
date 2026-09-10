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

# caminho_tabela_final = diretorio_atual+barra+volta_nivel+barra+'saidas'+barra+'combinado2.csv'

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

#caminho_x_2 = diretorio_atual+barra+volta_nivel+barra+'saidas'+barra+'Ponto-Interpolados-Linear.csv'

# %%
crs = {'init': 'epsg:4326'}

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
nome_coluna = 'UNI_TR_MT'  # Substitua pelo nome da coluna que você quer analisar

# # Contar a quantidade de vezes que cada valor se repete na coluna
# contagem_valores = dados[nome_coluna].value_counts()

# # Classificar os resultados em ordem decrescente
# contagem_valores_ordenada = contagem_valores.sort_values(ascending=False)

#dados_2 = dados.copy()

cod_alimentador = ['ISL07']

dados_todos_2 = dados_todos[dados_todos['CD_ALIMENTADOR'].isin(
    cod_alimentador)]

## Dados celesc : falta "['UNI_TR_MT', 'NR_LOCZ_EQPTO_RD'] not in index"

bairros = dados_todos_2.Bairro.drop_duplicates()


# %%
dados_todos_2 = dados_todos_2[['UC', 'Tipo', 'Medidor', 'Marca e Modelo', 'Ano', 'Endereço',	'Complemento', 'Bairro',	'Cidade',	'Regional',	'Tensão Fat.',
                               'Tensão Forn.',	'Tipo Fase',	'Situação UC',	'Disjuntor', 'UNI_TR_MT', 'NR_LOCZ_EQPTO_RD',	'PO_NOMINAL_TRAFO',	'CD_ALIMENTADOR',	'LATITUDE',	'LONGITUDE']]
# %%
dados_todos_2 = pd.merge(dados_todos_2, dados_eletricos,
                         how='left', on='UNI_TR_MT')

# %%
# le = LabelEncoder()
# label = le.fit_transform(dados_todos_2['UNI_TR_MT'])
# label_e = le.fit_transform(dados_eletricos['UNI_TR_MT'])
# #dados_todos_2.drop('UNI_TR_MT', axis=1, inplace=True)
# dados_todos_2.loc[:,'UNI_TR_MT'] = label

# contagem_valores = dados_todos_2['UNI_TR_MT'].value_counts()

# %%
# contagem_valores_ordenada = contagem_valores.sort_values(ascending=False)
# valores_para_filtrar = contagem_valores_ordenada.head(5).reset_index()
# valores_para_filtrar.drop('UNI_TR_MT', axis=1, inplace=True)
# valores_para_filtrar = valores_para_filtrar.rename(columns={'index':'UNI_TR_MT'})

# for x in valores_para_filtrar:
#         vf = valores_para_filtrar['UNI_TR_MT'].values


# novos_dados = dados_todos_2[dados_todos_2[nome_coluna].isin(vf)]

# %%
le = LabelEncoder()
label2 = le.fit_transform(dados_todos_2['Endereço'])
#dados_todos_2.drop('Endereço', axis=1, inplace=True)
dados_todos_2.loc[:, 'Endereco'] = label2

# %%

dados_todos_3 = dados_todos_2.reset_index()

contagem_valores2 = dados_todos_3['Endereco'].value_counts().reset_index()

#contagem_valores2 = contagem_valores2.rename(columns={'Endereco':'UCs'})

# %%
soma = contagem_valores2.head(79)
# somaUcs = soma.UCs.sum()
# %%
dados_todos_4 = dados_todos_3.copy()

# %%

dados_ivsUDH = dados_ivsUDH[dados_ivsUDH['ano'] == 2010]


dados_ivsUDH = dados_ivsUDH.rename(columns={'UDH': 'UDH_ATLAS'})


dados_ivsUDH_2 = dados_ivsUDH[['UDH_ATLAS', 'ano', 'ivs', 'ivs_infraestrutura_urbana',
                               'ivs_capital_humano', 'ivs_renda_e_trabalho', 'idhm', 'vulner_dia', 't_eletrica', 't_empregador_18m']]

dados_ivsUDH_2.loc[:, 'UDH_ATLAS'] = dados_ivsUDH_2['UDH_ATLAS'].astype(str)

# %%
dados_ibge_3 = dados_ibge.copy()


dados_ibge_3['geometry'] = dados_ibge_3.geometry.to_crs(4326)

# %%
dados_ibge_3['centroid'] = dados_ibge_3['geometry'].centroid

# %%

gdf_dados_todos_4 = gpd.GeoDataFrame(dados_todos_4, geometry=gpd.points_from_xy(
    dados_todos_4.LONGITUDE, dados_todos_4.LATITUDE))

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

join_ibge = pd.merge(join_ivs_ibge, dados_ibge_dom1,
                     how='left', on='Cod_setor')
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
join_ivs_2.to_excel(r'C:/celesc/entradas/join_ivs_2.xlsx',
                    sheet_name='Planilha1', index=False)

join_ibge.to_excel(r'C:/celesc/entradas/join_ibge.xlsx',
                   sheet_name='Planilha1', index=False)

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
dados_todos_7 = pd.merge(dados_todos_6, dados_escolhidos,
                         how='left', on='Cod_setor')

# %%

dados_ivs['centroid'] = dados_ivs['geometry'].centroid

# %%
gdf_dados_todos_7 = gpd.GeoDataFrame(dados_todos_7, geometry=gpd.points_from_xy(
    dados_todos_7.LONGITUDE, dados_todos_7.LATITUDE))

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
join_left_df_novo = pd.merge(
    join_left_df_novo, dados_ivsUDH, how='left', on='UDH_ATLAS')

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
# dados_todos_9 = dados_todos_8[['Cod_setor','UDH_ATLAS','UC', 'Tipo', 'Medidor', 'Bairro', 'CD_ALIMENTADOR',
#        'LATITUDE', 'LONGITUDE','centroid','UNI_TR_MT', 'domicilio', 'domicilio', 'medio_pessoas', 'rend_responsavel', 'rend_medio',
#        'saneamento', '%alfabetizados', '%ate5salarios', '%5a10']]
dados_todos_9 = dados_todos_8.copy()

dados_todos_9['UDH_ATLAS'] = dados_todos_9['UDH_ATLAS'].astype(str)
dados_todos_9.UDH_ATLAS = dados_todos_9.UDH_ATLAS.str.replace('.', '')

# %%

for ind in dados_todos_9.index:
    dados_todos_9.UDH_ATLAS.values[ind] = dados_todos_9.UDH_ATLAS.values[ind][:-1]


# %%
#dados_todos_10 = pd.merge(dados_todos_9, dados_ivsUDH_2, how='left', on='UDH_ATLAS')
dados_todos_10 = dados_todos_9.copy()

# %%
cod_setor = dados_todos_10['Cod_setor'].value_counts().reset_index()

# %%
#dados_todos_10 = dados_todos_10.drop(['vulner_dia'],axis=1)

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

# %%
#UNI_TR_MT = Trafo

y = dados_todos_10.copy()
y = y.pop('UNI_TR_MT')

# %%
# x = dados_todos_10.drop(['Bairro', 'CD_ALIMENTADOR','UC'],axis=1)

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
teste_4 = x_2[['UC', 'LATITUDE', 'LONGITUDE', 'UNI_TR_MT',
               'Cod_setor', 'PO_NOMINAL_TRAFO', 'Dif', 'Porc']]
teste_4['y_coords'] = teste_4['LATITUDE']
teste_4['x_coords'] = teste_4['LONGITUDE']

unknown_points_2 = teste_4[['x_coords', 'y_coords']].values

# %%
# fig = plt.figure(figsize=(10,10))
# ax = fig.add_subplot()
# dados_ibge.boundary.plot(ax=ax,color = 'grey',label = 'regiões')
# plt.scatter(teste_4['x_coords'], teste_4['y_coords'], 10, 'r', edgecolor='w', label='UC')
# plt.show()

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

out_pdf = r'C:\Pos\figuras\test2.pdf'

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

# # %%
# def get_poligono(long,lat):
#     pnt = Point(long,lat)
#     for i,j in enumerate(tabela_final.geometry):
#         if pnt.within(j):
#             return tabela_final.Clusters.iloc[i]

# %%
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
# teste_2 = teste_2.rename(columns={0: 'medio_pessoas', 1:'rend_responsavel', 2:'rend_medio',3:'saneamento',4:'%alfabetizados'
#                                   , 5:'%ate5salarios', 6:'%5a10', 7:'ivs', 8:'ivs_infraestrutura_urbana', 9:'ivs_capital_humano'
#                                   , 10:'ivs_renda_e_trabalho', 11: 'idhm', 12:'t_eletrica', 13:'t_empregador_18m'})


teste_2_3 = teste_2_2.rename(columns=dict(zip(old_names, new_names)))

# %%
teste_3 = teste_2_3.copy()

teste_3.dropna(inplace=True)

# %%

# 1) Filtrar as inspeções que resultaram em parecer diferente de "normal".
# 2) Fazer um laço while para percorrer todas as inspeções positivas, e
# quando ter sido realizada em algum consumidor de ISL07, atribuir ao transformador essa informação.

dados_inspe_2 = dados_inspe[['UC', 'SS', 'SEQ', 'DATA DA FISCALIZAÇÃO', 'UNIDADE', 'MUNICIPIO',
                             'BAIRRO', 'PARECER2']]

dados_inspe_3 = dados_inspe_2[dados_inspe_2['PARECER2'] != 0]
dados_inspe_3 = dados_inspe_3[dados_inspe_3['PARECER2'] != 3]

dados_inspe_4 = dados_inspe_3[['UC', 'PARECER2']]

dados_todos_UC = dados_todos_10[['Cod_setor', 'UNI_TR_MT',
                                 'UC', 'Bairro', 'CD_ALIMENTADOR', 'LATITUDE', 'LONGITUDE']]

dados_todos_11 = pd.merge(dados_todos_UC, dados_inspe_4,
                          how='left', on='UC').fillna(0)

dados_todos_11_count = dados_todos_11['PARECER2'].value_counts().reset_index()

dados_todos_11_count_2 = dados_todos_11.groupby(
    ['Bairro'])['PARECER2'].count().reset_index()

dados_todos_11_count_3 = dados_todos_11.groupby(
    ['UNI_TR_MT'])['PARECER2'].count().reset_index()

# %%

dados_todos_11_sum = dados_todos_11.groupby("UNI_TR_MT")["PARECER2"].agg(
    PARECER2_TOTAL=lambda x: x[x != 0].count()).reset_index()

dados_todos_11 = pd.merge(
    dados_todos_11, dados_todos_11_sum, how='left', on='UNI_TR_MT').fillna(0)

dados_todos_12 = dados_todos_11[[
    'UNI_TR_MT', 'PARECER2_TOTAL']].drop_duplicates()


# %%
dados_todos_10 = dados_todos_10.replace('X', np.nan)

dados_todos_13 = dados_todos_10[['UNI_TR_MT', 'UDH_ATLAS', 'domicilio', 'medio_pessoas', 'rend_responsavel', 'rend_medio', 'saneamento',
                                 '%alfabetizados', '%ate5salarios', '%5a10', 'ivs', 'ivs_infraestrutura_urbana', 'ivs_capital_humano',
                                 'ivs_renda_e_trabalho', 'idhm', 't_eletrica', 't_empregador_18m']].drop_duplicates()

dados_todos_14 = pd.merge(
    dados_todos_12, dados_todos_13, how='left', on='UNI_TR_MT')

# %%

dados_todos_15 = pd.merge(teste_3, dados_todos_12, how='left', on='UNI_TR_MT')
# %%
dados_todos_15_count = dados_todos_15['UNI_TR_MT'].value_counts().reset_index()

# %%
#dados_todos_11.to_csv(caminho_ISL07_INSPECOES,index=False, sep=';')
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

# %%
gdf_dados_todos_11 = gpd.GeoDataFrame(dados_todos_11, geometry=gpd.points_from_xy(
    dados_todos_11.LONGITUDE, dados_todos_11.LATITUDE))

gdf_dados_todos_11['PARECER2'] = gdf_dados_todos_11['PARECER2'].astype('Int64')

# %%
text1 = '1-IRREGULAR ADULTERACAO NA MEDICAO'
text2 = '2-IRREGULAR CONSUMO FORA MEDIDO'
text3 = '4-AVARIA NO EQUI.MEDICAO'
text4 = '5-LIGACAO CLANDESTINA AUTO-LIGADO'

gdf_dados_todos_12 = gdf_dados_todos_11[gdf_dados_todos_11['PARECER2'] != 0]

#%%
axis = sns.kdeplot(x = gdf_dados_todos_12.centroid.x, y = gdf_dados_todos_12.centroid.y,
                   weights = gdf_dados_todos_12["PARECER2"],
                   fill=True, gridsize=500, bw_adjust=0.25, cmap="coolwarm")

gdf_dados_todos_12.plot(facecolor="none", edgecolor="gray", ax=axis)

axis.set_axis_off()

plt.show()

#%%
import pysal.lib

import pysal.model

gdf_dados_todos_12["ID"] = range(gdf_dados_todos_12.shape[0])

state_weights = pysal.lib.weights.KNN.from_dataframe(gdf_dados_todos_12, ids="ID", k=3)

state_weights.plot(gdf=gdf_dados_todos_12, indexed_on="ID")

plt.show()


#%% https://michaelminn.net/tutorials/python-points/index.html
# https://pysal.org/notebooks/viz/splot/esda_morans_viz.html
# https://www.itl.nist.gov/div898/handbook/index.htm

import esda.moran

import splot.esda

moran = esda.moran.Moran(gdf_dados_todos_12["PARECER2_TOTAL"], state_weights)

splot.esda.plot_moran(moran, zstandard=True, figsize=(10,4))

plt.show()

#%%

moran_local = esda.moran.Moran_Local(gdf_dados_todos_12["PARECER2_TOTAL"], state_weights)

splot.esda.lisa_cluster(moran_local, gdf_dados_todos_12[["PARECER2_TOTAL", "geometry"]])

plt.show()

# %%
dados_fisca = dados_fisca.drop(columns=['ref', 'ref2'])

# %%
dados_fisca = dados_fisca[dados_fisca['PARECER2'] != 0]
dados_fisca = dados_fisca[dados_fisca['PARECER2'] != 3]

# %%
# dados_todos_fisca = dados_celesc[['UC', 'Situação UC', 'LATITUDE', 'LONGITUDE']]

# dados_todos_fisca_teste = dados_todos[['UC', 'Situação UC', 'LATITUDE', 'LONGITUDE']]

# dados_fisca_2 = pd.merge(dados_fisca, dados_todos_fisca, how='left', on='UC').dropna()

# %%
# dados_fisca_3 = pd.merge(dados_fisca, dados_todos_10_2,
#                          how='inner', on='UC').fillna(0)

# %%

local = gdf_dados_todos_11[["LATITUDE", "LONGITUDE", "PARECER2_TOTAL"]]

lat_longs = list(
    map(list, zip(local["LATITUDE"], local["LONGITUDE"], local["PARECER2_TOTAL"])))

# %%


map_h = folium.Map(
    location=[-27.6886, -48.5622],
    tiles="CartoDB positron",
    zoom_start=12,
    min_zoom=6,
    max_zoom=18)

HeatMap(lat_longs, min_opacity=0.2,

        radius=50, blur=50,
        max_zoom=1).add_to(map_h)

# for index, location_info in gdf_dados_todos_11.iterrows():
#     folium.Marker([location_info["LATITUDE"], location_info["LONGITUDE"]], popup=location_info["UC"]).add_to(map_h)

hm_name = "Heat_Map_FISCALI.html"
map_h.save(hm_name)
# %%

base_perdas = pd.read_excel(
    r'C:\Pos\entradas\entrada_MI.xlsx', sheet_name='Planilha1')
base_perdas_1 = base_perdas[['Cod_setor', 'Perdas']]
# %%

dados_todos_15['Cod_setor'] = dados_todos_15['Cod_setor'].astype('int64')

df_merged = dados_todos_15.merge(
    base_perdas_1, on='Cod_setor', how='left').dropna()

# %%

local_perdas = df_merged[["LATITUDE", "LONGITUDE", "Perdas"]]

lat_longs_perdas = list(
    map(list, zip(local_perdas["LATITUDE"], local_perdas["LONGITUDE"], local_perdas["Perdas"])))

#%%
map_h_perdas = folium.Map(
    location=[-27.6886, -48.5622],
    tiles="CartoDB positron",
    zoom_start=12,
    min_zoom=6,
    max_zoom=18)

HeatMap(lat_longs_perdas, min_opacity=0.2,

        radius=50, blur=50,
        max_zoom=1).add_to(map_h_perdas)

# for index, location_info in gdf_dados_todos_11.iterrows():
#     folium.Marker([location_info["LATITUDE"], location_info["LONGITUDE"]], popup=location_info["UC"]).add_to(map_h)

hm_name_perdas = "Heat_Map_Perdas.html"
map_h_perdas.save(hm_name_perdas)


# %%
df_merged["Porc"] = df_merged["Porc"]/100

# %%

df_merged["PARECER2_TOTAL"] = df_merged["PARECER2_TOTAL"].astype(float)

#%%
df_merged['Choice'] = np.where((df_merged['PARECER2_TOTAL'] > 0), 1, 0)

#%%
# BYTES_TO_MB_DIV = 0.000001
# def print_memory_usage_of_data_frame(df):
#     mem = round(df.memory_usage().sum() * BYTES_TO_MB_DIV, 3) 
#     print("Memory usage is " + str(mem) + " MB")
    
# print_memory_usage_of_data_frame(df_merged)

#%%
# sns.kdeplot(df_merged.loc[(df_merged['Choice']==1),'idhm_renda'], color='r', fill=True, label='1')
# sns.kdeplot(df_merged.loc[(df_merged['Choice']==0),'idhm_renda'], color='b', fill=True, label='0')
# plt.show()

#%%

out_pdf = r'C:\Pos\figuras\kdeplot.pdf'

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



for elem in names:
    if elem in df_merged.keys():
        # print(f"{elem}")
        a = f"{elem}"
        plt.figure()
        sns.set(rc={"figure.figsize":(8, 5)})
        
        sns_plot  = sns.kdeplot(df_merged.loc[(df_merged['Choice']==1),a], color='r', fill=True, label='1')
        sns_plot =  sns.kdeplot(df_merged.loc[(df_merged['Choice']==0),a], color='b', fill=True, label='0')
        fig = sns_plot.get_figure()
        #plt.show()
        pdf.savefig(fig)

   
pdf.close()

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
# dados_ivs.boundary.plot(ax=ax,color = 'red',label = 'regiões IVS')
# #dados_ivs.boundary.plot(ax=ax,color = 'grey',label = 'regiões')
# #dados_ibge.apply(lambda x: ax.annotate(text=x.ID, xy=x.geometry.centroid.coords[0], ha='center', fontsize=9),axis=1);
# #plt.plot(x_1.LONGITUDE, x_1.LATITUDE, 'o',color='blue', label='UCs com dados IBGE e IVS')
# #plt.plot(x_2.LONGITUDE, x_2.LATITUDE, 'x',color='magenta', label='UCs sem dados no IBGE')
# #gdf_dados_todos_12.centroid.plot(ax=ax,color='blue', label='UCs',markersize=4)
# gdf_dados_todos_12.apply(lambda x: ax.annotate(text=x.PARECER2, xy=x.geometry.centroid.coords[0], xytext=(1.5, 2.5), textcoords='offset points', fontsize=12),axis=1)
# ax.text(0.3,0.05,'1-IRREGULAR ADULTERACAO NA MEDICAO\n2-IRREGULAR CONSUMO FORA MEDIDO\n4-AVARIA NO EQUI.MEDICAO\n5-LIGACAO CLANDESTINA AUTO-LIGADO',transform=ax.transAxes, bbox=dict(facecolor='grey',edgecolor='black',boxstyle='square'))
# plt.plot(dados_todos_11.LONGITUDE, dados_todos_11.LATITUDE, 'x',color='magenta', label='UCs Inspeções')
# plt.axis('equal')
# ax.legend(bbox_to_anchor=(1, 1), bbox_transform=fig.transFigure)
# plt.show()

#%%

end = datetime.now()
end_time = end.strftime('%H:%M:%S')
#%%

delta = end - start

#%%
print("The time of execution start is :", start_time)
print("The time of execution end is :", end_time)

print("The time of execution is :", delta.total_seconds(), "seconds")
#%% ivs ivs urban infrastructure ivs human capital ivs income e work idhm vulner day t electrical  t employer 18m

dados_ivsUDH2_artigo = dados_ivsUDH_2.rename(columns={'ivs_infraestrutura_urbana':'ivs_urban_infrastructure','ivs_capital_humano':'ivs_human_capital',
                                                      'ivs_renda_e_trabalho':'ivs_income_e_work','vulner_dia':'vulner_day','t_eletrica': 't_electrical',
                                                      't_empregador_18m':'t_employer_18m'})
#%%
dados_ivsUDH2_artigo = dados_ivsUDH2_artigo.drop('ano', axis=1)
#%%
dados_ivsUDH2_artigo = dados_ivsUDH2_artigo.drop('vulner_day', axis=1)
dados_ivsUDH2_artigo =dados_ivsUDH2_artigo.drop('UDH_ATLAS', axis=1)

#%%
artigo_describe = dados_ivsUDH2_artigo.describe()
#%%

dados_ibge_artigo = dados_ibge_dom1[['domicilio','aguageral', 'aguadepoço','banheiroexclusivo','ban_esgoto','ban_outros']]

dados_ibge_artigo = dados_ibge_artigo.replace('X', np.nan)

artigo2_describe = dados_ibge_artigo.describe()

#boxplot = dados_ibge_artigo.boxplot()

boxplot =dados_ibge_artigo.boxplot(whis=[5, 95])
plt.show()
#%%

dados_ibge_artigo_2 = dados_ibge_artigo.copy()

dados_ibge_artigo_2['aguageral'] = np.arcsinh(dados_ibge_artigo_2['aguageral'].values) 

dados_ibge_artigo_2['aguadepoço'] = np.arcsinh(dados_ibge_artigo_2['aguadepoço'].values) 

dados_ibge_artigo_2['banheiroexclusivo'] = np.arcsinh(dados_ibge_artigo_2['banheiroexclusivo'].values) 

dados_ibge_artigo_2['domicilio'] = np.arcsinh(dados_ibge_artigo_2['domicilio'].values)
 
dados_ibge_artigo_2['ban_esgoto'] = np.arcsinh(dados_ibge_artigo_2['ban_esgoto'].values) 

dados_ibge_artigo_2['ban_outros'] = np.arcsinh(dados_ibge_artigo_2['ban_outros'].values) 


#%%
fig, ax = plt.subplots()
#boxplot  = dados_ibge_artigo_2[['aguageral', 'aguadepoço','banheiroexclusivo']].boxplot(whis=[5, 95])

boxplot  = dados_ibge_artigo.boxplot(whis=[5, 95])


plt.show()


#%%

# plt.subplot(2, 1, 1)
# #pylab.hist(dados_ibge_artigo_2['aguageral'][~np.isnan(dados_ibge_artigo_2['aguageral'])])
# ax1= sns.distplot(dados_ibge_artigo_2['aguadepoço'][~np.isnan(dados_ibge_artigo_2['aguadepoço'])])
# plt.title("aguageral arcsinh ", y=-0.20)

# plt.subplot(2, 1, 2)
# ax= sns.distplot(dados_ibge_artigo['aguadepoço'][~np.isnan(dados_ibge_artigo['aguadepoço'])])
# plt.title("aguageral raw", y=-0.30)
# plt.show()

#%%

