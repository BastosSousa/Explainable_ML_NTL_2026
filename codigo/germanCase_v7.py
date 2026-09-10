# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 14:45:43 2025

@author: Natalia
"""

import folium
import matplotlib.backends.backend_pdf
from scipy.spatial import distance_matrix
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import geopandas as gpd

import matplotlib.pyplot as plt

from datetime import datetime

import numpy as np


import os
import platform

from scipy.interpolate import griddata


import warnings
# %%
import pysal.lib
from pysal.lib import weights
import pysal.model
from esda.moran import Moran, Moran_Local
import esda.moran
import splot.esda
from splot.esda import moran_scatterplot
from shapely.geometry import Point
# %%
import pyproj
from geopandas import GeoDataFrame

from pyproj import Transformer
from geopy.geocoders import Nominatim
from time import sleep
from shapely import wkt
import requests
import time
from geopandas import sjoin

# %%
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
caminho_dados_DSO = diretorio_atual+barra + \
    volta_nivel+barra+'entradas'+barra+'DSO-Sachsen-Anhalt-file.xlsx'

# %%

caminho_anhalt = diretorio_atual+barra+volta_nivel+barra + \
    'DVG_ALKIS'+barra+'DVG_ALKIS.shp'


caminho_shp = diretorio_atual+barra+volta_nivel+barra + \
    'vg-hist.utm32s.shape'+barra+'daten'+barra+'utm32s' + \
    barra+'shape'+barra+'VG-Hist_1990-10-03_KRS.shp'


# %%

crs = {'init': 'epsg:4326'}

dados_anhalt = gpd.read_file(caminho_anhalt)
type(dados_anhalt)
dados_anhalt.crs


dados_shp = gpd.read_file(caminho_shp)
type(dados_shp)
dados_shp.crs

#%%
dados_todos = pd.read_excel(caminho_dados_DSO)

# %%

# fig = plt.figure(figsize=(10, 5))  # vulner_dia
# ax = fig.add_subplot()
# #dados_anhalt.boundary.plot(ax=ax, color='red', label='Regions sachsen anhalt')
# dados_shp.plot(ax=ax)
# plt.legend()
# plt.show()
#%%
### EPSG:4326
def geocode_city(city, country="Germany"):
    url = "https://nominatim.openstreetmap.org/search"
    
    params = {
        "q": f"{city}, {country}",
        "format": "json",
        "limit": 1
    }
    
    headers = {
        "User-Agent": "seu_app_nome (seu_email@exemplo.com)"
    }
    
    response = requests.get(url, params=params, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        if len(data) > 0:
            return float(data[0]["lat"]), float(data[0]["lon"])
    
    return None, None
#%%

lats = []
long = []

for city in dados_todos["Ort"]:
    lat, lon = geocode_city(city)
    lats.append(lat)
    long.append(lon)
    time.sleep(1)  # respeita o limite do Nominatim

#%%
dados_todos["latitude"] = lats
dados_todos["longitude"] = long

#%%

gdf_dados_todos = gpd.GeoDataFrame(
    dados_todos, geometry=gpd.points_from_xy(dados_todos.longitude, dados_todos.latitude), crs="EPSG:4326"
)

#%%
gdf_dados_todos = gdf_dados_todos.to_crs(25832)

#%%
# Plot
# fig, ax = plt.subplots(figsize=(10, 5))
# dados_shp.plot(ax=ax)  # shapefile de fundo
# gdf_dados_todos.plot(ax=ax, color='red', markersize=20)
# plt.show()

#%%

# Spatial join: 'inner' só mantém os pontos que caem dentro de algum polígono
gdf_joined = sjoin(
    gdf_dados_todos,  # pontos
    dados_shp,        # polígonos
    how="left",       # se quiser manter todos os pontos mesmo que não caia em polígono, use 'left'
    predicate="within"  # verifica se ponto está dentro do polígono
)


# %%
caminho_dados = diretorio_atual+barra + \
    volta_nivel+barra+'entradas'+barra+'inkar_2024'+barra+'inkar_2024.csv'
    
# %%
caminho_dados_2 = diretorio_atual+barra + \
    volta_nivel+barra+'entradas'+barra+'Zensus2022_Energietraeger_10km-Gitter.csv'

#%%

dados = pd.read_csv(caminho_dados, sep=';', decimal=",")

#%%
dados_Zensus2022 = pd.read_csv(caminho_dados_2, sep=',')

#%%
dados_Zensus2022['geometry'] = [Point(xy) for xy in zip(dados_Zensus2022.x_mp_10km, dados_Zensus2022.y_mp_10km)] 

#%%
gdf = GeoDataFrame(dados_Zensus2022, crs="EPSG:3035", geometry=dados_Zensus2022.geometry)

#%%
gdf = gdf.to_crs(epsg=25832)

# %%

dados_2 = dados[dados['Zeitbezug'] == 2020]

dados_head = dados.head(2000)
# %%

dados_3 = dados_2[['Kuerzel', 'Name', 'Wert']].reset_index(drop=True)

# %%
contagem_valores = dados_3['Name'].value_counts()

# %%
df_pivot = dados_3.pivot_table(
    index="Name", columns="Kuerzel", values="Wert", aggfunc="first").reset_index()

# %%
# df_grouped = dados_3.groupby(["Name", "Kuerzel"])["Wert"].apply(list).unstack()

# %%
dados_head = df_pivot.head(1000)

#%%

df_pivot = df_pivot.rename(columns={'Name': 'GEN'})

# #%%

# lats = []
# long = []

# for city in df_pivot["GEN"]:
#     lat, lon = geocode_city(city)
#     lats.append(lat)
#     long.append(lon)
#     time.sleep(1)  # respeita o limite do Nominatim

# #%%
# df_pivot["latitude"] = lats
# df_pivot["longitude"] = long

# #%%

# gdf_df_pivot = gpd.GeoDataFrame(
#     df_pivot, geometry=gpd.points_from_xy(df_pivot.longitude, df_pivot.latitude), crs="EPSG:4326"
# )

# #%%
# gdf_df_pivot = gdf_df_pivot.to_crs(25832)

#%%

# gdf_df_pivot.to_csv('C:\Pos\germancase\entradas\gdf_df_pivot.csv', encoding='utf-8', index=False)

# %%
caminho_dados_gdf_df_pivot = diretorio_atual+barra + \
    volta_nivel+barra+'entradas'+barra+'gdf_df_pivot.csv'

#%%
gdf_df_pivot = pd.read_csv(caminho_dados_gdf_df_pivot, sep=',', encoding='utf-8')

#%%
gdf_df_pivot["geometry"] = gdf_df_pivot["geometry"].apply(wkt.loads)

gdf_df_pivot = gpd.GeoDataFrame(
    gdf_df_pivot,
    geometry="geometry",
    crs="EPSG:25832"
)

#%%

gdf_joined_2 = sjoin(
    gdf_df_pivot,  # pontos
    dados_shp,        # polígonos
    how="left",       # se quiser manter todos os pontos mesmo que não caia em polígono, use 'left'
    predicate="within"  # verifica se ponto está dentro do polígono
)

#%%

dados_head = gdf_joined_2.head(1000)

#%%

gdf_df_pivot = gdf_df_pivot[~gdf_df_pivot.geometry.isna()].copy()

#%%

def idw_interpolation(xy_known, values_known, xy_unknown, power=2):
    """
    xy_known: array (n, 2) com coordenadas conhecidas
    values_known: array (n,) com valores conhecidos
    xy_unknown: array (m, 2) com coordenadas a interpolar
    """
    interpolated = []

    for x0, y0 in xy_unknown:
        distances = np.sqrt((xy_known[:, 0] - x0)**2 + (xy_known[:, 1] - y0)**2)

        # evita divisão por zero
        if np.any(distances == 0):
            interpolated.append(values_known[distances == 0][0])
        else:
            weights = 1 / distances**power
            interpolated.append(np.sum(weights * values_known) / np.sum(weights))

    return np.array(interpolated)

#%%

coords = np.array([
    (geom.x, geom.y) for geom in gdf_df_pivot.geometry
])

#%%

res = list(gdf_df_pivot.columns)
print(res)

#%%

cols_to_interpolate = ["col1", "col2", "col3"]

for col in cols_to_interpolate:
    if gdf_df_pivot[col].isna().any():
        mask_known = gdf_df_pivot[col].notna()
        mask_unknown = gdf_df_pivot[col].isna()

        xy_known = coords[mask_known]
        values_known = gdf_df_pivot.loc[mask_known, col].values
        xy_unknown = coords[mask_unknown]

        gdf_df_pivot.loc[mask_unknown, col] = idw_interpolation(
            xy_known,
            values_known,
            xy_unknown
        )


#%%

df_merged = df_pivot.merge(dados_shp, on='GEN', how='left').drop_duplicates()

#%%
df_limpo = df_merged.dropna(subset=["geometry"])

#%%
dados_head = df_limpo.head(100)

#%% 
df_limpo = df_limpo.set_geometry('geometry')

#%%
# fig = plt.figure(figsize=(10, 10))  
# col = "OEV20_ANT"

# ax = fig.add_subplot()
# # df_limpo.plot(ax=ax,
# #     column=col,     # coluna usada no choropleth
# #     cmap="OrRd",            # colormap (ou 'viridis', 'Blues', etc.)
# #     legend=True,            # mostra legenda
# #     figsize=(8, 6),         # tamanho do gráfico
# #     edgecolor="black",      # contorno dos polígonos
# #     linewidth=0.5
# # )
# #dados_anhalt.boundary.plot(ax=ax, color='red', label='Regions sachsen anhalt')
# dados_shp.boundary.plot(ax=ax, color='blue')
# gdf.plot(ax=ax, color='purple',marker='o', markersize=9)
# plt.title(f"Choropleth map of '{col}' value", fontsize=14)
# plt.legend()
# plt.show()


#%%
from shapely.geometry import box

def make_grid_fixed_area(gdf, area_km2=10, crs=None):
    """
    Gera grid de quadrados com área fixa sobre o bounding box de um GeoDataFrame.

    Parameters
    ----------
    gdf : GeoDataFrame
        Dados de referência (o grid cobre o bounding box).
    area_km2 : float
        Área desejada de cada quadrado em km².
    crs : str ou None
        CRS para o grid. Se None, herda do gdf.

    Returns
    -------
    GeoDataFrame com o grid.
    """
    xmin, ymin, xmax, ymax = gdf.total_bounds
    
    # lado do quadrado em metros
    cell_size = (area_km2 * 1e6) ** 0.5  
    
    if crs is None:
        crs = gdf.crs
    
    grid_cells = []
    for x0 in np.arange(xmin, xmax, cell_size):
        for y0 in np.arange(ymin, ymax, cell_size):
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            grid_cells.append(box(x0, y0, x1, y1))
    
    grid = gpd.GeoDataFrame(geometry=grid_cells, crs=crs)
    return grid

#%%

grid = make_grid_fixed_area(gdf, area_km2=10, crs="EPSG:25832")

#%%

# Convert 'Numeric_Column' to string using astype()
dados_anhalt['AGS'] = dados_anhalt['AGS'].astype(str)

dados_anhalt_2 = dados_anhalt[dados_anhalt['AGS'].str.startswith('15')]


gdados_anhalt_2 = GeoDataFrame(dados_anhalt_2, crs="EPSG:25832", geometry=dados_anhalt_2.geometry)

#%%
gdf_dados_todos = gdf_dados_todos.rename(columns={'Ort': 'GEN'})

#%%
# fig = plt.figure(figsize=(10, 10))  
# col = "a_hybrid"

# ax = fig.add_subplot()
# df_limpo.plot(ax=ax,
#     column=col,     # coluna usada no choropleth
#     cmap="OrRd",            # colormap (ou 'viridis', 'Blues', etc.)
#     legend=True,            # mostra legenda
#     figsize=(8, 6),         # tamanho do gráfico
#     edgecolor="black",      # contorno dos polígonos
#     linewidth=0.5
# )
# dados_anhalt.boundary.plot(ax=ax, color='red', label='Regions sachsen anhalt')
# #dados_shp.boundary.plot(ax=ax, color='blue')
# dados_shp.boundary.plot(ax=ax)
# gdados_anhalt_2.boundary.plot(ax=ax, color='green')
# #grid.boundary.plot(ax=ax, linewidth=0.5,color="red")
# #gdf.plot(ax=ax, color='purple',marker='o', markersize=9)
# plt.title(f"Choropleth map of '{col}' value", fontsize=14)
# plt.legend()
# plt.show()

#%%
# Separate rows containing any NaN values
df_nan = df_limpo[df_limpo.isna().any(axis=1)]

#%%
# Keep only rows without NaN values
df_no_nan = df_limpo.dropna()


#%%
df_limpo["y"] = df_limpo.centroid.map(lambda p: p.y)

df_limpo["x"] = df_limpo.centroid.map(lambda p: p.x)

#%%
sample_points = df_limpo[['x', 'y']].values

#%% pre processing para a etapa de interpolacao

from sklearn import preprocessing

norma_teste_10 = df_limpo.copy()
#%%
norma_teste_10 = norma_teste_10.drop(["geometry","y","x","GEN"], axis=1)

#%%

norma_teste_11 = norma_teste_10[['Abfahrten_Bahn', 'Abfahrten_Bus', 'Abfahrten_Tram', 'Abfahrten_insg', 'Anzahl_weiterfSchu', 'Betten_Ins', 'Bib', 
                        'BibEntl', 'BibOEB', 'BibSpB', 'BibWiB', 'HalteBahn_28Abf', 'HalteBus_28Abf', 'HalteInsgesamt_28Abf', 'HalteTram_28Abf',
                        'Haltestelle Bahn', 'Haltestelle Bus', 'Haltestelle Tram', 'Haltestelle insgesamt', 'KH_ABNVS_0', 'KH_ABNVS_1', 
                        'KH_ABNVS_2', 'KH_ABNVS_3', 'KH_SPNV_ChestP', 'KH_SPNV_Ki', 'KH_SPNV_SchV', 'KH_SPNV_Spez', 'KH_SPNV_Stroke', 
                        'KH_insg', 'Kitas', 'O03_AG', 'O03_Anzahl', 'O03_LG', 'O03_OLG', 'OEV20_ANT', 'OEV20_DIST', 'SuS_weiterfSchulen', 
                        'a_ALGII_SGBII', 'a_Aufstocker', 'a_BG1P', 'a_BG5um', 'a_BGKind', 'a_GSa', 'a_GSa_f', 'a_GSa_m', 'a_HHwg', 'a_KuA', 
                        'a_Minijobs', 'a_SGBII_f', 'a_Stellen_Experte', 'a_Stellen_Fachkraft', 'a_Stellen_Helfer', 'a_Stellen_Spezialist', 
                        'a_Unterkunft_SGBII', 'a_VerbrInsolv_Selbstständige', 'a_WGlastzu', 'a_WGmietzu', 'a_WZC_Industrie', 
                        'a_WZKN_Dienstleistung', 'a_aloLang', 'a_aloLang_f', 'a_aloLang_m', 'a_alo_ausländer', 'a_alo_ausländer_f', 
                        'a_alo_ausländer_m', 'a_alo_experte', 'a_alo_f', 'a_alo_fachkraft', 'a_alo_helfer', 'a_alo_m', 'a_alo_oAusb', 
                        'a_alo_spezialist', 'a_alo_u25', 'a_alo_u25_f', 'a_alo_u25_m', 'a_alo_ü55', 'a_alo_ü55_f', 'a_alo_ü55_m', 
                        'a_ausl_bev', 'a_ausl_bev_f', 'a_ausp_svw', 'a_azubi', 'a_azubi_m', 'a_azubi_w', 'a_bb_1000Mbits', 'a_bb_100Mbits', 
                        'a_bb_50Mbits', 'a_betr0003_bev0003', 'a_betr0306_bev0306', 'a_betr_groß', 'a_betr_klein', 'a_betr_kleinst', 
                        'a_betr_mittel', 'a_bev0003', 'a_bev0306', 'a_bev0618', 'a_bev1825', 'a_bev1825_f', 'a_bev2530', 'a_bev2530_f', 
                        'a_bev3050', 'a_bev5065', 'a_bev6575', 'a_bev6575_f', 'a_bev65um', 'a_bev65um_f', 'a_bev7585', 'a_bev7585_f', 
                        'a_bev75um', 'a_bev75um_f', 'a_bev85um', 'a_bev85um_f', 'a_bevMZ', 'a_bevOZ', 'a_bev_0006', 'a_bevd150_', 'a_bevf', 
                        'a_bevf_2040', 'a_bs_a', 'a_bs_w', 'a_bws_1sektor', 'a_bws_2sektor', 'a_bws_3sektor', 'a_diesel', 'a_einp_sva', 
                        'a_elektro', 'a_erh', 'a_ewfBG', 'a_ewfBG_55um_', 'a_ewfBG_allein', 'a_ewfBG_f', 'a_ewfBG_u25_', 'a_ewt_primär', 
                        'a_ewt_sekundär', 'a_ewt_tertiär', 'a_fert_wg12', 'a_fert_wg_EE', 'a_fert_wo12', 'a_fert_wo_EE', 'a_fert_wohn', 
                        'a_freifläche', 'a_gas', 'a_gb_Frauen', 'a_gb_Männer', 'a_gb_knj', 'a_gb_knj_prBv', 'a_gb_nj', 'a_gb_nj_prBv', 
                        'a_gb_ü65', 'a_gb_ü65_Frauen', 'a_gb_ü65_Männer', 'a_geb1520', 'a_geb4045', 'a_geb_bev', 'a_gen_wo12', 'a_gen_wo3um', 
                        'a_ges0001_bev0001', 'a_gest_bev', 'a_haus_sperr', 'a_hh_kind', 'a_hheink_hoch', 'a_hheink_mittel', 'a_hheink_niedrig',
                        'a_hybrid', 'a_kb_IgEinr', 'a_landwirtschaft', 'a_naturnah', 'a_organ', 'a_plug_in_hybrid', 'a_sMin', 'a_sch_a', 
                        'a_schul_abi', 'a_schul_abi_m', 'a_schul_abi_w', 'a_schul_haupt', 'a_schul_haupt_m', 'a_schul_haupt_w', 'a_schul_oA', 
                        'a_schul_oA_m', 'a_schul_oA_w', 'a_schul_real', 'a_schul_real_m', 'a_schul_real_w', 'a_schutzs_abev', 'a_schutzs_bev', 
                        'a_sgbII_III_bev', 'a_stud_1', 'a_stud_a', 'a_stud_m', 'a_stud_w', 'a_suv_fl', 'a_sva_akadem', 'a_sva_exp', 'a_sva_fach',
                        'a_sva_helf', 'a_sva_mAbschluss', 'a_sva_oAbschluss', 'a_sva_spez', 'a_sva_tz', 'a_sva_tz_f', 'a_svb_Bau', 
                        'a_svb_Handwerk', 'a_svb_IT', 'a_svb_kreativ', 'a_svb_primär', 'a_svb_sekundär', 'a_svb_tertiär', 
                        'a_svb_unternehmensdienstlstg', 'a_svb_wissen', 'a_svw_akadem', 'a_svw_mAbschluss', 'a_svw_oAbschluss', 'a_wald', 
                        'a_wasser', 'a_wertstoff', 'a_wg_wo12', 'a_wg_wo3um', 'a_wo_r12', 'a_wo_r5um', 'a_wo_wg12', 'a_wo_wg3um', 'a_übern_ausl',
                        'ag_nst', 'allgbSchule_Anzahl', 'allgbSchulen_Förder', 'alo', 'auspend', 'bev_korr', 'bip', 'd_ALGII_Lstg', 
                        'd_ALGI_Frau', 'd_ALGI_Lstg', 'd_ALGI_Mann', 'd_ArbStd', 'd_Bruttoverdienst', 'd_Bruttoverdienst_Prod', 
                        'd_Hausshaltseinkommen', 'd_KW_BL', 'd_SGBII_Lstg', 'd_Unterkunft_Lstg', 'd_VerbrInsolv', 'd_VerbrInsolv_Forderungen', 
                        'd_Wohngeld_Lstg', 'd_uinsolv_ford', 'e10_bev', 'e5_bev', 'einpend', 'et1000', 'ewf_1565_ges', 'fläche_ges_qkm', 
                        'grundschule_anzahl', 'i_TFR', 'i_beschq_mf', 'i_fz_bab', 'i_fz_bh', 'i_fz_fh', 'i_fz_klv', 'i_fz_mz', 'i_fz_oz', 
                        'i_saldo_nat', 'i_wans', 'i_wans_a', 'i_wans_a_f', 'i_wans_a_m', 'i_wans_b', 'i_wans_b_00183050', 'i_wans_b_1825', 
                        'i_wans_b_1825_f', 'i_wans_b_1825_m', 'i_wans_b_2530', 'i_wans_b_2530_f', 'i_wans_b_2530_m', 'i_wans_b_3050', 
                        'i_wans_b_65um', 'i_wans_b_f', 'i_wans_b_m', 'i_wans_vol', 'i_wans_vol_f', 'i_wans_vol_m', 'lg_nst', 'm_bev_alter', 
                        'm_ek', 'm_ek_2555', 'm_ek_5565', 'm_ek_akad', 'm_ek_f', 'm_ek_m', 'm_ek_oakad', 'm_mietpr', 'm_übern', 'npend', 
                        'olg_nst', 'q_Grundschüler', 'q_HH', 'q_HH1', 'q_PBG_bev', 'q_SGBII', 'q_SUSförder1000EW', 'q_SUSweiterfSchulen', 
                        'q_Schuldner', 'q_Stud_1000Einw', 'q_abf_bev', 'q_abhg_alt', 'q_abhg_jung', 'q_allgemeinärzte_bev', 'q_alo', 'q_alo_f', 
                        'q_alo_m', 'q_alo_u25', 'q_alo_u25_einw', 'q_alo_u25_einw_f', 'q_alo_u25_einw_m', 'q_alo_ü55', 'q_alo_ü55_einw', 
                        'q_alo_ü55_einw_f', 'q_alo_ü55_einw_m', 'q_arbeitslosigkeit', 'q_asyl_bev', 'q_auspl', 'q_azubi_1525', 'q_baugew_umsatz', 
                        'q_bauueb', 'q_benzin', 'q_betten_bev', 'q_bev_arzt', 'q_bev_fl', 'q_bevsva_qkm', 'q_bip_et', 'q_bip_ew', 'q_bs', 
                        'q_bws_1sektor', 'q_bws_2sektor', 'q_bws_3sektor', 'q_bws_et', 'q_dienstleistung', 'q_ebuerg_aus', 'q_ebuerg_ew', 
                        'q_ehe', 'q_einzelhandelskaufkraft', 'q_erhfl_bev', 'q_ew_suv_qkm', 'q_ewp', 'q_ewp_f', 'q_ewp_m', 'q_fert_wo12_bev', 
                        'q_fert_wo3um_bev', 'q_fert_wo_bev', 'q_fort_bev', 'q_fort_f_bev', 'q_fort_m_bev', 'q_freifläche_bev', 'q_gen_wo_ew', 
                        'q_gsch_kk', 'q_handw_umsatz', 'q_hausabf_bev', 'q_hausarzt_bev', 'q_industrie', 'q_internist_bev', 'q_iumsatz', 
                        'q_iumsatz_ausl', 'q_kaufkraft', 'q_kb_Pers', 'q_kb_bev', 'q_kinderarzt_kinder', 'q_ladepunkte', 'q_ladepunkte_elek', 
                        'q_naturnah_bev', 'q_newfBGu15_bev', 'q_pendlersaldo', 'q_pkw_bev', 'q_rwm_amph_05_bev', 'q_rwm_amph_10_bev', 
                        'q_rwm_dpf_05_bev', 'q_rwm_dpf_10_bev', 'q_rwm_grw_gw_05_bev', 'q_rwm_grw_gw_10_bev', 'q_rwm_grw_ri_05_bev', 
                        'q_rwm_grw_ri_10_bev', 'q_rwm_hsf_05_bev', 'q_rwm_hsf_10_bev', 'q_rwm_sbf_05_bev', 'q_rwm_sbf_10_bev', 'q_sch', 
                        'q_sch_a618', 'q_scheid', 'q_schlafg_bev', 'q_schueler1000EW', 'q_selbst', 'q_straf', 'q_stud', 'q_stud_1825', 
                        'q_stud_fh', 'q_suv_ew', 'q_sva_bev', 'q_svw', 'q_svw_15u30', 'q_svw_ausl', 'q_svw_bev', 'q_svw_f', 'q_svw_m', 
                        'q_svw_ü55', 'q_uinsolv_betr', 'q_unf_bev', 'q_vunp_bev', 'q_vunpt_bev', 'q_wofl_bev', 'q_wohneinbr', 'q_zu_bev', 
                        'q_zu_f_bev', 'q_zu_m_bev', 'q_ärzte_bev', 'q_übern_bev', 'r_ewf_jungalt', 's_SvbAO_SvbWO', 'stud', 'sus_allgbSchul', 
                        'sus_allgbSchuleFörder', 'sus_grundschule', 'sva', 'svw', 'v_suvfl_vorjahr', 'xbev', 'xbevf', 'xbevm']]

#%%
x = norma_teste_11.values #returns a numpy array
#%%
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

#%%
norma_teste_1 = pd.DataFrame(x_scaled, columns= norma_teste_11.columns)

#%% pre processing para a etapa de interpolacao

values = norma_teste_10[['Abfahrten_Bahn', 'Abfahrten_Bus', 'Abfahrten_Tram', 'Abfahrten_insg', 'Anzahl_weiterfSchu', 'Betten_Ins', 'Bib', 
                        'BibEntl', 'BibOEB', 'BibSpB', 'BibWiB', 'HalteBahn_28Abf', 'HalteBus_28Abf', 'HalteInsgesamt_28Abf', 'HalteTram_28Abf',
                        'Haltestelle Bahn', 'Haltestelle Bus', 'Haltestelle Tram', 'Haltestelle insgesamt', 'KH_ABNVS_0', 'KH_ABNVS_1', 
                        'KH_ABNVS_2', 'KH_ABNVS_3', 'KH_SPNV_ChestP', 'KH_SPNV_Ki', 'KH_SPNV_SchV', 'KH_SPNV_Spez', 'KH_SPNV_Stroke', 
                        'KH_insg', 'Kitas', 'O03_AG', 'O03_Anzahl', 'O03_LG', 'O03_OLG', 'OEV20_ANT', 'OEV20_DIST', 'SuS_weiterfSchulen', 
                        'a_ALGII_SGBII', 'a_Aufstocker', 'a_BG1P', 'a_BG5um', 'a_BGKind', 'a_GSa', 'a_GSa_f', 'a_GSa_m', 'a_HHwg', 'a_KuA', 
                        'a_Minijobs', 'a_SGBII_f', 'a_Stellen_Experte', 'a_Stellen_Fachkraft', 'a_Stellen_Helfer', 'a_Stellen_Spezialist', 
                        'a_Unterkunft_SGBII', 'a_VerbrInsolv_Selbstständige', 'a_WGlastzu', 'a_WGmietzu', 'a_WZC_Industrie', 
                        'a_WZKN_Dienstleistung', 'a_aloLang', 'a_aloLang_f', 'a_aloLang_m', 'a_alo_ausländer', 'a_alo_ausländer_f', 
                        'a_alo_ausländer_m', 'a_alo_experte', 'a_alo_f', 'a_alo_fachkraft', 'a_alo_helfer', 'a_alo_m', 'a_alo_oAusb', 
                        'a_alo_spezialist', 'a_alo_u25', 'a_alo_u25_f', 'a_alo_u25_m', 'a_alo_ü55', 'a_alo_ü55_f', 'a_alo_ü55_m', 
                        'a_ausl_bev', 'a_ausl_bev_f', 'a_ausp_svw', 'a_azubi', 'a_azubi_m', 'a_azubi_w', 'a_bb_1000Mbits', 'a_bb_100Mbits', 
                        'a_bb_50Mbits', 'a_betr0003_bev0003', 'a_betr0306_bev0306', 'a_betr_groß', 'a_betr_klein', 'a_betr_kleinst', 
                        'a_betr_mittel', 'a_bev0003', 'a_bev0306', 'a_bev0618', 'a_bev1825', 'a_bev1825_f', 'a_bev2530', 'a_bev2530_f', 
                        'a_bev3050', 'a_bev5065', 'a_bev6575', 'a_bev6575_f', 'a_bev65um', 'a_bev65um_f', 'a_bev7585', 'a_bev7585_f', 
                        'a_bev75um', 'a_bev75um_f', 'a_bev85um', 'a_bev85um_f', 'a_bevMZ', 'a_bevOZ', 'a_bev_0006', 'a_bevd150_', 'a_bevf', 
                        'a_bevf_2040', 'a_bs_a', 'a_bs_w', 'a_bws_1sektor', 'a_bws_2sektor', 'a_bws_3sektor', 'a_diesel', 'a_einp_sva', 
                        'a_elektro', 'a_erh', 'a_ewfBG', 'a_ewfBG_55um_', 'a_ewfBG_allein', 'a_ewfBG_f', 'a_ewfBG_u25_', 'a_ewt_primär', 
                        'a_ewt_sekundär', 'a_ewt_tertiär', 'a_fert_wg12', 'a_fert_wg_EE', 'a_fert_wo12', 'a_fert_wo_EE', 'a_fert_wohn', 
                        'a_freifläche', 'a_gas', 'a_gb_Frauen', 'a_gb_Männer', 'a_gb_knj', 'a_gb_knj_prBv', 'a_gb_nj', 'a_gb_nj_prBv', 
                        'a_gb_ü65', 'a_gb_ü65_Frauen', 'a_gb_ü65_Männer', 'a_geb1520', 'a_geb4045', 'a_geb_bev', 'a_gen_wo12', 'a_gen_wo3um', 
                        'a_ges0001_bev0001', 'a_gest_bev', 'a_haus_sperr', 'a_hh_kind', 'a_hheink_hoch', 'a_hheink_mittel', 'a_hheink_niedrig',
                        'a_hybrid', 'a_kb_IgEinr', 'a_landwirtschaft', 'a_naturnah', 'a_organ', 'a_plug_in_hybrid', 'a_sMin', 'a_sch_a', 
                        'a_schul_abi', 'a_schul_abi_m', 'a_schul_abi_w', 'a_schul_haupt', 'a_schul_haupt_m', 'a_schul_haupt_w', 'a_schul_oA', 
                        'a_schul_oA_m', 'a_schul_oA_w', 'a_schul_real', 'a_schul_real_m', 'a_schul_real_w', 'a_schutzs_abev', 'a_schutzs_bev', 
                        'a_sgbII_III_bev', 'a_stud_1', 'a_stud_a', 'a_stud_m', 'a_stud_w', 'a_suv_fl', 'a_sva_akadem', 'a_sva_exp', 'a_sva_fach',
                        'a_sva_helf', 'a_sva_mAbschluss', 'a_sva_oAbschluss', 'a_sva_spez', 'a_sva_tz', 'a_sva_tz_f', 'a_svb_Bau', 
                        'a_svb_Handwerk', 'a_svb_IT', 'a_svb_kreativ', 'a_svb_primär', 'a_svb_sekundär', 'a_svb_tertiär', 
                        'a_svb_unternehmensdienstlstg', 'a_svb_wissen', 'a_svw_akadem', 'a_svw_mAbschluss', 'a_svw_oAbschluss', 'a_wald', 
                        'a_wasser', 'a_wertstoff', 'a_wg_wo12', 'a_wg_wo3um', 'a_wo_r12', 'a_wo_r5um', 'a_wo_wg12', 'a_wo_wg3um', 'a_übern_ausl',
                        'ag_nst', 'allgbSchule_Anzahl', 'allgbSchulen_Förder', 'alo', 'auspend', 'bev_korr', 'bip', 'd_ALGII_Lstg', 
                        'd_ALGI_Frau', 'd_ALGI_Lstg', 'd_ALGI_Mann', 'd_ArbStd', 'd_Bruttoverdienst', 'd_Bruttoverdienst_Prod', 
                        'd_Hausshaltseinkommen', 'd_KW_BL', 'd_SGBII_Lstg', 'd_Unterkunft_Lstg', 'd_VerbrInsolv', 'd_VerbrInsolv_Forderungen', 
                        'd_Wohngeld_Lstg', 'd_uinsolv_ford', 'e10_bev', 'e5_bev', 'einpend', 'et1000', 'ewf_1565_ges', 'fläche_ges_qkm', 
                        'grundschule_anzahl', 'i_TFR', 'i_beschq_mf', 'i_fz_bab', 'i_fz_bh', 'i_fz_fh', 'i_fz_klv', 'i_fz_mz', 'i_fz_oz', 
                        'i_saldo_nat', 'i_wans', 'i_wans_a', 'i_wans_a_f', 'i_wans_a_m', 'i_wans_b', 'i_wans_b_00183050', 'i_wans_b_1825', 
                        'i_wans_b_1825_f', 'i_wans_b_1825_m', 'i_wans_b_2530', 'i_wans_b_2530_f', 'i_wans_b_2530_m', 'i_wans_b_3050', 
                        'i_wans_b_65um', 'i_wans_b_f', 'i_wans_b_m', 'i_wans_vol', 'i_wans_vol_f', 'i_wans_vol_m', 'lg_nst', 'm_bev_alter', 
                        'm_ek', 'm_ek_2555', 'm_ek_5565', 'm_ek_akad', 'm_ek_f', 'm_ek_m', 'm_ek_oakad', 'm_mietpr', 'm_übern', 'npend', 
                        'olg_nst', 'q_Grundschüler', 'q_HH', 'q_HH1', 'q_PBG_bev', 'q_SGBII', 'q_SUSförder1000EW', 'q_SUSweiterfSchulen', 
                        'q_Schuldner', 'q_Stud_1000Einw', 'q_abf_bev', 'q_abhg_alt', 'q_abhg_jung', 'q_allgemeinärzte_bev', 'q_alo', 'q_alo_f', 
                        'q_alo_m', 'q_alo_u25', 'q_alo_u25_einw', 'q_alo_u25_einw_f', 'q_alo_u25_einw_m', 'q_alo_ü55', 'q_alo_ü55_einw', 
                        'q_alo_ü55_einw_f', 'q_alo_ü55_einw_m', 'q_arbeitslosigkeit', 'q_asyl_bev', 'q_auspl', 'q_azubi_1525', 'q_baugew_umsatz', 
                        'q_bauueb', 'q_benzin', 'q_betten_bev', 'q_bev_arzt', 'q_bev_fl', 'q_bevsva_qkm', 'q_bip_et', 'q_bip_ew', 'q_bs', 
                        'q_bws_1sektor', 'q_bws_2sektor', 'q_bws_3sektor', 'q_bws_et', 'q_dienstleistung', 'q_ebuerg_aus', 'q_ebuerg_ew', 
                        'q_ehe', 'q_einzelhandelskaufkraft', 'q_erhfl_bev', 'q_ew_suv_qkm', 'q_ewp', 'q_ewp_f', 'q_ewp_m', 'q_fert_wo12_bev', 
                        'q_fert_wo3um_bev', 'q_fert_wo_bev', 'q_fort_bev', 'q_fort_f_bev', 'q_fort_m_bev', 'q_freifläche_bev', 'q_gen_wo_ew', 
                        'q_gsch_kk', 'q_handw_umsatz', 'q_hausabf_bev', 'q_hausarzt_bev', 'q_industrie', 'q_internist_bev', 'q_iumsatz', 
                        'q_iumsatz_ausl', 'q_kaufkraft', 'q_kb_Pers', 'q_kb_bev', 'q_kinderarzt_kinder', 'q_ladepunkte', 'q_ladepunkte_elek', 
                        'q_naturnah_bev', 'q_newfBGu15_bev', 'q_pendlersaldo', 'q_pkw_bev', 'q_rwm_amph_05_bev', 'q_rwm_amph_10_bev', 
                        'q_rwm_dpf_05_bev', 'q_rwm_dpf_10_bev', 'q_rwm_grw_gw_05_bev', 'q_rwm_grw_gw_10_bev', 'q_rwm_grw_ri_05_bev', 
                        'q_rwm_grw_ri_10_bev', 'q_rwm_hsf_05_bev', 'q_rwm_hsf_10_bev', 'q_rwm_sbf_05_bev', 'q_rwm_sbf_10_bev', 'q_sch', 
                        'q_sch_a618', 'q_scheid', 'q_schlafg_bev', 'q_schueler1000EW', 'q_selbst', 'q_straf', 'q_stud', 'q_stud_1825', 
                        'q_stud_fh', 'q_suv_ew', 'q_sva_bev', 'q_svw', 'q_svw_15u30', 'q_svw_ausl', 'q_svw_bev', 'q_svw_f', 'q_svw_m', 
                        'q_svw_ü55', 'q_uinsolv_betr', 'q_unf_bev', 'q_vunp_bev', 'q_vunpt_bev', 'q_wofl_bev', 'q_wohneinbr', 'q_zu_bev', 
                        'q_zu_f_bev', 'q_zu_m_bev', 'q_ärzte_bev', 'q_übern_bev', 'r_ewf_jungalt', 's_SvbAO_SvbWO', 'stud', 'sus_allgbSchul', 
                        'sus_allgbSchuleFörder', 'sus_grundschule', 'sva', 'svw', 'v_suvfl_vorjahr', 'xbev', 'xbevf', 'xbevm']].values

#%%
# Define the unknown point coordinates (e.g., a 2D grid)
df_nan['y_coords'] = df_nan.centroid.map(lambda p: p.y)
df_nan['x_coords'] = df_nan.centroid.map(lambda p: p.x)

unknown_points = df_nan[['x_coords', 'y_coords']].values

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
    
    # Convert to NumPy arrays
    sample_points = np.asarray(sample_points)
    unknown_points = np.asarray(unknown_points)
    values = np.asarray(values)
    
    # Remover pontos com valores NaN
    mask = ~np.isnan(values)
    sample_points = sample_points[mask]
    values = values[mask]
    
    # Calcular distâncias
    distances = distance_matrix(sample_points, unknown_points)
    
    # Evitar divisão por zero
    distances[distances == 0] = 1e-10
    
    # Calcular pesos com base na distância inversa
    weights = 1 / np.power(distances, power)
    
    # Calcular os valores interpolados
    interpolated_values = np.sum(weights * values[:, np.newaxis], axis=0) / np.sum(weights, axis=0)
    
    return interpolated_values


#%% Dividir dados em pacotes menores de numero 'chunk_size' de linhas

chunk_size = 1500

chunk_unknown_points = np.split(unknown_points, range(chunk_size, unknown_points.shape[0], chunk_size))


#%% preparar dataframe que ira receber os dados interpolados
#%% funcao para aplicar a funcao de interpolacao para os pacotes
columns = len(values[0])

def process_chunk(chunk):
    
    global resultado
    global result
    global chunk_result
    #for chunk in chunk_unknown_points:
    chunk_result = pd.DataFrame()
    result =  pd.DataFrame()
    resultado = pd.DataFrame()

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

resultado_2 = resultado.reset_index(drop=True)


#%%
new_names = ['Abfahrten_Bahn', 'Abfahrten_Bus', 'Abfahrten_Tram', 'Abfahrten_insg', 'Anzahl_weiterfSchu', 'Betten_Ins', 'Bib', 
                        'BibEntl', 'BibOEB', 'BibSpB', 'BibWiB', 'HalteBahn_28Abf', 'HalteBus_28Abf', 'HalteInsgesamt_28Abf', 'HalteTram_28Abf',
                        'Haltestelle Bahn', 'Haltestelle Bus', 'Haltestelle Tram', 'Haltestelle insgesamt', 'KH_ABNVS_0', 'KH_ABNVS_1', 
                        'KH_ABNVS_2', 'KH_ABNVS_3', 'KH_SPNV_ChestP', 'KH_SPNV_Ki', 'KH_SPNV_SchV', 'KH_SPNV_Spez', 'KH_SPNV_Stroke', 
                        'KH_insg', 'Kitas', 'O03_AG', 'O03_Anzahl', 'O03_LG', 'O03_OLG', 'OEV20_ANT', 'OEV20_DIST', 'SuS_weiterfSchulen', 
                        'a_ALGII_SGBII', 'a_Aufstocker', 'a_BG1P', 'a_BG5um', 'a_BGKind', 'a_GSa', 'a_GSa_f', 'a_GSa_m', 'a_HHwg', 'a_KuA', 
                        'a_Minijobs', 'a_SGBII_f', 'a_Stellen_Experte', 'a_Stellen_Fachkraft', 'a_Stellen_Helfer', 'a_Stellen_Spezialist', 
                        'a_Unterkunft_SGBII', 'a_VerbrInsolv_Selbstständige', 'a_WGlastzu', 'a_WGmietzu', 'a_WZC_Industrie', 
                        'a_WZKN_Dienstleistung', 'a_aloLang', 'a_aloLang_f', 'a_aloLang_m', 'a_alo_ausländer', 'a_alo_ausländer_f', 
                        'a_alo_ausländer_m', 'a_alo_experte', 'a_alo_f', 'a_alo_fachkraft', 'a_alo_helfer', 'a_alo_m', 'a_alo_oAusb', 
                        'a_alo_spezialist', 'a_alo_u25', 'a_alo_u25_f', 'a_alo_u25_m', 'a_alo_ü55', 'a_alo_ü55_f', 'a_alo_ü55_m', 
                        'a_ausl_bev', 'a_ausl_bev_f', 'a_ausp_svw', 'a_azubi', 'a_azubi_m', 'a_azubi_w', 'a_bb_1000Mbits', 'a_bb_100Mbits', 
                        'a_bb_50Mbits', 'a_betr0003_bev0003', 'a_betr0306_bev0306', 'a_betr_groß', 'a_betr_klein', 'a_betr_kleinst', 
                        'a_betr_mittel', 'a_bev0003', 'a_bev0306', 'a_bev0618', 'a_bev1825', 'a_bev1825_f', 'a_bev2530', 'a_bev2530_f', 
                        'a_bev3050', 'a_bev5065', 'a_bev6575', 'a_bev6575_f', 'a_bev65um', 'a_bev65um_f', 'a_bev7585', 'a_bev7585_f', 
                        'a_bev75um', 'a_bev75um_f', 'a_bev85um', 'a_bev85um_f', 'a_bevMZ', 'a_bevOZ', 'a_bev_0006', 'a_bevd150_', 'a_bevf', 
                        'a_bevf_2040', 'a_bs_a', 'a_bs_w', 'a_bws_1sektor', 'a_bws_2sektor', 'a_bws_3sektor', 'a_diesel', 'a_einp_sva', 
                        'a_elektro', 'a_erh', 'a_ewfBG', 'a_ewfBG_55um_', 'a_ewfBG_allein', 'a_ewfBG_f', 'a_ewfBG_u25_', 'a_ewt_primär', 
                        'a_ewt_sekundär', 'a_ewt_tertiär', 'a_fert_wg12', 'a_fert_wg_EE', 'a_fert_wo12', 'a_fert_wo_EE', 'a_fert_wohn', 
                        'a_freifläche', 'a_gas', 'a_gb_Frauen', 'a_gb_Männer', 'a_gb_knj', 'a_gb_knj_prBv', 'a_gb_nj', 'a_gb_nj_prBv', 
                        'a_gb_ü65', 'a_gb_ü65_Frauen', 'a_gb_ü65_Männer', 'a_geb1520', 'a_geb4045', 'a_geb_bev', 'a_gen_wo12', 'a_gen_wo3um', 
                        'a_ges0001_bev0001', 'a_gest_bev', 'a_haus_sperr', 'a_hh_kind', 'a_hheink_hoch', 'a_hheink_mittel', 'a_hheink_niedrig',
                        'a_hybrid', 'a_kb_IgEinr', 'a_landwirtschaft', 'a_naturnah', 'a_organ', 'a_plug_in_hybrid', 'a_sMin', 'a_sch_a', 
                        'a_schul_abi', 'a_schul_abi_m', 'a_schul_abi_w', 'a_schul_haupt', 'a_schul_haupt_m', 'a_schul_haupt_w', 'a_schul_oA', 
                        'a_schul_oA_m', 'a_schul_oA_w', 'a_schul_real', 'a_schul_real_m', 'a_schul_real_w', 'a_schutzs_abev', 'a_schutzs_bev', 
                        'a_sgbII_III_bev', 'a_stud_1', 'a_stud_a', 'a_stud_m', 'a_stud_w', 'a_suv_fl', 'a_sva_akadem', 'a_sva_exp', 'a_sva_fach',
                        'a_sva_helf', 'a_sva_mAbschluss', 'a_sva_oAbschluss', 'a_sva_spez', 'a_sva_tz', 'a_sva_tz_f', 'a_svb_Bau', 
                        'a_svb_Handwerk', 'a_svb_IT', 'a_svb_kreativ', 'a_svb_primär', 'a_svb_sekundär', 'a_svb_tertiär', 
                        'a_svb_unternehmensdienstlstg', 'a_svb_wissen', 'a_svw_akadem', 'a_svw_mAbschluss', 'a_svw_oAbschluss', 'a_wald', 
                        'a_wasser', 'a_wertstoff', 'a_wg_wo12', 'a_wg_wo3um', 'a_wo_r12', 'a_wo_r5um', 'a_wo_wg12', 'a_wo_wg3um', 'a_übern_ausl',
                        'ag_nst', 'allgbSchule_Anzahl', 'allgbSchulen_Förder', 'alo', 'auspend', 'bev_korr', 'bip', 'd_ALGII_Lstg', 
                        'd_ALGI_Frau', 'd_ALGI_Lstg', 'd_ALGI_Mann', 'd_ArbStd', 'd_Bruttoverdienst', 'd_Bruttoverdienst_Prod', 
                        'd_Hausshaltseinkommen', 'd_KW_BL', 'd_SGBII_Lstg', 'd_Unterkunft_Lstg', 'd_VerbrInsolv', 'd_VerbrInsolv_Forderungen', 
                        'd_Wohngeld_Lstg', 'd_uinsolv_ford', 'e10_bev', 'e5_bev', 'einpend', 'et1000', 'ewf_1565_ges', 'fläche_ges_qkm', 
                        'grundschule_anzahl', 'i_TFR', 'i_beschq_mf', 'i_fz_bab', 'i_fz_bh', 'i_fz_fh', 'i_fz_klv', 'i_fz_mz', 'i_fz_oz', 
                        'i_saldo_nat', 'i_wans', 'i_wans_a', 'i_wans_a_f', 'i_wans_a_m', 'i_wans_b', 'i_wans_b_00183050', 'i_wans_b_1825', 
                        'i_wans_b_1825_f', 'i_wans_b_1825_m', 'i_wans_b_2530', 'i_wans_b_2530_f', 'i_wans_b_2530_m', 'i_wans_b_3050', 
                        'i_wans_b_65um', 'i_wans_b_f', 'i_wans_b_m', 'i_wans_vol', 'i_wans_vol_f', 'i_wans_vol_m', 'lg_nst', 'm_bev_alter', 
                        'm_ek', 'm_ek_2555', 'm_ek_5565', 'm_ek_akad', 'm_ek_f', 'm_ek_m', 'm_ek_oakad', 'm_mietpr', 'm_übern', 'npend', 
                        'olg_nst', 'q_Grundschüler', 'q_HH', 'q_HH1', 'q_PBG_bev', 'q_SGBII', 'q_SUSförder1000EW', 'q_SUSweiterfSchulen', 
                        'q_Schuldner', 'q_Stud_1000Einw', 'q_abf_bev', 'q_abhg_alt', 'q_abhg_jung', 'q_allgemeinärzte_bev', 'q_alo', 'q_alo_f', 
                        'q_alo_m', 'q_alo_u25', 'q_alo_u25_einw', 'q_alo_u25_einw_f', 'q_alo_u25_einw_m', 'q_alo_ü55', 'q_alo_ü55_einw', 
                        'q_alo_ü55_einw_f', 'q_alo_ü55_einw_m', 'q_arbeitslosigkeit', 'q_asyl_bev', 'q_auspl', 'q_azubi_1525', 'q_baugew_umsatz', 
                        'q_bauueb', 'q_benzin', 'q_betten_bev', 'q_bev_arzt', 'q_bev_fl', 'q_bevsva_qkm', 'q_bip_et', 'q_bip_ew', 'q_bs', 
                        'q_bws_1sektor', 'q_bws_2sektor', 'q_bws_3sektor', 'q_bws_et', 'q_dienstleistung', 'q_ebuerg_aus', 'q_ebuerg_ew', 
                        'q_ehe', 'q_einzelhandelskaufkraft', 'q_erhfl_bev', 'q_ew_suv_qkm', 'q_ewp', 'q_ewp_f', 'q_ewp_m', 'q_fert_wo12_bev', 
                        'q_fert_wo3um_bev', 'q_fert_wo_bev', 'q_fort_bev', 'q_fort_f_bev', 'q_fort_m_bev', 'q_freifläche_bev', 'q_gen_wo_ew', 
                        'q_gsch_kk', 'q_handw_umsatz', 'q_hausabf_bev', 'q_hausarzt_bev', 'q_industrie', 'q_internist_bev', 'q_iumsatz', 
                        'q_iumsatz_ausl', 'q_kaufkraft', 'q_kb_Pers', 'q_kb_bev', 'q_kinderarzt_kinder', 'q_ladepunkte', 'q_ladepunkte_elek', 
                        'q_naturnah_bev', 'q_newfBGu15_bev', 'q_pendlersaldo', 'q_pkw_bev', 'q_rwm_amph_05_bev', 'q_rwm_amph_10_bev', 
                        'q_rwm_dpf_05_bev', 'q_rwm_dpf_10_bev', 'q_rwm_grw_gw_05_bev', 'q_rwm_grw_gw_10_bev', 'q_rwm_grw_ri_05_bev', 
                        'q_rwm_grw_ri_10_bev', 'q_rwm_hsf_05_bev', 'q_rwm_hsf_10_bev', 'q_rwm_sbf_05_bev', 'q_rwm_sbf_10_bev', 'q_sch', 
                        'q_sch_a618', 'q_scheid', 'q_schlafg_bev', 'q_schueler1000EW', 'q_selbst', 'q_straf', 'q_stud', 'q_stud_1825', 
                        'q_stud_fh', 'q_suv_ew', 'q_sva_bev', 'q_svw', 'q_svw_15u30', 'q_svw_ausl', 'q_svw_bev', 'q_svw_f', 'q_svw_m', 
                        'q_svw_ü55', 'q_uinsolv_betr', 'q_unf_bev', 'q_vunp_bev', 'q_vunpt_bev', 'q_wofl_bev', 'q_wohneinbr', 'q_zu_bev', 
                        'q_zu_f_bev', 'q_zu_m_bev', 'q_ärzte_bev', 'q_übern_bev', 'r_ewf_jungalt', 's_SvbAO_SvbWO', 'stud', 'sus_allgbSchul', 
                        'sus_allgbSchuleFörder', 'sus_grundschule', 'sva', 'svw', 'v_suvfl_vorjahr', 'xbev', 'xbevf', 'xbevm']

#%%

old_names = list(range(len(new_names)))

resultado_3 = resultado_2.rename(
    columns=dict(zip(old_names, new_names))
)


#%%
teste_4 = df_limpo[["geometry","y","x","GEN"]]

teste_4 = teste_4.mask(teste_4.eq('None')).dropna()

teste_4['y_coords'] = teste_4['y']
teste_4['x_coords'] = teste_4['x']

unknown_points_2 = teste_4[['x_coords', 'y_coords']].values

chunk_unknown_points_2 = np.split(unknown_points_2, range(chunk_size, unknown_points_2.shape[0], chunk_size))

#%%

for i, chunk in enumerate(chunk_unknown_points_2):
    print(f'Processing Chunk {i+1}')
    process_chunk(chunk)

#%%

resultado_4 = resultado.reset_index(drop=True)

#%%
resultado_4 = resultado_4.rename(
    columns=dict(zip(old_names, new_names))
)
#%%
teste_4 = teste_4.reset_index(drop=True)
#%%

resultado_5 = pd.merge(teste_4, resultado_4, left_index=True, right_index=True)

#%%

fig = plt.figure(figsize=(10, 10))  
col = "a_hybrid"

ax = fig.add_subplot()
resultado_5.plot(ax=ax,
    column=col,     # coluna usada no choropleth
    cmap="OrRd",            # colormap (ou 'viridis', 'Blues', etc.)
    legend=True,            # mostra legenda
    figsize=(8, 6),         # tamanho do gráfico
    edgecolor="black",      # contorno dos polígonos
    linewidth=0.5
)
#dados_anhalt.boundary.plot(ax=ax, color='red', label='Regions sachsen anhalt')
#dados_shp.boundary.plot(ax=ax, color='blue')
dados_shp.boundary.plot(ax=ax)
#gdados_anhalt_2.boundary.plot(ax=ax, color='green')
#grid.boundary.plot(ax=ax, linewidth=0.5,color="red")
#gdf.plot(ax=ax, color='purple',marker='o', markersize=9)
plt.title(f"Choropleth map of '{col}' value", fontsize=14)
plt.legend()
plt.show()

#%%

df_merged_2 = pd.merge(dados_todos, resultado_5, on='GEN', how='left')

#%%
# Criar DataFrame de exemplo
df_nan_3 = df_merged_2.copy()

#%%%
df_nan_3 = df_nan_3[df_nan_3['a_hybrid'].notna()]

#%%
# Inicializar geocodificador
# geolocator = Nominatim(user_agent="geo_example")

# #%%
# def get_lat_lon(city, postal_code, country="Germany"):
#     """
#     Retorna latitude e longitude de uma cidade alemã e seu Postleitzahl.
#     """
#     try:
#         query = f"{postal_code} {city}, {country}"
#         location = geolocator.geocode(query)
#         if location:
#             return location.latitude, location.longitude
#     except Exception as e:
#         print(f"Erro ao buscar {city} {postal_code}: {e}")
#     return None, None

# # Aplicar a função a cada linha
# latitudes = []
# longitudes = []

# for _, row in df_nan_3.iterrows():
#     lat, lon = get_lat_lon(row['GEN'], row['Postleitzahl'])
#     latitudes.append(lat)
#     longitudes.append(lon)
#     sleep(1)  # pausa de 1 segundo para não sobrecarregar o servidor

# # Adicionar colunas ao DataFrame
# df_nan_3['latitude'] = latitudes
# df_nan_3['longitude'] = longitudes

#%%
# Criar transformador: de EPSG:4326 -> EPSG:25832 (ETRS89 / UTM zone 32N)
# transformer = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)

#%%

# print(df_nan_3.columns.tolist())

df_nan_3 = df_nan_3.drop(['y', 'x', 'y_coords', 'x_coords'], axis=1)

#%%
df_nan_3 = GeoDataFrame(df_nan_3, crs="EPSG:25832", geometry=df_nan_3.geometry)

#%%

fig = plt.figure(figsize=(10, 10))  
col = "Summe der Netzverluste [kWh]"

ax = fig.add_subplot()
df_nan_3.plot(ax=ax,
    column=col,     # coluna usada no choropleth
    cmap="OrRd",            # colormap (ou 'viridis', 'Blues', etc.)
    legend=True,            # mostra legenda
    figsize=(8, 6),         # tamanho do gráfico
    edgecolor="black",      # contorno dos polígonos
    linewidth=0.5
)
dados_anhalt.boundary.plot(ax=ax, color='gray', label='Regions sachsen anhalt')
#dados_shp.boundary.plot(ax=ax, color='blue')
#dados_shp.boundary.plot(ax=ax)
#gdados_anhalt_2.boundary.plot(ax=ax, color='green')
#grid.boundary.plot(ax=ax, linewidth=0.5,color="red")
#gdf.plot(ax=ax, color='purple',marker='o', markersize=9)
plt.title(f"Choropleth map of '{col}' value", fontsize=14)
plt.legend()
plt.show()