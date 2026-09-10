# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 15:32:07 2024

@author: Natalia Bastos
"""

import folium

import numpy as np
import subprocess

import os
import platform
import pandas as pd
import shutil
from folium.plugins import MarkerCluster

import geopandas as gpd

import h3
import seaborn as sns
from shapely.geometry import Polygon

import matplotlib.pyplot as plt

from sklearn.preprocessing import Normalizer

from sklearn.cluster import KMeans

from pylab import bone, pcolor, colorbar

from sklearn.preprocessing import LabelEncoder

import umap.umap_ as umap

from kneed import KneeLocator

from minisom import MiniSom

from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler


from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import shap
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from pathlib import Path

# ==============================================================================
# CONFIGURAÇÃO AUTOMÁTICA DE DIRETÓRIOS PARA O GITHUB
# ==============================================================================
REMOTE_URL = "https://github.com/BastosSousa/Explainable_ML_NTL_2026.git"
# Identifica a raiz do projeto
BASE_DIR = Path(__file__).resolve().parent
repo_dir = Path(r"C:\Pos\codigo")
# Define a estrutura de pastas para entrada e saída
DIR_ENTRADA = BASE_DIR / "dados_entrada"
DIR_SAIDA = BASE_DIR / "dados_saida"
DIR_FIGURAS = BASE_DIR / "figuras_html"

# Cria as pastas automaticamente caso não existam
for pasta in [DIR_ENTRADA, DIR_SAIDA, DIR_FIGURAS]:
    pasta.mkdir(parents=True, exist_ok=True)

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

caminho_dados_final1 = diretorio_atual+barra + \
    volta_nivel+barra+'saidas'+barra+'data_final1.csv'
   
caminho_dados_final3 = diretorio_atual+barra + \
    volta_nivel+barra+'saidas'+barra+'data_toFuzzy.csv'
  
    
caminho_dados_final2 = diretorio_atual+barra + \
    volta_nivel+barra+'saidas'+barra+'data_finalFuzzy2.csv'

#%%
mapeamento_copias = {
    caminho_dados2: DIR_ENTRADA / "dados_papaer2.csv",
    caminho_dados_final1: DIR_ENTRADA / "data_final1.csv",
    caminho_dados_final3: DIR_ENTRADA / "data_toFuzzy.csv"
}

#%%

for caminho_origem, caminho_destino in mapeamento_copias.items():
    if os.path.exists(caminho_origem):
        shutil.copy(caminho_origem, caminho_destino)
        print(f"[OK] File copied: {Path(caminho_origem).name} -> {DIR_ENTRADA}")
    else:
        print(f"[AVISO] Source file not found: {caminho_origem}")

# 4. Atualização das variáveis do script para ler a partir de DIR_ENTRADA
caminho_dados2 = DIR_ENTRADA / "dados_papaer2.csv"
caminho_dados_final1 = DIR_ENTRADA / "data_final1.csv"
caminho_dados_final3 = DIR_ENTRADA / "data_toFuzzy.csv"

# Caminho dos arquivos gerados na SAÍDA
caminho_dados_final2 = DIR_SAIDA / "data_finalFuzzy2.csv"
caminho_mapa_hex = DIR_FIGURAS / "hexmap.html"
caminho_mapa_fuzzy = DIR_FIGURAS / "DSO_Fuzzy_Trapezoidal_Risk_Profiles.html"


#%%
dados2 = pd.read_csv(caminho_dados2,sep=';')

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
map.save(str(DIR_FIGURAS / "BLANK_map.html"))

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


parecer['count'] =(parecer['ids']
                      .apply(lambda w_PARECER2_TOTAL:len(w_PARECER2_TOTAL)))
#%%

def add_geometry(row):
  points = h3.cell_to_boundary(row['h3_cell'])
  return Polygon(points)

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
map.save(str(caminho_mapa_hex))
#%%

gdf_to_normalize = gdf.copy()
gdf_to_normalize.drop(columns = ['LATITUDE', 'LONGITUDE','geometry', 'pop_count_local_moran','pop_count_local_moran_p_sim','h3_cell', 'w_PARECER2_TOTAL','PARECER2_TOTAL','Perdas'], inplace = True)

#%% Normalizar

normal_values = Normalizer().fit_transform(gdf_to_normalize.values)

#%% Aplicar UMAP

reducer = umap.UMAP()
embedding = reducer.fit_transform(normal_values)

#%%

csfont = {'fontname':'Times New Roman'}
plt.rc('font',family='Times New Roman')
#plt.title('UMAP embedding');
plt.scatter(embedding[:, 0], embedding[:, 1], cmap='Spectral', s=5)
plt.ylabel("UMAP component 2",size=15)
plt.xlabel("UMAP component 1",size=15)
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
#%%
# df1['segment'] = df_umap_kmeans['segment']
sns.scatterplot(data=df_umap_kmeans, x="comp1", y="comp2", hue='segment')

#%%

df_umap_kmeans_2 = df_umap_kmeans[['w_PARECER2_TOTAL', 'pop_count_local_moran',
                                   'pop_count_local_moran_p_sim','h3_cell','segment_kmeans_umap','PARECER2_TOTAL','Perdas']]


#%%

df_umap_kmeans_2 = df_umap_kmeans_2.drop_duplicates(["h3_cell", "pop_count_local_moran_p_sim"])

df_umap_kmeans_2 = df_umap_kmeans_2.loc[df_umap_kmeans_2['pop_count_local_moran'].isin(['HH','LH'])]

df_umap_kmeans_2['geometry'] = df_umap_kmeans_2['h3_cell'].apply(lambda x: cell_to_shapely(x))
df_umap_kmeans_2 = gpd.GeoDataFrame(data=df_umap_kmeans_2, geometry='geometry', crs=4326)

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

df3_data.drop(columns = ['LATITUDE', 'LONGITUDE','geometry','comp1', 'comp2','pop_count_local_moran',
                         'segment_kmeans_umap', 'segment', 'target'], inplace = True)


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

#df3_data["clusters"] = cluster_df.clusters

#df3_data_2 = df3_data[['PARECER2_TOTAL','h3_cell', 'ivs','idhm' , 'Perdas','geometry','Porc','PO_NOMINAL_TRAFO','clusters']]

#%%
from scipy.stats import gaussian_kde

h3_counts = df3_data.groupby("h3_cell").size()

plt.figure(figsize=(10,5))

plt.hist(h3_counts, bins=50, density=True, alpha=0.5)

kde = gaussian_kde(h3_counts)
x_range = np.linspace(h3_counts.min(), h3_counts.max(), 500)
plt.plot(x_range, kde(x_range), linewidth=2)

plt.xscale("log")

plt.title("Distribution of samples by H3 (with KDE, log scale)")
plt.xlabel("Number of samples per H3 (log)")
plt.ylabel("Density")

plt.show()

#%%
# # 1. contagem por H3
# h3_counts = df3_data_2.groupby("h3_cell").size().reset_index(name="n_samples")

# # 2. juntar com geometria (assumindo que df3_data_2 já tem geometry por linha)
# h3_geom = df3_data_2[["h3_cell", "geometry"]].drop_duplicates()

# h3_map = h3_geom.merge(h3_counts, on="h3_cell", how="left")

# # 3. converter para GeoDataFrame
# h3_gdf = gpd.GeoDataFrame(h3_map, geometry="geometry", crs=4326)

# # 4. plot
# fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# h3_gdf.plot(
#     column="n_samples",
#     cmap="viridis",
#     legend=True,
#     ax=ax
# )

# ax.set_title("Densidade de amostras por célula H3")
# ax.set_axis_off()

# plt.show()

#%%
# plt.figure(figsize=(10,5))

# plt.hist(h3_counts, bins=50, density=True, alpha=0.5)

# kde = gaussian_kde(h3_counts)
# x_range = np.linspace(h3_counts.min(), h3_counts.max(), 500)
# plt.plot(x_range, kde(x_range), linewidth=2)

# plt.xscale("log")

# plt.title("Distribuição de amostras por H3 (com KDE, escala log)")
# plt.xlabel("Número de amostras por H3 (log)")
# plt.ylabel("Densidade")

# plt.show()
#%%

base_features = [
    c for c in df3_data.columns
    if c not in ["h3_cell", "geometry", "Perdas","clusters"]
]

agg_dict = {}

for col in base_features:
    agg_dict[col] = ["mean", "std"]

agg_dict["Perdas"] = ["mean", "std", "count"]

h3_agg = df3_data.groupby("h3_cell").agg(agg_dict)

h3_agg.columns = [
    f"{col}_{stat}" for col, stat in h3_agg.columns
]

h3_agg = h3_agg.reset_index()
#%%
from sklearn.feature_selection import VarianceThreshold

X_temp = h3_agg.drop(columns=["h3_cell", "Perdas_mean"])

selector = VarianceThreshold(threshold=0.0)
X_filtered = selector.fit_transform(X_temp)

kept_cols = X_temp.columns[selector.get_support()]

X = pd.DataFrame(X_filtered, columns=kept_cols)


#%% XGBoost
y = h3_agg["Perdas_mean"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R²:", r2_score(y_test, y_pred))

#%%

importance = pd.DataFrame({
    "feature": X.columns,
    "gain": model.feature_importances_
}).sort_values("gain", ascending=False)


#%%
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

shap_df = pd.DataFrame(shap_values, columns=X.columns)
shap_df["h3_cell"] = h3_agg["h3_cell"].values

#%% Plotar figura 1 chap 5

# 2. Configuração de estilo visual: Times New Roman, Tamanho 21
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 21,
    "axes.labelsize": 21,
    "axes.titlesize": 21,
    "xtick.labelsize": 21,
    "ytick.labelsize": 21,
    "legend.fontsize": 19,
    "figure.titlesize": 21,
})

# Seleciona apenas as colunas dos recursos (features)
feature_cols = [col for col in X.columns if col != "h3_cell"]
X_plot = X[feature_cols]

# Se shap_values for um numpy array, selecionamos apenas os índices das colunas de features
if isinstance(shap_values, pd.DataFrame):
  shap_values_plot = shap_values[feature_cols].values
else:
  # Mantém a matriz alinhada caso venha como numpy array direto
  shap_values_plot = (
      shap_df[feature_cols].values if "h3_cell" in shap_df else shap_values
  )

# 4. Renderização da Figura
fig, ax = plt.subplots(figsize=(11, 7))

# Plot do SHAP Summary
shap.summary_plot(shap_values_plot, X_plot, show=False)

# Aplicação de títulos e formatação com a fonte 21
plt.title("GeoSHAP Feature Importance & Attribution", fontsize=21, pad=20)
plt.xlabel("SHAP Value (Impact on Model Output)", fontsize=21)

# Forçar fonte Times New Roman tamanho 21 nos eixos X e Y
ax_curr = plt.gca()
for item in (
    [ax_curr.title, ax_curr.xaxis.label, ax_curr.yaxis.label]
    + ax_curr.get_xticklabels()
    + ax_curr.get_yticklabels()
):
  item.set_fontsize(21)
  item.set_fontfamily("serif")

plt.tight_layout()

# 5. Salvar o arquivo da figura em formato SVG para o LaTeX
plt.savefig("geoshap_summary_plot.svg", format="svg", bbox_inches="tight")
plt.savefig(str(DIR_FIGURAS / "geoshap_summary_plot.svg"), format="svg", bbox_inches="tight")
plt.show()


#%%

top_features = (
    shap_df.drop(columns=["h3_cell"])
    .abs()
    .mean()
    .sort_values(ascending=False)
    .head(20)
    .index
)

print(top_features)


#%%
train_pred = model.predict(X_train)

print(
    "Train R²:",
    r2_score(y_train, train_pred)
)

print(
    "Test R²:",
    r2_score(y_test, y_pred)
)

#%%
h3_agg["pred"] = model.predict(X)
h3_agg["residual"] = y - h3_agg["pred"]


#%%

shap_df["h3_cell"] = h3_agg["h3_cell"].values

shap_h3 = shap_df.groupby("h3_cell").mean().reset_index()

#%%

geo_h3 = df3_data[["h3_cell", "geometry"]].drop_duplicates()

map_df = (
    h3_agg[["h3_cell", "pred", "residual"]]
    .merge(shap_h3, on="h3_cell")
    .merge(geo_h3, on="h3_cell")
)

gdf = gpd.GeoDataFrame(map_df, geometry="geometry", crs=4326)
#%%
top_shap_feature = (
    shap_df.drop(columns=["h3_cell"])
    .abs()
    .mean()
    .sort_values(ascending=False)
    .index[0]
)

#%%

fig, axes = plt.subplots(1, 3, figsize=(20, 7))

# -------------------------
# 1. RISCO (predição)
# -------------------------
gdf.plot(
    column="pred",
    cmap="viridis",
    legend=True,
    ax=axes[0]
)
axes[0].set_title("Projected risk (Losses)")
axes[0].set_axis_off()

# -------------------------
# 2. ERRO (residual)
# -------------------------
gdf.plot(
    column="residual",
    cmap="coolwarm",
    legend=True,
    ax=axes[1]
)
axes[1].set_title("Model error (Residual)")
axes[1].set_axis_off()

# -------------------------
# 3. SHAP (impacto da variável)
# -------------------------
gdf.plot(
    column=top_shap_feature,
    cmap="plasma",
    legend=True,
    ax=axes[2]
)
axes[2].set_title(f"Impact SHAP: {top_shap_feature}")
axes[2].set_axis_off()

plt.tight_layout()
plt.show()

#%%

shap_importance = (
    shap_df.drop(columns=["h3_cell"])
    .abs()
    .mean()
    .sort_values(ascending=False)
)

top_cols = shap_importance.head(15).index.tolist()

#%%

som_df = shap_h3[top_cols].copy()

scaler = StandardScaler()

X_som = scaler.fit_transform(som_df[top_cols])


#%%

SOM_X_AXIS_NODES = 16
SOM_Y_AXIS_NODES = 16

som = MiniSom(
    x=SOM_X_AXIS_NODES,
    y=SOM_Y_AXIS_NODES,
    input_len=X_som.shape[1],
    sigma=2,
    learning_rate=0.5,
    random_seed=42
)

som.pca_weights_init(X_som)
N_ITERATIONS = 500*SOM_X_AXIS_NODES*SOM_Y_AXIS_NODES
som.train_random(X_som, N_ITERATIONS)
#%%

# Obtendo a matriz de distâncias do SOM
distance_map = som.distance_map()
# Plotando o mapa de calor
plt.figure(figsize=(10, 10))
plt.pcolor(distance_map, cmap='bone')
plt.colorbar()
#plt.title('SOM heatmap')
plt.ylabel("SOM_Y_AXIS",**csfont,fontsize=15)
plt.xlabel("SOM_X_AXIS",**csfont,fontsize=15)
plt.show()

#%%

weights = som.get_weights()
weights_flat = weights.reshape(-1, weights.shape[-1])

#%%
# scores = []

# for k in range(2,11):

#     ward = AgglomerativeClustering(
#         n_clusters=k,
#         linkage="ward"
#     )

#     labels = ward.fit_predict(weights_flat)

#     score = silhouette_score(
#         weights_flat,
#         labels
#     )

#     scores.append(score)

# best_k = np.argmax(scores) + 2

#%%
Z = linkage(weights_flat, method="ward")

plt.figure(figsize=(12,6))

dendrogram(
    Z,
    truncate_mode="lastp",
    p=30,              # mostra últimos agrupamentos
    leaf_rotation=90,
    leaf_font_size=10,
    show_contracted=True
)

plt.xlabel("Clusters")
plt.ylabel("Ward Distance")
plt.title("Dendrogram of SOM + Ward")

plt.show()

#%%
ward = AgglomerativeClustering(
    n_clusters=3,
    linkage="ward"
)

neuron_clusters = ward.fit_predict(
    weights_flat
)

#%%

winner = np.array(
    [som.winner(x) for x in X_som]
)

winner_idx = np.ravel_multi_index(
    winner.T,
    (SOM_X_AXIS_NODES,
     SOM_Y_AXIS_NODES)
)

som_cluster = neuron_clusters[winner_idx]

#%%

cluster_df = pd.DataFrame({

    "h3_cell": shap_h3["h3_cell"],

    "som_cluster": som_cluster

})

#%%
risk_df = h3_agg.merge(
    cluster_df,
    on="h3_cell"
)
#%%

plt.rc('font', family='Times New Roman')

fig, ax = plt.subplots(figsize=(10,10))

bone()

# U-Matrix
pcolor(som.distance_map().T)

cbar = colorbar()
cbar.ax.tick_params(labelsize=15)

# neurônios SOM
for i in range(SOM_X_AXIS_NODES):
    for j in range(SOM_Y_AXIS_NODES):

        idx = np.ravel_multi_index(
            ([i], [j]),
            (SOM_X_AXIS_NODES, SOM_Y_AXIS_NODES)
        )[0]

        cluster = neuron_clusters[idx]

        ax.text(
            i + 0.5,
            j + 0.5,
            str(cluster),
            ha='center',
            va='center',
            fontsize=10,
            color='white'
        )

ax.set_xlabel("SOM X", fontsize=15)
ax.set_ylabel("SOM Y", fontsize=15)

plt.title("U-Matrix with Ward clusters")
plt.show()

##CONTINUAR


#%%%
# plt.figure(figsize=(10,10))

# bone()
# pcolor(som.distance_map().T)

# for i, w in enumerate(winner):

#     plt.scatter(
#         w[0] + 0.5,
#         w[1] + 0.5,
#         c=risk_df["pred"].iloc[i],
#         cmap="viridis",
#         s=30
#     )

# plt.colorbar(label="Perdas previstas")

# plt.title("SOM colorido pelo risco previsto")

# plt.show()

 
#%%

import seaborn as sns

plt.figure(figsize=(10,6))

sns.boxplot(
    data=risk_df,
    x="som_cluster",
    y="pred"
)

plt.xlabel("Cluster SOM-WARD")
plt.ylabel("Expected losses")

plt.title("Risk distribution by cluster")

plt.show()


#%%

risk_df.groupby("som_cluster")["pred"].agg([
    "count",
    "mean",
    "median",
    "std"
])

#%%
risk_df["som_cluster"].value_counts()

#%% verifica qual melhor escolha de k para ward

# scores = []

# for k in range(2, 11):

#     ward = AgglomerativeClustering(
#         n_clusters=k,
#         linkage="ward"
#     )

#     labels = ward.fit_predict(weights_flat)

#     score = silhouette_score(
#         weights_flat,
#         labels
#     )

#     scores.append(score)

#     print(f"K = {k} | Silhouette = {score:.4f}")

# #%%
# plt.figure(figsize=(8,5))

# plt.plot(
#     range(2,11),
#     scores,
#     marker='o'
# )

# plt.xlabel("Número de clusters (K)")
# plt.ylabel("Silhouette Score")
# plt.title("Silhouette Score - SOM + Ward")

# plt.grid(True)

# plt.show()


#%%

cluster_df = pd.DataFrame({
    "h3_cell": shap_h3["h3_cell"],
    "som_cluster": som_cluster
})

h3_final = h3_agg.merge(
    cluster_df,
    on="h3_cell"
)


#%%

perfil = (
    h3_final
    .groupby("som_cluster")[top_cols]
    .mean()
)

#%% padronizar para comparar

perfil_z = pd.DataFrame(
    StandardScaler().fit_transform(perfil),
    columns=perfil.columns,
    index=perfil.index
)

#%% Graficos
plt.figure(figsize=(12,6))

sns.heatmap(
    perfil_z,
    cmap="RdBu_r",
    center=0
)

plt.title("Perfis territoriais SOM-WARD")
plt.show()

#%%
h3_final.groupby("som_cluster")["pred"].agg([
    "mean",
    "median",
    "std"
])

#%%
h3_final["geometry"] = h3_final["h3_cell"].apply(cell_to_shapely)

gdf_clusters = gpd.GeoDataFrame(
    h3_final,
    geometry="geometry",
    crs="EPSG:4326"
)
#%%
# cluster_colors = {
#     0: "#1f78b4",  # azul
#     1: "#e31a1c",  # vermelho
#     2: "#33a02c"   # verde
# }

# import matplotlib.colors as mcolors

# clusters = sorted(gdf_clusters["som_cluster"].unique())

# cmap = plt.cm.Set1

# cluster_colors = {
#     c: mcolors.to_hex(cmap(i))
#     for i, c in enumerate(clusters)
# }


#%%

cluster_score = {
    2: 1,
    0: 2,
    3: 3,
    1: 4
}


risk_df["cluster_score"] = (
    risk_df["som_cluster"]
    .map(cluster_score)
)


top_cols = shap_importance.head(5).index.tolist()

#%%

base_vars = [
    "pred",
    
    "Perdas_count",
    "cluster_score"
]
shap_vars = top_cols

#%%
fuzzy_vars = base_vars + shap_vars

#%%
fuzzy_input = risk_df[
    ["h3_cell"] + fuzzy_vars
].copy()
#%%

#fuzzy_input.to_csv(caminho_dados_final2 , index=False)

#%%
from folium import GeoJsonPopup
from folium.plugins import MousePosition

#%%
perfil_nome = {
    0: "Medium Risk Territory",
    1: "High Risk Territory",
    2: "Low Risk Territory"
}

gdf_clusters["perfil"] = (
    gdf_clusters["som_cluster"]
    .map(perfil_nome)
)

perfil_colors = {
    "High Risk Territory": "#d73027",
    "Medium Risk Territory": "#fc8d59",
    "Low Risk Territory": "#1a9850"
}

center = [
    gdf_clusters.geometry.centroid.y.mean(),
    gdf_clusters.geometry.centroid.x.mean()
]

m = folium.Map(
    location=center,
    zoom_start=13,
    tiles=None
)

folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Esri',
    name='Satellite',
    overlay=False,
    control=True
).add_to(m)

folium.TileLayer(
    'CartoDB positron',
    name='Street Map'
).add_to(m)

popup = GeoJsonPopup(
    fields=[
        "perfil",
        "pred",
        "Perdas_mean",
        "Perdas_std",
        "Perdas_count"
    ],
    aliases=[
        "Territorial Profile",
        "Predicted Losses",
        "Average Losses",
        "Std Dev",
        "Samples"
    ],
    localize=True
)

folium.GeoJson(
    gdf_clusters,
    name="Territorial Profiles",

    style_function=lambda feat: {

        "fillColor":
            perfil_colors[
                feat["properties"]["perfil"]
            ],

        "color":"white",

        "weight":0.3,

        "fillOpacity":0.65

    },

    popup=popup

).add_to(m)
    
legend_html = """
<div style="
position: fixed;
top: 180px;
right: 20px;
width: 270px;
background-color: white;
border-radius:10px;
box-shadow: 0 0 15px rgba(0,0,0,0.3);
padding:15px;
z-index:9999;
font-size:14px;
font-family:Arial;
">

<h4 style="margin-top:0;">
DSO Risk Profiles
</h4>

<p>
<b>Territorial Classification</b>
</p>

<div>
<span style="
background:#d73027;
width:18px;
height:18px;
display:inline-block;
margin-right:10px;">
</span>
High Risk Territory
</div>

<div>
<span style="
background:#fc8d59;
width:18px;
height:18px;
display:inline-block;
margin-right:10px;">
</span>
Medium Risk Territory
</div>

<div>
<span style="
background:#1a9850;
width:18px;
height:18px;
display:inline-block;
margin-right:10px;">
</span>
Low Risk Territory
</div>

<hr>

<p>
<b>Source:</b><br>
XGBoost + SHAP + SOM-WARD
</p>

</div>
"""

m.get_root().html.add_child(
    folium.Element(legend_html)
)    

title_html = """
<div style="
position: fixed;
top: 20px;
left: 50%;
transform: translateX(-50%);
background-color:white;
padding:10px 20px;
border-radius:8px;
box-shadow:0 0 15px rgba(0,0,0,0.2);
z-index:9999;
font-size:18px;
font-weight:bold;
font-family:Arial;
">

Electric Loss Risk Assessment
(H3 Spatial Profiles)

</div>
"""

m.get_root().html.add_child(
    folium.Element(title_html)
)
    
MousePosition().add_to(m)
folium.LayerControl().add_to(m)
m.save("DSO_Risk_Profiles.html")
m.save(str(DIR_FIGURAS / "DSO_Risk_Profiles.html"))

#%% Preparacao para fuzzy


# Ranking automático dos clusters

cluster_rank = (
    fuzzy_input
    .groupby("cluster_score")["pred"]
    .mean()
    .sort_values()
)

print(cluster_rank)


#%%

clusters = cluster_rank.index.tolist()

cluster_weight = {
    c: i/(len(clusters)-1)
    for i,c in enumerate(clusters)
}

cluster_weight

#%%

fuzzy_input["cluster_weight"] = (
    fuzzy_input["cluster_score"]
    .map(cluster_weight)
)

#%%

base_vars = [
    "pred",
    "Perdas_count",
    "t_sem_lixo_mean",
    "idhm_renda_mean",
    "t_c6a14_fora_mean",
    "t_pop5a6_escola_mean",
    "t_env_mean"
]

scaler = MinMaxScaler()

X_norm = pd.DataFrame(
    scaler.fit_transform(fuzzy_input[base_vars]),
    columns=base_vars,
    index=fuzzy_input.index
)
#%%

weights_shap = (
    shap_importance.loc[top_cols]
    /
    shap_importance.loc[top_cols].sum()
)

print(weights_shap)

#%%

X_norm["social_score"] = 0

for col in top_cols:

    X_norm["social_score"] += (
        weights_shap[col] *
        X_norm[col]
    )

#%%

X_norm["fuzzy_base"] = (

      0.50 * X_norm["pred"]

    + 0.20 * (1 - X_norm["Perdas_count"])

    + 0.30 * X_norm["social_score"]

)

#%%

X_norm["fuzzy_final"] = (

    0.80 * X_norm["fuzzy_base"]

    +

    0.20 * fuzzy_input["cluster_weight"]

)

#%%

q1,q2 = X_norm["fuzzy_final"].quantile(
    [0.33,0.66]
)

X_norm["risk_level"] = pd.cut(

    X_norm["fuzzy_final"],

    bins=[
        -np.inf,
        q1,
        q2,
        np.inf
    ],

    labels=[
        "Low Risk",
        "Medium Risk",
        "High Risk"
    ]
)

#%%

risk_map = fuzzy_input.copy()

risk_map["social_score"] = X_norm["social_score"]
risk_map["cluster_weight"] = fuzzy_input["cluster_weight"]
risk_map["fuzzy_index"] = X_norm["fuzzy_final"]
risk_map["Risk_score"] = X_norm["fuzzy_final"]
risk_map["risk_level"] = X_norm["risk_level"]
risk_map["classificacao"] = X_norm["risk_level"]

# Create the missing membership degree columns
risk_map["Low Risk"] = (risk_map["risk_level"] == "Low Risk").astype(float)
risk_map["Medium Risk"] = (risk_map["risk_level"] == "Medium Risk").astype(float)
risk_map["High Risk"] = (risk_map["risk_level"] == "High Risk").astype(float)

#%%

risk_map.to_csv(caminho_dados_final2 , index=False)

#%%
# Configuração de estilo: Times New Roman, Tamanho 21
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 21,
    'axes.labelsize': 21,
    'axes.titlesize': 21,
    'xtick.labelsize': 21,
    'ytick.labelsize': 21,
    'legend.fontsize': 19,
    'figure.titlesize': 21
})

# Cálculo das frequências por célula H3
h3_counts = df3_data.groupby("h3_cell").size()

# Criar Figura 
fig, ax = plt.subplots(figsize=(10, 6))

# Histograma
ax.hist(h3_counts, bins=50, density=True, alpha=0.5, color='#1f77b4', edgecolor='none')

# Curva KDE (Kernel Density Estimation)
kde = gaussian_kde(h3_counts)
x_range = np.linspace(h3_counts.min(), h3_counts.max(), 500)
ax.plot(x_range, kde(x_range), color='#d62728', linewidth=2.5, label='KDE Density')

# Escala Logarítmica e Rótulos
ax.set_xscale("log")
ax.set_title("Distribution of Samples per H3 Cell (Log Scale)", pad=20)
ax.set_xlabel("Number of Samples per H3 Cell (Log Scale)")
ax.set_ylabel("Density")
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(framealpha=0.9)

# Salvando em formato vetorial SVG
fig.savefig('h3_sample_distribution_kde.svg', format='svg', bbox_inches='tight')
fig.savefig(str(DIR_FIGURAS / 'h3_sample_distribution_kde.svg'), format='svg', bbox_inches='tight')
plt.show()

#%%
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.cluster import AgglomerativeClustering

# 1. Configuração de estilo: Times New Roman, Tamanho 21
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 24,
    "axes.labelsize": 24,
    "axes.titlesize": 24,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "legend.fontsize": 24,
    "figure.titlesize": 24,
})

# ==============================================================================
# 2. EXTRAÇÃO DOS DADOS DO SOM E DO WARD
# ==============================================================================
# Assumindo que seu modelo MiniSom treinado se chama 'som' (ou substitua pelo seu objeto):
# Extrai a U-Matrix (Matriz de Distâncias Unificada)
u_matrix = som.distance_map()  # Retorna a matriz 16x16 com as distâncias

# Extrai os pesos dos neurônios
weights = som.get_weights()  # Formato (16, 16, n_features)
weights_flat = weights.reshape(-1, weights.shape[-1])  # Achata para (256, n_features)

# Aplica o agrupamento hierárquico de Ward nos pesos dos neurônios
ward = AgglomerativeClustering(n_clusters=3, linkage="ward")
cluster_labels = ward.fit_predict(weights_flat)

# Remapeia os rótulos dos clusters de volta para a grade 16x16 do SOM
cluster_grid = cluster_labels.reshape(weights.shape[0], weights.shape[1])

# ==============================================================================
# 3. GERAÇÃO DA FIGURA DUPLA (U-Matrix + Ward Clusters)
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# --- Subplot (a): U-Matrix ---
im1 = ax1.imshow(u_matrix, cmap="bone_r", origin="lower")
ax1.set_title("(a) SOM U-Matrix", pad=15)
ax1.set_xlabel("Neuron X")
ax1.set_ylabel("Neuron Y")

cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cbar1.set_label("Distance", fontsize=21)
cbar1.ax.tick_params(labelsize=21)

# --- Subplot (b): Ward Clustering Overlay ---
# Paleta operacional: Verde (Low), Laranja (Medium), Vermelho (High)
cmap_clusters = ListedColormap(["#2ca02c", "#ff7f0e", "#d62728"])

im2 = ax2.imshow(cluster_grid, cmap=cmap_clusters, origin="lower")
ax2.set_title("(b) Ward Clustering Overlay", pad=15)
ax2.set_xlabel("Neuron X")
ax2.set_ylabel("Neuron Y")

cbar2 = fig.colorbar(im2, ax=ax2, ticks=[0, 1, 2], fraction=0.046, pad=0.04)
cbar2.ax.set_yticklabels(["Low Risk", "Medium Risk", "High Risk"])
cbar2.ax.tick_params(labelsize=19)

# Formatação das fontes nos eixos
for ax in [ax1, ax2]:
  ax.tick_params(axis="both", which="major", labelsize=21)
  for item in (
      [ax.title, ax.xaxis.label, ax.yaxis.label]
      + ax.get_xticklabels()
      + ax.get_yticklabels()
  ):
    item.set_fontfamily("serif")

plt.tight_layout()

# 4. Salvar arquivo SVG
plt.savefig("som_umatrix_ward_clusters.svg", format="svg", bbox_inches="tight")
plt.savefig(str(DIR_FIGURAS / "som_umatrix_ward_clusters.svg"), format="svg", bbox_inches="tight")
plt.show()

#%%
gdf_fuzzy = gdf_clusters.copy()

# Atribuição dos resultados Fuzzy diretamente ao GeoDataFrame
gdf_fuzzy["Risk_score"] = risk_map["fuzzy_index"].values
gdf_fuzzy["mu_low"] = risk_map["Low Risk"].values
gdf_fuzzy["mu_med"] = risk_map["Medium Risk"].values
gdf_fuzzy["mu_high"] = risk_map["High Risk"].values
gdf_fuzzy["fuzzy_class"] = risk_map["classificacao"].values

# Mapeamento de Cores para os Níveis Fuzzy
fuzzy_colors = {
    "High Risk": "#d73027",  # Vermelho
    "Medium Risk": "#fc8d59",  # Laranja
    "Low Risk": "#1a9850",  # Verde
}

# ==============================================================================
# 2. INICIALIZAÇÃO DO MAPA
# ==============================================================================
center = [
    gdf_fuzzy.geometry.centroid.y.mean(),
    gdf_fuzzy.geometry.centroid.x.mean(),
]

m = folium.Map(location=center, zoom_start=13, tiles=None)

# Camadas de Fundo (Basemaps)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri",
    name="Satellite",
    overlay=False,
    control=True,
).add_to(m)

folium.TileLayer("CartoDB positron", name="Street Map").add_to(m)

# ==============================================================================
# 3. CONFIGURAÇÃO DO POPUP INTERATIVO (Com Métricas Fuzzy)
# ==============================================================================
# Formatação dos graus de pertinência e score no Popup para 3 casas decimais
gdf_fuzzy["Risk_score_str"] = gdf_fuzzy["Risk_score"].map(
    lambda x: f"{x:.3f}" if pd.notnull(x) else "N/A"
)
gdf_fuzzy["mu_low_str"] = gdf_fuzzy["mu_low"].map(
    lambda x: f"{x:.3f}" if pd.notnull(x) else "N/A"
)
gdf_fuzzy["mu_med_str"] = gdf_fuzzy["mu_med"].map(
    lambda x: f"{x:.3f}" if pd.notnull(x) else "N/A"
)
gdf_fuzzy["mu_high_str"] = gdf_fuzzy["mu_high"].map(
    lambda x: f"{x:.3f}" if pd.notnull(x) else "N/A"
)

popup = GeoJsonPopup(
    fields=[
        "fuzzy_class",
        "Risk_score_str",
        "mu_high_str",
        "mu_med_str",
        "mu_low_str",
        "pred",
        "Perdas_count",
    ],
    aliases=[
        "Fuzzy Risk Class",
        "Normalized Risk Score",
        "µ(High Risk)",
        "µ(Medium Risk)",
        "µ(Low Risk)",
        "Model Prediction",
        "Loss Sample Uncertainty",
    ],
    localize=True,
)

# ==============================================================================
# 4. ADIÇÃO DA CAMADA GEOJSON AO MAPA
# ==============================================================================
folium.GeoJson(
    gdf_fuzzy,
    name="Fuzzy Trapezoidal Risk Map",
    style_function=lambda feat: {
        "fillColor": fuzzy_colors.get(
            feat["properties"]["fuzzy_class"], "#gray"
        ),
        "color": "white",
        "weight": 0.4,
        "fillOpacity": 0.65,
    },
    popup=popup,
).add_to(m)


# ==============================================================================
# STREET VIEW FOR HIGH RISK POINTS
# ==============================================================================

# 1. Filter only High Risk cells from your fuzzy GeoDataFrame
gdf_high_risk = gdf_fuzzy[gdf_fuzzy["fuzzy_class"] == "High Risk"].copy()

# 2. Perform Spatial Join to get original points inside High Risk polygons
# Assumes 'df' or 'gdf' contains your point-level LATITUDE and LONGITUDE
gdf_points = gpd.GeoDataFrame(
    df, 
    geometry=gpd.points_from_xy(df.LONGITUDE, df.LATITUDE), 
    crs="EPSG:4326"
)

high_risk_points = gpd.sjoin(
    gdf_points, 
    gdf_high_risk[["geometry", "fuzzy_class"]], 
    how="inner", 
    predicate="within"
)

# 3. Add High Risk points with Street View links to the Folium Map
fg_high_risk_points = folium.FeatureGroup(name="High Risk Points (Street View)", show=True)

for _, row in high_risk_points.iterrows():
    lat, lng = row["LATITUDE"], row["LONGITUDE"]
    
    # Generate Google Street View URL
    street_view_url = f"https://www.google.com/maps?layer=c&cbll={lat},{lng}"
    
    popup_html = f"""
    <div style="font-family: Arial; font-size: 12px; width: 180px;">
        <b>High Risk Point</b><br>
        <b>Lat:</b> {lat:.5f}<br>
        <b>Lng:</b> {lng:.5f}<br><br>
        <a href="{street_view_url}" target="_blank" style="
            background-color: #d73027;
            color: white;
            padding: 5px 10px;
            text-decoration: none;
            border-radius: 4px;
            display: inline-block;
            font-weight: bold;">
            OPEN STREET VIEW
        </a>
    </div>
    """
    
    folium.Marker(
        location=[lat, lng],
        popup=folium.Popup(popup_html, max_width=220),
        icon=folium.Icon(color="red", icon="camera", prefix="fa")
    ).add_to(fg_high_risk_points)

# 4. Attach Feature Group to your existing map 'm' and save
fg_high_risk_points.add_to(m)

# ==============================================================================
# 5. COMPONENTES HTML SOBREPOSTOS (Título e Legenda Personalizados)
# ==============================================================================

# Legenda Atualizada com a Metodologia Fuzzy Trapezoidal
legend_html = """
<div style="
position: fixed; 
top: 170px; 
right: 20px;
width: 290px;
background-color: white;
border-radius:10px;
box-shadow: 0 0 15px rgba(0,0,0,0.3);
padding:15px;
z-index:9999;
font-size:13px;
font-family:Arial;
">

<h4 style="margin-top:0; margin-bottom:8px; color:#2c3e50;">
DSO Fuzzy Risk Map
</h4>

<p style="margin-bottom:10px;">
<b>Trapezoidal Membership Classification</b>
</p>

<div style="margin-bottom:6px;">
<span style="background:#d73027; width:16px; height:16px; display:inline-block; margin-right:8px; vertical-align:middle; border-radius:3px;"></span>
<b>High Risk</b> (µ_High = max)
</div>

<div style="margin-bottom:6px;">
<span style="background:#fc8d59; width:16px; height:16px; display:inline-block; margin-right:8px; vertical-align:middle; border-radius:3px;"></span>
<b>Medium Risk</b> (µ_Med = max)
</div>

<div style="margin-bottom:10px;">
<span style="background:#1a9850; width:16px; height:16px; display:inline-block; margin-right:8px; vertical-align:middle; border-radius:3px;"></span>
<b>Low Risk</b> (µ_Low = max)
</div>

<hr style="border: 0.5px solid #eee; margin: 10px 0;">

<p style="font-size:11px; color:#555; margin-bottom:0;">
<b>Methodology Pipeline:</b><br>
XGBoost + SHAP + SOM-Ward → Integrated Risk Score → Partition of Unity Fuzzy Trapezoidal Logic.
</p>

</div>
"""

title_html = """
<div style="
position: fixed;
top: 15px;
left: 50%;
transform: translateX(-50%);
background-color:white;
padding:10px 24px;
border-radius:8px;
box-shadow:0 0 15px rgba(0,0,0,0.2);
z-index:9999;
font-size:16px;
font-weight:bold;
font-family:Arial;
color:#2c3e50;
">
Remote Inspection Dashboard: H3 Grid Fuzzy Assessment
</div>
"""

m.get_root().html.add_child(folium.Element(legend_html))
m.get_root().html.add_child(folium.Element(title_html))

# Ferramenta de leitura de coordenadas do mouse e controle de camadas
MousePosition().add_to(m)
folium.LayerControl().add_to(m)

# Salvar o painel em HTML
m.save("DSO_Fuzzy_Trapezoidal_Risk_Profiles.html")
m.save(str(caminho_mapa_fuzzy))
print("Dashboard atualizado e salvo com sucesso como 'DSO_Fuzzy_Trapezoidal_Risk_Profiles.html'!")

#%%

from pathlib import Path

def commit_dados_figuras_e_codigos(mensagem="feat: atualiza dados, figuras_html e scripts python"):
    BASE_DIR = Path(r"C:\Pos\codigo")
    remote_url = "https://github.com/BastosSousa/Explainable_ML_NTL_2026.git"

    if not BASE_DIR.exists():
        print(f"[ERRO] O diretório '{BASE_DIR}' não foi encontrado.")
        return

    try:
        print(f"\n[GIT] Apontando para o repositório local: {BASE_DIR}")

        # 1. Garante a inicialização do Git
        if not (BASE_DIR / ".git").exists():
            subprocess.run(["git", "init"], cwd=BASE_DIR, check=True)

        # 2. Configura o remote origin
        res_remote = subprocess.run(["git", "remote"], cwd=BASE_DIR, capture_output=True, text=True)
        if "origin" not in res_remote.stdout:
            subprocess.run(["git", "remote", "add", "origin", remote_url], cwd=BASE_DIR, check=True)
        else:
            subprocess.run(["git", "remote", "set-url", "origin", remote_url], cwd=BASE_DIR, check=True)

        # 3. Limpa a área de staging
        subprocess.run(["git", "reset"], cwd=BASE_DIR, capture_output=True)

        # 4. Adiciona as pastas de dados/figuras E todos os scripts *.py da raiz
        print("[GIT] Adicionando dados_entrada/, dados_saida/, figuras_html/ e scripts *.py...")
        subprocess.run(
            ["git", "add", "-f", "dados_entrada/", "dados_saida/", "figuras_html/", "*.py"],
            cwd=BASE_DIR,
            check=True
        )

        # 5. Registra o commit
        print("[GIT] Executando commit...")
        subprocess.run(
            ["git", "commit", "--allow-empty", "-m", mensagem],
            cwd=BASE_DIR,
            check=True
        )

        # 6. Sincroniza e faz o push
        subprocess.run(["git", "branch", "-M", "main"], cwd=BASE_DIR, check=True)

        print("[GIT] Enviando dados e scripts Python para o GitHub...")
        subprocess.run(["git", "push", "-u", "origin", "main"], cwd=BASE_DIR, check=True)

        print("\n[SUCESSO] Pastas e scripts Python sincronizados no GitHub com sucesso!")

    except subprocess.CalledProcessError as e:
        print(f"\n[ERRO GIT] Falha ao executar comando: {e}")

if __name__ == "__main__":
    commit_dados_figuras_e_codigos()