# # -*- coding: utf-8 -*-
# """
# Created on Tue Jul 28 17:45:25 2026

# @author: Natalia
# """

# # -*- coding: utf-8 -*-
# """
# Framework Integrado de Análise de Perdas na Rede Elétrica (Alemanha)
# Combina: Nominatim Geocoding + H3 Indexing + XGBoost Espacial (WX) +
#          GeoSHAP + SOM-Ward Clustering + Fuzzy Risk Index
#  Target: Summe der Netzverluste [GWh]
# """

# import os
# import platform
# import re
# import time
# import warnings
# import geopandas as gpd
# import h3
# import libpysal
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import requests
# import shap
# from minisom import MiniSom
# from scipy.cluster.hierarchy import dendrogram, linkage
# from scipy.spatial.distance import cdist
# from sklearn.cluster import AgglomerativeClustering
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from sklearn.model_selection import KFold
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from shapely.geometry import Polygon
# from shapely.wkt import loads as wkt_loads
# from xgboost import XGBRegressor
# import seaborn as sns

# import branca.colormap as cm
# import folium

# from folium.plugins import Fullscreen, MeasureControl



# warnings.filterwarnings("ignore")

# # ===============================
# # 1. CONFIGURAÇÃO DE DIRETÓRIOS E SISTEMA
# # ===============================
# diretorio_atual = os.getcwd()
# nome_sistema_operacional = platform.system()
# volta_nivel = "../" if nome_sistema_operacional == "Linux" else "..\\"

# out_dir = os.path.join(diretorio_atual, volta_nivel, "saidas")
# os.makedirs(out_dir, exist_ok=True)

# # ===============================
# # 2. CARREGAMENTO E GEOCODIFICAÇÃO (NOMINATIM)
# # ===============================
# caminho_dados_DSO = os.path.join(
#     diretorio_atual, volta_nivel, "entradas", "DSO-Sachsen-Anhalt-file.xlsx"
# )
# caminho_dados_DSO_L = os.path.join(
#     diretorio_atual, volta_nivel, "entradas", "DSO-LowerSaxony.xlsx"
# )

# dso_sa = pd.read_excel(caminho_dados_DSO)
# dso_ls = pd.read_excel(caminho_dados_DSO_L)
# dso_sa["state"] = "Sachsen-Anhalt"
# dso_ls["state"] = "Lower Saxony"
# dados_combinados = pd.concat([dso_sa, dso_ls], ignore_index=True)


# # Função de Geocodificação enviada
# def geocode_city(city, country="Germany"):
#     url = "https://nominatim.openstreetmap.org/search"
#     params = {"q": f"{city}, {country}", "format": "json", "limit": 1}
#     headers = {
#         "User-Agent": "dso_analysis_app (researcher_email@domain.com)"
#     }  # Insira seu user-agent

#     try:
#         response = requests.get(url, params=params, headers=headers, timeout=5)
#         if response.status_code == 200:
#             data = response.json()
#             if len(data) > 0:
#                 return float(data[0]["lat"]), float(data[0]["lon"])
#     except Exception:
#         pass
#     return None, None


# # Execução condicional da geocodificação
# if (
#     "latitude" not in dados_combinados.columns
#     or dados_combinados["latitude"].isnull().any()
# ):
#     print("Geocodificando cidades via Nominatim...")
#     lats, longs = [], []
#     for city in dados_combinados["Ort"]:
#         lat, lon = geocode_city(city)
#         lats.append(lat)
#         longs.append(lon)
#         time.sleep(1)  # Respeita o rate limit do OSM
#     dados_combinados["latitude"] = lats
#     dados_combinados["longitude"] = longs

# # Carregamento dos arquivos processados e polígonos
# caminho_shp = os.path.join(
#     diretorio_atual,
#     volta_nivel,
#     "vg-hist.utm32s.shape",
#     "daten",
#     "utm32s",
#     "shape",
#     "VG-Hist_1990-10-03_KRS.shp",
# )
# dados_shp = gpd.read_file(caminho_shp).to_crs(epsg=4326)

# gdf_final_5_interp = pd.read_csv(
#     os.path.join(out_dir, "gdf_final_5_interp.csv"), sep=";", encoding="utf-8"
# )
# gdf_final_5_interp["geometry"] = gdf_final_5_interp["geometry"].apply(
#     wkt_loads
# )
# gdf_final_5_interp = gpd.GeoDataFrame(
#     gdf_final_5_interp, geometry="geometry", crs=dados_shp.crs
# )

# gdf_final_4_3 = pd.read_csv(
#     os.path.join(out_dir, "gdf_final_3.csv"), sep=";", encoding="utf-8"
# )
# gdf_final_4_3 = gdf_final_4_3[
#     gdf_final_4_3["state"].isin(["Lower Saxony", "Sachsen-Anhalt"])
# ]
# gdf_final_4_3 = gdf_final_4_3.drop_duplicates().dropna(
#     subset=["Summe der Netzverluste [kWh]"]
# )
# gdf_final_4_3 = gpd.GeoDataFrame(
#     gdf_final_4_3,
#     geometry=gpd.points_from_xy(
#         gdf_final_4_3.longitude_y, gdf_final_4_3.latitude_y
#     ),
#     crs=gdf_final_5_interp.crs,
# )

# joined = gpd.sjoin(
#     gdf_final_4_3.reset_index(drop=True),
#     gdf_final_5_interp.reset_index(drop=True),
#     how="left",
#     predicate="within",
# )
# result = (
#     joined.groupby(level=0)
#     .first()
#     .reset_index(drop=True)
#     .drop(columns=["index_right"], errors="ignore")
# )

# # ===============================
# # 3. CONVERSÃO TARGET (GWh) E INDEXAÇÃO H3
# # ===============================
# target_col = "Summe der Netzverluste [kWh]"
# result[target_col] = (
#     result[target_col]
#     .astype(str)
#     .str.replace(".", "", regex=False)
#     .str.replace(",", ".", regex=False)
# )
# result[target_col] = pd.to_numeric(result[target_col], errors="coerce")

# # Target em GWh
# result["target_GWh"] = result[target_col] / 1e6
# result = result.dropna(subset=["target_GWh"]).reset_index(drop=True)

# # Aplicação H3 (Resolução 5 para abrangência regional de DSOs)
# H3_RESOLUTION = 4


# def latlng_to_h3(row):
#     return h3.latlng_to_cell(row.geometry.y, row.geometry.x, res=H3_RESOLUTION)


# result["h3_cell"] = result.apply(latlng_to_h3, axis=1)

# # ===============================
# # 4. PREPARAÇÃO DE FEATURES E LAGS ESPACIAIS (WX)
# # ===============================
# df_numeric = result.select_dtypes(include=[np.number]).copy()
# cols_to_drop = [
#     target_col,
#     "target_GWh",
#     "latitude",
#     "longitude",
#     "latitude_y",
#     "longitude_y",
#     "x",
#     "y",
#     "x_mp_10km",
#     "y_mp_10km",
#     "ADE",
#     "AGS",
#     "IBZ",
#     "SDV_AGS",
#     "EWZ",
#     "KFL",
#     "AGS_ST",
#     "OBJECTID",
# ]
# df_numeric = df_numeric.drop(
#     columns=[c for c in cols_to_drop if c in df_numeric.columns],
#     errors="ignore",
# )

# scaler = StandardScaler()
# X = scaler.fit_transform(df_numeric)
# standardized_df = pd.DataFrame(X, columns=df_numeric.columns)
# y = result["target_GWh"].values

# coords = np.array(list(zip(result.geometry.x, result.geometry.y)))
# w_knn = libpysal.weights.KNN.from_array(coords, k=4)
# w_knn.transform = "r"

# # Construção dos Spatial Lags (WX)
# WX = w_knn.full()[0] @ X
# X_spatial = np.hstack([X, WX])

# # ===============================
# # 5. MODELAGEM PREDITIVA (XGBoost Espacial)
# # ===============================
# xgb_spatial = XGBRegressor(
#     n_estimators=300,
#     max_depth=4,
#     learning_rate=0.05,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     random_state=42,
# )
# xgb_spatial.fit(X_spatial, y)

# # Previsões
# result["pred_GWh"] = xgb_spatial.predict(X_spatial)
# result["residual_GWh"] = y - result["pred_GWh"]

# # ===============================
# # 6. CÁLCULO SHAP GLOBAL E GEOSHAP
# # ===============================
# explainer = shap.TreeExplainer(xgb_spatial)
# shap_vals_all = explainer.shap_values(X_spatial)
# # Retém apenas os valores SHAP correspondentes às variáveis originais X
# shap_values = shap_vals_all[:, : X.shape[1]]

# # Kernel Gaussiano para GeoSHAP
# dist_matrix = cdist(coords, coords)
# bandwidth = np.percentile(dist_matrix, 10)
# gaussian_kernel = np.exp(-(dist_matrix**2) / (2 * bandwidth**2))
# W_kernel = gaussian_kernel / gaussian_kernel.sum(axis=1, keepdims=True)

# # GeoSHAP: Matriz ponderada por proximidade
# gw_shap = W_kernel @ np.abs(shap_values)

# # Agregação por célula H3
# result_h3 = (
#     result.groupby("h3_cell")
#     .agg({"target_GWh": "mean", "pred_GWh": "mean", "residual_GWh": "mean"})
#     .reset_index()
# )

# shap_h3_df = pd.DataFrame(gw_shap, columns=standardized_df.columns)
# shap_h3_df["h3_cell"] = result["h3_cell"]
# shap_h3_agg = shap_h3_df.groupby("h3_cell").mean().reset_index()

# h3_main = result_h3.merge(shap_h3_agg, on="h3_cell")

# # ===============================
# # 7. APRENDIZADO NÃO SUPERVISIONADO HÍBRIDO (SOM + WARD)
# # ===============================
# # Seleciona as variáveis top de impacto no GeoSHAP
# top_features = (
#     shap_h3_agg.drop(columns=["h3_cell"])
#     .mean()
#     .sort_values(ascending=False)
#     .head(15)
#     .index.tolist()
# )

# X_som = StandardScaler().fit_transform(h3_main[top_features])

# SOM_X, SOM_Y = 10, 10
# som = MiniSom(
#     x=SOM_X,
#     y=SOM_Y,
#     input_len=X_som.shape[1],
#     sigma=1.5,
#     learning_rate=0.5,
#     random_seed=42,
# )
# som.pca_weights_init(X_som)
# som.train_random(X_som, 5000)

# # Extração dos pesos dos neurônios para Ward
# weights_flat = som.get_weights().reshape(-1, X_som.shape[1])
# ward = AgglomerativeClustering(n_clusters=3, linkage="ward")
# neuron_clusters = ward.fit_predict(weights_flat)

# # Mapeamento das células H3 aos clusters SOM-Ward
# winners = np.array([som.winner(x) for x in X_som])
# winner_idxs = np.ravel_multi_index(winners.T, (SOM_X, SOM_Y))
# h3_main["som_ward_cluster"] = neuron_clusters[winner_idxs]

# # ===============================
# # 8. SISTEMA DE CLASSIFICAÇÃO MULTICRITÉRIO FUZZY
# # ===============================
# scaler_m = MinMaxScaler()

# # Mapeia score do cluster baseado nas perdas médias previstas
# cluster_risk_rank = (
#     h3_main.groupby("som_ward_cluster")["pred_GWh"].mean().sort_values()
# )
# cluster_score_map = {
#     c: i / (len(cluster_risk_rank) - 1)
#     for i, c in enumerate(cluster_risk_rank.index)
# }
# h3_main["cluster_score"] = h3_main["som_ward_cluster"].map(cluster_score_map)

# # Variáveis para a composição Fuzzy
# fuzzy_cols = ["pred_GWh"] + top_features[:3]
# norm_fuzzy = pd.DataFrame(
#     scaler_m.fit_transform(h3_main[fuzzy_cols]),
#     columns=fuzzy_cols,
#     index=h3_main.index,
# )

# # Pesos Fuzzy: Predição (50%), Impacto GeoSHAP (30%), Perfil SOM (20%)
# norm_fuzzy["geoshap_score"] = norm_fuzzy[top_features[:3]].mean(axis=1)

# h3_main["fuzzy_risk_index"] = (
#     0.50 * norm_fuzzy["pred_GWh"]
#     + 0.30 * norm_fuzzy["geoshap_score"]
#     + 0.20 * h3_main["cluster_score"]
# )

# # Categorização dos Níveis de Risco
# q1, q2 = h3_main["fuzzy_risk_index"].quantile([0.33, 0.66])
# h3_main["risk_category"] = pd.cut(
#     h3_main["fuzzy_risk_index"],
#     bins=[-np.inf, q1, q2, np.inf],
#     labels=["Low Risk", "Medium Risk", "High Risk"],
# )


# # Reconstrução da geometria H3 Shapely
# def h3_to_polygon(cell):
#     coords = h3.cell_to_boundary(cell)
#     return Polygon([(c[1], c[0]) for c in coords])


# h3_main["geometry"] = h3_main["h3_cell"].apply(h3_to_polygon)
# gdf_h3_final = gpd.GeoDataFrame(h3_main, geometry="geometry", crs="EPSG:4326")

# # Exportação do resultado final
# gdf_h3_final.to_file(
#     os.path.join(out_dir, "alemanha_h3_fuzzy_risk_profiles.geojson"),
#     driver="GeoJSON",
# )

# print("\n Processamento concluído com sucesso!")
# print(f"Total de Células H3 Analisadas: {len(gdf_h3_final)}")
# print("\nDistribuição dos Perfis de Risco Fuzzy:")
# print(gdf_h3_final["risk_category"].value_counts())

# #%%
# # # ===============================
# # # 1. FIGURA 1: FUNÇÕES DE PERTINÊNCIA FUZZY (TEÓRICA)
# # # ===============================
# # # Vetor do score Fuzzy (0 a 1)
# # x_fuzzy = np.linspace(0, 1, 500)


# # # Definição de funções de pertinência trapezoidais/triangulares
# # def low_risk_mf(x):
# #     return np.maximum(0, np.minimum(1, (0.4 - x) / 0.4))


# # def medium_risk_mf(x):
# #     return np.maximum(
# #         0, np.minimum((x - 0.2) / 0.3, np.minimum(1, (0.8 - x) / 0.3))
# #     )


# # def high_risk_mf(x):
# #     return np.maximum(0, np.minimum(1, (x - 0.6) / 0.4))


# # # Plot das Curvas de Pertinência
# # plt.figure(figsize=(10, 5))
# # plt.plot(
# #     x_fuzzy,
# #     low_risk_mf(x_fuzzy),
# #     label="Low Risk",
# #     color="#1a9850",
# #     linewidth=2.5,
# # )
# # plt.plot(
# #     x_fuzzy,
# #     medium_risk_mf(x_fuzzy),
# #     label="Medium Risk",
# #     color="#e6b800",
# #     linewidth=2.5,
# # )
# # plt.plot(
# #     x_fuzzy,
# #     high_risk_mf(x_fuzzy),
# #     label="High Risk",
# #     color="#d73027",
# #     linewidth=2.5,
# # )

# # plt.title("Fuzzy Membership Functions", fontsize=14, fontweight="bold", pad=15)
# # plt.xlabel("Fuzzy Risk Index", fontsize=12)
# # plt.ylabel("Degree of Membership", fontsize=12)
# # plt.grid(True, linestyle="--", alpha=0.5)
# # plt.legend(fontsize=11, loc="center right")
# # plt.ylim(-0.05, 1.05)
# # plt.xlim(0, 1)

# # plt.tight_layout()
# # plt.show()

# # ===============================
# # 2. FIGURA 2: DISTRIBUIÇÃO DOS SCORES REAIS DAS CÉLULAS H3
# # ===============================
# # Carregando dados gerados no script anterior (se disponíveis)
# diretorio_atual = os.getcwd()
# nome_sistema = platform.system()
# volta = "../" if nome_sistema == "Linux" else "..\\"
# out_dir = os.path.join(diretorio_atual, volta, "saidas")
# caminho_csv = os.path.join(out_dir, "alemanha_h3_fuzzy_risk_profiles.geojson")

# try:
#     import geopandas as gpd

#     gdf_h3 = gpd.read_file(caminho_csv)

#     q1, q2 = gdf_h3["fuzzy_risk_index"].quantile([0.33, 0.66])

#     plt.figure(figsize=(10, 5))
#     sns.histplot(
#         gdf_h3["fuzzy_risk_index"],
#         kde=True,
#         color="#2b5c8f",
#         bins=25,
#         stat="density",
#         alpha=0.4,
#     )

#     # Linhas verticais para os quantis de corte
#     plt.axvline(
#         q1,
#         color="#1a9850",
#         linestyle="--",
#         linewidth=2,
#         label=f"Low/Mid Cut ({q1:.2f})",
#     )
#     plt.axvline(
#         q2,
#         color="#d73027",
#         linestyle="--",
#         linewidth=2,
#         label=f"Mid/High Cut ({q2:.2f})",
#     )

#     plt.title(
#         "Actual Distribution of the Fuzzy Risk Index (H3 Cells)",
#         fontsize=14,
#         fontweight="bold",
#         pad=15,
#     )
#     plt.xlabel("Fuzzy Risk Index", fontsize=12)
#     plt.ylabel("Densidade de Células H3", fontsize=12)
#     plt.grid(True, linestyle="--", alpha=0.5)
#     plt.legend(fontsize=11)

#     plt.tight_layout()
#     plt.show()

# except Exception as e:
#     print(
#         f"Aviso: Não foi possível carregar o arquivo real para o segundo gráfico ({e}). Exibindo apenas a curva teórica."
#     )

# #%%

# # Carrega o GeoJSON final processado
# caminho_geojson = os.path.join(
#     out_dir, "alemanha_h3_fuzzy_risk_profiles.geojson"
# )
# gdf_h3 = gpd.read_file(caminho_geojson)

# # Assegura projeção WGS84 para exibição correta no Leaflet/Folium
# if gdf_h3.crs != "EPSG:4326":
#     gdf_h3 = gdf_h3.to_crs(epsg=4326)

# # ===============================
# # 2. DEFINIÇÃO DE CORES E ESTILOS
# # ===============================
# # Mapeamento de cores baseado na Categoria de Risco Fuzzy
# color_map_category = {
#     "High Risk": "#d73027",  # Vermelho
#     "Medium Risk": "#fee08b",  # Amarelo/Laranja
#     "Low Risk": "#1a9850",  # Verde
# }


# def get_feature_style(feature):
#     risk_cat = feature["properties"].get("risk_category", "Low Risk")
#     color = color_map_category.get(risk_cat, "#808080")

#     return {
#         "fillColor": color,
#         "color": "#ffffff",  # Bordas brancas para melhor contraste na imagem de satélite
#         "weight": 1.5,
#         "fillOpacity": 0.55,  # Leve transparência para enxergar a imagem abaixo
#     }


# def get_highlight_style(feature):
#     return {
#         "weight": 3.0,
#         "color": "#ffff00",  # Destaque amarelo brilhante ao passar o mouse
#         "fillOpacity": 0.80,
#     }


# # ===============================
# # 3. CRIAÇÃO DO MAPA BASE COM SATÉLITE
# # ===============================
# centro_lat = gdf_h3.geometry.centroid.y.mean()
# centro_lon = gdf_h3.geometry.centroid.x.mean()

# # Inicializa o mapa com fundo transparente
# mapa = folium.Map(
#     location=[centro_lat, centro_lon],
#     zoom_start=8,
#     tiles=None,  # Definiremos as camadas explicitamente abaixo
#     control_scale=True,
# )

# # Camada 1: Imagem de Satélite de Alta Resolução (Esri World Imagery)
# folium.TileLayer(
#     tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
#     attr="Esri, Maxar, Earthstar Geographics, and the GIS User Community",
#     name="Satélite (Esri World Imagery)",
#     overlay=False,
#     control=True,
# ).add_to(mapa)

# # Camada 2: Rótulos e Nomes de Cidades/Ruas por cima do Satélite (Opcional)
# folium.TileLayer(
#     tiles="https://{s}.basemaps.cartocdn.com/rastertiles/voyager_only_labels/{z}/{x}/{y}{r}.png",
#     attr="&copy; OpenStreetMap &copy; CARTO",
#     name="Rótulos de Cidades (Labels)",
#     overlay=True,
#     control=True,
# ).add_to(mapa)

# # Camada 3: Fundo Clássico Vetorial (Para alternar se necessário)
# folium.TileLayer(
#     tiles="CartoDB positron", name="Mapa Claro (Positron)", overlay=False, control=True
# ).add_to(mapa)

# # Adiciona ferramentas de medição e tela cheia
# Fullscreen().add_to(mapa)
# MeasureControl(position="topleft").add_to(mapa)

# # ===============================
# # 4. CAMADA DAS CÉLULAS H3
# # ===============================
# popup_fields = [
#     "h3_cell",
#     "fuzzy_risk_index",
#     "risk_category",
#     "som_ward_cluster",
#     "pred_GWh",
#     "target_GWh",
# ]

# popup_aliases = [
#     "Célula H3:",
#     "Índice Fuzzy de Risco:",
#     "Categoria de Risco:",
#     "Cluster SOM-Ward:",
#     "Perda Prevista (GWh):",
#     "Perda Real (GWh):",
# ]

# geojson_layer = folium.GeoJson(
#     gdf_h3,
#     name="Perfil de Risco Fuzzy (H3)",
#     style_function=get_feature_style,
#     highlight_function=get_highlight_style,
#     tooltip=folium.GeoJsonTooltip(
#         fields=["h3_cell", "risk_category", "fuzzy_risk_index"],
#         aliases=["H3:", "Risco:", "Score Fuzzy:"],
#         localize=True,
#         sticky=True,
#     ),
#     popup=folium.GeoJsonPopup(
#         fields=popup_fields,
#         aliases=popup_aliases,
#         localize=True,
#         labels=True,
#         style="font-family: Arial; font-size: 12px; width: 250px;",
#     ),
# )

# geojson_layer.add_to(mapa)

# # ===============================
# # 5. LEGENDA PERSONALIZADA
# # ===============================
# html_legenda = """
# <div style="
#     position: fixed; 
#     bottom: 30px; left: 30px; width: 180px; height: 130px; 
#     background-color: rgba(255, 255, 255, 0.9); z-index:9999; font-size:12px;
#     border:2px solid #333; border-radius:8px; padding: 10px;
#     box-shadow: 2px 2px 5px rgba(0,0,0,0.5);
#     font-family: Arial, sans-serif;">
#     <b style="font-size: 13px;">Risco Fuzzy H3</b><br>
#     <i style="background: #d73027; width: 15px; height: 15px; float: left; margin-right: 8px; opacity: 0.8; border-radius: 3px;"></i> Alto Risco<br><br>
#     <i style="background: #fee08b; width: 15px; height: 15px; float: left; margin-right: 8px; opacity: 0.8; border-radius: 3px;"></i> Médio Risco<br><br>
#     <i style="background: #1a9850; width: 15px; height: 15px; float: left; margin-right: 8px; opacity: 0.8; border-radius: 3px;"></i> Baixo Risco
# </div>
# """

# mapa.get_root().html.add_child(folium.Element(html_legenda))

# # Controle de camadas (Permite ligar/desligar rótulos, alternar mapa base, etc.)
# folium.LayerControl(collapsed=False).add_to(mapa)

# # ===============================
# # 6. SALVAR ARQUIVO HTML
# # ===============================
# caminho_saida_html = os.path.join(
#     out_dir, "mapa_satelite_h3_fuzzy_alemanha.html"
# )
# mapa.save(caminho_saida_html)

# print(f"Mapa de Satélite em HTML gerado com sucesso!")
# print(f"Arquivo salvo em: {caminho_saida_html}")

#%%

# import matplotlib.pyplot as plt
# from matplotlib.patches import Circle
# import numpy as np
# import pandas as pd
# import geopandas as gpd
# import h3
# from shapely.geometry import Point, Polygon

# # ===============================
# # 1. CONFIGURAÇÃO DE ESTILO E FONTE (24pt SERIF / LATEX)
# # ===============================
# plt.rcParams.update(
#     {
#         "font.family": "serif",
#         "font.serif": ["Times New Roman"],
#         "font.size": 24,
#         "axes.titlesize": 22,
    
#         "axes.labelsize": 24,
#         "xtick.labelsize": 16,
#         "ytick.labelsize": 16,
#     }
# )

# # ===============================
# # 2. GERAR DADOS SINTÉTICOS PARA ILUSTRAÇÃO
# # ===============================
# np.random.seed(42)

# num_points = 600
# lats = np.random.uniform(52.10, 52.18, num_points)
# lons = np.random.uniform(11.60, 11.72, num_points)

# shap_values = (lats - 52.10) * 20 + (lons - 11.60) * 15 + np.random.normal(
#     0, 0.5, num_points
# )

# df_obs = pd.DataFrame({"lat": lats, "lon": lons, "shap_val": shap_values})
# gdf_obs = gpd.GeoDataFrame(
#     df_obs,
#     geometry=gpd.points_from_xy(df_obs.lon, df_obs.lat),
#     crs="EPSG:4326",
# )

# H3_RES = 7
# gdf_obs["h3_cell"] = [
#     h3.latlng_to_cell(lat, lon, res=H3_RES) for lat, lon in zip(lats, lons)
# ]

# df_h3_agg = (
#     gdf_obs.groupby("h3_cell")
#     .agg(mean_shap=("shap_val", "mean"), count=("shap_val", "count"))
#     .reset_index()
# )


# def h3_to_poly(cell):
#     coords = h3.cell_to_boundary(cell)
#     return Polygon([(c[1], c[0]) for c in coords])


# df_h3_agg["geometry"] = df_h3_agg["h3_cell"].apply(h3_to_poly)
# gdf_h3 = gpd.GeoDataFrame(df_h3_agg, geometry="geometry", crs="EPSG:4326")

# centroids = np.array(
#     [[geom.centroid.x, geom.centroid.y] for geom in gdf_h3.geometry]
# )
# dist_matrix = np.linalg.norm(
#     centroids[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=-1
# )

# bandwidth = np.percentile(dist_matrix, 25)
# weights = np.exp(-(dist_matrix**2) / (2 * (bandwidth**2)))

# geoshap_vals = (weights @ gdf_h3["mean_shap"].values) / weights.sum(axis=1)
# gdf_h3["geoshap"] = geoshap_vals

# # ===============================
# # 3. PLOT EM GRADE 2x2 (2 LINHAS X 2 COLUNAS)
# # ===============================
# # Subplots definidos como 2 linhas e 2 colunas
# fig, axes = plt.subplots(2, 2, figsize=(16, 14), sharex=True, sharey=True)
# cmap = "coolwarm"
# vmin, vmax = gdf_obs["shap_val"].min(), gdf_obs["shap_val"].max()

# # --- PAINEL A (Linha 0, Coluna 0): SHAP Nível Observação ---
# ax1 = axes[0, 0]
# sc1 = ax1.scatter(
#     gdf_obs.geometry.x,
#     gdf_obs.geometry.y,
#     c=gdf_obs["shap_val"],
#     cmap=cmap,
#     s=40,
#     alpha=0.8,
#     vmin=vmin,
#     vmax=vmax,
# )
# ax1.set_title("(a) Observation SHAP\n" + r"$\phi_{ij}$", pad=15)
# ax1.set_ylabel("Latitude")
# ax1.grid(True, linestyle=":", alpha=0.5)

# # --- PAINEL B (Linha 0, Coluna 1): Média SHAP no Grid H3 ---
# ax2 = axes[0, 1]
# gdf_h3.plot(
#     column="mean_shap",
#     ax=ax2,
#     cmap=cmap,
#     edgecolor="black",
#     linewidth=0.5,
#     vmin=vmin,
#     vmax=vmax,
# )
# ax2.set_title(
#     "(b) Aggregated SHAP\n",
#     pad=15,
# )
# ax2.grid(True, linestyle=":", alpha=0.5)

# # --- PAINEL C (Linha 1, Coluna 0): Kernel Gaussiano & Vizinhança ---
# ax3 = axes[1, 0]
# gdf_h3.plot(
#     column="mean_shap",
#     ax=ax3,
#     cmap=cmap,
#     edgecolor="gray",
#     linewidth=0.3,
#     alpha=0.4,
#     vmin=vmin,
#     vmax=vmax,
# )

# target_idx = int(len(gdf_h3) / 2)
# target_geom = gdf_h3.iloc[target_idx].geometry
# center_x, center_y = target_geom.centroid.x, target_geom.centroid.y

# gpd.GeoSeries([target_geom]).plot(
#     ax=ax3, facecolor="none", edgecolor="black", linewidth=2.5
# )

# circle1 = Circle(
#     (center_x, center_y),
#     bandwidth * 0.6,
#     color="black",
#     fill=False,
#     linestyle="--",
#     linewidth=1.8,
# )
# circle2 = Circle(
#     (center_x, center_y),
#     bandwidth * 1.2,
#     color="black",
#     fill=False,
#     linestyle=":",
#     linewidth=1.5,
# )
# ax3.add_patch(circle1)
# ax3.add_patch(circle2)

# ax3.set_title(
#     "(c) Gaussian Kernel\n", pad=15
# )
# ax3.set_xlabel("Longitude")
# ax3.set_ylabel("Latitude")
# ax3.grid(True, linestyle=":", alpha=0.5)

# # --- PAINEL D (Linha 1, Coluna 1): Superfície GeoSHAP Suavizada ---
# ax4 = axes[1, 1]
# gdf_h3.plot(
#     column="geoshap",
#     ax=ax4,
#     cmap=cmap,
#     edgecolor="black",
#     linewidth=0.3,
#     vmin=vmin,
#     vmax=vmax,
# )
# ax4.set_title(
#     "(d) GeoSHAP Surface\n",
#     pad=15,
# )
# ax4.set_xlabel("Longitude")
# ax4.grid(True, linestyle=":", alpha=0.5)

# # ===============================
# # 4. BARRA DE CORES E AJUSTES FINAIS
# # ===============================
# # Espaçamento entre os subplots e margens para caber a barra lateral
# fig.subplots_adjust(
#     top=0.90, bottom=0.08, left=0.08, right=0.84, hspace=0.35, wspace=0.15
# )

# # Posicionamento da Barra de Cores Vertical à direita
# cbar_ax = fig.add_axes([0.86, 0.12, 0.025, 0.72])
# sm = plt.cm.ScalarMappable(
#     cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax)
# )
# sm._A = []
# cbar = fig.colorbar(sm, cax=cbar_ax)

# cbar.set_label(
#     "Feature Impact / SHAP Value",
#     rotation=270,
#     labelpad=25,
   
#     fontsize=22,
# )

# # Salvar figura SVG
# plt.savefig(
#     "geoshap_spatialization_pipeline_2x2.svg",
#     format="svg",
#     bbox_inches="tight",
# )
# plt.show()

# print("Figura em grade 2x2 salva em 'geoshap_spatialization_pipeline_2x2.svg'")

#%%%

# import matplotlib.pyplot as plt
# import numpy as np

# # Set publication style with Times New Roman and font size 21
# plt.rcParams.update({
#     'font.family': 'serif',
#     'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
#     'font.size': 24,
#     'axes.labelsize': 24,
#     'axes.titlesize': 24,
#     'xtick.labelsize': 19,
#     'ytick.labelsize': 19,
#     'legend.fontsize': 19,
#     'figure.titlesize': 23
# })

# # Layout: 2 Linhas x 1 Coluna
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 16))

# # ==============================================================================
# # PAINEL (a): Funções de Pertinência Hiper-Trapezoidais Adaptativas
# # ==============================================================================
# x = np.linspace(-0.5, 2.5, 500)

# def trapezoid(x, a, b, c, d):
#     y = np.zeros_like(x)
#     idx1 = (x >= a) & (x < b)
#     if b > a:
#         y[idx1] = (x[idx1] - a) / (b - a)
#     idx2 = (x >= b) & (x <= c)
#     y[idx2] = 1.0
#     idx3 = (x > c) & (x <= d)
#     if d > c:
#         y[idx3] = (d - x[idx3]) / (d - c)
#     return y

# mf_low = trapezoid(x, -0.5, -0.5, 0.2, 0.7)
# mf_med = trapezoid(x, 0.3, 0.7, 1.2, 1.6)
# mf_high = trapezoid(x, 1.2, 1.7, 2.5, 2.5)

# ax1.plot(x, mf_low, color='#1f77b4', linewidth=2.5, label='Low Impact ($A_{j,1}$)')
# ax1.plot(x, mf_med, color='#ff7f0e', linewidth=2.5, label='Moderate Impact ($A_{j,2}$)')
# ax1.plot(x, mf_high, color='#d62728', linewidth=2.5, label='High Impact ($A_{j,3}$)')

# # Anotações de Suporte e Núcleo (Core)
# ax1.annotate('', xy=(0.3, -0.05), xytext=(1.6, -0.05),
#              arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
# ax1.text(0.95, -0.14, 'Support $(a_j, d_j)$', ha='center', va='top', fontsize=21, color='black')

# ax1.annotate('', xy=(0.7, 1.05), xytext=(1.2, 1.05),
#              arrowprops=dict(arrowstyle='<->', color='#ff7f0e', lw=1.8))
# ax1.text(0.95, 1.13, 'Core / Plateau $(b_j, c_j)$ ($\mu=1.0$)', ha='center', va='bottom', fontsize=21, color='#ff7f0e', fontweight='bold')

# # Linhas verticais tracejadas dos parâmetros
# for val, label in zip([0.3, 0.7, 1.2, 1.6], ['$a_j$', '$b_j$', '$c_j$', '$d_j$']):
#     ax1.axvline(val, color='#ff7f0e', linestyle='--', alpha=0.5, linewidth=1.2)
#     ax1.text(val, -0.02, label, ha='center', va='top', fontsize=21, color='#ff7f0e', fontweight='bold')

# # Ponto de observação GeoSHAP x_j*
# x_star = 0.95
# mu_star = trapezoid(np.array([x_star]), 0.3, 0.7, 1.2, 1.6)[0]
# ax1.axvline(x_star, color='black', linestyle=':', linewidth=2)
# ax1.scatter([x_star], [mu_star], color='black', zorder=5, s=80)
# ax1.annotate(f'Observed GeoSHAP ($x_j^*$)\n$\mu(x_j^*)=1.0$', xy=(x_star, mu_star), xytext=(x_star+0.25, 0.65),
#              arrowprops=dict(arrowstyle='->', color='black', lw=1.5), fontsize=19,
#              bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8f9fa', edgecolor='gray', alpha=0.9))

# ax1.set_title('(a) Data-Driven Hyper-Trapezoidal Membership Functions', pad=25)
# ax1.set_xlabel('GeoSHAP Feature Attribution ($x_j$)')
# ax1.set_ylabel('Membership Degree $\mu(x_j)$')
# ax1.set_ylim(-0.22, 1.28)
# ax1.set_xlim(-0.2, 2.3)
# ax1.grid(True, linestyle=':', alpha=0.5)

# # Legenda abaixo do Painel A
# ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2, framealpha=0.9)


# # ==============================================================================
# # PAINEL (b): Agregação de Regras e Desfuzzificação por CoG
# # ==============================================================================
# y = np.linspace(0, 100, 500)

# mu_out_low = 0.1 * trapezoid(y, 0, 0, 20, 40)
# mu_out_med = 0.45 * trapezoid(y, 25, 45, 55, 75)
# mu_out_high = 0.85 * trapezoid(y, 60, 80, 100, 100)
# mu_agg = np.maximum(mu_out_low, np.maximum(mu_out_med, mu_out_high))

# ax2.plot(y, mu_out_low, color='#2ca02c', linestyle='--', linewidth=1.8, label='Low Risk Output')
# ax2.plot(y, mu_out_med, color='#bcbd22', linestyle='--', linewidth=1.8, label='Moderate Risk Output')
# ax2.plot(y, mu_out_high, color='#d62728', linestyle='--', linewidth=1.8, label='High Risk Output')

# ax2.fill_between(y, 0, mu_agg, color='#9467bd', alpha=0.35, label='Aggregated Fuzzy Area $B_{agg}$')
# ax2.plot(y, mu_agg, color='#7030a0', linewidth=2.5)

# cog_val = np.sum(y * mu_agg) / np.sum(mu_agg)

# ax2.axvline(cog_val, color='#d62728', linestyle='-', linewidth=2.5)
# ax2.scatter([cog_val], [0], color='#d62728', s=100, zorder=6)
# ax2.annotate(f'Defuzzified RIII Score\n$\mathrm{{RIII}} = {cog_val:.1f}$', 
#              xy=(cog_val, 0.4), xytext=(cog_val-42, 0.75),
#              arrowprops=dict(arrowstyle='->', color='#d62728', lw=2),
#              fontsize=21, fontweight='bold', color='#800000',
#              bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffe6e6', edgecolor='#d62728', alpha=0.95))

# # Zonas de Risco
# ax2.axvline(40, color='gray', linestyle=':', alpha=0.7)
# ax2.axvline(75, color='gray', linestyle=':', alpha=0.7)
# ax2.text(20, 1.05, 'Low Risk\n($<40$)', ha='center', va='bottom', fontsize=18, color='green')
# ax2.text(57.5, 1.05, 'Moderate Risk\n($40-75$)', ha='center', va='bottom', fontsize=18, color='#b8860b')
# ax2.text(87.5, 1.05, 'High Risk\n($>75$)', ha='center', va='bottom', fontsize=18, color='red')

# ax2.set_title('(b) Rule Aggregation & Center of Gravity (CoG) Defuzzification', pad=25)
# ax2.set_xlabel('Regional Inspection Importance Index ($\mathrm{RIII} \in [0, 100]$)')
# ax2.set_ylabel('Aggregated Membership Degree $\mu_{B}(y)$')
# ax2.set_ylim(-0.05, 1.28)
# ax2.set_xlim(0, 100)
# ax2.grid(True, linestyle=':', alpha=0.5)

# # Legenda abaixo do Painel B
# ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2, framealpha=0.9)

# # Ajuste de layout vertical
# plt.subplots_adjust(hspace=0.6)
# plt.savefig('fuzzy_decision_model_vertical.pdf', bbox_inches='tight')
# plt.savefig('fuzzy_decision_model_vertical.png', dpi=300, bbox_inches='tight')
# plt.show()

#%%

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 17:45:25 2026

@author: Natalia
"""

"""
Integrated Framework for Power Grid Loss Analysis (Germany)
Combines: Nominatim Geocoding + H3 Indexing + Spatial XGBoost (WX) +
GeoSHAP + SOM-Ward Clustering + Fuzzy Risk Index (Linear + Trapezoidal)
Target: Summe der Netzverluste [GWh]
"""

import os
import platform
import re
import time
import warnings
import geopandas as gpd
import h3
import libpysal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import shap
from minisom import MiniSom
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from shapely.geometry import Polygon
from shapely.wkt import loads as wkt_loads
from xgboost import XGBRegressor
import seaborn as sns

import branca.colormap as cm
import folium
from folium.plugins import Fullscreen, MeasureControl

warnings.filterwarnings("ignore")

# ===============================
# 1. CONFIGURAÇÃO DE DIRETÓRIOS E SISTEMA
# ===============================
diretorio_atual = os.getcwd()
nome_sistema_operacional = platform.system()
volta_nivel = "../" if nome_sistema_operacional == "Linux" else "..\\"

out_dir = os.path.join(diretorio_atual, volta_nivel, "saidas")
os.makedirs(out_dir, exist_ok=True)

# ===============================
# 2. CARREGAMENTO E GEOCODIFICAÇÃO (NOMINATIM)
# ===============================
caminho_dados_DSO = os.path.join(
    diretorio_atual, volta_nivel, "entradas", "DSO-Sachsen-Anhalt-file.xlsx"
)
caminho_dados_DSO_L = os.path.join(
    diretorio_atual, volta_nivel, "entradas", "DSO-LowerSaxony.xlsx"
)

dso_sa = pd.read_excel(caminho_dados_DSO)
dso_ls = pd.read_excel(caminho_dados_DSO_L)
dso_sa["state"] = "Sachsen-Anhalt"
dso_ls["state"] = "Lower Saxony"
dados_combinados = pd.concat([dso_sa, dso_ls], ignore_index=True)


def geocode_city(city, country="Germany"):
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": f"{city}, {country}", "format": "json", "limit": 1}
    headers = {
        "User-Agent": "dso_analysis_app (researcher_email@domain.com)"
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if len(data) > 0:
                return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception:
        pass
    return None, None


if (
    "latitude" not in dados_combinados.columns
    or dados_combinados["latitude"].isnull().any()
):
    print("Geocodificando cidades via Nominatim...")
    lats, longs = [], []
    for city in dados_combinados["Ort"]:
        lat, lon = geocode_city(city)
        lats.append(lat)
        longs.append(lon)
        time.sleep(1)
    dados_combinados["latitude"] = lats
    dados_combinados["longitude"] = longs

# Carregamento dos arquivos processados e polígonos
caminho_shp = os.path.join(
    diretorio_atual,
    volta_nivel,
    "vg-hist.utm32s.shape",
    "daten",
    "utm32s",
    "shape",
    "VG-Hist_1990-10-03_KRS.shp",
)
dados_shp = gpd.read_file(caminho_shp).to_crs(epsg=4326)

gdf_final_5_interp = pd.read_csv(
    os.path.join(out_dir, "gdf_final_5_interp.csv"), sep=";", encoding="utf-8"
)
gdf_final_5_interp["geometry"] = gdf_final_5_interp["geometry"].apply(
    wkt_loads
)
gdf_final_5_interp = gpd.GeoDataFrame(
    gdf_final_5_interp, geometry="geometry", crs=dados_shp.crs
)

gdf_final_4_3 = pd.read_csv(
    os.path.join(out_dir, "gdf_final_3.csv"), sep=";", encoding="utf-8"
)
gdf_final_4_3 = gdf_final_4_3[
    gdf_final_4_3["state"].isin(["Lower Saxony", "Sachsen-Anhalt"])
]
gdf_final_4_3 = gdf_final_4_3.drop_duplicates().dropna(
    subset=["Summe der Netzverluste [kWh]"]
)
gdf_final_4_3 = gpd.GeoDataFrame(
    gdf_final_4_3,
    geometry=gpd.points_from_xy(
        gdf_final_4_3.longitude_y, gdf_final_4_3.latitude_y
    ),
    crs=gdf_final_5_interp.crs,
)

joined = gpd.sjoin(
    gdf_final_4_3.reset_index(drop=True),
    gdf_final_5_interp.reset_index(drop=True),
    how="left",
    predicate="within",
)
result = (
    joined.groupby(level=0)
    .first()
    .reset_index(drop=True)
    .drop(columns=["index_right"], errors="ignore")
)

# ===============================
# 3. CONVERSÃO TARGET (GWh) E INDEXAÇÃO H3
# ===============================
target_col = "Summe der Netzverluste [kWh]"
result[target_col] = (
    result[target_col]
    .astype(str)
    .str.replace(".", "", regex=False)
    .str.replace(",", ".", regex=False)
)
result[target_col] = pd.to_numeric(result[target_col], errors="coerce")

result["target_GWh"] = result[target_col] / 1e6
result = result.dropna(subset=["target_GWh"]).reset_index(drop=True)

H3_RESOLUTION = 4


def latlng_to_h3(row):
    return h3.latlng_to_cell(row.geometry.y, row.geometry.x, res=H3_RESOLUTION)


result["h3_cell"] = result.apply(latlng_to_h3, axis=1)

# ===============================
# 4. PREPARAÇÃO DE FEATURES E LAGS ESPACIAIS (WX)
# ===============================
df_numeric = result.select_dtypes(include=[np.number]).copy()
cols_to_drop = [
    target_col,
    "target_GWh",
    "latitude",
    "longitude",
    "latitude_y",
    "longitude_y",
    "x",
    "y",
    "x_mp_10km",
    "y_mp_10km",
    "ADE",
    "AGS",
    "IBZ",
    "SDV_AGS",
    "EWZ",
    "KFL",
    "AGS_ST",
    "OBJECTID",
]
df_numeric = df_numeric.drop(
    columns=[c for c in cols_to_drop if c in df_numeric.columns],
    errors="ignore",
)

scaler = StandardScaler()
X = scaler.fit_transform(df_numeric)
standardized_df = pd.DataFrame(X, columns=df_numeric.columns)
y = result["target_GWh"].values

coords = np.array(list(zip(result.geometry.x, result.geometry.y)))
w_knn = libpysal.weights.KNN.from_array(coords, k=4)
w_knn.transform = "r"

WX = w_knn.full()[0] @ X
X_spatial = np.hstack([X, WX])


# ===============================
# 5. MODELAGEM PREDITIVA (XGBoost Espacial)
# ===============================
xgb_spatial = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
xgb_spatial.fit(X_spatial, y)

result["pred_GWh"] = xgb_spatial.predict(X_spatial)
result["residual_GWh"] = y - result["pred_GWh"]

# ===============================
# 6. CÁLCULO SHAP GLOBAL E GEOSHAP
# ===============================
explainer = shap.TreeExplainer(xgb_spatial)
shap_vals_all = explainer.shap_values(X_spatial)
shap_values = shap_vals_all[:, : X.shape[1]]

dist_matrix = cdist(coords, coords)
bandwidth = np.percentile(dist_matrix, 10)
gaussian_kernel = np.exp(-(dist_matrix**2) / (2 * bandwidth**2))
W_kernel = gaussian_kernel / gaussian_kernel.sum(axis=1, keepdims=True)

gw_shap = W_kernel @ np.abs(shap_values)

result_h3 = (
    result.groupby("h3_cell")
    .agg({"target_GWh": "mean", "pred_GWh": "mean", "residual_GWh": "mean"})
    .reset_index()
)

shap_h3_df = pd.DataFrame(gw_shap, columns=standardized_df.columns)
shap_h3_df["h3_cell"] = result["h3_cell"]
shap_h3_agg = shap_h3_df.groupby("h3_cell").mean().reset_index()

h3_main = result_h3.merge(shap_h3_agg, on="h3_cell")

# ===============================
# 7. APRENDIZADO NÃO SUPERVISIONADO HÍBRIDO (SOM + WARD)
# ===============================
top_features = (
    shap_h3_agg.drop(columns=["h3_cell"])
    .mean()
    .sort_values(ascending=False)
    .head(15)
    .index.tolist()
)

X_som = StandardScaler().fit_transform(h3_main[top_features])

SOM_X, SOM_Y = 10, 10
som = MiniSom(
    x=SOM_X,
    y=SOM_Y,
    input_len=X_som.shape[1],
    sigma=1.5,
    learning_rate=0.5,
    random_seed=42,
)
som.pca_weights_init(X_som)
som.train_random(X_som, 5000)

weights_flat = som.get_weights().reshape(-1, X_som.shape[1])
ward = AgglomerativeClustering(n_clusters=3, linkage="ward")
neuron_clusters = ward.fit_predict(weights_flat)

winners = np.array([som.winner(x) for x in X_som])
winner_idxs = np.ravel_multi_index(winners.T, (SOM_X, SOM_Y))
h3_main["som_ward_cluster"] = neuron_clusters[winner_idxs]

# ===============================
# 8. SISTEMA DE CLASSIFICAÇÃO MULTICRITÉRIO FUZZY (INCLUINDO TRAPEZOIDAL)
# ===============================
scaler_m = MinMaxScaler()

cluster_risk_rank = (
    h3_main.groupby("som_ward_cluster")["pred_GWh"].mean().sort_values()
)
cluster_score_map = {
    c: i / (len(cluster_risk_rank) - 1)
    for i, c in enumerate(cluster_risk_rank.index)
}
h3_main["cluster_score"] = h3_main["som_ward_cluster"].map(cluster_score_map)

fuzzy_cols = ["pred_GWh"] + top_features[:3]
norm_fuzzy = pd.DataFrame(
    scaler_m.fit_transform(h3_main[fuzzy_cols]),
    columns=fuzzy_cols,
    index=h3_main.index,
)

norm_fuzzy["geoshap_score"] = norm_fuzzy[top_features[:3]].mean(axis=1)

# Score continuo Fuzzy (0 a 1)
h3_main["fuzzy_risk_index"] = (
    0.50 * norm_fuzzy["pred_GWh"]
    + 0.30 * norm_fuzzy["geoshap_score"]
    + 0.20 * h3_main["cluster_score"]
)

# Categorização Baseada em Quantis
q1, q2 = h3_main["fuzzy_risk_index"].quantile([0.33, 0.66])
h3_main["risk_category"] = pd.cut(
    h3_main["fuzzy_risk_index"],
    bins=[-np.inf, q1, q2, np.inf],
    labels=["Low Risk", "Medium Risk", "High Risk"],
)

# -------------------------------------------------------------
# IMPLEMENTAÇÃO DAS FUNÇÕES DE PERTINÊNCIA TRAPEZOIDAL (CASO BRASIL)
# -------------------------------------------------------------
def fuzzy_trap_low(x):
    return np.maximum(0, np.minimum(1, (0.4 - x) / 0.4))


def fuzzy_trap_med(x):
    return np.maximum(
        0, np.minimum((x - 0.2) / 0.3, np.minimum(1, (0.8 - x) / 0.3))
    )


def fuzzy_trap_high(x):
    return np.maximum(0, np.minimum(1, (x - 0.6) / 0.4))


# Aplicação das funções de pertinência trapezoidais
x_vals = h3_main["fuzzy_risk_index"].values
h3_main["mu_low"] = fuzzy_trap_low(x_vals)
h3_main["mu_medium"] = fuzzy_trap_med(x_vals)
h3_main["mu_high"] = fuzzy_trap_high(x_vals)

# Atribuição da Categoria Trapezoidal Dominante e o Grau Máximo (mu_max)
categories = ["Low Risk", "Medium Risk", "High Risk"]
mu_matrix = h3_main[["mu_low", "mu_medium", "mu_high"]].values
max_idx = np.argmax(mu_matrix, axis=1)

h3_main["fuzzy_trap_category"] = [categories[i] for i in max_idx]
h3_main["fuzzy_trap_degree"] = np.max(mu_matrix, axis=1)


def h3_to_polygon(cell):
    coords = h3.cell_to_boundary(cell)
    return Polygon([(c[1], c[0]) for c in coords])


h3_main["geometry"] = h3_main["h3_cell"].apply(h3_to_polygon)
gdf_h3_final = gpd.GeoDataFrame(h3_main, geometry="geometry", crs="EPSG:4326")

gdf_h3_final.to_file(
    os.path.join(out_dir, "alemanha_h3_fuzzy_risk_profiles.geojson"),
    driver="GeoJSON",
)

print("\n Processing successfully completed!")
print(f"Total H3 Cells Analyzed: {len(gdf_h3_final)}")
print("\nDistribution of Fuzzy Risk Profiles (Quantile):")
print(gdf_h3_final["risk_category"].value_counts())
print("\nDistribution of Fuzzy Risk Profiles (Trapezoidal):")
print(gdf_h3_final["fuzzy_trap_category"].value_counts())


# ===============================
# 9. GERAÇÃO DO MAPA INTERATIVO HTML COM CAMADA FUZZY TRAPEZOIDAL
# ===============================
if gdf_h3_final.crs != "EPSG:4326":
    gdf_h3_final = gdf_h3_final.to_crs(epsg=4326)

color_map_category = {
    "High Risk": "#d73027",
    "Medium Risk": "#fee08b",
    "Low Risk": "#1a9850",
}


# Estilo para Categoria Padrão (Quantil)
def get_feature_style_quantile(feature):
    risk_cat = feature["properties"].get("risk_category", "Low Risk")
    color = color_map_category.get(risk_cat, "#808080")
    return {
        "fillColor": color,
        "color": "#ffffff",
        "weight": 1.5,
        "fillOpacity": 0.55,
    }


# Estilo para Categoria Fuzzy Trapezoidal
def get_feature_style_trapezoidal(feature):
    risk_cat = feature["properties"].get("fuzzy_trap_category", "Low Risk")
    color = color_map_category.get(risk_cat, "#808080")
    return {
        "fillColor": color,
        "color": "#000000",
        "weight": 1.5,
        "fillOpacity": 0.65,
    }


def get_highlight_style(feature):
    return {
        "weight": 3.0,
        "color": "#ffff00",
        "fillOpacity": 0.85,
    }


centro_lat = gdf_h3_final.geometry.centroid.y.mean()
centro_lon = gdf_h3_final.geometry.centroid.x.mean()

mapa = folium.Map(
    location=[centro_lat, centro_lon],
    zoom_start=8,
    tiles=None,
    control_scale=True,
)

# Camadas Base
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri, Maxar, Earthstar Geographics, and the GIS User Community",
    name="Satélite (Esri World Imagery)",
    overlay=False,
    control=True,
).add_to(mapa)

folium.TileLayer(
    tiles="https://{s}.basemaps.cartocdn.com/rastertiles/voyager_only_labels/{z}/{x}/{y}{r}.png",
    attr="&copy; OpenStreetMap &copy; CARTO",
    name="Rótulos de Cidades (Labels)",
    overlay=True,
    control=True,
).add_to(mapa)

folium.TileLayer(
    tiles="CartoDB positron", name="Mapa Claro (Positron)", overlay=False, control=True
).add_to(mapa)

Fullscreen().add_to(mapa)
MeasureControl(position="topleft").add_to(mapa)

# Fields/Aliases para os Popups
popup_fields = [
    "h3_cell",
    "fuzzy_risk_index",
    "risk_category",
    "fuzzy_trap_category",
    "fuzzy_trap_degree",
    "mu_low",
    "mu_medium",
    "mu_high",
    "som_ward_cluster",
    "pred_GWh",
    "target_GWh",
]

popup_aliases = [
    "Célula H3:",
    "Índice Fuzzy Risco:",
    "Risco (Quantil):",
    "Risco (Trapezoidal):",
    "Grau Pertinência Máx (μ):",
    "μ (Low):",
    "μ (Medium):",
    "μ (High):",
    "Cluster SOM-Ward:",
    "Perda Prevista (GWh):",
    "Perda Real (GWh):",
]

# Camada 1: Risco Fuzzy por Quantil
layer_quantile = folium.GeoJson(
    gdf_h3_final,
    name="Perfil de Risco Fuzzy (Quantil)",
    style_function=get_feature_style_quantile,
    highlight_function=get_highlight_style,
    tooltip=folium.GeoJsonTooltip(
        fields=["h3_cell", "risk_category", "fuzzy_risk_index"],
        aliases=["H3:", "Risco Quantil:", "Score Fuzzy:"],
        localize=True,
        sticky=True,
    ),
    popup=folium.GeoJsonPopup(
        fields=popup_fields,
        aliases=popup_aliases,
        localize=True,
        labels=True,
        style="font-family: Arial; font-size: 12px; width: 270px;",
    ),
)
layer_quantile.add_to(mapa)

# Camada 2: Risco Fuzzy Trapezoidal
layer_trapezoidal = folium.GeoJson(
    gdf_h3_final,
    name="Perfil de Risco Fuzzy (Trapezoidal Brasil)",
    style_function=get_feature_style_trapezoidal,
    highlight_function=get_highlight_style,
    tooltip=folium.GeoJsonTooltip(
        fields=[
            "h3_cell",
            "fuzzy_trap_category",
            "fuzzy_trap_degree",
            "fuzzy_risk_index",
        ],
        aliases=[
            "H3:",
            "Risco Trapezoidal:",
            "Grau Pertinência (μ):",
            "Score Fuzzy:",
        ],
        localize=True,
        sticky=True,
    ),
    popup=folium.GeoJsonPopup(
        fields=popup_fields,
        aliases=popup_aliases,
        localize=True,
        labels=True,
        style="font-family: Arial; font-size: 12px; width: 270px;",
    ),
)
layer_trapezoidal.add_to(mapa)

# Legenda HTML
html_legenda = """
<div style="
    position: fixed; 
    bottom: 30px; left: 30px; width: 200px; height: 135px; 
    background-color: rgba(255, 255, 255, 0.95); z-index:9999; font-size:12px;
    border:2px solid #333; border-radius:8px; padding: 10px;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.5);
    font-family: Arial, sans-serif;">
    <b style="font-size: 13px;">Classificação Fuzzy</b><br>
    <i style="background: #d73027; width: 15px; height: 15px; float: left; margin-right: 8px; opacity: 0.8; border-radius: 3px;"></i> High Risk<br><br>
    <i style="background: #fee08b; width: 15px; height: 15px; float: left; margin-right: 8px; opacity: 0.8; border-radius: 3px;"></i> Medium Risk<br><br>
    <i style="background: #1a9850; width: 15px; height: 15px; float: left; margin-right: 8px; opacity: 0.8; border-radius: 3px;"></i> Low Risk
</div>
"""
mapa.get_root().html.add_child(folium.Element(html_legenda))

folium.LayerControl(collapsed=False).add_to(mapa)

caminho_saida_html = os.path.join(
    out_dir, "mapa_satelite_h3_fuzzy_alemanha.html"
)
mapa.save(caminho_saida_html)

print(f"\n Mapa interativo HTML atualizado salvo com sucesso!")
print(f"Caminho do arquivo: {caminho_saida_html}")

#%%


