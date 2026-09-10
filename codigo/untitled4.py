# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 17:16:18 2026

@author: Natalia
"""

# -*- coding: utf-8 -*-
"""
Análise do Impacto Socioeconômico e Infraestrutural nas Perdas de Rede (Netzverluste)
Modelagem: XGBoost Global vs. Espacial com Explicabilidade SHAP e GeoSHAP
Target em GWh
"""

import os
import platform
import re
import time
import warnings
import geopandas as gpd
import libpysal
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import requests
import shap
from esda.moran import Moran
from geopandas import sjoin
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
from pysal.lib import weights
from scipy.spatial import cKDTree, distance_matrix
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from shapely import wkt
from shapely.geometry import Point, box
from shapely.ops import unary_union
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ===============================
# CONFIGURAÇÃO DE DIRETÓRIOS E SISTEMA
# ===============================
diretorio_atual = os.getcwd()
nome_sistema_operacional = platform.system()
barra = "/" if nome_sistema_operacional == "Linux" else "\\"
volta_nivel = "../" if nome_sistema_operacional == "Linux" else "..\\"

out_dir = os.path.join(diretorio_atual, volta_nivel, "saidas")
os.makedirs(out_dir, exist_ok=True)

# ===============================
# 1. CARREGAMENTO DOS DADOS
# ===============================
caminho_dados = os.path.join(
    diretorio_atual, volta_nivel, "entradas", "inkar_2024", "inkar_2024.csv"
)
caminho_dados_DSO = os.path.join(
    diretorio_atual,
    volta_nivel,
    "entradas",
    "DSO-Sachsen-Anhalt-file.xlsx",
)
caminho_dados_DSO_L = os.path.join(
    diretorio_atual, volta_nivel, "entradas", "DSO-LowerSaxony.xlsx"
)
caminho_dados_2 = os.path.join(
    diretorio_atual,
    volta_nivel,
    "entradas",
    "Zensus2022_Energietraeger_10km-Gitter.csv",
)

dso_sa = pd.read_excel(caminho_dados_DSO)
dso_ls = pd.read_excel(caminho_dados_DSO_L)
dso_sa["state"] = "Sachsen-Anhalt"
dso_ls["state"] = "Lower Saxony"
dados_combinados = pd.concat([dso_sa, dso_ls], ignore_index=True)

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
    wkt.loads
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

A = gdf_final_4_3.reset_index(drop=True)
B = gdf_final_5_interp.reset_index(drop=True)
joined = gpd.sjoin(A, B, how="left", predicate="within")
result = (
    joined.groupby(level=0)
    .first()
    .reset_index(drop=True)
    .drop(columns=["index_right"], errors="ignore")
)


#%%%


# ===============================
# 2. CONVERSÃO DO TARGET (kWh -> GWh)
# ===============================
target_col = "Summe der Netzverluste [kWh]"
result[target_col] = result[target_col].astype(str)
result[target_col] = result[target_col].str.replace(".", "", regex=False)
result[target_col] = result[target_col].str.replace(",", ".", regex=False)
result[target_col] = pd.to_numeric(result[target_col], errors="coerce")

# CONVERSÃO DE UNIDADE AQUI:
y = result[target_col].values / 1e6  # kWh divididos por 1e6 tornam-se GWh
mask_target = ~np.isnan(y)

# ===============================
# 3. FILTRAGEM E PADRONIZAÇÃO DE FEATURES (X)
# ===============================
df_numeric_2 = result.select_dtypes(include=[np.number]).copy()
cols_to_drop = [
    target_col,
    "latitude",
    "longitude",
    "Anzahl der Entnahmestellen jeweils für alle Netz",
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
    "H_SPNV_ChestP",
    "KH_SPNV_Stroke",
    "lg_nst",
]
df_numeric_2 = df_numeric_2.drop(
    columns=[c for c in cols_to_drop if c in df_numeric_2.columns],
    errors="ignore",
)

scaler = StandardScaler()
X = scaler.fit_transform(df_numeric_2)
standardized_df = pd.DataFrame(X, columns=df_numeric_2.columns)

coords = np.array(list(zip(result.geometry.x, result.geometry.y)))

# Matriz de Vizinhança Espacial
w_knn = libpysal.weights.KNN.from_array(coords, k=4)
w_knn.transform = "r"

# ===============================
# 4. COMPARAÇÃO DE PERFORMANCE XGBOOST (GLOBAL vs ESPACIAL)
# ===============================
WX = w_knn.full()[0] @ X
X_spatial = np.hstack([X, WX])

print("\n" + "=" * 50)
print("COMPARAÇÃO DE PERFORMANCE DOS MODELOS (VALOR EM GWh)")
print("=" * 50)


#%%


# Dicionário para armazenar métricas globais e plotar o gráfico
perf_metrics = {}

def evaluate_model(X_data, y_data, title="Model", model_key="global"):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_list, rmse_list, mae_list = [], [], []

    for train_idx, val_idx in kf.split(X_data):
        X_tr, X_val = X_data[train_idx], X_data[val_idx]
        y_tr, y_val = y_data[train_idx], y_data[val_idx]

        m = XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
        m.fit(X_tr, y_tr)
        preds = m.predict(X_val)

        r2_list.append(r2_score(y_val, preds))
        rmse_list.append(np.sqrt(mean_squared_error(y_val, preds)))
        mae_list.append(mean_absolute_error(y_val, preds))

    mean_r2 = np.mean(r2_list)
    mean_rmse = np.mean(rmse_list)
    mean_mae = np.mean(mae_list)

    perf_metrics[model_key] = {"R2": mean_r2, "RMSE": mean_rmse, "MAE": mean_mae}

    print(f"--- {title} ---")
    print(f"R² Médio: {mean_r2:.4f}")
    print(f"RMSE Médio [GWh]: {mean_rmse:.4f}")
    print(f"MAE Médio [GWh]: {mean_mae:.4f}\n")


evaluate_model(X, y, title="1. XGBoost Global (Apenas Features Socioeconômicas)", model_key="Global")
evaluate_model(
    X_spatial,
    y,
    title="2. XGBoost Espacial (Features + Spatial Lags W*X)",
    model_key="Spatial",
)

# Treinamento do Modelo XGBoost Principal para Interpretação SHAP
model = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
model.fit(X, y)

# ===============================
# 5. CÁLCULO SHAP & GEOSHAP
# ===============================
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

dist_matrix = cdist(coords, coords)
bandwidth = np.percentile(dist_matrix, 10)
gaussian_kernel = np.exp(-(dist_matrix**2) / (2 * bandwidth**2))
W_kernel = gaussian_kernel / gaussian_kernel.sum(axis=1, keepdims=True)

gw_shap = W_kernel @ np.abs(shap_values)
#%%
# ===============================
# 6. GRUPOS INKAR E GEOSHAP
# ===============================
def build_inkar_groups(columns):
    inkar_groups = {
        "arbeitslosigkeit_allgemein": [],
        "arbeitslosigkeit_struktur": [],
        "wirtschaft": [],
        "demografie": [],
        "bildung": [],
        "infrastruktur": [],
        "sonstige": [],
    }

    for col in columns:
        col_lower = col.lower()
        if re.search(r"(sus|arbeitslos|alo|alg|sbg)", col_lower):
            inkar_groups["arbeitslosigkeit_allgemein"].append(col)
        elif re.search(r"(svb|beschäft|erwerb|arbeitnehmer)", col_lower):
            inkar_groups["arbeitslosigkeit_struktur"].append(col)
        elif re.search(r"(bev|einwohner|alter|geburten|sterbe|dichte|rwm)", col_lower):
            inkar_groups["demografie"].append(col)
        elif re.search(r"(bip|wirtschaft|umsatz|unternehmen|betriebe|svw)", col_lower):
            inkar_groups["wirtschaft"].append(col)
        elif re.search(r"(bildung|schule|abitur|akadem|studium)", col_lower):
            inkar_groups["bildung"].append(col)
        elif re.search(r"(verkehr|straße|netz|energie|versorgung|infrastruktur)", col_lower):
            inkar_groups["infrastruktur"].append(col)
        else:
            inkar_groups["sonstige"].append(col)

    return {k: v for k, v in inkar_groups.items() if len(v) > 0}


inkar_groups = build_inkar_groups(standardized_df.columns)

gw_group_importance = {}
for group, cols in inkar_groups.items():
    idx = [standardized_df.columns.get_loc(c) for c in cols if c in standardized_df.columns]
    if len(idx) > 0:
        group_shap = np.abs(shap_values)[:, idx].mean(axis=1)
        gw_group = W_kernel @ group_shap
        gw_group_importance[group] = gw_group.mean()

ser_geoshap = pd.Series(gw_group_importance).sort_values(ascending=False)

print("\n" + "=" * 50)
print("Importância por Grupos INKAR (GeoSHAP em GWh):")
print("=" * 50)
print(ser_geoshap)

# #%%%%
# # ===============================
# # 7. GERADOR DA FIGURA DE RESULTADOS PARA A TESE (INTEGRADO)
# # ===============================
# plt.rcParams.update({
#     "font.family": "serif",
#     "font.serif": ["Times New Roman", "DejaVu Serif"],
#     "font.size": 21,
#     "axes.labelsize": 21,
#     "axes.titlesize": 21,
#     "xtick.labelsize": 21,
#     "ytick.labelsize": 21,
# })

# fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# # Subplot (a): Comparação de Desempenho (R² e MAE)
# models_names = ["XGBoost Global\n(Non-Spatial)", "XGBoost Spatial\n(Features + W·X)"]
# r2_vals = [perf_metrics["Global"]["R2"], perf_metrics["Spatial"]["R2"]]
# mae_vals = [perf_metrics["Global"]["MAE"], perf_metrics["Spatial"]["MAE"]]

# x = np.arange(len(models_names))
# width = 0.35

# ax1 = axes[0]
# color_r2 = "#1f77b4"
# color_mae = "#d62728"

# bars1 = ax1.bar(x - width/2, r2_vals, width, label="R² Score", color=color_r2, alpha=0.85)
# ax1.set_ylabel("R² Score", color=color_r2, fontweight="bold")
# ax1.set_ylim(0, 1.0)
# ax1.tick_params(axis="y", labelcolor=color_r2)
# ax1.set_xticks(x)
# ax1.set_xticklabels(models_names)
# ax1.set_title("(a) Cross-Validation Performance (R² & MAE)", pad=32)
# ax1.grid(True, linestyle="--", alpha=0.3)

# ax1_twin = ax1.twinx()
# bars2 = ax1_twin.bar(x + width/2, mae_vals, width, label="MAE [GWh]", color=color_mae, alpha=0.85)
# ax1_twin.set_ylabel("MAE [GWh]", color=color_mae, fontweight="bold")
# ax1_twin.set_ylim(0, max(mae_vals) * 1.3)
# ax1_twin.tick_params(axis="y", labelcolor=color_mae)

# # Adicionar rótulos de valores acima das barras
# for bar in bars1:
#     yval = bar.get_height()
#     ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.4f}", ha="center", va="bottom", fontsize=21, fontweight="bold", color=color_r2)

# for bar in bars2:
#     yval = bar.get_height()
#     ax1_twin.text(bar.get_x() + bar.get_width()/2.0, yval + 0.4, f"{yval:.2f}", ha="center", va="bottom", fontsize=21, fontweight="bold", color=color_mae)


# # Subplot (b): Importância por Grupos INKAR (GeoSHAP)
# group_labels_map = {
#     "arbeitslosigkeit_allgemein": "General Unemployment",
#     "demografie": "Demographics",
#     "sonstige": "Other Regional Features",
#     "wirtschaft": "Economy & Industry",
#     "arbeitslosigkeit_struktur": "Structural Employment",
#     "bildung": "Education",
# }

# labels_b = [group_labels_map.get(k, k) for k in ser_geoshap.index[::-1]]
# vals_b = ser_geoshap.values[::-1]
# total_geoshap_sum = ser_geoshap.sum()
# pcts_b = [(v / total_geoshap_sum) * 100 for v in vals_b]

# ax2 = axes[1]
# bars_b = ax2.barh(labels_b, vals_b, color="#2b5c8f", alpha=0.85, edgecolor="black", linewidth=0.5)
# ax2.set_xlabel("GeoSHAP Value (Mean Absolute Impact [GWh])")
# ax2.set_title("(b) INKAR Thematic Group Importance (GeoSHAP)", pad=32)
# ax2.grid(True, linestyle="--", alpha=0.3, axis="x")

# for bar, pct in zip(bars_b, pcts_b):
#     w_val = bar.get_width()
#     if w_val > 0:
#         ax2.text(w_val + 0.008, bar.get_y() + bar.get_height()/2.0, f"{w_val:.4f} GWh ({pct:.1f}%)", va="center", ha="left", fontsize=21, fontweight="bold")
#     else:
#         ax2.text(0.005, bar.get_y() + bar.get_height()/2.0, "0.0000 GWh (0.0%)", va="center", ha="left", fontsize=21, color="black")

# ax2.set_xlim(0, max(vals_b) * 1.25)

# plt.tight_layout()
# caminho_figura = os.path.join(out_dir, "dso_regional_results_summary.pdf")
# plt.savefig(caminho_figura, dpi=300, bbox_inches="tight")
# plt.show()

# print(f"\nFigura salva com sucesso em: {caminho_figura}")

#%%
# Alterado para 2 linhas e 1 coluna (figsize ajustado na vertical: 8x10)
fig, axes = plt.subplots(2, 1, figsize=(8, 10))

# -------------------------------
# Figura Superior (a): Comparação de Desempenho
# -------------------------------
models_names = ["XGBoost Global\n(Non-Spatial)", "XGBoost Spatial\n(Features + W·X)"]
r2_vals = [perf_metrics["Global"]["R2"], perf_metrics["Spatial"]["R2"]]
mae_vals = [perf_metrics["Global"]["MAE"], perf_metrics["Spatial"]["MAE"]]

x = np.arange(len(models_names))
width = 0.35

ax1 = axes[0]  # Primeiro subplot (topo)
color_r2 = "#1f77b4"
color_mae = "#d62728"

bars1 = ax1.bar(x - width/2, r2_vals, width, label="R² Score", color=color_r2, alpha=0.85)
ax1.set_ylabel("R² Score", color=color_r2, fontweight="bold")
ax1.set_ylim(0, 1.0)
ax1.tick_params(axis="y", labelcolor=color_r2)
ax1.set_xticks(x)
ax1.set_xticklabels(models_names)
ax1.set_title("(a) Cross-Validation Performance (R² & MAE)", pad=32)
ax1.grid(True, linestyle="--", alpha=0.3)

ax1_twin = ax1.twinx()
bars2 = ax1_twin.bar(x + width/2, mae_vals, width, label="MAE [GWh]", color=color_mae, alpha=0.85)
ax1_twin.set_ylabel("MAE [GWh]", color=color_mae, fontweight="bold")
ax1_twin.set_ylim(0, max(mae_vals) * 1.3)
ax1_twin.tick_params(axis="y", labelcolor=color_mae)

# Rótulos nas barras
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.4f}", ha="center", va="bottom", fontsize=18, fontweight="bold", color=color_r2)

for bar in bars2:
    yval = bar.get_height()
    ax1_twin.text(bar.get_x() + bar.get_width()/2.0, yval + 0.4, f"{yval:.2f}", ha="center", va="bottom", fontsize=18, fontweight="bold", color=color_mae)


# -------------------------------
# Figura Inferior (b): Grupos GeoSHAP
# -------------------------------
group_labels_map = {
    "arbeitslosigkeit_allgemein": "General Unemployment",
    "demografie": "Demographics",
    "sonstige": "Other Regional Features",
    "wirtschaft": "Economy & Industry",
    "arbeitslosigkeit_struktur": "Structural Employment",
    "bildung": "Education",
}

labels_b = [group_labels_map.get(k, k) for k in ser_geoshap.index[::-1]]
vals_b = ser_geoshap.values[::-1]
total_geoshap_sum = ser_geoshap.sum()
pcts_b = [(v / total_geoshap_sum) * 100 for v in vals_b]

ax2 = axes[1]  # Segundo subplot (baixo)
bars_b = ax2.barh(labels_b, vals_b, color="#2b5c8f", alpha=0.85, edgecolor="black", linewidth=0.5)
ax2.set_xlabel("GeoSHAP Value (Mean Absolute Impact [GWh])")
ax2.set_title("(b) INKAR Thematic Group Importance (GeoSHAP)", pad=32)
ax2.grid(True, linestyle="--", alpha=0.3, axis="x")

for bar, pct in zip(bars_b, pcts_b):
    w_val = bar.get_width()
    if w_val > 0:
        ax2.text(w_val + 0.008, bar.get_y() + bar.get_height()/2.0, f"{w_val:.4f} GWh ({pct:.1f}%)", va="center", ha="left", fontsize=18, fontweight="bold")
    else:
        ax2.text(0.005, bar.get_y() + bar.get_height()/2.0, "0.0000 GWh (0.0%)", va="center", ha="left", fontsize=18, color="black")

ax2.set_xlim(0, max(vals_b) * 1.25)

# Espaçamento entre os dois gráficos verticais
plt.tight_layout()

# Salvar figura atualizada
caminho_figura = os.path.join(out_dir, "dso_regional_results_summary_stacked.png")
plt.savefig(caminho_figura, dpi=300, bbox_inches="tight")
plt.show()

#%%

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 17:16:18 2026

@author: Natalia
"""

# -*- coding: utf-8 -*-
"""
Analysis of Socioeconomic and Infrastructural Impact on Grid Losses (Netzverluste)
Modeling: XGBoost Global vs. Spatial with SHAP and GeoSHAP Explainability
Target in GWh + Fuzzy Trapezoidal Logic + H3 Grid Layer (Res=4) + Folium Interactive Map (Google Maps / Street View)
"""

import os
import platform
import re
import time
import warnings
import folium
import geopandas as gpd
import h3
import libpysal
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import requests
import shap
from esda.moran import Moran
from folium import plugins
from geopandas import sjoin
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
from pysal.lib import weights
from scipy.spatial import cKDTree, distance_matrix
from scipy.spatial.distance import cdist
from shapely import wkt
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ===============================
# SYSTEM & DIRECTORY CONFIGURATION
# ===============================
diretorio_atual = os.getcwd()
nome_sistema_operacional = platform.system()
barra = "/" if nome_sistema_operacional == "Linux" else "\\"
volta_nivel = "../" if nome_sistema_operacional == "Linux" else "..\\"

out_dir = os.path.join(diretorio_atual, volta_nivel, "saidas")
os.makedirs(out_dir, exist_ok=True)

# ===============================
# 1. DATA LOADING
# ===============================
caminho_dados = os.path.join(
    diretorio_atual, volta_nivel, "entradas", "inkar_2024", "inkar_2024.csv"
)
caminho_dados_DSO = os.path.join(
    diretorio_atual,
    volta_nivel,
    "entradas",
    "DSO-Sachsen-Anhalt-file.xlsx",
)
caminho_dados_DSO_L = os.path.join(
    diretorio_atual, volta_nivel, "entradas", "DSO-LowerSaxony.xlsx"
)
caminho_dados_2 = os.path.join(
    diretorio_atual,
    volta_nivel,
    "entradas",
    "Zensus2022_Energietraeger_10km-Gitter.csv",
)

dso_sa = pd.read_excel(caminho_dados_DSO)
dso_ls = pd.read_excel(caminho_dados_DSO_L)
dso_sa["state"] = "Sachsen-Anhalt"
dso_ls["state"] = "Lower Saxony"
dados_combinados = pd.concat([dso_sa, dso_ls], ignore_index=True)

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
    wkt.loads
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

A = gdf_final_4_3.reset_index(drop=True)
B = gdf_final_5_interp.reset_index(drop=True)
joined = gpd.sjoin(A, B, how="left", predicate="within")
result = (
    joined.groupby(level=0)
    .first()
    .reset_index(drop=True)
    .drop(columns=["index_right"], errors="ignore")
)

# ===============================
# 2. TARGET CONVERSION (kWh -> GWh)
# ===============================
target_col = "Summe der Netzverluste [kWh]"
result[target_col] = result[target_col].astype(str)
result[target_col] = result[target_col].str.replace(".", "", regex=False)
result[target_col] = result[target_col].str.replace(",", ".", regex=False)
result[target_col] = pd.to_numeric(result[target_col], errors="coerce")

y = result[target_col].values / 1e6  # kWh to GWh
mask_target = ~np.isnan(y)

# ===============================
# 3. FEATURE FILTERING & STANDARDIZATION (X)
# ===============================
df_numeric_2 = result.select_dtypes(include=[np.number]).copy()
cols_to_drop = [
    target_col,
    "latitude",
    "longitude",
    "Anzahl der Entnahmestellen jeweils für alle Netz",
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
    "H_SPNV_ChestP",
    "KH_SPNV_Stroke",
    "lg_nst",
]
df_numeric_2 = df_numeric_2.drop(
    columns=[c for c in cols_to_drop if c in df_numeric_2.columns],
    errors="ignore",
)

scaler = StandardScaler()
X = scaler.fit_transform(df_numeric_2)
standardized_df = pd.DataFrame(X, columns=df_numeric_2.columns)

coords = np.array(list(zip(result.geometry.x, result.geometry.y)))

w_knn = libpysal.weights.KNN.from_array(coords, k=4)
w_knn.transform = "r"

# ===============================
# 4. XGBOOST PERFORMANCE COMPARISON
# ===============================
WX = w_knn.full()[0] @ X
X_spatial = np.hstack([X, WX])

print("\n" + "=" * 50)
print("MODEL PERFORMANCE COMPARISON (VALUES IN GWh)")
print("=" * 50)

perf_metrics = {}

def evaluate_model(X_data, y_data, title="Model", model_key="global"):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_list, rmse_list, mae_list = [], [], []

    for train_idx, val_idx in kf.split(X_data):
        X_tr, X_val = X_data[train_idx], X_data[val_idx]
        y_tr, y_val = y_data[train_idx], y_data[val_idx]

        m = XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
        m.fit(X_tr, y_tr)
        preds = m.predict(X_val)

        r2_list.append(r2_score(y_val, preds))
        rmse_list.append(np.sqrt(mean_squared_error(y_val, preds)))
        mae_list.append(mean_absolute_error(y_val, preds))

    mean_r2 = np.mean(r2_list)
    mean_rmse = np.mean(rmse_list)
    mean_mae = np.mean(mae_list)

    perf_metrics[model_key] = {"R2": mean_r2, "RMSE": mean_rmse, "MAE": mean_mae}

    print(f"--- {title} ---")
    print(f"Mean R²: {mean_r2:.4f}")
    print(f"Mean RMSE [GWh]: {mean_rmse:.4f}")
    print(f"Mean MAE [GWh]: {mean_mae:.4f}\n")


evaluate_model(X, y, title="1. Global XGBoost (Socioeconomic Features Only)", model_key="Global")
evaluate_model(
    X_spatial,
    y,
    title="2. Spatial XGBoost (Features + Spatial Lags W*X)",
    model_key="Spatial",
)

model = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
model.fit(X, y)

# ===============================
# 5. SHAP & GEOSHAP COMPUTATION
# ===============================
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

dist_matrix = cdist(coords, coords)
bandwidth = np.percentile(dist_matrix, 10)
gaussian_kernel = np.exp(-(dist_matrix**2) / (2 * bandwidth**2))
W_kernel = gaussian_kernel / gaussian_kernel.sum(axis=1, keepdims=True)

gw_shap = W_kernel @ np.abs(shap_values)

# ===============================
# 6. INKAR GROUPS & GEOSHAP
# ===============================
def build_inkar_groups(columns):
    inkar_groups = {
        "arbeitslosigkeit_allgemein": [],
        "arbeitslosigkeit_struktur": [],
        "wirtschaft": [],
        "demografie": [],
        "bildung": [],
        "infrastruktur": [],
        "sonstige": [],
    }

    for col in columns:
        col_lower = col.lower()
        if re.search(r"(sus|arbeitslos|alo|alg|sbg)", col_lower):
            inkar_groups["arbeitslosigkeit_allgemein"].append(col)
        elif re.search(r"(svb|beschäft|erwerb|arbeitnehmer)", col_lower):
            inkar_groups["arbeitslosigkeit_struktur"].append(col)
        elif re.search(r"(bev|einwohner|alter|geburten|sterbe|dichte|rwm)", col_lower):
            inkar_groups["demografie"].append(col)
        elif re.search(r"(bip|wirtschaft|umsatz|unternehmen|betriebe|svw)", col_lower):
            inkar_groups["wirtschaft"].append(col)
        elif re.search(r"(bildung|schule|abitur|akadem|studium)", col_lower):
            inkar_groups["bildung"].append(col)
        elif re.search(r"(verkehr|straße|netz|energie|versorgung|infrastruktur)", col_lower):
            inkar_groups["infrastruktur"].append(col)
        else:
            inkar_groups["sonstige"].append(col)

    return {k: v for k, v in inkar_groups.items() if len(v) > 0}


inkar_groups = build_inkar_groups(standardized_df.columns)

gw_group_importance = {}
for group, cols in inkar_groups.items():
    idx = [standardized_df.columns.get_loc(c) for c in cols if c in standardized_df.columns]
    if len(idx) > 0:
        group_shap = np.abs(shap_values)[:, idx].mean(axis=1)
        gw_group = W_kernel @ group_shap
        gw_group_importance[group] = gw_group.mean()

ser_geoshap = pd.Series(gw_group_importance).sort_values(ascending=False)

# ===============================
# 7. TRAPEZOIDAL FUZZY LOGIC CALCULATION
# ===============================
def trapezoidal_mf(x, a, b, c, d):
    """Calculates trapezoidal fuzzy membership degree within [0, 1]."""
    return np.maximum(0, np.minimum(np.minimum((x - a) / (b - a + 1e-9), 1), (d - x) / (d - c + 1e-9)))

# Fuzzy logic membership for network losses (y in GWh)
q25, q50, q75, q95 = np.percentile(y, [25, 50, 75, 95])
min_y, max_y = np.min(y), np.max(y)

fuzzy_low    = trapezoidal_mf(y, min_y, min_y, q25, q50)
fuzzy_medium = trapezoidal_mf(y, q25, q50, q50, q75)
fuzzy_high   = trapezoidal_mf(y, q50, q75, max_y, max_y)

result["Netzverluste_GWh"] = y
result["Fuzzy_Low"] = fuzzy_low
result["Fuzzy_Medium"] = fuzzy_medium
result["Fuzzy_High"] = fuzzy_high
result["Fuzzy_Loss_Index"] = (0.0 * fuzzy_low + 50.0 * fuzzy_medium + 100.0 * fuzzy_high)

# ===============================
# 8. UBER H3 SPATIAL GRID (RESOLUTION = 4)
# ===============================
H3_RESOLUTION = 4

def lat_lng_to_h3(row):
    # Support for updated h3 library syntax (latlng_to_cell / geo_to_h3)
    try:
        return h3.latlng_to_cell(row.geometry.y, row.geometry.x, H3_RESOLUTION)
    except AttributeError:
        return h3.geo_to_h3(row.geometry.y, row.geometry.x, H3_RESOLUTION)

result["h3_cell"] = result.apply(lat_lng_to_h3, axis=1)

# Aggregate Fuzzy Loss Results by H3 Hexagon
h3_summary = result.groupby("h3_cell").agg(
    avg_fuzzy_index=("Fuzzy_Loss_Index", "mean"),
    avg_loss_gwh=("Netzverluste_GWh", "mean"),
    avg_fuzzy_low=("Fuzzy_Low", "mean"),
    avg_fuzzy_med=("Fuzzy_Medium", "mean"),
    avg_fuzzy_high=("Fuzzy_High", "mean"),
    point_count=("Netzverluste_GWh", "count")
).reset_index()

def h3_to_polygon(hex_id):
    try:
        boundary = h3.cell_to_boundary(hex_id)
    except AttributeError:
        boundary = h3.h3_to_geo_boundary(hex_id)
    # Convert lat/lon coordinates into Shapely Polygon (lon, lat order)
    return Polygon([(pt[1], pt[0]) for pt in boundary])

h3_summary["geometry"] = h3_summary["h3_cell"].apply(h3_to_polygon)
gdf_h3 = gpd.GeoDataFrame(h3_summary, geometry="geometry", crs="EPSG:4326")

print(f"\nH3 Grid Layer generated at Resolution {H3_RESOLUTION}: {len(gdf_h3)} cells created.")

# ===============================
# 9. INTERACTIVE FOLIUM MAP (ENGLISH + H3 RES 4 + GOOGLE STREET VIEW)
# ===============================
lat_center = result.geometry.y.mean()
lon_center = result.geometry.x.mean()

m = folium.Map(location=[lat_center, lon_center], zoom_start=8, tiles=None)

# Google Maps Layers
folium.TileLayer(
    tiles="https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
    attr="Google Maps Street/Roadmap",
    name="Google Maps (Roadmap / Street View)",
    overlay=False,
    control=True,
).add_to(m)

folium.TileLayer(
    tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
    attr="Google Satellite",
    name="Google Satellite",
    overlay=False,
    control=True,
).add_to(m)

folium.TileLayer(
    tiles="https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}",
    attr="Google Terrain",
    name="Google Terrain",
    overlay=False,
    control=True,
).add_to(m)

folium.TileLayer(
    tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
    attr="Google Hybrid",
    name="Google Hybrid",
    overlay=False,
    control=True,
).add_to(m)

folium.TileLayer("cartodbpositron", name="CartoDB Positron").add_to(m)

# --- 1. LAYER: H3 Hexagonal Grid (Res = 4) with Fuzzy Index ---
import matplotlib.cm as cm
import matplotlib.colors as colors

norm = colors.Normalize(vmin=0, vmax=100)
cmap = cm.get_cmap("YlOrRd") # Yellow to Orange to Red

def style_h3_function(feature):
    val = feature["properties"]["avg_fuzzy_index"]
    color_hex = colors.to_hex(cmap(norm(val)))
    return {
        "fillColor": color_hex,
        "color": "#333333",
        "weight": 1.2,
        "fillOpacity": 0.55,
    }

fg_h3 = folium.FeatureGroup(name="H3 Grid Layer (Res=4 - Fuzzy Score)", show=True)

# Define GeoJson layer for H3 cells
h3_geojson = folium.GeoJson(
    gdf_h3,
    style_function=style_h3_function,
    tooltip=folium.GeoJsonTooltip(
        fields=["h3_cell", "avg_fuzzy_index", "avg_loss_gwh", "point_count"],
        aliases=["H3 Cell ID:", "Avg Fuzzy Index:", "Avg Grid Loss (GWh):", "DSO Samples Count:"],
        localize=True,
    ),
).add_to(fg_h3)

# Add custom popup for H3 Cells in English
for idx, row in gdf_h3.iterrows():
    centroid = row.geometry.centroid
    lat, lon = centroid.y, centroid.x
    street_view_url = f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat},{lon}"
    
    popup_h3 = f"""
    <div style="font-family: Arial; font-size: 12px; width: 240px;">
        <h4 style="margin: 0 0 5px 0;">H3 Cell: <code>{row['h3_cell']}</code></h4>
        <b>Spatial Resolution:</b> Level 4<br>
        <b>Sample Points inside:</b> {row['point_count']}<br>
        <b>Avg Grid Loss:</b> {row['avg_loss_gwh']:.2f} GWh<br>
        <hr style="margin: 5px 0;">
        <b><u>Aggregated Fuzzy Index: {row['avg_fuzzy_index']:.1f} / 100</u></b><br>
        • Avg Low Loss Membership: {row['avg_fuzzy_low']:.2f}<br>
        • Avg Medium Loss Membership: {row['avg_fuzzy_med']:.2f}<br>
        • Avg High Loss Membership: {row['avg_fuzzy_high']:.2f}<br>
        <hr style="margin: 5px 0;">
        <a href="{street_view_url}" target="_blank" style="color: blue; font-weight: bold;">
            📍 Open Centroid in Google Street View
        </a>
    </div>
    """
    
    # Place a transparent clickable popup marker on H3 cell centroids
    folium.CircleMarker(
        location=[lat, lon],
        radius=1,
        color="transparent",
        fill=False,
        popup=folium.Popup(popup_h3, max_width=300),
    ).add_to(fg_h3)

fg_h3.add_to(m)

# --- 2. LAYER: DSO Measurement Points ---
fg_points = folium.FeatureGroup(name="DSO Stations (Points)", show=True)

for idx, row in result.iterrows():
    lat = row.geometry.y
    lon = row.geometry.x
    loss = row["Netzverluste_GWh"]
    fuzzy_idx_val = row["Fuzzy_Loss_Index"]
    
    if fuzzy_idx_val > 60:
        color = "red"
    elif fuzzy_idx_val > 30:
        color = "orange"
    else:
        color = "green"

    street_view_url = f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat},{lon}"

    popup_text = f"""
    <div style="font-family: Arial; font-size: 12px; width: 230px;">
        <b>Region / State:</b> {row.get('state', 'N/A')}<br>
        <b>H3 Resolution 4 ID:</b> <code>{row['h3_cell']}</code><br>
        <b>Total Grid Loss:</b> {loss:.2f} GWh<br>
        <hr style="margin: 5px 0;">
        <b><u>Fuzzy Membership Degrees:</u></b><br>
        • Low Loss: {row['Fuzzy_Low']:.2f}<br>
        • Medium Loss: {row['Fuzzy_Medium']:.2f}<br>
        • High Loss: {row['Fuzzy_High']:.2f}<br>
        <b>Fuzzy Loss Index: {fuzzy_idx_val:.1f} / 100</b><br>
        <hr style="margin: 5px 0;">
        <a href="{street_view_url}" target="_blank" style="color: blue; font-weight: bold;">
            📍 Open Location in Google Street View
        </a>
    </div>
    """

    folium.CircleMarker(
        location=[lat, lon],
        radius=5,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.85,
        popup=folium.Popup(popup_text, max_width=300),
    ).add_to(fg_points)

fg_points.add_to(m)

# Layer Controls
folium.LayerControl(collapsed=False).add_to(m)

# Save Map
caminho_mapa_html = os.path.join(out_dir, "map_grid_losses_fuzzy_h3_res4_streetview.html")
m.save(caminho_mapa_html)
print(f"Interactive HTML map with H3 (Res 4) saved to: {caminho_mapa_html}")

# ===============================
# 10. STACKED FIGURE FOR THESIS (ENGLISH LABELS)
# ===============================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

fig, axes = plt.subplots(2, 1, figsize=(8, 10))

# Subplot (a): Cross-Validation Performance
models_names = ["XGBoost Global\n(Non-Spatial)", "\nXGBoost Spatial\n(Features + W·X)"]
r2_vals = [perf_metrics["Global"]["R2"], perf_metrics["Spatial"]["R2"]]
mae_vals = [perf_metrics["Global"]["MAE"], perf_metrics["Spatial"]["MAE"]]

x = np.arange(len(models_names))
width = 0.35

ax1 = axes[0]
color_r2 = "#1f77b4"
color_mae = "#d62728"

bars1 = ax1.bar(x - width/2, r2_vals, width, label="R² Score", color=color_r2, alpha=0.85)
ax1.set_ylabel("R² Score", color=color_r2, fontweight="bold")
ax1.set_ylim(0, 1.0)
ax1.tick_params(axis="y", labelcolor=color_r2)
ax1.set_xticks(x)
ax1.set_xticklabels(models_names)
ax1.set_title("(a) Cross-Validation Performance (R² & MAE)", pad=15)
ax1.grid(True, linestyle="--", alpha=0.3)

ax1_twin = ax1.twinx()
bars2 = ax1_twin.bar(x + width/2, mae_vals, width, label="MAE [GWh]", color=color_mae, alpha=0.85)
ax1_twin.set_ylabel("MAE [GWh]", color=color_mae, fontweight="bold")
ax1_twin.set_ylim(0, max(mae_vals) * 1.3)
ax1_twin.tick_params(axis="y", labelcolor=color_mae)

for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold", color=color_r2)

for bar in bars2:
    yval = bar.get_height()
    ax1_twin.text(bar.get_x() + bar.get_width()/2.0, yval + 0.4, f"{yval:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold", color=color_mae)

# Subplot (b): GeoSHAP INKAR Importance
group_labels_map = {
    "arbeitslosigkeit_allgemein": "General Unemployment",
    "demografie": "Demographics",
    "sonstige": "Other Regional Features",
    "wirtschaft": "Economy & Industry",
    "arbeitslosigkeit_struktur": "Structural Employment",
    "bildung": "Education",
}

labels_b = [group_labels_map.get(k, k) for k in ser_geoshap.index[::-1]]
vals_b = ser_geoshap.values[::-1]
total_geoshap_sum = ser_geoshap.sum()
pcts_b = [(v / total_geoshap_sum) * 100 for v in vals_b]

ax2 = axes[1]
bars_b = ax2.barh(labels_b, vals_b, color="#2b5c8f", alpha=0.85, edgecolor="black", linewidth=0.5)
ax2.set_xlabel("GeoSHAP Value (Mean Absolute Impact [GWh])")
ax2.set_title("(b) INKAR Thematic Group Importance (GeoSHAP)", pad=15)
ax2.grid(True, linestyle="--", alpha=0.3, axis="x")

for bar, pct in zip(bars_b, pcts_b):
    w_val = bar.get_width()
    if w_val > 0:
        ax2.text(w_val + 0.008, bar.get_y() + bar.get_height()/2.0, f"{w_val:.4f} GWh ({pct:.1f}%)", va="center", ha="left", fontsize=10, fontweight="bold")
    else:
        ax2.text(0.005, bar.get_y() + bar.get_height()/2.0, "0.0000 GWh (0.0%)", va="center", ha="left", fontsize=10, color="black")

ax2.set_xlim(0, max(vals_b) * 1.25)

plt.tight_layout()

caminho_figura = os.path.join(out_dir, "dso_regional_results_summary_stacked.png")
plt.savefig(caminho_figura, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nStacked summary figure saved to: {caminho_figura}")