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
# Construção do Lag Espacial das Features (W * X) para o Modelo Espacial
WX = w_knn.full()[0] @ X
X_spatial = np.hstack([X, WX])

print("\n" + "=" * 50)
print("COMPARAÇÃO DE PERFORMANCE DOS MODELOS (VALOR EM GWh)")
print("=" * 50)


def evaluate_model(X_data, y_data, title="Model"):
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

    print(f"--- {title} ---")
    print(f"R² Médio: {np.mean(r2_list):.4f}")
    print(f"RMSE Médio [GWh]: {np.mean(rmse_list):.4f}")
    print(f"MAE Médio [GWh]: {np.mean(mae_list):.4f}\n")


evaluate_model(X, y, title="1. XGBoost Global (Apenas Features Socioeconômicas)")
evaluate_model(
    X_spatial,
    y,
    title="2. XGBoost Espacial (Features + Spatial Lags W*X)",
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
# 5. [LOCAL SHAP GLOBAL] CÁLCULO DO SHAP CONVENCIONAL
# ===============================
# AQUI ESTÁ O SHAP GLOBAL: Calculado diretamente sobre a matriz de características X
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)  # Valores em GWh

# ===============================
# 6. [LOCAL GEOSHAP] CÁLCULO DO GEOSHAP (PONDERADO ESPACIALMENTE)
# ===============================
# AQUI ESTÁ O GEOSHAP: Aplicação da Matriz Kernel Gaussiana sobre os SHAP Values
dist_matrix = cdist(coords, coords)
bandwidth = np.percentile(dist_matrix, 10)  # Banda de suavização
gaussian_kernel = np.exp(-(dist_matrix**2) / (2 * bandwidth**2))
W_kernel = gaussian_kernel / gaussian_kernel.sum(axis=1, keepdims=True)

# Matriz GeoSHAP (Multiplicação Matricial)
gw_shap = W_kernel @ np.abs(shap_values)

# ===============================
# 7. VISUALIZAÇÃO COMPARATIVA PARA A TESE (FIGURA BEESWARM)
# ===============================
translation_dict = {
    "q_svw_bev": "Social Sec. Employees (%)",
    "a_alo_f": "Female Unemployment Rate",
    "a_gb_Frauen": "Female Marginal Employment",
    "a_bev3050": "Population Aged 30-50 (%)",
    "a_alo_spezialist": "Specialist Unemployment Rate",
    "a_Stellen_Spezialist": "Specialist Job Openings",
    "q_alo_ü55_einw_m": "Male Unemployment >55 y/o",
    "a_Stellen_Experte": "Expert Job Openings",
    "d_Haushaltseinkommen": "Household Income",
    "a_BG5um": "Large Enterprise Density",
    "Abfahrten_Bahn": "Railway Departures",
    "d_Bruttoverdienst_Prod": "Gross Earnings (Manufacturing)",
    "a_alo_ü55_m": "Male Unemployment >55 Rate",
    "a_alo_ü55": "Unemployment >55 Rate",
    "BibEntl": "Library Distance / Infra",
}

translated_feature_names = [
    translation_dict.get(col, col) for col in standardized_df.columns
]

exp_global = shap.Explanation(
    values=shap_values, data=X, feature_names=translated_feature_names
)
exp_gw = shap.Explanation(
    values=gw_shap, data=X, feature_names=translated_feature_names
)

plt.rcParams.update(
    {"font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 11}
)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Plot Subplot (a): SHAP Global
plt.sca(axes[0])
shap.plots.beeswarm(exp_global, max_display=10, show=False, plot_size=None)
axes[0].set_title("(a) SHAP Beeswarm (Global Model)", fontsize=13, pad=12)
axes[0].set_xlabel("SHAP value (Impact on Network Losses [GWh])")

# Plot Subplot (b): GeoSHAP
plt.sca(axes[1])
shap.plots.beeswarm(exp_gw, max_display=10, show=False, plot_size=None)
axes[1].set_title(
    "(b) GeoSHAP Beeswarm (Spatially Weighted)", fontsize=13, pad=12
)
axes[1].set_xlabel("GeoSHAP value (Local Spatial Impact [GWh])")

plt.tight_layout()
plt.savefig(
    os.path.join(out_dir, "figura_thesis_shap_vs_geoshap_gwh.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.show()


# ===============================
# 8. DEFINIÇÃO DA FUNÇÃO E ANÁLISE POR GRUPOS INKAR
# ===============================


def build_inkar_groups(columns):
    """Cria grupos temáticos INKAR com base em padrões nos nomes das variáveis."""
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
        elif re.search(
            r"(bev|einwohner|alter|geburten|sterbe|dichte|rwm)", col_lower
        ):
            inkar_groups["demografie"].append(col)
        elif re.search(
            r"(bip|wirtschaft|umsatz|unternehmen|betriebe|svw)", col_lower
        ):
            inkar_groups["wirtschaft"].append(col)
        elif re.search(r"(bildung|schule|abitur|akadem|studium)", col_lower):
            inkar_groups["bildung"].append(col)
        elif re.search(
            r"(verkehr|straße|netz|energie|versorgung|infrastruktur)", col_lower
        ):
            inkar_groups["infrastruktur"].append(col)
        else:
            inkar_groups["sonstige"].append(col)

    # Remover grupos sem variáveis associadas
    return {k: v for k, v in inkar_groups.items() if len(v) > 0}


# Chamada da função (agora devidamente definida acima)
inkar_groups = build_inkar_groups(standardized_df.columns)

gdf_GW_SHAP = gpd.GeoDataFrame(
    gw_shap, geometry=gpd.points_from_xy(coords[:, 0], coords[:, 1])
)

gw_group_importance = {}
for group, cols in inkar_groups.items():
    idx = [
        standardized_df.columns.get_loc(c)
        for c in cols
        if c in standardized_df.columns
    ]
    if len(idx) > 0:
        group_shap = np.abs(shap_values)[:, idx].mean(axis=1)
        gw_group = W_kernel @ group_shap
        gdf_GW_SHAP[group + "_gw"] = gw_group
        gw_group_importance[group] = gw_group.mean()

print("\nImportância por Grupos INKAR (GeoSHAP em GWh):")
print(pd.Series(gw_group_importance).sort_values(ascending=False))

# ===============================
# 8. ANÁLISE POR GRUPOS INKAR E MAPPING
# ===============================
inkar_groups = build_inkar_groups(standardized_df.columns)
gdf_GW_SHAP = gpd.GeoDataFrame(
    gw_shap, geometry=gpd.points_from_xy(coords[:, 0], coords[:, 1])
)

gw_group_importance = {}
for group, cols in inkar_groups.items():
    idx = [standardized_df.columns.get_loc(c) for c in cols if c in standardized_df.columns]
    if len(idx) > 0:
        group_shap = np.abs(shap_values)[:, idx].mean(axis=1)
        gw_group = W_kernel @ group_shap
        gdf_GW_SHAP[group + "_gw"] = gw_group
        gw_group_importance[group] = gw_group.mean()

print("\nImportância por Grupos INKAR (GeoSHAP):")
print(pd.Series(gw_group_importance).sort_values(ascending=False))