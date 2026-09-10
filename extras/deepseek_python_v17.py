import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import distance
import pandas as pd
import os
from sklearn.mixture import GaussianMixture
import platform
from shapely.geometry import Polygon
from matplotlib.patches import Polygon
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

caminho_data_toFuzzy = diretorio_atual+barra + \
    volta_nivel+barra+'saidas'+barra+'data_toFuzzy.csv'
    
#%%

dados = pd.read_csv(caminho_data_toFuzzy,sep=';')
   
#%%
dados2 = dados[['umap_1_scaled','umap_2_scaled']].values

#%%

corrs = dados[["Perdas_norm", "Parecer_norm", "umap_1_scaled", "umap_2_scaled"]].corr()
# print(corrs)

#%%

df = dados[["Perdas_norm", "Parecer_norm", "umap_1_scaled", "umap_2_scaled"]]


#%%%
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.decomposition import PCA
#%%
class RiskFuzzyClassifierWithTrapezoid:
    def __init__(self, risk_labels=['Low Risk', 'Medium Risk', 'High Risk'], largura_topo=0.3):
        """
        largura_topo: fração do intervalo (0 = triângulo, 1 = retângulo).
        Ex: 0.2 significa que o topo terá 20% da largura da base.
        """
        self.risk_labels = risk_labels
        self.n_classes = len(risk_labels)
        self.conjuntos = []
        self.largura_topo = largura_topo
        self.resultados = pd.DataFrame()  # DataFrame com classificações
    
        
    def preparar_dados(self, df):
        df_norm = df.copy()
        for col in ["umap_1_scaled", "umap_2_scaled"]:
            corr = dados[[col, "Perdas"]].corr().iloc[0,1] + dados[[col, "w_PARECER2_TOTAL"]].corr().iloc[0,1]
            if corr < 0:  # relação negativa → inverter
                df_norm[col] = 1 - df_norm[col]
        
        
        df_norm["Risk_score"] = df_norm[["Perdas_norm", "Parecer_norm", "umap_1_scaled", "umap_2_scaled"]].mean(axis=1)
        
        X = df_norm[["Perdas_norm", "Parecer_norm", "umap_1_scaled", "umap_2_scaled"]].values
        #y = pd.qcut(df_norm["Risk_score"], q=3, labels=["Low", "Medium", "High"])
        
        pca = PCA(n_components=1)
        risk_score_1d = pca.fit_transform(X).flatten()
        df_norm["Risk_score_new"] = (risk_score_1d - risk_score_1d.min()) / (risk_score_1d.max() - risk_score_1d.min())


        # # Ajustar LDA para reduzir a 1 dimensão (1D Risk_score)
        # lda = LDA(n_components=1)
        # risk_score = lda.fit_transform(X, y).flatten()  # vetor 1D       
        # df_norm["Risk_score_new"] = (risk_score - risk_score.min()) / (risk_score.max() - risk_score.min())  # normalizar 0-1
        
        # self.lda_model = lda
        self.df_norm = df_norm
        
        # Definir fronteiras fuzzy dinamicamente pelos quantis
        q20, q40, q60, q80 = df_norm["Risk_score_new"].quantile([0.25, 0.40, 0.60,0.80])
      
        base_width = q80 - q20
        top_width = self.largura_topo * base_width
        b = q20 + (base_width - top_width)/2
        c = q80 - (base_width - top_width)/2
        self.definicoes = {
            "Low Risk":    (0.0, 0.0, q20, q40),
            "Medium Risk": (q20, q40, q60, q80),
            "High Risk":   (q60, q80, 1.0, 1.0)
        }
        
        print("\n FUZZY TRAPEZOID PARAMETERS")
    
        for label, params in self.definicoes.items():
            a, b, c, d = params
            print(f"{label}: (a={a:.4f}, b={b:.4f}, c={c:.4f}, d={d:.4f})")

        
        return df_norm

    @staticmethod
    def trapezio_1d(x, a, b, c, d, theta=1.0):
        """
        Constrained trapezoidal membership function
        a <= b <= c <= d, theta > 0
        """
    
        # enforce constraints
        eps = 1e-9
        a = float(a)
        b = max(a + eps, b)
        c = max(b + eps, c)
        d = max(c + eps, d)
    
        mu = np.zeros_like(x, dtype=float)
    
        # rising edge
        idx = (x >= a) & (x <= b)
        mu[idx] = ((x[idx] - a) / (b - a)) ** theta
    
        # plateau
        idx = (x >= b) & (x <= c)
        mu[idx] = 1.0
    
        # falling edge
        idx = (x >= c) & (x <= d)
        mu[idx] = ((d - x[idx]) / (d - c)) ** theta
    
        return np.clip(mu, 0, 1)


    @staticmethod
    def _intersecao_trapezios(params1, params2, n=2000):
        """
        Retorna x onde dois trapézios se interceptam (µ1=µ2), ignorando regiões µ=0.
        """
    
        a1, b1, c1, d1 = params1
        a2, b2, c2, d2 = params2
    
        x = np.linspace(0, 1, n)
    
        def trap(x, a, b, c, d):
            return np.maximum(
                np.minimum(
                    np.minimum((x - a) / (b - a + 1e-9), 1),
                    (d - x) / (d - c + 1e-9)
                ),
                0
            )
    
        y1 = trap(x, a1, b1, c1, d1)
        y2 = trap(x, a2, b2, c2, d2)
    
        # considerar apenas pontos onde ambos são > 0
        mask = (y1 > 0) & (y2 > 0)
                  
        if not np.any(mask):
            return (None, None)
    
        x_valid = x[mask]
        y1_valid = y1[mask]
        y2_valid = y2[mask]
    
        diff = np.abs(y1_valid - y2_valid)
        idx = np.argmin(diff)
    
        return float(x_valid[idx]), float(y1_valid[idx])
    
    def pontos_intersecao(self):
        """
        Retorna os valores x onde os conjuntos fuzzy se interceptam.
        """
        low = self.definicoes["Low Risk"]
        med = self.definicoes["Medium Risk"]
        high = self.definicoes["High Risk"]
    
        inter_low_med = self._intersecao_trapezios(low, med)
        inter_med_high = self._intersecao_trapezios(med, high)
    
        return {
            "Low ∩ Medium": inter_low_med,
            "Medium ∩ High": inter_med_high
        }

    
    def classificar_com_score(self):
        """
        Classifica cada ponto com base no Risk_score 1D
        e normaliza as pertinências para que:
        µ_low + µ_medium + µ_high = 1
        """
        if self.df_norm is None or self.definicoes is None:
            raise ValueError("! Execute preparar_dados(df) antes de classificar.")
    
        resultados = []
    
        for _, row in self.df_norm.iterrows():
            x = row["Risk_score_new"]
    
            # Raw memberships
            mu_raw = {
                label: self.trapezio_1d(x, *params)
                for label, params in self.definicoes.items()
            }
    
            soma = sum(mu_raw.values())
    
            # 🔹 Normalize (partition of unity)
            if soma > 0:
                mu = {k: v / soma for k, v in mu_raw.items()}
            else:
                mu = mu_raw  # edge case (should not happen)
    
            classificacao = max(mu.items(), key=lambda kv: kv[1])[0]
    
            linha = {
                "Risk_score_new": x,
                "classificacao": classificacao,
                **mu,
                "sum_mu": sum(mu.values())
            }
    
            resultados.append(linha)
    
        self.resultados = pd.DataFrame(resultados)
        return self.resultados

    def plot_partition_of_unity(self):
        """
        Valida se a soma das funções de pertinência é 1
        para todo x ∈ [0,1].
        """
        x = np.linspace(0, 1, 500)
    
        mu_sum = np.zeros_like(x)
    
        for params in self.definicoes.values():
            mu_sum += self.trapezio_1d(x, *params)
    
        # Normalize globally (same rule used in classification)
        mu_sum = mu_sum / np.maximum(mu_sum, 1e-9)
    
        plt.figure(figsize=(7, 4))
        plt.plot(x, mu_sum, color="black", linewidth=2)
        plt.axhline(1.0, linestyle="--", color="red", label="µ sum = 1")
    
        plt.ylim(0.95, 1.05)
        plt.xlabel("Risk score")
        plt.ylabel("Σ µ(x)")
        plt.title("Partition of Unity Validation")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.show()

    def plotar_trapezio_1d(self, intervalo=None):
        """
        Plota as funções fuzzy trapezoidais 1D calculadas pelos quantis
        e marca os pontos de interseção (Low∩Medium, Medium∩High) com linhas,
        pontos e anotações.
        """
        csfont = {'fontname':'Times New Roman'}
        plt.rc('font',family='Times New Roman')
        if self.definicoes is None:
            raise ValueError("! Execute preparar_dados(df) antes de plotar.")
    
        x = np.linspace(0, 1, 1000)
        plt.figure(figsize=(9, 6))
    
        # cores (ajuste se preferir)
        cores = {
            'Low Risk': 'green',
            'Medium Risk': 'orange',
            'High Risk': 'red'
        }
    
        # plota trapézios
        for label, (a, b, c, d) in self.definicoes.items():
            y = self.trapezio_1d(x, a, b, c, d)
            plt.plot(x, y, label=label, color=cores.get(label, 'gray'), linewidth=2)
            plt.fill_between(x, y, alpha=0.18, color=cores.get(label, 'gray'))
    
            # opcional: marcar topo (b,c) e base (a,d)
            plt.plot([b, c], [1, 1], marker='o', linestyle='None', color=cores.get(label, 'gray'), alpha=0.6, markersize=4)
    
        # # calcular interseções entre conjuntos adjacentes (assumindo Low/Medium/High)
        # pairs = [("Low Risk", "Medium Risk"), ("Medium Risk", "High Risk")]
        # intersec_points = {}
    
        # for p1, p2 in pairs:
        #     params1 = self.definicoes.get(p1)
        #     params2 = self.definicoes.get(p2)
        #     if params1 is None or params2 is None:
        #         intersec_points[f"{p1} ∩ {p2}"] = (None, None)
        #         continue
        #     xi, mu = self._intersecao_trapezios(params1, params2, n=2000)
        #     intersec_points[f"{p1} ∩ {p2}"] = (xi, mu)
    
        # # marcar e anotar as interseções válidas
        # for nome, (xi, mu) in intersec_points.items():
        #     if xi is None:
        #         # opcional: mostrar aviso no gráfico
        #         plt.text(0.02, 0.02, f"{nome}: no overlap", transform=plt.gca().transAxes, fontsize=9, color='gray')
        #         continue
    
        #     # linha vertical tracejada
        #     plt.axvline(x=xi, color='black', linestyle='--', alpha=0.7)
    
        #     # marcador no ponto (xi, mu)
        #     plt.scatter([xi], [mu], color='black', s=60, zorder=6)
    
        #     # anotação legível acima do ponto
        #     plt.annotate(
        #         f"{nome}\n x={xi:.3f}\nµ≈{mu:.3f}",
        #         xy=(xi, mu),
        #         xytext=(0, 8),
        #         textcoords="offset points",
        #         ha="center",
        #         va="bottom",
        #         fontsize=9,
        #         bbox=dict(facecolor="white", alpha=0.8, edgecolor="none")
        #     )
    
        # estética
        plt.ylim(-0.05, 1.10)
        plt.xlim(0, 1)
        plt.xlabel("Risk score",**csfont,fontsize=15)
        plt.ylabel("µ(x)",**csfont,fontsize=15)
        plt.title("Fuzzy Risk Functions",**csfont,fontsize=15)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()    


    def plotar_interseccoes(self):
        """
        Plota apenas os triângulos correspondentes às interseções 
        dos conjuntos fuzzy que têm µ > 0.
        """
        if self.definicoes is None:
            raise ValueError("! Execute preparar_dados(df) antes de plotar.")
    
        x = np.linspace(0, 1, 2000)
    
        pairs = [
            ("Low Risk", "Medium Risk"),
            ("Medium Risk", "High Risk")
        ]
    
        plt.figure(figsize=(10, 6))
    
        encontrou = False
    
        for p1, p2 in pairs:
            a1, b1, c1, d1 = self.definicoes[p1]
            a2, b2, c2, d2 = self.definicoes[p2]
    
            y1 = self.trapezio_1d(x, a1, b1, c1, d1)
            y2 = self.trapezio_1d(x, a2, b2, c2, d2)
    
            # interseção = mínimo vertical entre dois fuzzy sets
            y_inter = np.minimum(y1, y2)
    
            # somente se houver interseção real
            if np.max(y_inter) > 0:
                encontrou = True
    
                # índice do pico da interseção
                idx = np.argmax(y_inter)
                xi = x[idx]
                mui = y_inter[idx]
                
                # bases
                idxs = np.where(y_inter > 0)[0]
                base_left = x[idxs[0]]
                base_right = x[idxs[-1]]
  
               # Triângulo resultante (a, b, c)
                a = base_left
                b = xi
                c = base_right  
  
                # plota a interseção triangular
                plt.fill_between(x, y_inter, alpha=0.6, label=f"{p1} ∩ {p2}")
    
                # ponto da interseção
                plt.scatter([a], [0], color="blue", s=80, zorder=5)
                plt.scatter([xi], [mui], color="black", s=60, zorder=5)
                plt.scatter([c], [0], color="blue", s=80, zorder=5)
                
                plt.plot([a, b, c], [0, mui, 0], '--', color="black")
    
                # print(f"   Interseção {p1} ∩ {p2}")
                # print(f"   x = {xi:.4f}, µ = {mui:.4f}")
    
        if not encontrou:
            print("! Nenhuma interseção real encontrada.")
            return
    
        plt.title(" Fuzzy Intersections (Triangular Subsets)")
        plt.xlabel("x")
        plt.ylabel("µ(x)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(-0.05, 1.05)
        plt.show()
       
#%%
# Criar e treinar classificador
classificador = RiskFuzzyClassifierWithTrapezoid()

#%%
df_norm = classificador.preparar_dados(df)
#%%
resultados = classificador.classificar_com_score()

#%%
inter = classificador.pontos_intersecao()
print(inter)

#%%
classificador.plotar_trapezio_1d()

#%%
classificador.plotar_interseccoes()

#%%
resultados['h3_cell'] = dados['h3_cell'].values

#%%
# Validate constraint
classificador.plot_partition_of_unity()

# Check numerically
print(classificador.resultados["sum_mu"].describe())


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

resultados2 = resultados.copy()

#%%
import h3
import geopandas as gpd

#%%

from shapely.geometry import Polygon
import h3

def cell_to_shapely(cell):
    # Get the boundary of the H3 cell (list of [lat, lng])
    coords = h3.cell_to_boundary(cell)
    # Flip to [lng, lat] for Shapely
    flipped = [[lng, lat] for lat, lng in coords]
    # Ensure the polygon is closed (first point == last point)
    if flipped[0] != flipped[-1]:
        flipped.append(flipped[0])
    return Polygon(flipped)

#%%# Convert H3 cells to Shapely polygons

h3_geoms = resultados2['h3_cell'].apply(cell_to_shapely)

# Create a GeoDataFrame
h3_gdf = gpd.GeoDataFrame(data=resultados2, geometry=h3_geoms, crs="EPSG:4326")

#%%

fig, ax = plt.subplots(figsize=(8, 6))
h3_gdf.plot(
    column="Risk_score_new",   # numeric column (0–1)
    cmap="Paired",        # or "viridis", "plasma"
    legend=True,
    edgecolor="black",
    linewidth=0.5,
    ax=ax
)

plt.title("Shapefile Geometries with Fuzzy Membership Coloring")
plt.show()


#%%
import folium

from folium.plugins import MarkerCluster

#%%
geojson_obj = h3_gdf.to_json()

#%%

caminho_12 = diretorio_atual+barra + \
    volta_nivel+barra+'entradas'+barra+'gdf_dados_todos_12-2.xlsx'
  
#%%

gdf_dados_todos_12 = pd.read_excel(caminho_12)

#%%
map = folium.Map(
    location=[-27.5969, -48.5495],
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri",
    zoom_start=17
)

# --- Layer 1: Choropleth (must be added directly to map) ---
choropleth = folium.Choropleth(
    geo_data=geojson_obj,
    name="Choropleth Layer",
    data=h3_gdf,
    columns=["h3_cell", "Risk_score_new"],
    key_on='feature.properties.h3_cell',
    fill_color="YlOrRd",
    legend_name="Energy Theft Cases Risk",
    highlight=True
).add_to(map)

# --- Layer 2: Highlight layer ---
highlight_layer = folium.FeatureGroup(name="Highlights")
highlights = folium.features.GeoJson(
    geojson_obj,
    style_function=lambda x: {'color':'transparent', 'fillColor':'transparent', 'weight':0},
    highlight_function=lambda x: {'fillColor': '#000000', 'color':'#000000', 'fillOpacity': 0.50, 'weight': 0.1},
    tooltip=folium.features.GeoJsonTooltip(
        fields=['h3_cell', 'Risk_score_new'],
        aliases=['h3_cell id: ', 'Level Risk: '],
        labels=True,
        sticky=False
    )
)
highlight_layer.add_child(highlights)
map.add_child(highlight_layer)


# --- Layer 3: Marker layer ---
marker_layer = folium.FeatureGroup(name="Markers (Google Street View)")
for i, row in gdf_dados_todos_12.iterrows():
    lat = row['LATITUDE']
    lng = row['LONGITUDE']
    popup = (
        '<br>'
        f'<a href="https://www.google.com/maps?layer=c&cbll={lat},{lng}" target="blank">'
        'GOOGLE STREET VIEW</a>'
    )
    cf_marker = folium.Marker(
        location=[lat, lng],
        popup=popup,
        icon=folium.Icon(color="blue", icon="remove-sign")
    )
    cf_marker.add_to(marker_layer)
marker_layer.add_to(map)

# Keep highlights on top
map.keep_in_front(highlight_layer)

# --- Add Layer Control ---
folium.LayerControl(collapsed=False).add_to(map)

# --- Custom CSS for Legend Styling ---
custom_css = """
<style>
    /* Legend box styling */
    .legend {
        background-color: rgba(255, 255, 255, 0.9) !important; /* white box with slight transparency */
        border-radius: 10px;
        padding: 10px 14px;
        color: black !important;
        font-size: 18px !important;
        font-weight: bold;
        box-shadow: 0 0 8px rgba(0, 0, 0, 0.3);
    }

    /* Optional: ensure the legend is always visible and above other controls */
    .legend.leaflet-control {
        z-index: 9999 !important;
    }
</style>
"""
map.get_root().header.add_child(folium.Element(custom_css))


# --- Save map ---

#%%

caminho_dados2 = diretorio_atual+barra + \
    volta_nivel+barra+'entradas'+barra+'ISL07.csv'
    

caminho_dados3 = diretorio_atual+barra + \
    volta_nivel+barra+'entradas'+barra+'PNTBT.csv'
    
#%%

dadosUCs1 = pd.read_csv(caminho_dados2,sep=',')
dadosUCs2 = pd.read_csv(caminho_dados3,sep=',')


#%%

df_merged = dadosUCs1.merge(dadosUCs2, on='UNI_TR_AT', how='left').dropna()

#%%

df_merged['PNTBT_sum'] = df_merged[['PNTBT_01', 'PNTBT_02', 'PNTBT_03', 'PNTBT_04', 'PNTBT_05',
       'PNTBT_06', 'PNTBT_07', 'PNTBT_08', 'PNTBT_09', 'PNTBT_10', 'PNTBT_11',
       'PNTBT_12']].sum(axis=1)
#%%

df_merged['PNTBT_avg'] = df_merged['PNTBT_sum']/12

#%%

df_merged_2 = df_merged[['UNI_TR_AT','LONGITUDE', 'LATITUDE','PNTBT_sum', 'PNTBT_avg']]

#%%

gdf_dadosUCs = gpd.GeoDataFrame(df_merged_2, geometry=gpd.points_from_xy(df_merged_2.LATITUDE, df_merged_2.LONGITUDE), crs='EPSG:4326')

#%%
import osmnx as ox

from shapely.geometry import Point
from shapely.ops import unary_union


# Create GeoDataFrame from points
gdf = gpd.GeoDataFrame(
    geometry=[Point(xy) for xy in zip(df_merged["LATITUDE"], df_merged["LONGITUDE"])],
    crs="EPSG:4326"
)

# Project to UTM (meters)
gdf_proj = gdf.to_crs(gdf.estimate_utm_crs())

# Dissolve + buffer (7 km)
polygon_proj = gdf_proj.unary_union.buffer(7000)

# Back to lat/lon
polygon = gpd.GeoSeries([polygon_proj], crs=gdf_proj.crs).to_crs("EPSG:4326").iloc[0]

# Download graph
G = ox.graph_from_polygon(polygon, network_type="walk")

#%%
top_points = gdf_dadosUCs.sort_values(by='PNTBT_avg', ascending=False).head(50)


top_coords = top_points[['LATITUDE','LONGITUDE']].values


#%%
# # --- Plot network ---
fig, ax = ox.plot_graph(G, show=False, close=False)

# # --- Plot all points for reference (optional) ---
# ax.scatter(
#     coords[:,1],  # longitudes
#     coords[:,0],  # latitudes
#     color="lightgray",
#     s=30,
#     zorder=3,
#     label="All points"
# )

# --- Plot only top-value points ---
ax.scatter(
    top_coords[:,0],  # longitudes
    top_coords[:,1],  # latitudes
    color="red",
    s=80,
    zorder=5,
    label="Highest value"
)

ax.legend()
plt.show()

#%%

# # --- Select top 500 points by 'value' ---
# top_points = dadosUCs2.sort_values(by='Porc', ascending=False).head(1000).reset_index(drop=True)

# # Coordinates for map centering
# center = top_points[['LATITUDE', 'LONGITUDE']].mean().tolist()

# # Create Folium map centered on points
# m = folium.Map(location=center, zoom_start=13)

# # --- Compute and plot routes between consecutive top points ---
# for i in range(len(top_points) - 1):
#     orig = (top_points.loc[i, "LATITUDE"], top_points.loc[i, "LONGITUDE"])
#     dest = (top_points.loc[i + 1, "LATITUDE"], top_points.loc[i + 1, "LONGITUDE"])

#     # Find nearest graph nodes
#     orig_node = ox.distance.nearest_nodes(G, orig[1], orig[0])
#     dest_node = ox.distance.nearest_nodes(G, dest[1], dest[0])

#     # Compute shortest path
#     route = ox.shortest_path(G, orig_node, dest_node, weight="length")

#     # Extract coordinates
#     route_coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in route]

#     # Add route line to Folium map
#     folium.PolyLine(
#         route_coords,
#         color="blue",
#         weight=3,
#         opacity=0.7
#     ).add_to(m)

# # --- Add markers for top points ---
# for i, row in top_points.iterrows():
#     folium.CircleMarker(
#         location=(row["LATITUDE"], row["LONGITUDE"]),
#         radius=5,
#         color="red",
#         fill=True,
#         fill_color="red",
#         fill_opacity=0.9,
#         popup=f"Value: {row['Porc']}"
#     ).add_to(m)

# # Save map
# m.save(r'C:\Pos\Run-py-Figures\walk_top500.html')

#%%
# --- Select top 1000 points by 'Porc' ---
top_points = gdf_dadosUCs.sort_values(by='PNTBT_avg', ascending=False).head(100).reset_index()

# --- FeatureGroup for Top Risk Routes ---
top_routes_layer = folium.FeatureGroup(name="Top 50 Risk Routes")

#%%
import json
import geopandas as gpd
from shapely.geometry import Point


#%%
map = folium.Map(
    location=[-27.5969, -48.5495],
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri",
    zoom_start=17
)

# --- Layer 1: Choropleth (must be added directly to map) ---
choropleth = folium.Choropleth(
    geo_data=geojson_obj,
    name="Choropleth Layer",
    data=h3_gdf,
    columns=["h3_cell", "Risk_score_new"],
    key_on='feature.properties.h3_cell',
    fill_color="YlOrRd",
    legend_name="Energy Theft Cases Risk",
    highlight=True
).add_to(map)

# --- Layer 2: Highlight layer ---
highlight_layer = folium.FeatureGroup(name="Highlights")
highlights = folium.features.GeoJson(
    geojson_obj,
    style_function=lambda x: {'color':'transparent', 'fillColor':'transparent', 'weight':0},
    highlight_function=lambda x: {'fillColor': '#000000', 'color':'#000000', 'fillOpacity': 0.50, 'weight': 0.1},
    tooltip=folium.features.GeoJsonTooltip(
        fields=['h3_cell', 'Risk_score_new'],
        aliases=['h3_cell id: ', 'Level Risk: '],
        labels=True,
        sticky=False
    )
)
highlight_layer.add_child(highlights)
map.add_child(highlight_layer)

# --- Compute and plot routes between consecutive top points ---
for i in range(len(top_points) - 1):
    orig = (top_points.loc[i, "LATITUDE"], top_points.loc[i, "LONGITUDE"])
    dest = (top_points.loc[i + 1, "LATITUDE"], top_points.loc[i + 1, "LONGITUDE"])

    # Find nearest graph nodes
    orig_node = ox.distance.nearest_nodes(G, orig[1], orig[0])
    dest_node = ox.distance.nearest_nodes(G, dest[1], dest[0])

    # Compute shortest path
    route = ox.shortest_path(G, orig_node, dest_node, weight="length")

    # Extract coordinates
    route_coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in route]

    # Add route line
    folium.PolyLine(
        route_coords,
        color="blue",
        weight=3,
        opacity=0.6
    ).add_to(top_routes_layer)

# --- Add markers for top points ---
for _, row in top_points.iterrows():
    folium.CircleMarker(
        location=(row["LONGITUDE"], row["LATITUDE"]),
        radius=4,
        color="red",
        fill=True,
        fill_color="red",
        fill_opacity=0.85,
        popup=f"PNTBT_avg: {row['PNTBT_avg']}"
    ).add_to(top_routes_layer)

top_routes_layer.add_to(map)

#%%
points_gdf = gpd.GeoDataFrame(
    top_points,
    geometry=[
        Point(lon, lat)
        for lon, lat in zip(top_points["LATITUDE"], top_points["LONGITUDE"])
    ],
    crs="EPSG:4326"
)


geojson_dict = json.loads(geojson_obj)

polygon_gdf = gpd.GeoDataFrame.from_features(
    geojson_dict["features"],
    crs="EPSG:4326"
)

polygon = polygon_gdf.unary_union


#%%
points_outside = points_gdf[~points_gdf.geometry.within(polygon)]
count_outside = len(points_outside)

print(f"Red points outside polygon: {count_outside}")


#%%
map.save(r'C:\Pos\Run-py-Figures\DeepSeekMap2.html')