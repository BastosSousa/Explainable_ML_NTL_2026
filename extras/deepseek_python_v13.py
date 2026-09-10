import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import distance
import pandas as pd
import os
from sklearn.mixture import GaussianMixture
import platform

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
    def trapezio_1d(x, a, b, c, d):
        """Função de pertinência trapezoidal 1D"""
        return np.maximum(
            np.minimum(
                np.minimum((x - a) / (b - a + 1e-9), 1),
                (d - x) / (d - c + 1e-9)
            ),
            0
        )    

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
            Classifica cada ponto com base no Risk_score 1D (quantis dinâmicos).
            """
            if self.df_norm is None or self.definicoes is None:
                raise ValueError(" ! Execute preparar_dados(df) antes de classificar.")
    
            resultados = []
            for _, row in self.df_norm.iterrows():
                x = row["Risk_score_new"]
    
                pertinencias = {
                    label: self.trapezio_1d(x, *params)
                    for label, params in self.definicoes.items()
                }
                classificacao = max(pertinencias.items(), key=lambda kv: kv[1])[0]
    
                linha = {"Risk_score_new": x, "classificacao": classificacao, **pertinencias}
                resultados.append(linha)
    
            self.resultados = pd.DataFrame(resultados)
            return self.resultados


    # def plotar_trapezio_1d(self, intervalo=None):
    #     """
    #     Plota as funções fuzzy trapezoidais 1D calculadas pelos quantis.
    #     """
    #     if self.definicoes is None:
    #         raise ValueError(" ! Execute preparar_dados(df) antes de plotar.")

    #     x = np.linspace(0, 1, 500)
    #     plt.figure(figsize=(8, 5))
    #     cores = {'Low Risk': 'green', 'Medium Risk': 'orange', 'High Risk': 'red'}

    #     for label, (a, b, c, d) in self.definicoes.items():
    #         y = self.trapezio_1d(x, a, b, c, d)
    #         plt.plot(x, y, label=label, color=cores[label])
    #         plt.fill_between(x, y, alpha=0.2, color=cores[label])

    #     plt.ylim(-0.05, 1.05)
    #     plt.xlabel("x")
    #     plt.ylabel("µ(x)")
    #     plt.title("1D Fuzzy Risk Functions (quantile-based)")
    #     plt.grid(True, alpha=0.3)
    #     plt.legend()
    #     plt.show()

    def plotar_trapezio_1d(self, intervalo=None):
        """
        Plota as funções fuzzy trapezoidais 1D calculadas pelos quantis
        e marca os pontos de interseção (Low∩Medium, Medium∩High) com linhas,
        pontos e anotações.
        """
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
    
        # calcular interseções entre conjuntos adjacentes (assumindo Low/Medium/High)
        pairs = [("Low Risk", "Medium Risk"), ("Medium Risk", "High Risk")]
        intersec_points = {}
    
        for p1, p2 in pairs:
            params1 = self.definicoes.get(p1)
            params2 = self.definicoes.get(p2)
            if params1 is None or params2 is None:
                intersec_points[f"{p1} ∩ {p2}"] = (None, None)
                continue
            xi, mu = self._intersecao_trapezios(params1, params2, n=2000)
            intersec_points[f"{p1} ∩ {p2}"] = (xi, mu)
    
        # marcar e anotar as interseções válidas
        for nome, (xi, mu) in intersec_points.items():
            if xi is None:
                # opcional: mostrar aviso no gráfico
                plt.text(0.02, 0.02, f"{nome}: no overlap", transform=plt.gca().transAxes, fontsize=9, color='gray')
                continue
    
            # linha vertical tracejada
            plt.axvline(x=xi, color='black', linestyle='--', alpha=0.7)
    
            # marcador no ponto (xi, mu)
            plt.scatter([xi], [mu], color='black', s=60, zorder=6)
    
            # anotação legível acima do ponto
            plt.annotate(
                f"{nome}\n x={xi:.3f}\nµ≈{mu:.3f}",
                xy=(xi, mu),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none")
            )
    
        # estética
        plt.ylim(-0.05, 1.10)
        plt.xlim(0, 1)
        plt.xlabel("Risk_score_new")
        plt.ylabel("µ(x)")
        plt.title("Fuzzy Risk Functions + Intersection Points", fontsize=13)
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
import h3
import geopandas as gpd

#%%

def cell_to_shapely(cell):
    coords = h3.cell_to_boundary(cell)
    flipped = tuple(coord[::-1] for coord in coords)
    return Polygon(flipped)

#%%
h3_geoms = resultados['h3_cell'].apply(lambda x: cell_to_shapely(x))
#%%
resultados['geometry'] = h3_geoms.values

#%%
resultados_2 = gpd.GeoDataFrame(data=resultados, geometry='geometry', crs=4326)

#%%

fig, ax = plt.subplots(figsize=(8, 6))
resultados_2.plot(
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
geojson_obj = resultados_2.to_json()

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
    data=resultados_2,
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
map.save(r'C:\Pos\Run-py-Figures\DeepSeekMap.html')

#%%

# caminho_data_UCs = diretorio_atual+barra + \
    
#%%

# gdf_dados_todos_12 = pd.read_excel(caminho_12)

#%%
# dadosUCs2 = dadosUCs.rename(
#     columns={'LATITUDE_x': 'LATITUDE', 'LONGITUDE_x':'LONGITUDE'})
# #%%

# dadosUCs2 = dadosUCs2[['Cod_setor', 'UNI_TR_MT', 'UC','Porc',
#        'LATITUDE', 'LONGITUDE',]]

# #%%

# gdf_dadosUCs = gpd.GeoDataFrame(dadosUCs2, geometry=gpd.points_from_xy(dadosUCs2.LONGITUDE, dadosUCs2.LATITUDE), crs='EPSG:4326')




