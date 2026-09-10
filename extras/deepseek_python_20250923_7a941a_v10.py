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
        q20, q50, q80 = df_norm["Risk_score_new"].quantile([0.25, 0.50, 0.75])
      
        base_width = q80 - q20
        top_width = self.largura_topo * base_width
        b = q20 + (base_width - top_width)/2
        c = q80 - (base_width - top_width)/2
        self.definicoes = {
            "Low Risk":    (0.0, 0.0, q20, q50),
            "Medium Risk": (q20, b, c, q80),
            "High Risk":   (q50, q80, 1.0, 1.0)
        }
        
        
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

   
    def classificar_com_score(self):
            """
            Classifica cada ponto com base no Risk_score 1D (quantis dinâmicos).
            """
            if self.df_norm is None or self.definicoes is None:
                raise ValueError("⚠️ Execute preparar_dados(df) antes de classificar.")
    
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


    def plotar_trapezio_1d(self, intervalo=None):
        """
        Plota as funções fuzzy trapezoidais 1D calculadas pelos quantis.
        """
        if self.definicoes is None:
            raise ValueError("⚠️ Execute preparar_dados(df) antes de plotar.")

        x = np.linspace(0, 1, 500)
        plt.figure(figsize=(8, 5))
        cores = {'Low Risk': 'green', 'Medium Risk': 'orange', 'High Risk': 'red'}

        for label, (a, b, c, d) in self.definicoes.items():
            y = self.trapezio_1d(x, a, b, c, d)
            plt.plot(x, y, label=label, color=cores[label])
            plt.fill_between(x, y, alpha=0.2, color=cores[label])

        plt.ylim(-0.05, 1.05)
        plt.xlabel("x")
        plt.ylabel("µ(x)")
        plt.title("1D Fuzzy Risk Functions (quantile-based)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()
        
#%%
# Criar e treinar classificador
classificador = RiskFuzzyClassifierWithTrapezoid()
#%%
df_norm = classificador.preparar_dados(df)
#%%
resultados = classificador.classificar_com_score()
#%%
classificador.plotar_trapezio_1d()

#%%

resultados['h3_cell'] = dados['h3_cell'].values
#%%
from shapely.geometry import Polygon

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

