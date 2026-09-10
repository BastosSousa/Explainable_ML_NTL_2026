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


#%%

# def preparar_dados(self, df):
#     for col in ["umap_1_scaled", "umap_2_scaled"]:
#         corr = dados[[col, "Perdas"]].corr().iloc[0,1] + dados[[col, "w_PARECER2_TOTAL"]].corr().iloc[0,1]
#         if corr < 0:  # relação negativa → inverter
#             df_norm[col] = 1 - df_norm[col]
    
    
#     df_norm["Risk_score"] = df_norm[["Perdas_norm", "Parecer_norm", "umap_1_scaled", "umap_2_scaled"]].mean(axis=1)
#     self.df_norm = df_norm
#     return df_norm

#%%%%%%%%%%%%%%%%%%%%%%%

# import pandas as pd

# class RiskFuzzyClassifierWithTrapezoid:
#     def __init__(self, risk_labels=['Low Risk', 'Medium Risk', 'High Risk'], largura_topo=0.3, sigma=0.05):
#         """
#         largura_topo: fração do intervalo (0 = triângulo, 1 = retângulo).
#         sigma: desvio padrão da Gaussiana usada para suavizar as transições.
#         """
#         self.risk_labels = risk_labels
#         self.n_classes = len(risk_labels)
#         self.conjuntos = []
#         self.largura_topo = largura_topo
#         self.sigma = sigma
#         self.resultados = pd.DataFrame()  # DataFrame com classificações
#         self.df_norm = None
#         self.definicoes = None

#     def preparar_dados(self, df_norm):
        

#         for col in ["umap_1_scaled", "umap_2_scaled"]:
#             corr = dados[[col, "Perdas"]].corr().iloc[0,1] + dados[[col, "w_PARECER2_TOTAL"]].corr().iloc[0,1]
#             if corr < 0:  # relação negativa → inverter
#                 df_norm[col] = 1 - df_norm[col]
        
        
#         df_norm["Risk_score"] = df_norm[["Perdas_norm", "Parecer_norm", "umap_1_scaled", "umap_2_scaled"]].mean(axis=1)
#         self.df_norm = df_norm

#         # Definir fronteiras fuzzy dinamicamente pelos quantis
#         q20, q50, q80 = df_norm["Risk_score"].quantile([0.20, 0.50, 0.80])
#         base_width = q80 - q20
#         top_width = self.largura_topo * base_width
#         b = q20 + (base_width - top_width)/2
#         c = q80 - (base_width - top_width)/2

#         # Trapezoides com transições suaves
#         self.definicoes = {
#             "Low Risk":    (0.0, 0.0, q20, q50),
#             "Medium Risk": (q20, b, c, q80),
#             "High Risk":   (q50, q80, 1.0, 1.0)
#         }

#         return df_norm

#     @staticmethod
#     def trapezio_1d(x, a, b, c, d):
#         """Função de pertinência trapezoidal 1D"""
#         return np.maximum(
#             np.minimum(
#                 np.minimum((x - a) / (b - a + 1e-9), 1),
#                 (d - x) / (d - c + 1e-9)
#             ),
#             0
#         )

#     def gaussian_smooth(self, x, center):
#         """Fator de suavização Gaussiana"""
#         return np.exp(-0.5 * ((x - center) / self.sigma) ** 2)

#     def trapezio_gaussian(self, x, a, b, c, d):
#         """
#         Combina trapezoide com Gaussiana para suavizar as bordas
#         """
#         trap = self.trapezio_1d(x, a, b, c, d)
#         smooth_left = self.gaussian_smooth(x, b)
#         smooth_right = self.gaussian_smooth(x, c)
#         return trap * np.minimum(1, smooth_left + smooth_right)

#     def classificar_com_score(self):
#         """Classifica cada ponto com base no Risk_score 1D (quantis dinâmicos)."""
#         if self.df_norm is None or self.definicoes is None:
#             raise ValueError("⚠️ Execute preparar_dados(df) antes de classificar.")

#         resultados = []
#         for _, row in self.df_norm.iterrows():
#             x = row["Risk_score"]

#             pertinencias = {
#                 label: self.trapezio_gaussian(x, *params)
#                 for label, params in self.definicoes.items()
#             }
#             classificacao = max(pertinencias.items(), key=lambda kv: kv[1])[0]

#             linha = {"Risk_score": x, "classificacao": classificacao, **pertinencias}
#             resultados.append(linha)

#         self.resultados = pd.DataFrame(resultados)
#         return self.resultados

#     def plotar_trapezio_1d(self, intervalo=None):
#         """Plota as funções fuzzy trapezoidais suavizadas 1D"""
#         if self.definicoes is None:
#             raise ValueError("⚠️ Execute preparar_dados(df) antes de plotar.")

#         x = np.linspace(0, 1, 500)
#         plt.figure(figsize=(8, 5))
#         cores = {'Low Risk': 'green', 'Medium Risk': 'orange', 'High Risk': 'red'}

#         for label, (a, b, c, d) in self.definicoes.items():
#             y = self.trapezio_gaussian(x, a, b, c, d)
#             plt.plot(x, y, label=label, color=cores[label])
#             plt.fill_between(x, y, alpha=0.2, color=cores[label])

#         plt.ylim(-0.05, 1.05)
#         plt.xlabel("x")
#         plt.ylabel("µ(x)")
#         plt.title("1D Fuzzy Risk Functions (Gaussian-tuned Trapezoids)")
#         plt.grid(True, alpha=0.3)
#         plt.legend()
#         plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

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

#%%%%%%

# class RiskFuzzyClassifierWithTrapezoid:
#     def __init__(self, 
#                  risk_labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'], 
#                  largura_topo=0.3):
#         """
#         largura_topo: fraction of the trapezoid base used for the top (0=triangle, 1=rectangle)
#         """
#         self.risk_labels = risk_labels
#         self.n_classes = len(risk_labels)
#         self.largura_topo = largura_topo
#         self.df_norm = None
#         self.definicoes = None
#         self.resultados = pd.DataFrame()

#     def preparar_dados(self, df):
#         df_norm = df.copy()

#         # Normalize columns if correlation is negative
#         for col in ["umap_1_scaled", "umap_2_scaled"]:
#             corr = df_norm[[col, "Perdas_norm"]].corr().iloc[0,1] + df_norm[[col, "Parecer_norm"]].corr().iloc[0,1]
#             if corr < 0:
#                 df_norm[col] = 1 - df_norm[col]

#         # Compute Risk_score as mean of normalized features
#         df_norm["Risk_score"] = df_norm[["Perdas_norm", "Parecer_norm", "umap_1_scaled", "umap_2_scaled"]].mean(axis=1)
#         self.df_norm = df_norm

#         # Compute quantiles for 5 classes
#         q0, q20, q40, q60, q80, q100 = df_norm["Risk_score"].quantile([0,0.2,0.4,0.6,0.8,1.0])
#         top_frac = self.largura_topo
        
#         self.definicoes = {
#             "Very Low":  (q0, q0, q20, q40),
#             "Low":       (q20, q20 + top_frac*(q40-q20), q40 - top_frac*(q40-q20), q60),
#             "Medium":    (q40, q40 + top_frac*(q60-q40), q60 - top_frac*(q60-q40), q60),
#             "High":      (q60, q60 + top_frac*(q80-q60), q80 - top_frac*(q80-q60), q80),
#             "Very High": (q80, q100, q100, q100)
#         }

#         return df_norm

#     @staticmethod
#     def trapezio_1d(x, a, b, c, d):
#         """Standard trapezoidal membership"""
#         return np.maximum(
#             np.minimum(
#                 np.minimum((x - a) / (b - a + 1e-9), 1),
#                 (d - x) / (d - c + 1e-9)
#             ),
#             0
#         )

#     def classificar_com_score(self):
#         """Classify each row based on Risk_score"""
#         if self.df_norm is None or self.definicoes is None:
#             raise ValueError("⚠️ Execute preparar_dados(df) antes de classificar.")

#         resultados = []
#         for _, row in self.df_norm.iterrows():
#             x = row["Risk_score"]
#             pertinencias = {label: self.trapezio_1d(x, *params)
#                             for label, params in self.definicoes.items()}
#             classificacao = max(pertinencias.items(), key=lambda kv: kv[1])[0]

#             linha = {"Risk_score": x, "classificacao": classificacao, **pertinencias}
#             resultados.append(linha)

#         self.resultados = pd.DataFrame(resultados)
#         return self.resultados

#     def plotar_trapezio_1d(self):
#         """Plot standard 1D trapezoidal fuzzy functions"""
#         if self.definicoes is None:
#             raise ValueError("⚠️ Execute preparar_dados(df) antes de plotar.")

#         x = np.linspace(0, 1, 500)
#         plt.figure(figsize=(10, 6))
#         cores = ['#2ca02c', '#98df8a', '#ffbb78', '#ff7f0e', '#d62728']  # green → red

#         for color, (label, params) in zip(cores, self.definicoes.items()):
#             y = self.trapezio_1d(x, *params)
#             plt.plot(x, y, label=label, color=color)
#             plt.fill_between(x, y, alpha=0.2, color=color)

#         plt.ylim(-0.05, 1.05)
#         plt.xlabel("Risk_score")
#         plt.ylabel("µ(x)")
#         plt.title("1D Fuzzy Risk Functions (Very Low → Very High)")
#         plt.grid(True, alpha=0.3)
#         plt.legend()
#         plt.show()


        
#%%
# Criar e treinar classificador
classificador = RiskFuzzyClassifierWithTrapezoid()
#%%
df_norm = classificador.preparar_dados(df)
#%%
resultados = classificador.classificar_com_score()
#%%
# 4. Visualizar conjuntos trapezoidais 2D com UMAPs
# classificador.definir_conjuntos_trapezoidais(df_proc[["umap_1_scaled", "umap_2_scaled"]].values)
# classificador.plotar_trapezoides()
#%%
classificador.plotar_trapezio_1d()
