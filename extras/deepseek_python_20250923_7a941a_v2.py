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
# class RiskFuzzyClassifier:
#     def __init__(self, risk_labels=['Low Risk', 'Medium Risk', 'High Risk']):
#         self.risk_labels = risk_labels
#         self.n_classes = len(risk_labels)
#         self.conjuntos = []
        
#     def definir_risco_automaticamente(self, dados, risk_criteria='distance'):
#         """
#         Define automaticamente os conjuntos de risco
#         risk_criteria: 'distance' (baseado na distância da origem) ou 
#                       'density' (baseado na densidade dos dados)
#         """
#         dados = np.array(dados)
        
#         # Ordenar as classes por nível de risco
#         if risk_criteria == 'distance':
#             # Risco aumenta com a distância da origem (0,0)
#             centro_global = np.mean(dados, axis=0)
#             distancias = distance.cdist([centro_global], dados)[0]
#         else:
#             # Risco aumenta com a densidade (mais dados = mais risco)
#             from sklearn.neighbors import KernelDensity
#             kde = KernelDensity(bandwidth=0.5)
#             kde.fit(dados)
#             densidades = np.exp(kde.score_samples(dados))
#             distancias = -densidades  # Inverter para ordenação
            
#         # Clusterização considerando o critério de risco
#         gmm = GaussianMixture(n_components=self.n_classes, random_state=42)
#         gmm.fit(dados)
#         labels = gmm.predict(dados)
        
#         # Calcular risco médio de cada cluster
#         riscos_clusters = []
#         for i in range(self.n_classes):
#             if np.sum(labels == i) > 0:
#                 risco_medio = np.mean(distancias[labels == i])
#                 riscos_clusters.append((i, risco_medio))
        
#         # Ordenar clusters por risco (menor para maior)
#         riscos_clusters.sort(key=lambda x: x[1])
        
#         # Mapear para labels de risco
#         mapeamento_risco = {}
#         for idx, (cluster_id, _) in enumerate(riscos_clusters):
#             mapeamento_risco[cluster_id] = self.risk_labels[idx]
        
#         # Definir conjuntos hipertrapezoidais
#         self.definir_conjuntos_risco(dados, labels, mapeamento_risco, gmm.means_)
        
#         return self.conjuntos
    
#     def definir_conjuntos_risco(self, dados, labels, mapeamento_risco, centros):
#         """Define os conjuntos com características específicas de risco"""
        
#         self.conjuntos = []
        
#         for cluster_id, risco_label in mapeamento_risco.items():
#             classe_data = dados[labels == cluster_id]
            
#             if len(classe_data) > 0:
#                 centro = centros[cluster_id]
#                 cov = np.cov(classe_data.T)
                
#                 # Ajustar parâmetros baseado no nível de risco
#                 if risco_label == 'Low Risk':
#                     raio_nucleo = np.percentile(distance.cdist([centro], classe_data)[0], 50)
#                     raio_suporte = raio_nucleo * 1.8  # Área maior para baixo risco
#                     alpha = 0.2  # Transição suave
                    
#                 elif risco_label == 'Medium Risk':
#                     raio_nucleo = np.percentile(distance.cdist([centro], classe_data)[0], 60)
#                     raio_suporte = raio_nucleo * 1.5
#                     alpha = 0.4
                    
#                 else:  # High Risk
#                     raio_nucleo = np.percentile(distance.cdist([centro], classe_data)[0], 70)
#                     raio_suporte = raio_nucleo * 1.3  # Área mais concentrada
#                     alpha = 0.6
                
#                 conjunto = {
#                     'classe': cluster_id,
#                     'risco_label': risco_label,
#                     'centro': centro,
#                     'raio_nucleo': raio_nucleo,
#                     'raio_suporte': raio_suporte,
#                     'covariancia': cov,
#                     'alpha': alpha,
#                     'nivel_risco': self.risk_labels.index(risco_label)
#                 }
#                 self.conjuntos.append(conjunto)
        
#         # Ordenar por nível de risco
#         self.conjuntos.sort(key=lambda x: x['nivel_risco'])
#  #%%   

# def calcular_pertinencia_risco(self, ponto):
#     """Calcula pertinência considerando características de risco"""
    
#     pertinencias = {}
    
#     for conjunto in self.conjuntos:
#         # Distância Mahalanobis ajustada
#         diff = ponto - conjunto['centro']
#         try:
#             inv_cov = np.linalg.pinv(conjunto['covariancia'])
#             dist_mahal = np.sqrt(diff @ inv_cov @ diff.T)
#         except:
#             dist_mahal = np.linalg.norm(diff)
        
#         # Função de pertinência baseada no risco
#         if dist_mahal <= conjunto['raio_nucleo']:
#             pertinencia = 1.0
#         elif dist_mahal <= conjunto['raio_suporte']:
#             # Transição mais suave para baixo risco, mais abrupta para alto risco
#             decaimento = (dist_mahal - conjunto['raio_nucleo']) / \
#                         (conjunto['raio_suporte'] - conjunto['raio_nucleo'])
#             pertinencia = 1.0 - (decaimento ** (1 - conjunto['alpha']))
#         else:
#             pertinencia = 0.0
            
#         pertinencias[conjunto['risco_label']] = pertinencia
    
#     return pertinencias

# def classificar_risco(self, ponto, method='max'):
#     """Classifica um ponto em uma categoria de risco"""
#     pertinencias = self.calcular_pertinencia_risco(ponto)
    
#     if method == 'max':
#         # Máxima pertinência
#         risco_class = max(pertinencias.items(), key=lambda x: x[1])[0]
#     elif method == 'weighted':
#         # Média ponderada pelo nível de risco
#         soma_ponderada = sum(pertinencias[label] * (i + 1) 
#                            for i, label in enumerate(self.risk_labels))
#         soma_pertinencias = sum(pertinencias.values())
        
#         if soma_pertinencias > 0:
#             risco_medio = soma_ponderada / soma_pertinencias
#             if risco_medio < 1.5:
#                 risco_class = 'Low Risk'
#             elif risco_medio < 2.5:
#                 risco_class = 'Medium Risk'
#             else:
#                 risco_class = 'High Risk'
#         else:
#             risco_class = 'Indeterminado'
    
#     return risco_class, pertinencias
    
#%%
dados2 = dados[['umap_1_scaled','umap_2_scaled']].values

#%%
# # Criar e treinar classificador
# classificador = RiskFuzzyClassifier()
# conjuntos = classificador.definir_risco_automaticamente(dados2, 'distance')

# #%%
# # Adicionar métodos à classe
# classificador.calcular_pertinencia_risco = lambda x: calcular_pertinencia_risco(classificador, x)
# classificador.classificar_risco = lambda x, method='max': classificar_risco(classificador, x, method)

#%%

# def visualizar_sistema_risco(dados, classificador):
#     plt.figure(figsize=(15, 5))
    
#     # Cores para os níveis de risco
#     cores_risco = {'Low Risk': 'green', 'Medium Risk': 'orange', 'High Risk': 'red'}
    
#     # Plot 1: Dados e regiões de risco
#     # plt.subplot(1, 3, 1)
    
#     # Plotar dados com cores baseadas na classificação real
#     for i, conjunto in enumerate(classificador.conjuntos):
#         # Área de influência
#         circle_suporte = plt.Circle(
#             conjunto['centro'], conjunto['raio_suporte'], 
#             color=cores_risco[conjunto['risco_label']], alpha=0.15, fill=True
#         )
#         circle_nucleo = plt.Circle(
#             conjunto['centro'], conjunto['raio_nucleo'],
#             color=cores_risco[conjunto['risco_label']], alpha=0.4, fill=True
#         )
        
#         plt.gca().add_patch(circle_suporte)
#         plt.gca().add_patch(circle_nucleo)
        
#         # Centro
#         plt.scatter(conjunto['centro'][0], conjunto['centro'][1],
#                    c=cores_risco[conjunto['risco_label']], s=200, 
#                    marker='X', edgecolors='black', linewidth=2)
        
#         plt.text(conjunto['centro'][0], conjunto['centro'][1],
#                 conjunto['risco_label'], ha='center', fontweight='bold')
    
#     plt.scatter(dados[:, 0], dados[:, 1], alpha=0.6, c='gray', s=30)
#     plt.title('Sistema de Classificação de Risco')
#     plt.xlabel('Feature 1 (ex: Severidade)')
#     plt.ylabel('Feature 2 (ex: Probabilidade)')
#     plt.axis('equal')
#     plt.grid(True, alpha=0.3)
    
      
#     plt.tight_layout()
#     plt.show()

# #%%
# # Visualizar o sistema
# visualizar_sistema_risco(dados2, classificador)

#%%
class RiskFuzzyClassifierWithTrapezoid:
    def __init__(self, risk_labels=['Low Risk', 'Medium Risk', 'High Risk']):
        self.risk_labels = risk_labels
        self.n_classes = len(risk_labels)
        self.conjuntos = []
        
    def definir_conjuntos_trapezoidais(self, dados):
        """Define conjuntos trapezoidais para dados 2D"""
        dados = np.array(dados)
        
        # Clusterização
        gmm = GaussianMixture(n_components=self.n_classes, random_state=42)
        gmm.fit(dados)
        labels = gmm.predict(dados)
        centros = gmm.means_
        
        # Ordenar por distância da origem (para definir risco)
        distancias_centros = np.linalg.norm(centros, axis=1)
        indices_ordenados = np.argsort(distancias_centros)
        
        self.conjuntos = []
        for idx, original_idx in enumerate(indices_ordenados):
            classe_data = dados[labels == original_idx]
            risco_label = self.risk_labels[idx]
            
            if len(classe_data) > 0:
                # Definir parâmetros do trapézio
                x_min, x_max = classe_data[:, 0].min(), classe_data[:, 0].max()
                y_min, y_max = classe_data[:, 1].min(), classe_data[:, 1].max()
                
                # Ajustar tamanho baseado no nível de risco
                if risco_label == 'Low Risk':
                    expansao = 0.3  # Trapézio mais amplo
                elif risco_label == 'Medium Risk':
                    expansao = 0.2
                else:  # High Risk
                    expansao = 0.1  # Trapézio mais estreito
                
                # Definir vértices do trapézio
                trapézio = self.definir_vertices_trapezio(x_min, x_max, y_min, y_max, expansao, risco_label)
                
                # Para pertinência 1D (usamos apenas eixo x dos vértices)
                trap_1d = (x_min, (x_min + x_max) / 2,
                           (x_min + x_max) / 2, x_max)
                
                conjunto = {
                    'classe': original_idx,
                    'risco_label': risco_label,
                    'nivel_risco': idx,
                    'trapezio': trapézio,
                    'trapezio_1d': trap_1d,
                    'centro': centros[original_idx],
                    'dados': classe_data
                }
                self.conjuntos.append(conjunto)
        
        return self.conjuntos
    
    def definir_vertices_trapezio(self, x_min, x_max, y_min, y_max, expansao, risco_label):
        """Define os vértices do trapézio baseado no nível de risco"""
        
        # Expandir área
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        x_sup_min = x_min - x_range * expansao
        x_sup_max = x_max + x_range * expansao
        y_sup_min = y_min - y_range * expansao
        y_sup_max = y_max + y_range * expansao
        
        # Ajustar forma do trapézio baseado no risco
        if risco_label == 'Low Risk':
            # Trapézio mais retangular (base larga)
            vertices = np.array([
                [x_sup_min, y_sup_min],  # Vértice inferior esquerdo
                [x_sup_max, y_sup_min],  # Vértice inferior direito
                [x_sup_max, y_sup_max],  # Vértice superior direito
                [x_sup_min, y_sup_max]   # Vértice superior esquerdo
            ])
            
        elif risco_label == 'Medium Risk':
            # Trapézio ligeiramente inclinado
            inclinacao = 0.2
            vertices = np.array([
                [x_sup_min + x_range * inclinacao, y_sup_min],
                [x_sup_max - x_range * inclinacao, y_sup_min],
                [x_sup_max - x_range * inclinacao * 0.5, y_sup_max],
                [x_sup_min + x_range * inclinacao * 0.5, y_sup_max]
            ])
            
        else:  # High Risk
            # Trapézio mais estreito na base superior
            vertices = np.array([
                [x_sup_min, y_sup_min],
                [x_sup_max, y_sup_min],
                [x_sup_max - x_range * 0.3, y_sup_max],
                [x_sup_min + x_range * 0.3, y_sup_max]
            ])
        
        return vertices
    
 
  # ---------- NOVO: FUNÇÃO DE PERTINÊNCIA 1D ----------
    @staticmethod
    def trapezio_1d(x, a, b, c, d):
        return np.maximum(
            np.minimum(
                np.minimum((x - a) / (b - a + 1e-9), 1),
                (d - x) / (d - c + 1e-9)
            ),
            0
        )

    def plotar_funcoes_fuzzy(self):
        """Plota as funções de pertinência trapezoidais em 1D"""
        # pega os limites em x
        todos_vertices = [v for conj in self.conjuntos for v in conj['trapezio_1d']]
        x_min, x_max = min(todos_vertices), max(todos_vertices)
        x = np.linspace(x_min, x_max, 500)

        plt.figure(figsize=(8, 5))
        for conjunto in self.conjuntos:
            a, b, c, d = conjunto['trapezio_1d']
            y = self.trapezio_1d(x, a, b, c, d)
            plt.plot(x, y, label=conjunto['risco_label'])
            plt.fill_between(x, y, alpha=0.2)

        plt.ylim(-0.1, 1.1)
        plt.xlabel("x")
        plt.ylabel("µ(x)")
        plt.title("Funções Fuzzy - Trapézios")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()

    def plotar_trapezoides(self, dados=None):
        """Plota os trapezoides para cada classe de risco em 2D"""
        fig, ax = plt.subplots(figsize=(8, 6))

        cores = {'Low Risk': 'green', 'Medium Risk': 'orange', 'High Risk': 'red'}
        alphas = {'Low Risk': 0.2, 'Medium Risk': 0.3, 'High Risk': 0.4}

        for conjunto in self.conjuntos:
            vertices = conjunto['trapezio']
            risco_label = conjunto['risco_label']

            trapezoid = Polygon(vertices, closed=True,
                                alpha=alphas[risco_label],
                                color=cores[risco_label],
                                label=risco_label)
            ax.add_patch(trapezoid)

            ax.scatter(conjunto['centro'][0], conjunto['centro'][1],
                       color=cores[risco_label], s=100, marker='X',
                       edgecolors='black', linewidth=2)

            if dados is not None:
                classe_data = conjunto['dados']
                ax.scatter(classe_data[:, 0], classe_data[:, 1],
                           color=cores[risco_label], alpha=0.6, s=30)

            ax.text(conjunto['centro'][0], conjunto['centro'][1] + 0.3,
                    risco_label, ha='center', fontweight='bold', fontsize=10)

            ax.scatter(vertices[:, 0], vertices[:, 1],
                       color=cores[risco_label], s=50, marker='o')

        ax.set_xlabel('Feature 1 (ex: Severidade)')
        ax.set_ylabel('Feature 2 (ex: Probabilidade)')
        ax.set_title('Conjuntos Trapezoidais de Risco - Visão Geral')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.show()
#%%

# Criar e treinar classificador
classificador = RiskFuzzyClassifierWithTrapezoid()
conjuntos = classificador.definir_conjuntos_trapezoidais(dados2)

#%%
# Plotar os trapezoides
classificador.plotar_trapezoides(dados2)

#%%
classificador.plotar_funcoes_fuzzy()   