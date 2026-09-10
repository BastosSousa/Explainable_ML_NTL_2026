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

class RiskFuzzyClassifierWithTrapezoid:
    def __init__(self, risk_labels=['Low Risk', 'Medium Risk', 'High Risk'], largura_topo=0.2):
        """
        largura_topo: fração do intervalo (0 = triângulo, 1 = retângulo).
        Ex: 0.2 significa que o topo terá 20% da largura da base.
        """
        self.risk_labels = risk_labels
        self.n_classes = len(risk_labels)
        self.conjuntos = []
        self.largura_topo = largura_topo
        self.resultados = pd.DataFrame()  # DataFrame com classificações

    def definir_conjuntos_trapezoidais(self, dados):
        dados = np.array(dados)

        gmm = GaussianMixture(n_components=self.n_classes, random_state=42)
        gmm.fit(dados)
        labels = gmm.predict(dados)
        centros = gmm.means_

        distancias_centros = np.linalg.norm(centros, axis=1)
        indices_ordenados = np.argsort(distancias_centros)

        self.conjuntos = []
        for idx, original_idx in enumerate(indices_ordenados):
            classe_data = dados[labels == original_idx]
            risco_label = self.risk_labels[idx]

            if len(classe_data) > 0:
                x_min, x_max = classe_data[:, 0].min(), classe_data[:, 0].max()
                y_min, y_max = classe_data[:, 1].min(), classe_data[:, 1].max()

                if risco_label == 'Low Risk':
                    expansao = 0.3
                elif risco_label == 'Medium Risk':
                    expansao = 0.2
                else:
                    expansao = 0.1

                vertices = self.definir_vertices_trapezio(x_min, x_max, y_min, y_max, expansao, risco_label)

                conjunto = {
                    'classe': original_idx,
                    'risco_label': risco_label,
                    'nivel_risco': idx,
                    'trapezio': vertices,
                    'centro': centros[original_idx],
                    'dados': classe_data
                }
                self.conjuntos.append(conjunto)

        # Após criar os conjuntos, classificar todos os dados originais
        self.dados_originais_df = pd.DataFrame(dados, columns=['Feature1', 'Feature2'])
        self.classificar_lote(dados)

        return self.conjuntos


    def definir_vertices_trapezio(self, x_min, x_max, y_min, y_max, expansao, risco_label):
        """Define os vértices do trapézio baseado no nível de risco (2D)"""
        x_range = x_max - x_min
        y_range = y_max - y_min

        x_sup_min = x_min - x_range * expansao
        x_sup_max = x_max + x_range * expansao
        y_sup_min = y_min - y_range * expansao
        y_sup_max = y_max + y_range * expansao

        if risco_label == 'Low Risk':
            vertices = np.array([
                [x_sup_min, y_sup_min],
                [x_sup_max, y_sup_min],
                [x_sup_max, y_sup_max],
                [x_sup_min, y_sup_max]
            ])
        elif risco_label == 'Medium Risk':
            inclinacao = 0.2
            vertices = np.array([
                [x_sup_min + x_range * inclinacao, y_sup_min],
                [x_sup_max - x_range * inclinacao, y_sup_min],
                [x_sup_max - x_range * inclinacao * 0.5, y_sup_max],
                [x_sup_min + x_range * inclinacao * 0.5, y_sup_max]
            ])
        else:  # High Risk
            vertices = np.array([
                [x_sup_min, y_sup_min],
                [x_sup_max, y_sup_min],
                [x_sup_max - x_range * 0.3, y_sup_max],
                [x_sup_min + x_range * 0.3, y_sup_max]
            ])
        return vertices

    @staticmethod
    def ponto_dentro_poligono(x, y, poly):
        """Verifica se ponto está dentro de polígono (Ray Casting)."""
        n = len(poly)
        inside = False
        p1x, p1y = poly[0]
        for i in range(n + 1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside
    
    def funcao_pertinencia_trapezoidal(self, ponto, conjunto):
        """Calcula pertinência 2D com base no trapézio."""
        vertices = conjunto['trapezio']
        x, y = ponto

        if self.ponto_dentro_poligono(x, y, vertices):
            return 1.0
        else:
            # Pertinência decai com distância ao centro
            dist = np.linalg.norm(ponto - conjunto['centro'])
            max_dist = max(np.linalg.norm(v - conjunto['centro']) for v in vertices)
            return max(0, 1 - dist / (max_dist + 1e-9))

    def classificar_ponto(self, ponto):
        pertinencias = {}
        ponto = np.array(ponto)

        for conjunto in self.conjuntos:
            pert = self.funcao_pertinencia_trapezoidal(ponto, conjunto)
            pertinencias[conjunto['risco_label']] = pert

        classificacao = max(pertinencias.items(), key=lambda x: x[1])[0]

        linha = {"Feature1": ponto[0], "Feature2": ponto[1], "classificacao": classificacao, **pertinencias}
        self.resultados = pd.concat([self.resultados, pd.DataFrame([linha])], ignore_index=True)

        return classificacao, pertinencias

    def classificar_lote(self, pontos):
        """Classifica vários pontos 2D e preenche resultados."""
        for ponto in pontos:
            self.classificar_ponto(ponto)

    def salvar_resultados_csv(self, arquivo="resultados_classificacao.csv"):
        """Salva os resultados no CSV."""
        self.resultados.to_csv(arquivo, index=False)

    def plotar_trapezoides(self, dados=None):
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
classificador = RiskFuzzyClassifierWithTrapezoid(largura_topo=0.3)
conjuntos = classificador.definir_conjuntos_trapezoidais(dados2)

#%%
# Plotar os trapezoides
classificador.plotar_trapezoides(dados2)

#%%
classificador.salvar_resultados_csv("classificacao_risco.csv")
