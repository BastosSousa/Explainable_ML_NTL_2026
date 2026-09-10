import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import distance
import pandas as pd
import os
import platform
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
class AutoFuzzy2D:
    def __init__(self, n_classes=3, overlap_factor=0.2):
        self.n_classes = n_classes
        self.overlap_factor = overlap_factor
        
    def definir_conjuntos_automaticamente(self, dados):
        """Define automaticamente os conjuntos para 3 classes"""
        
        dados = np.array(dados)
        if dados.shape[1] != 2:
            raise ValueError("Dados devem ser 2D")
            
        # 1. Clusterização para encontrar regiões naturais
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(n_components=self.n_classes, random_state=42)
        gmm.fit(dados)
        labels = gmm.predict(dados)
        
        conjuntos = []
        
        for i in range(self.n_classes):
            # Dados da classe atual
            classe_data = dados[labels == i]
            
            if len(classe_data) > 0:
                # Estatísticas da classe
                centro = np.mean(classe_data, axis=0)
                cov = np.cov(classe_data.T)
                
                # Definir núcleo (área de pertinência máxima)
                distancias = distance.cdist([centro], classe_data)[0]
                raio_nucleo = np.percentile(distancias, 60)
                
                # Definir suporte (área de pertinência > 0)
                raio_suporte = np.max(distancias) * (1 + self.overlap_factor)
                
                conjunto = {
                    'classe': i,
                    'centro': centro,
                    'raio_nucleo': raio_nucleo,
                    'raio_suporte': raio_suporte,
                    'covariancia': cov
                }
                conjuntos.append(conjunto)
                
        self.conjuntos = conjuntos
        return conjuntos
 #%%   
    def fuzzificar_ponto(self, ponto):
        """Fuzzifica um ponto 2D para as 3 classes"""
        pertinencias = []
        
        for conjunto in self.conjuntos:
            # Distância Mahalanobis para considerar a forma da distribuição
            diff = ponto - conjunto['centro']
            inv_cov = np.linalg.pinv(conjunto['covariancia'])
            dist_mahal = np.sqrt(diff @ inv_cov @ diff.T)
            
            # Função de pertinência baseada na distância
            if dist_mahal <= conjunto['raio_nucleo']:
                pertinencia = 1.0
            elif dist_mahal <= conjunto['raio_suporte']:
                pertinencia = 1.0 - (dist_mahal - conjunto['raio_nucleo']) / \
                             (conjunto['raio_suporte'] - conjunto['raio_nucleo'])
            else:
                pertinencia = 0.0
                
            pertinencias.append(pertinencia)
            
        return np.array(pertinencias)
    
#%%
dados2 = dados[['umap_1_scaled','umap_2_scaled']].values

#%%
# Definir conjuntos automaticamente
fuzzy_system = AutoFuzzy2D(n_classes=3)
conjuntos = fuzzy_system.definir_conjuntos_automaticamente(dados2)

#%%

def visualizar_conjuntos(dados, conjuntos):
    plt.figure(figsize=(12, 5))
    
    # Plotar dados originais
    plt.subplot(1, 2, 1)
    plt.scatter(dados[:, 0], dados[:, 1], alpha=0.6, c='gray')
    
    # Plotar centros e regiões
    cores = ['red', 'blue', 'green']
    for i, conjunto in enumerate(conjuntos):
        centro = conjunto['centro']
        plt.scatter(centro[0], centro[1], c=cores[i], s=200, marker='x', linewidth=3)
        
        # Circulo do núcleo
        circle_nucleo = plt.Circle(centro, conjunto['raio_nucleo'], 
                                 color=cores[i], alpha=0.3, fill=True)
        plt.gca().add_patch(circle_nucleo)
        
        # Circulo do suporte
        circle_suporte = plt.Circle(centro, conjunto['raio_suporte'], 
                                  color=cores[i], alpha=0.1, fill=True)
        plt.gca().add_patch(circle_suporte)
        
    plt.title('Conjuntos Hipertrapezoidais Definidos')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()

#%%
# Visualizar
visualizar_conjuntos(dados2, conjuntos)