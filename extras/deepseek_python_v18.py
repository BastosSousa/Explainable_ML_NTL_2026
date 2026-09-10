import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import pandas as pd
import os

import platform


from sklearn.preprocessing import MinMaxScaler


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

caminho_dados_final2 = diretorio_atual+barra + \
    volta_nivel+barra+'saidas'+barra+'data_finalFuzzy2.csv'
    
#%%

risk_map = pd.read_csv(caminho_dados_final2,sep=',')
   
#%%

class RiskFuzzyClassifier:

    def __init__(self):

        self.df_norm = None
        self.definicoes = None
        self.resultados = None

    ##########################################################
    # Preparação
    ##########################################################

    def preparar_dados(self, risk_map):

        df = risk_map.copy()

        fuzzy_vars = [
            "pred",
            "social_score",
            "cluster_weight",
            "Perdas_count"
        ]

        scaler = MinMaxScaler()

        df[fuzzy_vars] = scaler.fit_transform(df[fuzzy_vars])

        # pouca amostra = maior incerteza
        df["Perdas_count"] = 1-df["Perdas_count"]

        ##########################################################
        # Índice fuzzy
        ##########################################################

        df["Risk_score"] = (

              0.50*df["pred"]

            + 0.30*df["social_score"]

            + 0.20*df["cluster_weight"]

        )

        df["Risk_score"]=(

            df["Risk_score"]-

            df["Risk_score"].min()

        )/(

            df["Risk_score"].max()

            -

            df["Risk_score"].min()

        )

        ##########################################################
        # Trapézios
        ##########################################################

        q20,q40,q60,q80 = df["Risk_score"].quantile(

            [0.20,0.40,0.60,0.80]

        )

        self.definicoes={

            "Low Risk":(

                0,
                0,
                q20,
                q40

            ),

            "Medium Risk":(

                q20,
                q40,
                q60,
                q80

            ),

            "High Risk":(

                q60,
                q80,
                1,
                1

            )

        }

        self.df_norm=df

        return df

    ##########################################################
    # Trapezoidal
    ##########################################################

    @staticmethod

    def trapezio(x, a, b, c, d):

        x = np.asarray(x, dtype=float)
    
        mu = np.zeros_like(x)
    
        # Caso especial: trapézio aberto à direita
        if c == d:
            mu[x >= b] = 1
    
            if b > a:
                idx = (x >= a) & (x < b)
                mu[idx] = (x[idx] - a) / (b - a)
    
            return np.clip(mu, 0, 1)
    
        # Caso especial: trapézio aberto à esquerda
        if a == b:
            mu[x <= c] = 1
    
            if d > c:
                idx = (x > c) & (x <= d)
                mu[idx] = (d - x[idx]) / (d - c)
    
            return np.clip(mu, 0, 1)
    
        # subida
        idx = (x >= a) & (x < b)
        mu[idx] = (x[idx] - a) / (b - a)
    
        # topo
        idx = (x >= b) & (x <= c)
        mu[idx] = 1
    
        # descida
        idx = (x > c) & (x <= d)
        mu[idx] = (d - x[idx]) / (d - c)
    
        return np.clip(mu, 0, 1)

    ##########################################################
    # Classificação
    ##########################################################

    def classificar(self):

        resultados=[]

        for _,row in self.df_norm.iterrows():

            x=row["Risk_score"]

            mu_low=self.trapezio(

                np.array([x]),

                *self.definicoes["Low Risk"]

            )[0]

            mu_med=self.trapezio(

                np.array([x]),

                *self.definicoes["Medium Risk"]

            )[0]

            mu_high=self.trapezio(

                np.array([x]),

                *self.definicoes["High Risk"]

            )[0]

            soma=mu_low+mu_med+mu_high

            if soma>0:

                mu_low/=soma

                mu_med/=soma

                mu_high/=soma

            mus={

                "Low Risk":mu_low,

                "Medium Risk":mu_med,

                "High Risk":mu_high

            }

            classe=max(mus,key=mus.get)

            resultados.append({

                "Risk_score":x,

                "Low Risk":mu_low,

                "Medium Risk":mu_med,

                "High Risk":mu_high,

                "classificacao":classe,

                "sum_mu":mu_low+mu_med+mu_high

            })

        self.resultados=pd.DataFrame(resultados)

        return self.resultados

    ##########################################################
    # Plot
    ##########################################################

    def plotar(self):

        x=np.linspace(0,1,500)

        plt.figure(figsize=(9,5))

        cores={

            "Low Risk":"green",

            "Medium Risk":"orange",

            "High Risk":"red"

        }

        for nome,param in self.definicoes.items():

            y=self.trapezio(

                x,

                *param

            )

            plt.plot(

                x,

                y,

                lw=3,

                label=nome,

                color=cores[nome]

            )

            plt.fill_between(

                x,

                y,

                alpha=.20,

                color=cores[nome]

            )

        plt.grid(alpha=.3)

        plt.xlim(0,1)

        plt.ylim(0,1.05)

        plt.xlabel("Risk Score")

        plt.ylabel("Membership")

        plt.legend()

        plt.show()
    
    def plot_partition_of_unity(self):

        x = np.linspace(0,1,1000)
    
        mu_low = self.trapezio(
            x,
            *self.definicoes["Low Risk"]
        )
    
        mu_med = self.trapezio(
            x,
            *self.definicoes["Medium Risk"]
        )
    
        mu_high = self.trapezio(
            x,
            *self.definicoes["High Risk"]
        )
    
        soma = mu_low + mu_med + mu_high
    
        # normalização igual à usada na classificação
        soma_norm = soma / np.maximum(soma,1e-9)
    
        plt.figure(figsize=(8,4))
    
        plt.plot(
            x,
            soma_norm,
            color="black",
            linewidth=2
        )
    
        plt.axhline(
            1,
            color="red",
            linestyle="--",
            label="Σ μ = 1"
        )
    
        plt.xlabel("Risk Score")
    
        plt.ylabel("Σ μ")
    
        plt.title("Partition of Unity")
    
        plt.ylim(0.95,1.05)
    
        plt.grid(alpha=.3)
    
        plt.legend()
    
        plt.show()
       
#%%
# Criar e treinar classificador
fuzzy = RiskFuzzyClassifier()
#%%
df_norm = fuzzy.preparar_dados(risk_map)
#%%
resultados = fuzzy.classificar()

#%%
print(resultados["sum_mu"].describe())
#%%
erro = np.abs(resultados["sum_mu"] - 1)

print("Erro máximo:", erro.max())

print("Erro médio:", erro.mean())

#%%
fuzzy.plotar()

#%%

risk_map = pd.concat(

    [

        risk_map,

        resultados

    ],

    axis=1

)

#%%
#classificador.plotar_trapezio_1d()

#%%
#classificador.plotar_interseccoes()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
