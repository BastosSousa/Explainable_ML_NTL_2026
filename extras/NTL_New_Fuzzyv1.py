# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 15:21:37 2026

@author: Natalia
"""

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

caminho_dados_final2 = diretorio_atual+barra + \
    volta_nivel+barra+'saidas'+barra+'data_finalFuzzy2.csv'

#%%

fuzzy_input  = pd.read_csv(caminho_dados_final2,sep=',')

#%%
numeric_cols = fuzzy_input.select_dtypes(
    include=np.number
).columns

#%%
fuzzy_memberships = {}

numeric_cols = [
    c for c in fuzzy_input.columns
    if c != "h3_cell"
]

for col in numeric_cols:

    vals = fuzzy_input[col].dropna()

    q10,q25,q50,q75,q90 = np.percentile(
        vals,
        [10,25,50,75,90]
    )

    fuzzy_memberships[col] = {

        "low":[
            vals.min(),
            vals.min(),
            q25,
            q50
        ],

        "medium":[
            q25,
            q50,
            q50,
            q75
        ],

        "high":[
            q50,
            q75,
            q90,
            vals.max()
        ]
    }
    
#%%
for k,v in fuzzy_memberships.items():

    print("\n",k)

    print(v)