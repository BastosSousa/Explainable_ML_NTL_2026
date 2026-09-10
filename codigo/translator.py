# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 11:38:49 2025

@author: Natalia
"""

import pandas as pd
from deep_translator import GoogleTranslator

# Arquivo de entrada e saída
input_file = r'C:\Pos\germancase\INKAR 2024 Indikatorenübersicht.csv'
output_file = r'C:\Pos\germancase\dados_traduzidos.csv'

# Carregar CSV
df = pd.read_csv(input_file)
#%%
# Escolher a coluna que deseja traduzir
coluna = "Name"   # <- substitua pelo nome da sua coluna

# Traduzir apenas essa coluna (linha por linha)
df[coluna + "_en"] = df[coluna].astype(str).apply(
    lambda x: GoogleTranslator(source="de", target="en").translate(x)
)

#%%
# Salvar resultado
df.to_csv(output_file, index=False)

#%%

input_file = r'C:\Pos\germancase\BBSR_Raumgliederungen_Referenzen_2022.csv'
output_file = r'C:\Pos\germancase\dados_traduzidos2.csv'
# Carregar CSV
df = pd.read_csv(input_file)

# Escolher o índice da linha que deseja traduzir (ex.: a primeira linha = 0)
linha_idx = 0  

#%%
# Traduzir cada célula dessa linha
traduzida = df.iloc[linha_idx].astype(str).apply(
    lambda x: GoogleTranslator(source="de", target="en").translate(x)
)

# Substituir a linha original pela traduzida
df.iloc[linha_idx] = traduzida
#%%

# Salvar o resultado em novo CSV
df.to_csv(output_file, index=False, encoding="utf-8")

