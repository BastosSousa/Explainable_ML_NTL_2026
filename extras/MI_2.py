import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

# %%
##################          ENTRADA DE DADOS          ##################################################
# Carrega os dados para Mutual Information
# base = np.array(pd.read_csv(r'C:/celesc/saidas/Pontos-interpolados-idw.csv', sep=';',header=None).iloc[:,:-1])
base_ivs = pd.read_excel(r'C:/celesc/entradas/join_ivs_2.xlsx', sheet_name='Planilha1')

base_ibge = pd.read_excel(r'C:/celesc/entradas/join_ibge.xlsx', sheet_name='Planilha1')

# %%
#base_perdas =np.array(pd.read_excel(r'C:\celesc\entrada_MI.xlsx',sheet_name='Planilha1',header=None).iloc[:,:])
base_perdas = pd.read_excel(r'C:\celesc\entrada_MI.xlsx', sheet_name='Planilha1')

# Separa os dados em perdas dados socioeconomicos
base_perdas_1 = base_perdas[['Cod_setor','Perdas']]

#%%
df_merged_ivs = base_ivs.merge(base_perdas_1, on = 'Cod_setor', how = 'left')

df_merged_ibge = base_ibge.merge(base_perdas_1, on = 'Cod_setor', how = 'left')

#%%
df_merged_ivs.dropna(inplace=True)

df_merged_ibge.dropna(inplace=True)

#%%

df_merged_ivs.to_excel(r'C:/celesc/entradas/entrada_MI_ivs.xlsx', sheet_name='Planilha1', index=False)

df_merged_ibge.to_excel(r'C:/celesc/entradas/entrada_MI_ibge.xlsx', sheet_name='Planilha1', index=False)

