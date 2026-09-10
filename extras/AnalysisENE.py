# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 13:22:04 2025

@author: Natalia
"""

import os
import platform
import pandas as pd


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

caminho_tabela1 = diretorio_atual+barra + \
    volta_nivel+barra+'entradas'+barra+'FlorianopolisConsumoPerdas2.csv'
    
caminho_tabela2 = diretorio_atual+barra + \
        volta_nivel+barra+'entradas'+barra+'Consumo_PNotaveis.csv'

caminho_tabela3 = diretorio_atual+barra + \
        volta_nivel+barra+'entradas'+barra+'Consumo_TransformadoresMT.csv'
        
caminho_tabela4 = diretorio_atual+barra + \
        volta_nivel+barra+'entradas'+barra+'UCALOC.csv'
    
#%%   
dados_todos_1 = pd.read_csv(caminho_tabela1, sep=',')

dados_todos_2 = pd.read_csv(caminho_tabela2, sep=';',decimal=',')

dados_todos_3 = pd.read_csv(caminho_tabela3, sep=';',decimal=',')

dados_todos_4 = pd.read_csv(caminho_tabela3, sep=';',decimal=',')

#%%
df2 =  dados_todos_2.groupby(['UNI_TR_AT', 'PN_CON']).agg({'ENE_01':'sum', 'ENE_02':'sum', 'ENE_03':'sum', 'ENE_04':'sum', 'ENE_05':'sum', 
                                                           'ENE_06':'sum', 'ENE_07':'sum','ENE_08':'sum', 'ENE_09':'sum', 'ENE_10':'sum', 
                                                           'ENE_11':'sum', 'ENE_12':'sum'})

#%%

df3 =  df2.groupby('UNI_TR_AT').agg({'ENE_01':'sum', 'ENE_02':'sum', 'ENE_03':'sum', 'ENE_04':'sum', 'ENE_05':'sum', 
                                                           'ENE_06':'sum', 'ENE_07':'sum','ENE_08':'sum', 'ENE_09':'sum', 'ENE_10':'sum', 
                                                           'ENE_11':'sum', 'ENE_12':'sum'}).reset_index()

#%%
df4 =  dados_todos_1.groupby(['UNI_TR_AT']).agg({'ENE_01':'sum', 'ENE_02':'sum', 'ENE_03':'sum', 'ENE_04':'sum', 'ENE_05':'sum', 
                                                           'ENE_06':'sum', 'ENE_07':'sum','ENE_08':'sum', 'ENE_09':'sum', 'ENE_10':'sum', 
                                                           'ENE_11':'sum', 'ENE_12':'sum'}).reset_index()

#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
col_list_Total = ['ENE_01','ENE_02','ENE_03','ENE_04','ENE_05','ENE_06','ENE_07','ENE_08','ENE_09','ENE_10','ENE_11', 'ENE_12']

#%%

# for spot, group in df4.set_index('UNI_TR_AT', append=True).groupby(level='UNI_TR_AT'):
#     df_teste = group  
#     df_teste = df_teste.transpose()
#     df_teste.plot(kind = 'bar', figsize = (12,6), align='center')
#     plt.xticks(rotation=30)
#     plt.savefig('C:\Pos\Run-py-Figures\Plots\{}-TRAFO.png'.format(spot))
    
#%%
csfont = {'fontname':'Times New Roman'}
plt.rc('font',family='Times New Roman')
#%%


for spot, group in df4.set_index('UNI_TR_AT', append=True).groupby(level='UNI_TR_AT'):
    for spot2, group2 in df3.set_index('UNI_TR_AT', append=True).groupby(level='UNI_TR_AT'):
            if spot2 == spot:
                df_teste = group
                df_teste2 = group2
                df_teste = df_teste.transpose()
                df_teste2 = df_teste2.transpose()
                df_list = [df_teste, df_teste2]
                ax = None
                for df in df_list: 
                    ax = df.plot(ax=ax)
                    plt.xticks(np.arange(0, 12, 1.0))
                    #plt.xticks(rotation=30)
                    plt.tick_params(axis='both', which='major', labelsize=10,rotation=30)
                    ax.set_ylabel("Energy in kWh",**csfont,fontsize=10)
                    ax.set_xlabel("Months",**csfont,fontsize=10)
                    ax.legend(['Consumed', 'Billed'])
                    plt.tight_layout()
                    plt.grid()
                    plt.savefig('C:\Pos\Run-py-Figures\Plots\{}-BilingVsTRAFO.png'.format(spot))
                    
                df_teste.plot(kind = 'bar', figsize = (12,6), align='center')
                df_teste2.plot(kind = 'bar', figsize = (12,6), align='center')
            plt.xticks(rotation=30)
            plt.savefig('C:\Pos\Run-py-Figures\Plots\{}-TRAFO.png'.format(spot))

#%%
contagem_trafosMT = dados_todos_3["UNI_TR_MT"].value_counts()

contagem_trafosAT = dados_todos_2["UNI_TR_AT"].value_counts()

contagem_PN = dados_todos_2["PN_CON"].value_counts()

contagem_Feeder = dados_todos_2["CTMT"].value_counts()

contagem_SUB = dados_todos_1["SUB"].value_counts()