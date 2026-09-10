# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 16:31:28 2024

@author: Natalia Bostos
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

##################          ENTRADA DE DADOS          ##################################################
#Carrega os dados para Mutual Information
base_ivs = np.array(pd.read_excel(r'C:\celesc\entradas\entrada_MI_ivs.xlsx',sheet_name='Planilha1',header=None).iloc[:,:])

#%%
perdas_ivs = base_ivs[:,base_ivs.shape[1]-1].reshape(-1,1) #Valor de Perdas

dados_ivs = base_ivs[:,2:base_ivs.shape[1]-1] #Valor das variáveis socioeconomicas

compara_ivs = np.random.choice([0, 1], size=80, p=[.3, .7]).reshape(1,-1)

#%%
###################           PROCESSAMENTO       ######################################################

'''
CALCULO DO MUTUAL INFORMATION COM TODOS OS DADOS
'''
#Calculo do Mutual Information
mi_global=np.append(np.array(dados_ivs[0,:]).reshape(1,-1),np.zeros([2,dados_ivs.shape[1]]).reshape(2,-1),axis=0)

for i in range(dados_ivs.shape[1]):
    mi_global[1,i]=mutual_info_score(perdas_ivs[1:,0],dados_ivs[1:,i])
    mi_global[2,i]=normalized_mutual_info_score(perdas_ivs[1:,0],dados_ivs[1:,i])


'''
CALCULO DA MEDIA DE MUTUAL INFORMATION DOS MAIORES E MENORES VALORES PERCENTUAIS
A VARIAVEL "VARIAÇÃO" É DEFINIDA PARA INDICAR QUAL O PERCENTUAL DE MAIORES E MENORES VALORES SERÃO UTILIZADOS PARA MEDIA

SE VARIACAO=0.01, INDICA QUE OS 1% MAIORES E MENORES CONJUNTOS TERÃO A MÉDIA REALIZADA
SE VARIACAO=0.1, INDICA QUE OS 10% MAIORES E MENORES CONJUNTOS TERÃO A MÉDIA REALIZADA
'''

#Monta um dicionário de dados e organiza em ordem crescente
dicdados={}
for i in range(dados_ivs.shape[1]):
    dicdados[i]=np.append(dados_ivs[1:,i].reshape(-1,1),perdas_ivs[1:,0].reshape(-1,1),axis=1)

for i in range(len(dicdados)):
    dicdados[i]=dicdados[i][np.argsort(dicdados[i][:, 0])]
 
del i

#Calcula os maximos e minimos
def maxmin(dicdados,dados,variacao):

    lim=int((dados.shape[0]-1)*variacao)
    
    dicresul={}
    for i in range(len(dicdados)):
        aux=dicdados[i]
        aux2=np.zeros([2,1]).reshape(-1,1)
        men=aux[0:lim,:].reshape(-1,aux.shape[1])
        mai=aux[aux.shape[0]-lim:aux.shape[0],:].reshape(-1,aux.shape[1])
        aux2[0,0]=np.mean(men[:,1])
        aux2[1,0]=np.mean(mai[:,1])
        dicresul[i]=aux2
        
    matrizresult=np.array(['var','menor','maior']).reshape(-1,1)
    for i in range(len(dicresul)):
        matrizresult=np.append(matrizresult,
                               np.append(np.array(dados[0,i]).reshape(1,1),dicresul[i],axis=0),
                               axis=1)
    
    return matrizresult

variacao=0.01
matrizresult=maxmin(dicdados,dados_ivs,variacao)


'''
CALCULO DO MUTUAL INFORMATION COM OS CONJUNTOS DE DADOS SIMPLIFICADOS DE ACORDO COM A VARIÁVEL COMPARA
PREVIAMENTE EXPLICADA. EM UM PRIMEIRO MOMENTO OS DADOS SÃO FILTRADOS EM RELAÇÃO A MAIOR OCORRÊNCIA E
APÓS CALCULADO O MUTUAL INFORMATION.
'''

#Filtro de dados
dicresul={}
for i in range(len(dicdados)):
    aux=dicdados[i]
    aux2=np.zeros([1,2])
    lim=compara_ivs[0,i]
    for j in range(aux.shape[0]):
        if (aux[j,0]-lim)!=0:
            aux2=np.append(aux2,aux[j,:].reshape(1,2),axis=0)
    aux2=aux2[1:,:]
    dicresul[i]=aux2
        
      
#Calculo do Mutual Information
mi_simpl=np.append(np.array(dados_ivs[0,:]).reshape(1,-1),np.zeros([2,dados_ivs.shape[1]]).reshape(2,-1),axis=0)

for i in range(len(dicresul)):
    mi_simpl[1,i]=mutual_info_score(dicresul[i][:,1],dicresul[i][:,0])
    mi_simpl[2,i]=normalized_mutual_info_score(dicresul[i][:,1],dicresul[i][:,0])    
  
df_mi_simpl = pd.DataFrame.from_dict(mi_simpl)
df_mi_global = pd.DataFrame.from_dict(mi_global)   
  
#%%
base_2 = pd.read_excel(r'C:\celesc\entradas\entrada_MI_ivs.xlsx',sheet_name='Planilha1')

#%%
indep_vars = ['ivs', 'ivs_infraestrutura_urbana', 'ivs_capital_humano', 'ivs_renda_e_trabalho', 'idhm', 'idhm_long', 'idhm_educ', 
              'idhm_renda', 'idhm_educ_sub_esc', 'idhm_educ_sub_freq', 't_sem_agua_esgoto', 't_sem_lixo', 't_vulner_mais1h', 't_mort1', 
              't_c0a5_fora', 't_c6a14_fora', 't_m10a17_filho', 't_mchefe_fundin_fmenor', 't_analf_15m', 't_cdom_fundin', 't_p15a24_nada', 
              't_vulner', 't_desocup18m', 't_p18m_fundin_informal', 't_vulner_depende_idosos', 't_atividade10a14', 'espvida', 
              't_pop18m_fundc', 't_pop5a6_escola', 't_pop11a13_ffun', 't_pop15a17_fundc', 't_pop18a20_medioc', 'renda_per_capita', 
              'populacao', 't_fmor5', 't_razdep', 't_fectot', 't_env', 'mchefe_fmenor', 'pop0a1', 'pop1a3', 'pop4', 'pop5', 'pop6', 
              'pop6a10', 'pop6a17', 'pop11a13', 'pop11a14', 'pop12a14', 'pop15m', 'pop15a17', 'pop15a24', 'pop16a18', 'pop18m', 
              'pop18a20', 'pop18a24', 'pop19a21', 'pop25m', 'pop65m', 'pea10a14', 'pea15a17', 'pea18m', 't_eletrica', 't_densidadem2', 
              't_analf_18m', 't_analf_25m', 'rdpc_def_vulner', 't_renda_trab', 'i_gini', 't_carteira_18m', 't_scarteira_18m', 
              't_setorpublico_18m', 't_contapropria_18m', 't_empregador_18m', 't_formal_18m', 't_fundc_ocup18m', 't_medioc_ocup18m', 
              't_supec_ocup18m', 't_renda_todos_trabalhos', 't_nremunerado_18m'] # set independent vars

dep_vars = ['Perdas'] # set dependent vars

#%%
from sklearn.feature_selection import mutual_info_regression as mi_reg

df_mi = pd.DataFrame([mi_reg(base_2[indep_vars], base_2[dep_var]) for dep_var in dep_vars], index = dep_vars, columns = indep_vars).apply(lambda x: x / x.max(), axis = 1)


#%%

from sklearn.feature_selection import SelectKBest

X_clf = base_2.iloc[:,2:82]

y_clf = base_2.iloc[:,82]

#%%
selector = SelectKBest(score_func=mutual_info_regression, k=10).fit_transform(X_clf,y_clf)

#%%

selector2 = SelectKBest(score_func=mutual_info_regression, k=10).fit(X_clf,y_clf)
x_new = selector2.transform(X_clf) # not needed to get the score
scores = selector2.scores_

#%%
pairs = zip(X_clf, scores) # zip with features_list
pairs= sorted(pairs, key=lambda x: x[1], reverse= True)

#%%
newx, newy = zip(*pairs)

#%%
df = pd.DataFrame(list(zip(newy,newx))).set_index(1)

# df.head(20).plot.bar(rot=0)

#%%
# plt.bar(range(len(pairs)), [val[1] for val in pairs], align='center')
# plt.xticks(range(len(pairs)), [val[0] for val in pairs])
# plt.xticks(rotation=45)
# plt.show()

#%%

df_mi_B = df_mi.sort_values(df_mi.last_valid_index(), axis=1,ascending=False)

# df_mi_B.plot.bar(rot=0)

#%%
r = base_2[base_2.columns[2:]].corr()['Perdas'][:]

#%%
base_3 = base_2[['ivs', 'ivs_infraestrutura_urbana', 'ivs_capital_humano', 'ivs_renda_e_trabalho', 'idhm', 'idhm_long', 'idhm_educ', 
              'idhm_renda', 'idhm_educ_sub_esc', 'idhm_educ_sub_freq', 't_sem_agua_esgoto', 't_sem_lixo', 't_vulner_mais1h', 't_mort1', 
              't_c0a5_fora', 't_c6a14_fora', 't_m10a17_filho', 't_mchefe_fundin_fmenor', 't_analf_15m', 't_cdom_fundin', 't_p15a24_nada', 
              't_vulner', 't_desocup18m', 't_p18m_fundin_informal', 't_vulner_depende_idosos', 't_atividade10a14', 'espvida', 
              't_pop18m_fundc', 't_pop5a6_escola', 't_pop11a13_ffun', 't_pop15a17_fundc', 't_pop18a20_medioc', 'renda_per_capita', 
              'populacao', 't_fmor5', 't_razdep', 't_fectot', 't_env', 'mchefe_fmenor', 'pop0a1', 'pop1a3', 'pop4', 'pop5', 'pop6', 
              'pop6a10', 'pop6a17', 'pop11a13', 'pop11a14', 'pop12a14', 'pop15m', 'pop15a17', 'pop15a24', 'pop16a18', 'pop18m', 
              'pop18a20', 'pop18a24', 'pop19a21', 'pop25m', 'pop65m', 'pea10a14', 'pea15a17', 'pea18m', 't_eletrica', 't_densidadem2', 
              't_analf_18m', 't_analf_25m', 'rdpc_def_vulner', 't_renda_trab', 'i_gini', 't_carteira_18m', 't_scarteira_18m', 
              't_setorpublico_18m', 't_contapropria_18m', 't_empregador_18m', 't_formal_18m', 't_fundc_ocup18m', 't_medioc_ocup18m', 
              't_supec_ocup18m', 't_renda_todos_trabalhos', 't_nremunerado_18m','Perdas']]


corr = base_3.corr()

#plt.matshow(corr)
#%%
import plotly.express as px

corr = base_3.corr()
mask= np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
# sns.heatmap(corr,
#             vmax=1, vmin=-1,
#             annot=True, annot_kws={'fontsize':7},
#             mask=mask,
#             cmap=sns.diverging_palette(20,220,as_cmap=True))

#%%
fig = px.imshow(corr, text_auto=True)
fig.show()
fig.write_html("seaborn_plot_ivs.html")

#%%
corr2=corr.style.background_gradient(cmap='coolwarm', axis=None)
a = corr2.to_html(open('my_file_ivs.html', 'w'))


#%%   Recursive Feature Elimination
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# define dataset
lab = preprocessing.LabelEncoder()
y_transformed = lab.fit_transform(y_clf)
# define RFE
rfe = RFE(estimator=LogisticRegression(), n_features_to_select=10)
# fit RFE
rfe.fit(X_clf, y_transformed)


# summarize all features
for i in range(X_clf.shape[1]):
  print('Column: %d, Selected %s, Rank: %.3f' % (i, rfe.support_[i], rfe.ranking_[i]))

#%%
from sklearn.feature_selection import SelectPercentile

def select_features(X_clf,y_transformed):
    global  nomes
    mutual_info = mutual_info_regression(X_clf, y_transformed,random_state=0)
    mutual_info = pd.Series(mutual_info)
    mutual_info.index = X_clf.columns
    mutual_info.sort_values(ascending=False)
    selected_top_columns = SelectPercentile(mutual_info_regression, percentile=25)
    selected_top_columns.fit(X_clf, y_transformed)
    selected_top_columns.get_support()
    nomes = X_clf.columns[selected_top_columns.get_support()]
    
    return
    
 
select_features(X_clf,y_transformed)   
#%%

ag = base_2[nomes[0]]
bg1 = base_2[nomes[1]]
bg2 = base_2[nomes[2]]
bg3 = base_2[nomes[3]]

#%%
new_feature1 = np.add(ag,bg1)
new_feature2 = np.add(ag,bg2)
new_feature3 = np.add(ag,bg3)

#%%
base_2_selecao = base_2[['Cod_setor','UDH_ATLAS',nomes[0],nomes[1],nomes[2],nomes[3]]]

#%%
base_2_selecao_ibge = pd.read_excel(r'C:\celesc\entradas\IBGE_selecao.xlsx',sheet_name='Planilha1')

#%%
base_2_selecao_combi = pd.merge(base_2_selecao_ibge, base_2_selecao, how='left', on=['Cod_setor','UDH_ATLAS'])

#%%
base_2_selecao.to_excel(r'C:/celesc/entradas/IVS_selecao.xlsx', sheet_name='Planilha1', index=False)

#%%
base_2_selecao_combi.to_excel(r'C:/celesc/entradas/IBGE_IVS_selecao.xlsx', sheet_name='Planilha1', index=False)


#%%
base_2['new_1'] =  new_feature1/100
base_2['new_2'] =  new_feature2/100
base_2['new_3'] =  new_feature3/100

#%%

def select_features2(X_clf2,y_transformed):
    
    mutual_info = mutual_info_regression(X_clf2, y_transformed,random_state=0)
    mutual_info = pd.Series(mutual_info)
    mutual_info.index = X_clf2.columns
    mutual_info.sort_values(ascending=False)
    selected_top_columns = SelectPercentile(mutual_info_regression, percentile=15)
    selected_top_columns.fit(X_clf2, y_transformed)
    selected_top_columns.get_support()
    
    nomes2 = X_clf2.columns[selected_top_columns.get_support()]
    
    return nomes2

#%%

X_clf2 = base_2.drop(base_2.columns[[0,1,82]], axis=1)

nomes2 = select_features2(X_clf2,y_transformed) 

#%% plot Recursive Feature Elimination

# rfecv = RFECV(estimator=LogisticRegression(), cv=StratifiedKFold(5, random_state=42, shuffle=True),step=1)
# rfecv.fit(X_clf, y_transformed)

# plt.figure(figsize=(8, 6))
# plt.plot(range(1, len(rfecv.cv_results_['mean_test_score'])+1), rfecv.cv_results_['mean_test_score'])
# plt.grid()
# plt.xticks(range(1, X_clf.shape[1]+1))
# plt.xlabel("Number of Selected Features")
# plt.ylabel("CV Score")
# plt.title("Recursive Feature Elimination (RFE)")
# plt.show()

# print("The optimal number of features: {}".format(rfecv.n_features_))