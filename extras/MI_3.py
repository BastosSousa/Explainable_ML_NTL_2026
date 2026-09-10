'''
ENTRADA DE DADOS:
- entrada_MI: Arquivo que contém codigo|municipio|bairro|situação|tiposetor|domicilios| VARIÁVEIS SOCIOECONMICAS ...| Perdas do setor censitário

    Pode ser simplificado com infos iniciais. Deve mudar a linha com "dados", o número 6 de início de aquisição de dados

VARIÁVEIS DE RESULTADO:
-mi_global: resultado de mutual information geral e normalizado para todos os dados.
-matrizresult: Média dos indicadores de perdas dos valores máximos e mínimos das variaveis socioeconomicas.
-mi_simpl: resultado de mutual information geral e normalizado para as simplificações de informação predominante para cada variável socioeconomica 
'''
from sklearn.feature_selection import SelectPercentile
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
import plotly.express as px
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression as mi_reg
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

##################          ENTRADA DE DADOS          ##################################################
# Carrega os dados para Mutual Information

base_ibge = np.array(pd.read_excel(r'C:\celesc\entradas\entrada_MI_ibge.xlsx',
                     sheet_name='Planilha1', header=None).iloc[:, :])


# %%
# Separa os dados em perdas dados socioeconomicos

# %%
perdas_ibge = base_ibge[:, base_ibge.shape[1] -
                        1].reshape(-1, 1)  # Valor de Perdas

# Valor das variáveis socioeconomicas
dados_ibge = base_ibge[:, 2:base_ibge.shape[1]-1]

# %%
'''
O VETOR "COMPARA" É UM VETOR UTILIZADO PARA FILTRAR AS INFORMAÇÕES QUE MAIS SE REPETEM NO UNIVERSO DE DADOS DE CADA VARIÁVEL
ESSE VETOR DEVE TER O NUMERO DE TERMOS EQUIVALENTE AO NUMERO DE VARIÁVEIS SOCIOECONOMICAS PROCESSADAS, SENDO QUE O VALOR A SER
ALOCADO CORRESPONDE AQUELE VALOR A SER ELIMINADO.

POR EXEMPLO: SE VERIFICA QUE NO UNIVERSO DA VARIAVEL SANEAMENTO, DOS 576 CONJUNTOS, 500 POSSUEM SANEAMENTO PARA TODAS AS RESIDÊNCIAS.
LOGO EM 500 CONJUNTOS ESSA VARIÁVEL É "IGUAL A 1". COM ESSE VETOR, COLOCANDO 1 NA POSIÇÃO DA VARIÁVEL NOS DADOS, SE EXCLUI TODAS OS
CONJUNTOS COM ESSE VALOR, E ENTÃO SE CALCULA O MUTUAL INFORMATION APENAS COM OS 76 CONJUNTOS RESTANTES, A FIM DE AFERIR A RELAÇÃO ENTRE
ESSAS VARIÁVEIS EM CONJUNTOS QUE NÃO POSSUEM SANEAMENTO.

PARA FINS DE MONTAGEM, O VETOR COMPARA DEVE TER O MESMO NÚMERO DE TERMOS EM RELAÇÃO AO NÚMERO DE VARIÁVEIS SOCIOECONOMICAS
CADA POSIÇÃO DEVE SER RESPEITADA EM RELAÇÃO A ORDEM QUE AS VARIÁVEIS FORAM INSERIDAS
EXEMPLO:
    POSIÇÃO 0 - SALÁRIO
    POSIÇAO 1 - NUMERO DE HABITANTES
    ...

'''
compara = np.array([0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1,
                   1, 1, 1, 1, 0, 0]).reshape(1, -1)

# %%
###################           PROCESSAMENTO       ######################################################

'''
CALCULO DO MUTUAL INFORMATION COM TODOS OS DADOS
'''
# Calculo do Mutual Information
mi_global = np.append(np.array(dados_ibge[0, :]).reshape(
    1, -1), np.zeros([2, dados_ibge.shape[1]]).reshape(2, -1), axis=0)

for i in range(dados_ibge.shape[1]):
    mi_global[1, i] = mutual_info_score(perdas_ibge[1:, 0], dados_ibge[1:, i])
    mi_global[2, i] = normalized_mutual_info_score(
        perdas_ibge[1:, 0], dados_ibge[1:, i])


'''
CALCULO DA MEDIA DE MUTUAL INFORMATION DOS MAIORES E MENORES VALORES PERCENTUAIS
A VARIAVEL "VARIAÇÃO" É DEFINIDA PARA INDICAR QUAL O PERCENTUAL DE MAIORES E MENORES VALORES SERÃO UTILIZADOS PARA MEDIA

SE VARIACAO=0.01, INDICA QUE OS 1% MAIORES E MENORES CONJUNTOS TERÃO A MÉDIA REALIZADA
SE VARIACAO=0.1, INDICA QUE OS 10% MAIORES E MENORES CONJUNTOS TERÃO A MÉDIA REALIZADA
'''

# Monta um dicionário de dados e organiza em ordem crescente
dicdados = {}
for i in range(dados_ibge.shape[1]):
    dicdados[i] = np.append(dados_ibge[1:, i].reshape(-1, 1),
                            perdas_ibge[1:, 0].reshape(-1, 1), axis=1)

for i in range(len(dicdados)):
    dicdados[i] = dicdados[i][np.argsort(dicdados[i][:, 0])]

del i

# Calcula os maximos e minimos


def maxmin(dicdados, dados, variacao):

    lim = int((dados.shape[0]-1)*variacao)

    dicresul = {}
    for i in range(len(dicdados)):
        aux = dicdados[i]
        aux2 = np.zeros([2, 1]).reshape(-1, 1)
        men = aux[0:lim, :].reshape(-1, aux.shape[1])
        mai = aux[aux.shape[0]-lim:aux.shape[0], :].reshape(-1, aux.shape[1])
        aux2[0, 0] = np.mean(men[:, 1])
        aux2[1, 0] = np.mean(mai[:, 1])
        dicresul[i] = aux2

    matrizresult = np.array(['var', 'menor', 'maior']).reshape(-1, 1)
    for i in range(len(dicresul)):
        matrizresult = np.append(matrizresult,
                                 np.append(np.array(dados[0, i]).reshape(
                                     1, 1), dicresul[i], axis=0),
                                 axis=1)

    return matrizresult


variacao = 0.01
matrizresult = maxmin(dicdados, dados_ibge, variacao)


'''
CALCULO DO MUTUAL INFORMATION COM OS CONJUNTOS DE DADOS SIMPLIFICADOS DE ACORDO COM A VARIÁVEL COMPARA
PREVIAMENTE EXPLICADA. EM UM PRIMEIRO MOMENTO OS DADOS SÃO FILTRADOS EM RELAÇÃO A MAIOR OCORRÊNCIA E
APÓS CALCULADO O MUTUAL INFORMATION.
'''

# Filtro de dados
dicresul = {}
for i in range(len(dicdados)):
    aux = dicdados[i]
    aux2 = np.zeros([1, 2])
    lim = compara[0, i]
    for j in range(aux.shape[0]):
        if (aux[j, 0]-lim) != 0:
            aux2 = np.append(aux2, aux[j, :].reshape(1, 2), axis=0)
    aux2 = aux2[1:, :]
    dicresul[i] = aux2


# Calculo do Mutual Information
mi_simpl = np.append(np.array(dados_ibge[0, :]).reshape(
    1, -1), np.zeros([2, dados_ibge.shape[1]]).reshape(2, -1), axis=0)

for i in range(len(dicresul)):
    mi_simpl[1, i] = mutual_info_score(dicresul[i][:, 1], dicresul[i][:, 0])
    mi_simpl[2, i] = normalized_mutual_info_score(
        dicresul[i][:, 1], dicresul[i][:, 0])

df_mi_simpl = pd.DataFrame.from_dict(mi_simpl)
df_mi_global = pd.DataFrame.from_dict(mi_global)
# %%
base_2 = pd.read_excel(
    r'C:\celesc\entradas\entrada_MI_ibge.xlsx', sheet_name='Planilha1')


indep_vars = ['aguageral', 'aguadepoço', 'aguaoutras', 'banheiroexclusivo', 'ban_esgoto', 'ban_fossa', 'ban_fossarudi',
              'ban_vala', 'ban_riomar', 'ban_outros', 'semban', 'coletalixo', 'comEE', 'semEE', 'comEE_semmedidor']  # set independent vars

dep_vars = ['Perdas']  # set dependent vars


df_mi = pd.DataFrame([mi_reg(base_2[indep_vars], base_2[dep_var]) for dep_var in dep_vars],
                     index=dep_vars, columns=indep_vars).apply(lambda x: x / x.max(), axis=1)


# %%


X_clf = base_2.iloc[:, 2:17]

y_clf = base_2.iloc[:, 17]

# %%
selector = SelectKBest(score_func=mutual_info_regression,
                       k=15).fit_transform(X_clf, y_clf)

# %%

selector2 = SelectKBest(
    score_func=mutual_info_regression, k=15).fit(X_clf, y_clf)
x_new = selector2.transform(X_clf)  # not needed to get the score
scores = selector2.scores_

# %%
pairs = zip(X_clf, scores)  # zip with features_list
pairs = sorted(pairs, key=lambda x: x[1], reverse=True)

# %%
newx, newy = zip(*pairs)
df = pd.DataFrame(list(zip(newy, newx))).set_index(1)

plt.bar(range(len(pairs)), [val[1] for val in pairs], align='center')
plt.xticks(range(len(pairs)), [val[0] for val in pairs])
plt.xticks(rotation=45)
plt.show()

# %%

df_mi_B = df_mi.sort_values(df_mi.last_valid_index(), axis=1, ascending=False)

df_mi_B.plot.bar(rot=0)

# %%

r = base_2[base_2.columns[2:]].corr()['Perdas'][:]

# %%
plt.scatter(X_clf.iloc[:, 0], y_clf)

# %%
base_3 = base_2[['aguageral', 'aguadepoço', 'aguaoutras', 'banheiroexclusivo', 'ban_esgoto', 'ban_fossa', 'ban_fossarudi',
                 'ban_vala', 'ban_riomar', 'ban_outros', 'semban', 'coletalixo', 'comEE', 'semEE', 'comEE_semmedidor', 'Perdas']]


corr = base_3.corr()

# plt.matshow(corr)
# %%

corr = base_3.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr,
            vmax=1, vmin=-1,
            annot=True, annot_kws={'fontsize': 7},
            mask=mask,
            cmap=sns.diverging_palette(20, 220, as_cmap=True))

plt.savefig(r'C:\celesc\entradas\seaborn_plot.webp')

# %%
fig = px.imshow(corr, text_auto=True)
fig.show()
fig.write_html(r'C:\celesc\entradas\seaborn_plot.html')

# %%
corr2 = corr.style.background_gradient(cmap='coolwarm', axis=None)
a = corr2.to_html(open(r'C:\celesc\entradas\my_file.html', 'w'))

# %%
# define dataset
lab = preprocessing.LabelEncoder()
y_transformed = lab.fit_transform(y_clf)
# define RFE
rfe = RFE(estimator=LogisticRegression(), n_features_to_select=4)
# fit RFE
rfe.fit(X_clf, y_transformed)

# rfecv = RFECV(estimator=LogisticRegression(), cv=StratifiedKFold(5, random_state=42, shuffle=True),step=1)
# rfecv.fit(X_clf, y_transformed)

# summarize all features
for i in range(X_clf.shape[1]):
    print('Column: %d, Selected %s, Rank: %.3f' %
          (i, rfe.support_[i], rfe.ranking_[i]))

# %%


def select_features(X_clf, y_transformed):
    global nomes
    mutual_info = mutual_info_regression(X_clf, y_transformed, random_state=0)
    mutual_info = pd.Series(mutual_info)
    mutual_info.index = X_clf.columns
    mutual_info.sort_values(ascending=False)
    selected_top_columns = SelectPercentile(
        mutual_info_regression, percentile=20)
    selected_top_columns.fit(X_clf, y_transformed)
    selected_top_columns.get_support()
    nomes = X_clf.columns[selected_top_columns.get_support()]

    return


select_features(X_clf, y_transformed)
# %%


ag = base_2[nomes[0]]
bg1 = base_2[nomes[1]]
bg2 = base_2[nomes[2]]
# bg3 = base_2[nomes[3]]

# %%
new_feature1 = np.multiply(ag, bg1)
new_feature2 = np.add(ag, bg2)
# new_feature3 = np.multiply(ag,bg3)

# %%
base_2['saneamento_b'] = new_feature1/1000
base_2['limpeza_b'] = new_feature2/10
# base_2['rede_basica'] =  new_feature3/1000

# %%

base_2_selecao = base_2[['Cod_setor', 'UDH_ATLAS',
                         nomes[0], nomes[1], nomes[2], nomes[3]]].drop_duplicates()

# %%
base_2_selecao.to_excel(
    r'C:/celesc/entradas/IBGE_selecao.xlsx', sheet_name='Planilha1', index=False)

# %%


def select_features2(X_clf2, y_transformed):

    mutual_info = mutual_info_regression(X_clf2, y_transformed)
    mutual_info = pd.Series(mutual_info)
    mutual_info.index = X_clf2.columns
    mutual_info.sort_values(ascending=False)
    selected_top_columns = SelectPercentile(
        mutual_info_regression, percentile=20)
    selected_top_columns.fit(X_clf2, y_transformed)
    selected_top_columns.get_support()

    nomes2 = X_clf2.columns[selected_top_columns.get_support()]

    return nomes2

# %%


X_clf2 = base_2.iloc[:, 1:20]

nomes2 = select_features2(X_clf2, y_transformed)

# %%
# plt.figure(figsize=(8, 6))
# plt.plot(range(1, len(rfecv.cv_results_['mean_test_score'])+1), rfecv.cv_results_['mean_test_score'])
# plt.grid()
# plt.xticks(range(1, X_clf.shape[1]+1))
# plt.xlabel("Number of Selected Features")
# plt.ylabel("CV Score")
# plt.title("Recursive Feature Elimination (RFE)")
# plt.show()

# print("The optimal number of features: {}".format(rfecv.n_features_))

# %%

# from sklearn.decomposition import PCA

# from sklearn.preprocessing import StandardScaler
# std_scaler = StandardScaler()
# scaled_df = std_scaler.fit_transform(base_2)

# pca = PCA(n_components=3)
# pca.fit_transform(scaled_df)

# print(sum(pca.explained_variance_ratio_))
#%%
dados_ibge_artigo = base_2[['aguageral', 'aguadepoço','banheiroexclusivo','ban_esgoto','ban_outros']]

dados_ibge_artigo = dados_ibge_artigo.replace('X', np.nan)

artigo2_describe = dados_ibge_artigo.describe()

boxplot =dados_ibge_artigo.boxplot()
plt.show()



