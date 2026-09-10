'''
ENTRADA DE DADOS:
- entrada_MI: Arquivo que contém codigo|municipio|bairro|situação|tiposetor|domicilios| VARIÁVEIS SOCIOECONMICAS ...| Perdas do setor censitário

    Pode ser simplificado com infos iniciais. Deve mudar a linha com "dados", o número 6 de início de aquisição de dados

VARIÁVEIS DE RESULTADO:
-mi_global: resultado de mutual information geral e normalizado para todos os dados.
-matrizresult: Média dos indicadores de perdas dos valores máximos e mínimos das variaveis socioeconomicas.
-mi_simpl: resultado de mutual information geral e normalizado para as simplificações de informação predominante para cada variável socioeconomica 
'''

import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

##################          ENTRADA DE DADOS          ##################################################
#Carrega os dados para Mutual Information
base=np.array(pd.read_excel(r'C:/celesc/entradas/entrada_MI.xlsx',sheet_name='Planilha1',header=None).iloc[:,:])

#Separa os dados em perdas dados socioeconomicos
perdas=base[:,base.shape[1]-1].reshape(-1,1) #Valor de Perdas
dados=base[:,6:base.shape[1]-1] #Valor das variáveis socioeconomicas


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
compara=np.array([0,0,0,0,1,1,0,1,1,1,1,1,1,1,1,0,0]).reshape(1,-1)

###################           PROCESSAMENTO       ######################################################

'''
CALCULO DO MUTUAL INFORMATION COM TODOS OS DADOS
'''
#Calculo do Mutual Information
mi_global=np.append(np.array(dados[0,:]).reshape(1,-1),np.zeros([2,dados.shape[1]]).reshape(2,-1),axis=0)

for i in range(dados.shape[1]):
    mi_global[1,i]=mutual_info_score(perdas[1:,0],dados[1:,i])
    mi_global[2,i]=normalized_mutual_info_score(perdas[1:,0],dados[1:,i])


'''
CALCULO DA MEDIA DE MUTUAL INFORMATION DOS MAIORES E MENORES VALORES PERCENTUAIS
A VARIAVEL "VARIAÇÃO" É DEFINIDA PARA INDICAR QUAL O PERCENTUAL DE MAIORES E MENORES VALORES SERÃO UTILIZADOS PARA MEDIA

SE VARIACAO=0.01, INDICA QUE OS 1% MAIORES E MENORES CONJUNTOS TERÃO A MÉDIA REALIZADA
SE VARIACAO=0.1, INDICA QUE OS 10% MAIORES E MENORES CONJUNTOS TERÃO A MÉDIA REALIZADA
'''

#Monta um dicionário de dados e organiza em ordem crescente
dicdados={}
for i in range(dados.shape[1]):
    dicdados[i]=np.append(dados[1:,i].reshape(-1,1),perdas[1:,0].reshape(-1,1),axis=1)

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
matrizresult=maxmin(dicdados,dados,variacao)


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
    lim=compara[0,i]
    for j in range(aux.shape[0]):
        if (aux[j,0]-lim)!=0:
            aux2=np.append(aux2,aux[j,:].reshape(1,2),axis=0)
    aux2=aux2[1:,:]
    dicresul[i]=aux2
        
      
#Calculo do Mutual Information
mi_simpl=np.append(np.array(dados[0,:]).reshape(1,-1),np.zeros([2,dados.shape[1]]).reshape(2,-1),axis=0)

for i in range(len(dicresul)):
    mi_simpl[1,i]=mutual_info_score(dicresul[i][:,1],dicresul[i][:,0])
    mi_simpl[2,i]=normalized_mutual_info_score(dicresul[i][:,1],dicresul[i][:,0])
    
    

x=[4,3,1,0,9,7,9,2,1,4,3]
y=[4,2,6,10,3,7,2,0,6,7,5]

teste1=mutual_info_score(x,y)
teste2=normalized_mutual_info_score(x,y)