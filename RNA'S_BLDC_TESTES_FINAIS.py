# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 16:29:12 2020

@author: luanpesquisas
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#----------------------- data processing currents ----------------------------

arquivo = open('IAdatame5.txt','r')
lista = arquivo.readlines()  # readlinesssssss
arquivo.close()


corrente_a = []
corrente_a1 = []
corrente_a2 = []
corrente_b = []
corrente_b1 = []
corrente_b2 = []
corrente_c = []
corrente_c1 = []
corrente_c2 = []

cont = 0

for i in lista: #current list
    cont += 1
    if cont < 101073:
        corrente_a.append(float(i))
        if cont > 1:
            corrente_a1.append(float(i))
        if cont > 2:
            corrente_a2.append(float(i))
    elif 101074 < cont <= 202146:
        corrente_b.append(float(i))
    elif cont > 202147:
        corrente_c.append(float(i))

corrente_a1.insert(101072,0)
corrente_a2.insert(101071,0)
corrente_a2.insert(101072,0)
#print(corrente_a)
#print(corrente_b)
#print(corrente_c)

time = range(1,101073,1)

#----- plotagem grafica correntes -------

fig, ax1 = plt.subplots()
ax1.plot(time, corrente_a, label = 'current A')
ax1.legend()

fig1 , current_b = plt.subplots()
current_b.plot(time, corrente_b, label = 'current B')
current_b.legend()

fig2, current_c = plt.subplots()
current_c.plot(time, corrente_c, label = 'current C')
current_c.legend()

#--------------------- data processing EA -----------------------------------

arquivo1 = open('Eadatame5.txt','r')
lista1 = arquivo1.readlines()
arquivo1.close()

Ea = []
Ea1 = []
Ea2 = []
Eb = []
Ec = []
cont1 = 0


for j in lista1: #lista de Ea
    cont1 += 1
    if cont1 < 101073:
        Ea.append(float(j))
        if cont1 > 1:
            Ea1.append(float(j))
        if cont1 > 2:
            Ea2.append(float(j))
    elif 101074 < cont1 <= 202146:
        Eb.append(float(j))
    elif cont1 > 202147:
        Ec.append(float(j))

Ea1.insert(101071,0)
Ea2.insert(101070,0)
Ea2.insert(101071,0)
#print(Ea)
#print(Eb)
#print(Ec)

#----- plotagem grafica Ea -------
        
fig3, Eaa = plt.subplots()
Eaa.plot(time,Ea, label = "Ea")
Eaa.legend()

fig4, Eab = plt.subplots()
Eab.plot(time,Eb, label = "Eb")
Eab.legend()

fig5, Eac = plt.subplots()
Eac.plot(time[39700:40000],Ec[39700:40000], label = "Ec")
Eac.legend()

#--------------------- data processing speed ------------------------------

arquivo2 = open('speeddatame5.txt','r')
lista2 = arquivo2.readlines()
arquivo2.close()

ent = []
speed = []
speed1 = []
speed2 = []
cont2 = 0
soma=[]
total = 0
num = 0
speedm = []
j = 0


for k in lista2:
    ent.append(float(k))
        

for i in ent:
    j=0
    num += 1
    soma.append(i)
    if num == 100:
        soma1 = sum(soma)
        total = (soma1 / 100)
        while (j < 100):
            speedm.append(total)
            j += 1
            num = 0
            soma.clear()

for k in speedm:
    cont2 += 1
    if cont2 < 101073:
        speed.append(float(k))
        if cont2 > 1:
            speed1.append(float(k))
        if cont2 > 2:
            speed2.append(float(k))

speed1.insert(101071,0)
speed2.insert(101070,0)
speed2.insert(101071,0)

veloc = speed[97750]
veloc_max = max(speed)
#----- plotagem grafica speed -------
        
fig6, speedfig = plt.subplots()
speedfig.plot(time[90000:101000],speedm[90000:], label = 'speed')
speedfig.legend()

#-------------------- data processing teta ---------------------------------

arquivo3 = open('tetadatame5.txt','r')
lista3 = arquivo3.readlines()
arquivo3.close()

tetat = []
tetatt = []
teta = []
teta1= []
teta2 = []
cont3 = 0

for l in lista3:
    tetat.append(float(l))

for i in tetat:  #comutação
    if i > -180 and i <= -120:
        i = -180
        tetatt.append(i)
    elif i > -120 and i <= -60:
        i = -120
        tetatt.append(i)
    elif i > -60 and i <= 0:
        i = -60
        tetatt.append(i)
    elif i > 0 and i <= 60:
        i = 0
        tetatt.append(i)
    elif i > 60 and i <= 120:
        i = 60
        tetatt.append(i)
    elif i > 120 and i <= 180:
        i = 120
        tetatt.append(i)

for l in tetatt:
    cont3 += 1
    if cont3 < 101073:
        teta.append(float(l))
        if cont3 > 1:
            teta1.append(float(l))
        if cont3 > 2:
            teta2.append(float(l))

teta1.insert(101071,0)
teta2.insert(101070,0)
teta2.insert(101071,0)



#----- plotagem grafica teta -------
        
fig7, tetafig = plt.subplots()
tetafig.plot(time[5000:10000], teta1[5000:10000], label = 'teta')
tetafig.legend()



# ------------ modulando tabelas(arrays) de entradas e saidas---------------

# criando um dataframe de entrada de dados com todas as correntes, tensões contra-eletromotriz e velocidade

inputdata1 = pd.DataFrame(data = (corrente_a[0:101000],corrente_a1[0:101000], corrente_a2[0:101000],Ea[0:101000],
                                  Ea1[0:101000],Ea2[0:101000],speed[0:101000],speed1[0:101000],speed2[0:101000]), 
                          index = ('corrente A','corrente A [k-1]','corrente A [k-2]','Ea','Ea [k-1]','Ea [k-2]','Speed','Speed [k-1]','Speed [k-2]'))

# transposta do in dataframe

inputdata = inputdata1.T
#print(inputdata2.head(10))

val_nulos_in = inputdata.isnull().values.any() #verificando se existe valores nulos no dataframe

if val_nulos_in == False:
    print("Não existem valores zeros no seu input data")
else:
    print("existem valores zeros no seu input data")
    
# transformando o dataframe de entrada em numpy array 

#inputdata = inputdata2.values
#print(inputdata)

# criando um dataframe de saida contendo os valores de teta

outputdata = pd.DataFrame(data = (teta[0:101000]), columns = ['teta1'])

#outputdata1 = pd.DataFrame(data = (teta[0:101000],teta1[0:101000],teta2[0:101000]), index = ['teta','teta1','teta2'])

#outputdata = outputdata1.T

val_nulos_out = outputdata.isnull().values.any() #verificando se existe valores nulos no dataframe

if val_nulos_out == False:
    print("Não existem valores zeros no seu output data")
else:
    print("existem valores zeros no seu output data")
    

# transformando o dataframe de saida em numpy array 

#outputdata = outputdata1.values
#print(outputdata)

#--------------------------- CORRELAÇÃO ENTRE AS VARIAVEIS -----------------

# correlação entre as variaveis preditoras 
# +1 = forte correlação positiva
# 0 = nenhuma correlação
# -1 = forte correlação negativa

def plot_corr(inputdata, size = 10):
    corr = inputdata.corr()
    fig, ax = plt.subplots(figsize = (size,size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns)                                                              
    plt.yticks(range(len(corr.columns)), corr.columns)
    
plot_corr(inputdata)  # imprimindo como uma imagem de correlação

print(inputdata.corr())  # imprimiindo como uma tabela de correlação

# estudar correlação e causalidade


#%%-------------- SELECIONANDO VARIAVEIS PREDITORAS -------------------------
#------------------------ scikit - learn -----------------------------------
#------------------- IMPLEMENTANDO OS REGRESSORES -------------------------

from sklearn.model_selection import train_test_split

atributos = ['corrente A','corrente A [k-1]','corrente A [k-2]','Ea','Ea [k-1]','Ea [k-2]','Speed','Speed [k-1]','Speed [k-2]']

atrib_prev = ['teta1']

inputdata.to_excel('inputdata.xlsx', index=False)
outputdata.to_excel('outputdata.xlsx')

X = inputdata[atributos].values
Y = outputdata[atrib_prev].values

print(X,Y)

print('exatidão teste (accuracy): {0:.4f}'.format(metrics.r2_score(y_test, SGD_predict_test)))

#%% ------------------ implemento da RNA ---------------------------------
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

x_train = X[:70000]
x_test = X[70000:]
y_train = Y[:70000]
y_test = Y[70000:]

modelo1 = make_pipeline(StandardScaler(), MLPRegressor())
modelo2 = modelo1.fit(x_train, y_train.ravel())

from sklearn import metrics
T = range(0, 30300, 1)



# verificando a exatidão dos dados de treinamento
SGD_predict_train = modelo2.predict(x_train)
print('exatidão treinamento (accuracy): {0:.4f}'.format(metrics.r2_score(y_train, SGD_predict_train)))

#verificando a exatidão dos dados de teste
SGD_predict_test = modelo2.predict(x_test)
print('exatidão teste (accuracy): {0:.4f}'.format(metrics.r2_score(y_test, SGD_predict_test)))

predictRNA = pd.DataFrame(SGD_predict_test, columns=['RNA'])
predictRNA.to_excel('predictRNA.xlsx')

predictRNA = pd.DataFrame(time, columns=['time'])
predictRNA.to_excel('time.xlsx')

#%%
veloc = speed[28000]
veloc_max = max(speed)
print(veloc)
T1 = range(0, 31000, 1)

fig15, speedfig = plt.subplots()
speedfig.plot(time[92500:95500],speedm[92500:95500], label = 'speed')
speedfig.legend()

plt.subplot(2,1,1)
plt.plot(T1[22500:25500], y_test[22500:25500], label = 'dados motor')
plt.ylabel('Posição Angular')
plt.xlabel('Amostragens')
plt.title('Comparativo do motor e da RNA em 600 rpm')
plt.legend()

plt.subplot(2,1,2)
plt.ylabel('Posição Angular')
plt.xlabel('Amostragens')
plt.plot(T1[22500:25500], SGD_predict_test[22500:25500], label = 'RNA')
plt.legend()









#%%
#----------------------- data processing currents ----------------------------

arquivo = open('IAdatame7.txt','r')
lista = arquivo.readlines()  # readlinesssssss
arquivo.close()


corrente_a = []
corrente_a1 = []
corrente_a2 = []
corrente_b = []
corrente_c = []


cont = 0

for i in lista: #current list
    cont += 1
    if cont < 17209:
        corrente_a.append(float(i))
        if cont > 1:
            corrente_a1.append(float(i))
        if cont > 2:
            corrente_a2.append(float(i))
    elif 17209 < cont <= 34418:
        corrente_b.append(float(i))
    elif cont > 34418:
        corrente_c.append(float(i))

corrente_a1.insert(101072,0)
corrente_a2.insert(101071,0)
corrente_a2.insert(101072,0)
#print(corrente_a)
#print(corrente_b)
#print(corrente_c)

time = range(1,17209,1)

#----- plotagem grafica correntes -------

fig, ax1 = plt.subplots()
ax1.plot(time, corrente_a, label = 'current A')
ax1.legend()

# fig1 , current_b = plt.subplots()
# current_b.plot(time, corrente_b, label = 'current B')
# current_b.legend()

# fig2, current_c = plt.subplots()
# current_c.plot(time, corrente_c, label = 'current C')
# current_c.legend()

#--------------------- data processing EA -----------------------------------

arquivo1 = open('Eadatame7.txt','r')
lista1 = arquivo1.readlines()
arquivo1.close()

Ea = []
Ea1 = []
Ea2 = []
Eb = []
Ec = []
cont1 = 0


for j in lista1: #lista de Ea
    cont1 += 1
    if cont1 < 17209:
        Ea.append(float(j))
        if cont1 > 1:
            Ea1.append(float(j))
        if cont1 > 2:
            Ea2.append(float(j))
    elif 17209 < cont1 <= 34418:
        Eb.append(float(j))
    elif cont1 > 34418:
        Ec.append(float(j))

Ea1.insert(101071,0)
Ea2.insert(101070,0)
Ea2.insert(101071,0)
#print(Ea)
#print(Eb)
#print(Ec)

#----- plotagem grafica Ea -------
        
fig3, Eaa = plt.subplots()
Eaa.plot(time,Ea, label = "Ea")
Eaa.legend()

# fig4, Eab = plt.subplots()
# Eab.plot(time,Eb, label = "Eb")
# Eab.legend()

# fig5, Eac = plt.subplots()
# Eac.plot(time[39700:40000],Ec[39700:40000], label = "Ec")
# Eac.legend()

#--------------------- data processing speed ------------------------------

arquivo2 = open('speeddatame7.txt','r')
lista2 = arquivo2.readlines()
arquivo2.close()

ent = []
speed = []
speed1 = []
speed2 = []
cont2 = 0
soma=[]
total = 0
num = 0
speedm = []
j = 0


for k in lista2:
    ent.append(float(k))
        

for i in ent:
    j=0
    num += 1
    soma.append(i)
    if num == 100:
        soma1 = sum(soma)
        total = (soma1 / 100)
        while (j < 100):
            speedm.append(total)
            j += 1
            num = 0
            soma.clear()

for k in speedm:
    cont2 += 1
    if cont2 < 17209:
        speed.append(float(k))
        if cont2 > 1:
            speed1.append(float(k))
        if cont2 > 2:
            speed2.append(float(k))

speed1.insert(101071,0)
speed2.insert(101070,0)
speed2.insert(101071,0)

veloc = speed[10000]
veloc_max = max(speed)
#----- plotagem grafica speed -------
        
fig6, speedfig = plt.subplots()
speedfig.plot(time[2700:3500],speedm[2700:3500], label = 'speed')
speedfig.legend()

#-------------------- data processing teta ---------------------------------

arquivo3 = open('tetadatame7.txt','r')
lista3 = arquivo3.readlines()
arquivo3.close()

tetat = []
tetatt = []
teta = []
teta1= []
teta2 = []
cont3 = 0

for l in lista3:
    tetat.append(float(l))

for i in tetat:  #comutação
    if i > -180 and i <= -120:
        i = -180
        tetatt.append(i)
    elif i > -120 and i <= -60:
        i = -120
        tetatt.append(i)
    elif i > -60 and i <= 0:
        i = -60
        tetatt.append(i)
    elif i > 0 and i <= 60:
        i = 0
        tetatt.append(i)
    elif i > 60 and i <= 120:
        i = 60
        tetatt.append(i)
    elif i > 120 and i <= 180:
        i = 120
        tetatt.append(i)

for l in tetatt:
    cont3 += 1
    if cont3 < 17209:
        teta.append(float(l))
        if cont3 > 1:
            teta1.append(float(l))
        if cont3 > 2:
            teta2.append(float(l))

teta1.insert(101071,0)
teta2.insert(101070,0)
teta2.insert(101071,0)



#----- plotagem grafica teta -------
        
fig7, tetafig = plt.subplots()
tetafig.plot(time[5500:6000], teta1[5500:6000], label = 'teta')
tetafig.legend()



# ------------ modulando tabelas(arrays) de entradas e saidas---------------

# criando um dataframe de entrada de dados com todas as correntes, tensões contra-eletromotriz e velocidade

inputdata1 = pd.DataFrame(data = (corrente_a[0:17200],corrente_a1[0:17200], corrente_a2[0:17200],Ea[0:17200],
                                  Ea1[0:17200],Ea2[0:17200],speed[0:17200],speed1[0:17200],speed2[0:17200]), 
                          index = ('corrente A','corrente A [k-1]','corrente A [k-2]','Ea','Ea [k-1]','Ea [k-2]','Speed','Speed [k-1]','Speed [k-2]'))

# transposta do in dataframe

inputdata = inputdata1.T
#print(inputdata2.head(10))

val_nulos_in = inputdata.isnull().values.any() #verificando se existe valores nulos no dataframe

if val_nulos_in == False:
    print("Não existem valores zeros no seu input data")
else:
    print("existem valores zeros no seu input data")

# criando um dataframe de saida contendo os valores de teta

outputdata = pd.DataFrame(data = (teta[0:17200]), columns = ['teta1'])


val_nulos_out = outputdata.isnull().values.any() #verificando se existe valores nulos no dataframe

if val_nulos_out == False:
    print("Não existem valores zeros no seu output data")
else:
    print("existem valores zeros no seu output data")
    
#--------------------------- CORRELAÇÃO ENTRE AS VARIAVEIS -----------------

# correlação entre as variaveis preditoras 
# +1 = forte correlação positiva
# 0 = nenhuma correlação
# -1 = forte correlação negativa

def plot_corr(inputdata, size = 10):
    corr = inputdata.corr()
    fig, ax = plt.subplots(figsize = (size,size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns)                                                              
    plt.yticks(range(len(corr.columns)), corr.columns)
    
plot_corr(inputdata)  # imprimindo como uma imagem de correlação

print(inputdata.corr())  # imprimiindo como uma tabela de correlação

# estudar correlação e causalidade


#%%-------------- SELECIONANDO VARIAVEIS PREDITORAS -------------------------
#------------------------ scikit - learn -----------------------------------
#------------------- IMPLEMENTANDO OS REGRESSORES -------------------------

from sklearn.model_selection import train_test_split

atributos = ['corrente A','corrente A [k-1]','corrente A [k-2]','Ea','Ea [k-1]','Ea [k-2]','Speed','Speed [k-1]','Speed [k-2]']

atrib_prev = ['teta1']

inputdata.to_excel('inputdata.xlsx', index=False)
outputdata.to_excel('outputdata.xlsx')

X = inputdata[atributos].values
Y = outputdata[atrib_prev].values

print(X,Y)


#%% ------------------ implemento da RNA ---------------------------------
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm, linear_model, neighbors, gaussian_process

# x_train = X[:12040]
# x_test = X[:12040]
# y_train = Y[:12040]
# y_test = Y[:12040]

#modelo1 = make_pipeline(StandardScaler(), svm.NuSVR()) #0.81

#modelo1 = make_pipeline(StandardScaler(), MLPRegressor())
#modelo2 = modelo1.fit(x_train, y_train.ravel())

from sklearn import metrics
T1 = range(0, 31000, 1)



# verificando a exatidão dos dados de treinamento
# SGD_predict_train = modelo1.predict(x_train)
# print('exatidão treinamento (accuracy): {0:.4f}'.format(metrics.r2_score(y_train, SGD_predict_train)))

#verificando a exatidão dos dados de teste
#SGD_predict_test = modelo1.predict(X)
print('exatidão teste (accuracy): {0:.4f}'.format(metrics.r2_score(Y, SGD_predict_test[:17200])))

# predictRNA = pd.DataFrame(SGD_predict_test, columns=['RNA'])
# predictRNA.to_excel('predictRNA.xlsx')

# predictRNA = pd.DataFrame(time, columns=['time'])
# predictRNA.to_excel('time.xlsx')

#%%

veloc = speed[450]
veloc_max = max(speed)
print(veloc)
T1 = range(0, 31000, 1)
plt.subplot(2,1,1)
plt.plot(T1[4000:6000], Y[4000:6000], label = 'dados motor')
plt.ylabel('Posição Angular')
plt.xlabel('Amostragens')
plt.title('Comparativo do motor e da RNA em 2000 rpm')
plt.legend()

plt.subplot(2,1,2)
plt.ylabel('Posição Angular')
plt.xlabel('Amostragens')
plt.plot(T1[4000:6000], SGD_predict_test[4000:6000], label = 'RNA')
plt.legend()