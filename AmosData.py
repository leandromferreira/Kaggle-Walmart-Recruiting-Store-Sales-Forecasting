# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 10:28:37 2017

@author: ferreira
"""

import numpy as np
import pandas as pd
import seaborn as sns

#Lendo a base de dados 
dfTrain = pd.read_csv('./walmart/train.csv')
dfFeature = pd.read_csv('./walmart/features.csv')
dfTest = pd.read_csv('./walmart/test.csv')
dfStores = pd.read_csv('./walmart/stores.csv')
submission = pd.read_csv('./walmart/sampleSubmission.csv')
   
#Adicionando informacoes sobre as lojas nas bases de traino e test
dfTrainTmp           = pd.merge(dfTrain, dfStores)
dfTestTmp            = pd.merge(dfTest, dfStores)   
  
#Adicionado as features 
train                = pd.merge(dfTrainTmp, dfFeature)
test                 = pd.merge(dfTestTmp, dfFeature)

#Separando data em campos separados Train
train['Year']        = pd.to_datetime(train['Date']).dt.year
train['Month']       = pd.to_datetime(train['Date']).dt.month
train['Day']         = pd.to_datetime(train['Date']).dt.day
train['Days']        = train['Month']*30+train['Day']

train['logSales']    = np.log(4990+train['Weekly_Sales'])

#Mesmo procedimento para test
test['Year']         = pd.to_datetime(test['Date']).dt.year
test['Month']        = pd.to_datetime(test['Date']).dt.month
test['Day']          = pd.to_datetime(test['Date']).dt.day
test['Days']         = test['Month']*30+test['Day']

train_count = train['Store'].count()
test_count = test['Store'].count()
print("Total de instancias no treino: " + str(train_count))
print("Total de instancias em teste " + str(test_count))
print("Total de dados ausentes nas base de treino e de teste")
print("\t MarkDown1:   "+ str(train_count-train['MarkDown1'].count())   +"\t"+ str(test_count-test['MarkDown1'].count()) )
print("\t MarkDown2:   "+ str(train_count-train['MarkDown2'].count())   +"\t"+ str(test_count-test['MarkDown2'].count()) )
print("\t MarkDown3:   "+ str(train_count-train['MarkDown3'].count())   +"\t"+ str(test_count-test['MarkDown3'].count()) )
print("\t MarkDown4:   "+ str(train_count-train['MarkDown4'].count())   +"\t"+ str(test_count-test['MarkDown4'].count()) )
print("\t MarkDown5:   "+ str(train_count-train['MarkDown5'].count())   +"\t"+ str(test_count-test['MarkDown5'].count()) )
print("\t Fuel_Price:  "+ str(train_count-train['Fuel_Price'].count())  +"\t"+ str(test_count-test['Fuel_Price'].count()) )
print("\t CPI:         "+ str(train_count-train['CPI'].count())         +"\t"+ str(test_count-test['CPI'].count()) )
print("\t Unemployment:"+ str(train_count-train['Unemployment'].count())+"\t"+ str(test_count-test['Unemployment'].count()) )

sns.set(style="whitegrid", color_codes=True)
fig = sns.countplot(y="Type", hue="Type", data=train)
fig = fig.get_figure()
fig.savefig('./Graphs/Tipos.png')
fig.clf()

fig = sns.regplot(x="Fuel_Price", y="logSales", data=train)
fig = fig.get_figure()
fig.savefig('./Graphs/Fuel_Price.png')
fig.clf()

fig = sns.regplot(x="Temperature", y="logSales", data=train)
fig = fig.get_figure()
fig.savefig('./Graphs/Temperature.png')
fig.clf()

fig = sns.regplot(x="Size", y="logSales", data=train)
fig = fig.get_figure()
fig.savefig('./Graphs/Size.png')
fig.clf()

train['Type']        = train['Type'].replace('A',1)
train['Type']        = train['Type'].replace('B',2)
train['Type']        = train['Type'].replace('C',3)

fig = sns.regplot(x="Type", y="logSales", data=train)
fig = fig.get_figure()
fig.savefig('./Graphs/Type.png')
fig.clf()

fig = sns.regplot(x="IsHoliday", y="logSales", data=train)
fig = fig.get_figure()
fig.savefig('./Graphs/isHoliday.png')
fig.clf()

fig = sns.regplot(x="Year", y="logSales", data=train)
fig = fig.get_figure()
fig.savefig('./Graphs/Year.png')
fig.clf()

fig = sns.regplot(x="Month", y="logSales", data=train)
fig = fig.get_figure()
fig.savefig('./Graphs/Month.png')
fig.clf()

#So rode esta parte depois de utilizar o script principal
results_AB = pd.read_csv('./Results/resultABmetrics.csv')
results_RF = pd.read_csv('./Results/resultRFmetrics.csv')

fig = sns.regplot(x="Store", y="absolute_error", data=results_AB)
fig = fig.get_figure()
fig.savefig('./Graphs/AbError_Store_AB.png')
fig.clf()

fig = sns.regplot(x="Store", y="absolute_error", data=results_RF)
fig = fig.get_figure()
fig.savefig('./Graphs/AbError_Store_RF.png')
fig.clf()

fig = sns.regplot(x="Store", y="squared_error", data=results_AB)
fig = fig.get_figure()
fig.savefig('./Graphs/SqError_Store_AB.png')
fig.clf()

fig = sns.regplot(x="Store", y="squared_error", data=results_RF)
fig = fig.get_figure()
fig.savefig('./Graphs/SqError_Store_RF.png')
fig.clf()

fig = sns.regplot(x="Store", y="sqrt(squared_error)", data=results_AB)
fig = fig.get_figure()
fig.savefig('./Graphs/Sqrt_SqError_Store_AB.png')
fig.clf()

fig = sns.regplot(x="Store", y="sqrt(squared_error)", data=results_RF)
fig = fig.get_figure()
fig.savefig('./Graphs/Sqrt_SqError_Store_RF.png')
fig.clf()

fig = sns.regplot(x="Store", y="Acuracia", data=results_AB)
fig = fig.get_figure()
fig.savefig('./Graphs/Acuracia_Store_AB.png')
fig.clf()

fig = sns.regplot(x="Store", y="Acuracia", data=results_RF)
fig = fig.get_figure()
fig.savefig('./Graphs/Acuracia_Store_RF.png')
fig.clf()

#Departamento
fig = sns.regplot(x="Departamento", y="absolute_error", data=results_AB)
fig = fig.get_figure()
fig.savefig('./Graphs/AbError_Departamento_AB.png')
fig.clf()

fig = sns.regplot(x="Departamento", y="absolute_error", data=results_RF)
fig = fig.get_figure()
fig.savefig('./Graphs/AbError_Departamento_RF.png')
fig.clf()

fig = sns.regplot(x="Departamento", y="squared_error", data=results_AB)
fig = fig.get_figure()
fig.savefig('./Graphs/SqError_Departamento_AB.png')
fig.clf()

fig = sns.regplot(x="Departamento", y="squared_error", data=results_RF)
fig = fig.get_figure()
fig.savefig('./Graphs/SqError_Departamento_RF.png')
fig.clf()

fig = sns.regplot(x="Departamento", y="sqrt(squared_error)", data=results_AB)
fig = fig.get_figure()
fig.savefig('./Graphs/Sqrt_SqError_Departamento_AB.png')
fig.clf()

fig = sns.regplot(x="Departamento", y="sqrt(squared_error)", data=results_RF)
fig = fig.get_figure()
fig.savefig('./Graphs/Sqrt_SqError_Departamento_RF.png')
fig.clf()

fig = sns.regplot(x="Departamento", y="Acuracia", data=results_AB)
fig = fig.get_figure()
fig.savefig('./Graphs/Acuracia_Departamento_AB.png')
fig.clf()

fig = sns.regplot(x="Departamento", y="Acuracia", data=results_RF)
fig = fig.get_figure()
fig.savefig('./Graphs/Acuracia_Departamento_RF.png')
fig.clf()