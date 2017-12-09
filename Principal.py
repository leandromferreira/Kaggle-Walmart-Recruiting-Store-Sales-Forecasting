# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 13:36:29 2017

@author: Ferreira
"""
import numpy as np
import pandas as pd
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import seaborn as sns


def prosData():

    #Reading Database 
    dfTrain = pd.read_csv('./walmart/train.csv')
    dfFeature = pd.read_csv('./walmart/features.csv')
    dfTest = pd.read_csv('./walmart/test.csv')
    dfStores = pd.read_csv('./walmart/stores.csv')
    submission = pd.read_csv('./walmart/sampleSubmission.csv')
       
    #Merging information between the data [Train and Test]
    dfTrainTmp           = pd.merge(dfTrain, dfStores)
    dfTestTmp            = pd.merge(dfTest, dfStores)   
      
    #Merging the feature with the data [Train and Test]
    train                = pd.merge(dfTrainTmp, dfFeature)
    test                 = pd.merge(dfTestTmp, dfFeature)
    
    #Split the field Date
    train['Year']        = pd.to_datetime(train['Date']).dt.year
    train['Month']       = pd.to_datetime(train['Date']).dt.month
    train['Day']         = pd.to_datetime(train['Date']).dt.day
    train['Days']        = train['Month']*30+train['Day'] 

    #Pass Type to numeric 
    train['Type']        = train['Type'].replace('A',1)
    train['Type']        = train['Type'].replace('B',2)
    train['Type']        = train['Type'].replace('C',3)
    
    #Conting the passend days util the holiday
    train['daysHoliday'] = train['IsHoliday']*train['Days']
    #Coverting the sales to 
    train['logSales']    = np.log(4990+train['Weekly_Sales'])

    #Mesmo procedimento para test
    test['Year']         = pd.to_datetime(test['Date']).dt.year
    test['Month']        = pd.to_datetime(test['Date']).dt.month
    test['Day']          = pd.to_datetime(test['Date']).dt.day
    test['Days']         = test['Month']*30+test['Day']
    test['Type']         = test['Type'].replace('A',1)
    test['Type']         = test['Type'].replace('B',2)
    test['Type']         = test['Type'].replace('C',3)
    test['daysHoliday']  = test['IsHoliday']*test['Days']
    
    #Retirando date que esta em um formato nao usal, e os demais dados que possuem dados ausentes
    #Weekly_Sales foi transformado entao vamos retira-lo
    
    train                = train.drop(['CPI','Unemployment','Date',
                                       'MarkDown1','MarkDown2','MarkDown3', 
                                       'MarkDown4','MarkDown5','Weekly_Sales'],axis=1)
                                       
    test                 = test.drop(['CPI','Unemployment','Date',
                                      'MarkDown1','MarkDown2','MarkDown3',
                                      'MarkDown4','MarkDown5'],axis=1)

    return (train,test,submission)

if __name__=="__main__":
    
   sns.set(color_codes=True)
   #Output files
   f_Submission_RF       = open('./Results/resultRF.csv','w')         #File Submission for RF
   f_Submission_AB       = open('./Results/resultAB.csv','w')         #File Submission for AB
   fmetrics_RF           = open('./Results/resultRFmetrics.csv','w')  #File with the metrics for RF
   fmetrics_AB           = open('./Results/resultABmetrics.csv','w')  #File with the metrics for AB
   #Outputs Header   
   f_Submission_RF.write('Id,Weekly_Sales\n')
   f_Submission_AB.write('Id,Weekly_Sales\n')
   fmetrics_RF.write('absolute_error,squared_error,sqrt(squared_error),Acuracia,Store,Departamento\n')
   fmetrics_AB.write('absolute_error,squared_error,sqrt(squared_error),Acuracia,Store,Departamento\n')      
#   If you want to continue a stop process
#   f_Submission_RF.close()
#   f_Submission_AB.close()
#   fmetrics_RF.close()
#   fmetrics_AB.close()
#   
#   #Troquei o tipo de escrita 'a' para se caso precisar comecar da onde parou
#   f_Submission_RF       = open('resultRF.csv','a')
#   f_Submission_AB       = open('resultAB.csv','a')
#   fmetrics_RF           = open('resultRFmetrics.csv','a')
#   fmetrics_AB           = open('resultABmetrics.csv','a')
   
   #Process the data   
   train,test,submission = prosData()
   
   #iniciado algoritmos
   RFreg    = RandomForestRegressor(n_estimators=200,min_samples_split=3,n_jobs=2)
   ABreg    = AdaBoostRegressor(n_estimators=200)
   #Grid Search
#   param_grid = {
#                   "n_estimators": [100, 300]
#    }
#   RFreg = GridSearchCV(estimator=RFreg, param_grid=param_grid, cv= 10)
#   ABreg = GridSearchCV(estimator=ABreg, param_grid=param_grid, cv= 10)
   #Model
   size = submission['Id'].count()
   i=0;
   while (i < size):

       tmpId        = submission['Id'][i]
       tmpStr       = tmpId.split('_')
       tmpStore     = int(tmpStr[0])				                #Store ID
       tmpDept      = int(tmpStr[1])				                #Dept ID 
       dataF1       = train.loc[train['Dept']==tmpDept]				#Get the data from Dept  ID from all data
       tmpDf        = dataF1.loc[dataF1['Store']==tmpStore]			#Get the data form Store ID from the filtring data 
       tmpSL        = tmpDf['Store'].count()
       tmpDL	    = dataF1['Dept'].count()	
       tmpF         = dataF1.loc[train['IsHoliday']==1]
       dataF1       = pd.concat([dataF1,tmpF*4])		          	#Reforcing holiday data
       dataF2       = dataF1.loc[dataF1['Store']==tmpStore]      		#Filtring 
       testF1       = test.loc[test['Dept']==tmpDept]		      		#DataFrame de teste para a predicao
       testF1       = testF1.loc[testF1['Store']==tmpStore]
       testRows     = testF1['Store'].count()
       k            = i + testRows
		
       if (tmpSL < 10) and (tmpDL!=0): #Quando o numero de lojas do dataframe for muito pequeno RF falha, entao vamos trabalhar apenas com os dados do departamento
          X_train, X_test, y_train, y_test = train_test_split(dataF1.drop(['logSales'],axis=1),np.asarray(dataF1['logSales'], dtype="|S6"))       
          tmpModel_RF_trabalho = RFreg.fit(X_train,np.asarray(y_train,dtype=float))
          tmpModel_RF_Submiss  = RFreg.fit(dataF1.drop(['logSales'],axis=1),
                                 np.asarray(dataF1['logSales'],dtype=float))
          tmpModel_AB_trabalho = ABreg.fit(X_train,np.asarray(y_train,dtype=float))
          tmpModel_AB_Submiss  = ABreg.fit(dataF1.drop(['logSales'],axis=1),
                                 np.asarray(dataF1['logSales'],dtype=float)) 
       else:
          X_train, X_test, y_train, y_test = train_test_split(dataF2.drop(['logSales'],axis=1),np.asarray(dataF2['logSales'], dtype="|S6"))          
          tmpModel_RF_trabalho = RFreg.fit(X_train,np.asarray(y_train,dtype=float))
          tmpModel_RF_Submiss  = RFreg.fit(dataF2.drop(['logSales'],axis=1),
                                 np.asarray(dataF2['logSales'],dtype=float))
          tmpModel_AB_trabalho = ABreg.fit(X_train,np.asarray(y_train,dtype=float))
          tmpModel_AB_Submiss  = ABreg.fit(dataF2.drop(['logSales'],axis=1),
                                 np.asarray(dataF2['logSales'],dtype=float))                                   
       #Temporarios para o preditc de cada algoritimo e cada um para seu objetivo
       tmpP_RF_Submiss      = ( np.exp(pd.to_numeric(tmpModel_RF_Submiss.predict(testF1))) - 4990 )
       tmpP_AB_Submiss      = ( np.exp(pd.to_numeric(tmpModel_AB_Submiss.predict(testF1))) - 4990 )
       tmpP_RF_trabalho     = tmpModel_RF_trabalho.predict(X_test)
       tmpP_AB_trabalho     = tmpModel_AB_trabalho.predict(X_test)
       
       #Gravando os resultados dos valores de erros minimo
       fmetrics_RF.write('%f,%f,%f,%f,%f,%f\n'%(metrics.mean_absolute_error(np.asarray(y_test,dtype=float),tmpP_RF_trabalho),
                                          metrics.mean_squared_error(np.asarray(y_test,dtype=float) ,tmpP_RF_trabalho),
                                  np.sqrt(metrics.mean_squared_error(np.asarray(y_test,dtype=float) ,tmpP_RF_trabalho)),
                                  RFreg.score(X_test, np.asarray(y_test,dtype=float))*100,tmpStore,tmpDept))
       
       fmetrics_AB.write('%f,%f,%f,%f,%f,%f\n'%(metrics.mean_absolute_error(np.asarray(y_test,dtype=float),tmpP_AB_trabalho),
                                          metrics.mean_squared_error(np.asarray(y_test,dtype=float) ,tmpP_AB_trabalho),
                                  np.sqrt(metrics.mean_squared_error(np.asarray(y_test,dtype=float) ,tmpP_AB_trabalho)),
                                  ABreg.score(X_test, np.asarray(y_test,dtype=float))*100,tmpStore,tmpDept))
       #submission['Weekly_Sales'][i:k] = tmpP_RF_Submiss
       #submission['Weekly_Sales'][i:k] = tmpP_AB_Submiss
       for j in range(i,k):                                     	#Escrita no arquivo de submissao
           f_Submission_RF.write('%s,%s\n'%(submission['Id'][j],tmpP_RF_Submiss[j-i]))
           f_Submission_AB.write('%s,%s\n'%(submission['Id'][j],tmpP_AB_Submiss[j-i]))
       i+=testRows       
       print (i)
   f_Submission_RF.close()
   f_Submission_AB.close()
   fmetrics_RF.close()
   fmetrics_AB.close() 

    
