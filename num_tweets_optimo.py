from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

import warnings
warnings.filterwarnings('ignore')
 

def converter(instr):
    return np.fromstring(instr[1:-1],sep=' ')

modelo='vector_w2v_1'

#Leer los datos
tweets=pd.DataFrame(pd.read_csv('E:\\tweet\\Tweets_clasificados_estimados_4x4_prueba.csv',encoding='utf-8-sig',converters={modelo:converter}))

#Carga de modelos

model = XGBClassifier()# XG Boost  
#model= SVC(kernel='linear')
#model=LogisticRegression()
#model=GaussianNB()

num_repeticiones=2 # Número de repeticiones del entrenamiento

def estimar_modelo(model,rs,tamano_train):
    X_train, X_test, y_train, y_test = train_test_split(np.array(tweets[modelo].tolist()),tweets['Clasificación'].values, train_size=tamano_train, random_state=rs)       
    model.fit(X_train,y_train)
    y_pred_test=model.predict(X_test)
    y_pred_train=model.predict(X_train)
    efi_train=(y_train == y_pred_train).sum()/X_train.shape[0]
    efi_test=(y_test == y_pred_test).sum()/X_test.shape[0]
    return efi_train,efi_test

 
def estimar_eficiencia(model,num_repeticiones):

    list_efi_train=[]
    list_efi_test=[]
    for q in range(1,81):
      print(q)     
      mean_list_train=[]
      mean_list_test=[]
      for p in range(0,num_repeticiones): 
         efi_train,efi_test=(estimar_modelo(model,p,q/100)) 
         mean_list_train.append(efi_train)
         mean_list_test.append(efi_test)

      list_efi_train.append(sum(mean_list_train)/len(mean_list_train))
      list_efi_test.append(sum(mean_list_test)/len(mean_list_test))
    return list_efi_train,list_efi_test

list_efi_train,list_efi_test=estimar_eficiencia(model,num_repeticiones)

df_train=pd.DataFrame(list_efi_train)
df_test=pd.DataFrame(list_efi_test)

df=pd.concat([df_train,df_test],axis=1)

df.to_csv('C:\\Users\\IDEX2060\\Desktop\\shupashups.csv')



