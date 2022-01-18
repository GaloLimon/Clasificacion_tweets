from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense

import warnings
warnings.filterwarnings('ignore')
 

def converter(instr):
    return np.fromstring(instr[1:-1],sep=' ')

modelo='vector_w2v_1'

#Leer los datos
tweets=pd.DataFrame(pd.read_csv('E:\\tweet\\Tweets_clasificados_estimados_4x4_prueba.csv',encoding='utf-8-sig',converters={modelo:converter}))

#Carga de modelos
model_1 = GaussianNB() #Naive-Bayes
model_2= SVC(kernel='linear')#Support Vector Machine  ['sigmoid','rbf','poly'(degree),'linear']
model_3 = XGBClassifier()# XG Boost
model_4=LogisticRegression() #Regresión Logística
     

tamano_train=0.8#Tamaño de la muestra de enternamiento
num_repeticiones=50 # Número de repeticiones del entrenamiento


def estimar_modelo(model,rs,tamano_train):
    #Divide los tweets en muestras de entrenamiento y muestras para realizar el test del modelo
    X_train, X_test, y_train, y_test = train_test_split(np.array(tweets[modelo].tolist()),tweets['Clasificación'].values, train_size=tamano_train, random_state=rs)       
    #Entrena el modelo   
    model.fit(X_train,y_train)
    #Predice sobre la muestra test
    y_pred_test=model.predict(X_test)
    #Predice sobre la misma muestra de entrenamiento
    y_pred_train=model.predict(X_train)
    
    #Calcula el % de cuántos tweets fueron clasificados correctamente
    efi_train=(y_train == y_pred_train).sum()/X_train.shape[0]
    efi_test=(y_test == y_pred_test).sum()/X_test.shape[0]
    
    #Devuelve la eficiencia para la muestra de entrenamiento y la muestra test
    return efi_train,efi_test


def estimar_eficiencia(model,tamano_train,num_repeticiones):
    
    #Se crean dos listas
    mean_list_train=[]
    mean_list_test=[]
    
    #Reliza n repeticiones para obtener la eficiencia del modelo
    for p in range(0,num_repeticiones): 
        
         efi_train,efi_test=(estimar_modelo(model,p,tamano_train)) 
    #Se llenan las listas con las eficiencias para cada repetición    
         mean_list_train.append(efi_train)
         mean_list_test.append(efi_test)
    #Calcula  el promedio de todas las eficiencias para aproximarnos a la eficiencia real   
    print('Eficiencia en la muestra train:' )   
    print(sum(mean_list_train)/len(mean_list_train))
    print('Eficiencia en la muestra test:' )
    print(sum(mean_list_test)/len(mean_list_test))


#imprime los resultados para cada modelo
print('Naive-Bayes')    
estimar_eficiencia(model_1,tamano_train,num_repeticiones)
print('SVC')
estimar_eficiencia(model_2,tamano_train,num_repeticiones)
print('XG Boost')
estimar_eficiencia(model_3,tamano_train,num_repeticiones)
print('Logística')   
estimar_eficiencia(model_4,tamano_train,num_repeticiones)


print('Red neuronal')

#Divide los tweets en muestras de entrenamiento y muestras para realizar el test del modelo

X_train, X_test, y_train, y_test = train_test_split(np.array(tweets[modelo].tolist()),tweets['Clasificación_coded'].values, train_size=tamano_train, random_state=0)       
y_train = to_categorical(y_train)

#Especificación de la red neuronal
model = Sequential()
model.add(Dense(500, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(18, activation='softmax'))
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

#Entrenamiento de la red neuronal
model.fit(X_train, y_train,epochs=10, batch_size=2, verbose=2)
model.fit(X_train, y_train)

#Predicciópn de la muestra de entrenamiento
y_pred_test=model.predict(X_test)
y_pred_test=np.round(y_pred_test)  
y_pred_test= np.argmax(y_pred_test, axis=-1) 

#Resultado de la eficiencia
print((y_test == y_pred_test).sum()/X_test.shape[0])     


