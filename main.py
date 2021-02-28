# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 09:18:42 2021

@author: 
"""
#∟importation des bibliothèques
import pandas as pd
import numpy as np
import os
from scipy.io import loadmat
import matplotlib.pyplot as plt

#téléchargement des données
def load_dataf(nom_fichier):
    """Cette fonction renvoie le dataframe d'un seul fichier"""
    dic= loadmat(nom_fichier)
    d= dic['d_iner']
    df=pd.DataFrame(d)
    if nom_fichier[4]=='s':
        df["subject"]=nom_fichier[5]
    else:
        df["subject"]=nom_fichier[4]
    if nom_fichier[7]=='t':
        df["experience"]=nom_fichier[8]
    else:
        df["experience"]=nom_fichier[7]
    if nom_fichier[2]=='_':
        df["action"]=nom_fichier[1]
    else:
        df["action"]=nom_fichier[1:3]
   
    return df

def load_data(chemin_dossier):
    """Cette fonction renvoie le dataframe de tous les fichiers d'un dossier donné"""
    files=os.listdir(chemin_dossier)
    dataf=load_dataf(files[0])
    for f in files[1:861]:
        dataframe=load_dataf(f)
        dataf=pd.concat([dataf,dataframe])
    return dataf

#d= load_data('C:/Users/abdou/Desktop/Projet-apprentissage-automatique-main/IMU')
#d.columns=['xa','ya','za','xg','yg','zg','subject','experience','action']
            
def tracer_signal(capteur,numaction,numsujet,numessai):
    """Cette fonction renvoie le signal mesuré par un capteur sur 3 axes pour un fichier donné
    en fonction du numéro de l'action numaction, le numéro du sujet numsujet, et le numéro de l'essai numessai.
    Si capteur=1, alors le capteur est l'acceléromètre.
    Si capteur=2, alors le capteur est le gyroscope."""
    nom_fichier='a'+str(numaction)+'_s'+str(numsujet)+'_t'+str(numessai)+'_inertial.mat'
    dataframe=load_dataf(nom_fichier)
    dataframe.columns=['xa','ya','za','xg','yg','zg','subject','experience','action']
    if capteur==1:
        X=dataframe['xa']
        Y=dataframe['ya']
        Z=dataframe['za']
    if capteur==2:
        X=dataframe['xg']
        Y=dataframe['yg']
        Z=dataframe['zg']
    nbrepoints=len(dataframe.index)
    t=np.linspace(0,nbrepoints/50,nbrepoints)
    gx=plt.plot(t,X,'r')
    gy=plt.plot(t,Y,'g')
    gz=plt.plot(t,Z,'k')
    
    return gx,gy,gz


def pandas_entropy(column, base=None):
    """Cette fonction permet de calculer l'entropie d'un vecteur de données"""
  vc = pd.Series(column).value_counts(normalize=True, sort=False)
  base = np.e if base is None else base
  return -(vc * np.log(vc)/np.log(base)).sum()

def feature_extraction(chemin_dossier):
    """Cette fonction permet de calculer les attributs des données des fichiers d'un dossier donné."""
    files=os.listdir(chemin_dossier)
    V=[[]]*861
    a=0
    for f in files[0:861]:
        Va=[]
        dataframe=load_dataf(f)
        dataframe.columns=['xa','ya','za','xg','yg','zg','subject','experience','action']
        Va.append(dataframe['xa'].mean())
        Va.append(dataframe['xa'].std())
        Va.append(dataframe['xa'].median())
        Va.append(pandas_entropy(dataframe['xa']))
        Va.append(dataframe['ya'].mean())
        Va.append(dataframe['ya'].std())
        Va.append(dataframe['ya'].median())
        Va.append(pandas_entropy(dataframe['ya']))
        Va.append(dataframe['za'].mean())
        Va.append(dataframe['za'].std())
        Va.append(dataframe['za'].median())
        Va.append(pandas_entropy(dataframe['za']))
        Va.append(dataframe['xg'].mean())
        Va.append(dataframe['xg'].std())
        Va.append(dataframe['xg'].median())
        Va.append(pandas_entropy(dataframe['xg']))
        Va.append(dataframe['yg'].mean())
        Va.append(dataframe['yg'].std())
        Va.append(dataframe['yg'].median())
        Va.append(pandas_entropy(dataframe['yg']))
        Va.append(dataframe['zg'].mean())
        Va.append(dataframe['zg'].std())
        Va.append(dataframe['zg'].median())
        Va.append(pandas_entropy(dataframe['zg']))
        if f[4]=='s':
            Va.append(f[5])
        else:
            Va.append(f[4])
        if f[2]=='_':
            Va.append(f[1])
        else:
            Va.append(f[1:3])
        V[a]=Va
        a=a+1
    return V

# V est le vecteur des attributs
V=feature_extraction('C:/Users/abdou/Desktop/Projet-apprentissage-automatique-main/IMU') 
# D est le dataframe généré à partir de V
D= pd.DataFrame(V,columns=['mean(Ax)','std(Ax)','median(Ax)','entr(Ax)','mean(Ay)','std(Ay)','median(Ay)','entr(Ay)','mean(Az)','std(Az)','median(Az)','entr(Az)','mean(Gx)','std(Gx)','median(Gx)','entr(Gx)','mean(Gy)','std(Gy)','median(Gy)','entr(Gy)','mean(Gz)','std(Gz)','median(Gz)','entr(Gz)','subject','action'])

def separer(D):
    """Cette fonction permet de distinguer les données d'apprentissage et les données test"""
    donneesapp=pd.DataFrame(columns=['mean(Ax)','std(Ax)','median(Ax)','entr(Ax)','mean(Ay)','std(Ay)','median(Ay)','entr(Ay)','mean(Az)','std(Az)','median(Az)','entr(Az)','mean(Gx)','std(Gx)','median(Gx)','entr(Gx)','mean(Gy)','std(Gy)','median(Gy)','entr(Gy)','mean(Gz)','std(Gz)','median(Gz)','entr(Gz)'])
    donneestest=pd.DataFrame(columns=['mean(Ax)','std(Ax)','median(Ax)','entr(Ax)','mean(Ay)','std(Ay)','median(Ay)','entr(Ay)','mean(Az)','std(Az)','median(Az)','entr(Az)','mean(Gx)','std(Gx)','median(Gx)','entr(Gx)','mean(Gy)','std(Gy)','median(Gy)','entr(Gy)','mean(Gz)','std(Gz)','median(Gz)','entr(Gz)'])
    labelapp=np.ndarray((431,1))
    labeltest=np.ndarray((430,1))
    ka=0
    kt=0
    for i in range(861):
        if int(D.iloc[i,24])%2==1:
            donneesapp=pd.concat([donneesapp,D.iloc[i:i+1,0:24]])
            labelapp[ka][0]=int(D.iloc[i,25])
            ka=ka+1
        if int(D.iloc[i,24])%2==0:
            donneestest=pd.concat([donneestest,D.iloc[i:i+1,0:24]])
            labeltest[kt][0]=int(D.iloc[i,25])
            kt=kt+1
    
    
    return donneesapp,donneestest,labelapp,labeltest
 
           
S=separer(D)
donneesapp=S[0]
donneestest=S[1]

#On calcule la moyenne et l'écart type pour pouvoir normaliser les données.
moy=donneesapp.mean()
ecart=donneesapp.std()
donneesapp=(donneesapp-moy)/ecart
donneestest=(donneestest-moy)/ecart

labelapp=S[2]
labeltest=S[3]   

#importation des différents classifieurs
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

#phase d'apprentissage
clf1 = tree.DecisionTreeClassifier()
clf1 = clf1.fit(donneesapp,labelapp)

clf2=KMeans(8,max_iter=600)
clf2= clf2.fit(donneesapp,labelapp)

clf3=MLPClassifier(100,max_iter=500)
clf3=clf3.fit(donneesapp,labelapp.ravel())


clf4= SVC(1,gamma='scale')
clf4=clf4.fit(donneesapp,labelapp)

clf5= GaussianNB()
clf5=clf5.fit(donneesapp,labelapp)

#phase de test
y1_pred= clf1.predict(donneestest)
print(metrics.accuracy_score(labeltest,y1_pred))

y2_pred=clf2.predict(donneestest)
print(metrics.accuracy_score(labeltest,y2_pred))

y3_pred= clf3.predict(donneestest)
print(metrics.accuracy_score(labeltest,y3_pred))

y4_pred= clf4.predict(donneestest)
print(metrics.accuracy_score(labeltest,y4_pred))

y5_pred= clf5.predict(donneestest)
print(metrics.accuracy_score(labeltest,y5_pred))

#phase d'évaluation
def analyse(y_pred):
    """Cette fonction permet d'afficher les différentes valeurs des indicateurs de performance
    qui sont la précision, le recall, le f1-score, et le support"""
    print(classification_report(labeltest,y_pred,labels=range(1,28)))
    return None
   
#affichage des matrices de confusion
    
#print(confusion_matrix(labeltest,y1_pred))
#print(confusion_matrix(labeltest,y2_pred))
print(confusion_matrix(labeltest,y3_pred))
#print(confusion_matrix(labeltest,y4_pred))
