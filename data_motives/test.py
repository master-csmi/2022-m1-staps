from pydoc import describe
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler  



df = pd.read_excel("/home/congo/Bureau/2022-m1-staps/data_motives/data_motives_final.xlsx","haller",) 
X = df.to_numpy()


# mise a zero des valeurs manquantes
df = df.fillna(0)


X = X[:,16:]


nlin ,ncol =X.shapedata = data[:,39:]

nlin ,ncol =X.shape
X_train = X[:int(nlin*0.70),:int(ncol*0.70)]
X_test = X[int(nlin*0.70):,int(ncol*0.70):]
print("nlin =" , nlin, " ,ncol =" ,ncol)

#Describtion of all numerical variables
desc = df.describe()
#print("df.describe()",desc)


#mutiplier nos val negative pour avoir des val positif ????!!!


# calcul de l'ecart-type par ligne et par  colonne
stdlin = df.std(axis=0)
stdcol = df.std(axis=1)



print( " std colonne = ",5/4*df.std(axis=0))











#normalisation par ligne des valeurs numériques de Q(16) à la fin 
for i in range(nlin):
    max_lin = np.max(X[i,16:])
    min_lin = np.max(X[i,16:])
   
# utiliser les fit_transform

#scaler.fit_transform(X)

imputer = SimpleImputer(missing_values= np.nan,strategy='mean')
#imputer.fit_transform(X)

#print(df.head(6))
