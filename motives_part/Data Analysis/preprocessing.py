import pandas as pd
import numpy as np



def imputation(df):
    # mise a zero des valeurs manquantes
    return df.fillna(0)
    

def preprocessing(data):
    data = imputation(data)
    # garder les valeurs qui nous concerne
    data = data.to_numpy()
    data = data[:,39:]

    # rendre toute les valeurs positives
    data_correct = data
    data = np.absolute(data)

    #Supression des valeurs abérantes par rapport au seuil
    seuil = 5*np.std(data,axis=0,dtype = np.float64)
    
    data = np.array(data)
    data_tmp = np.zeros_like(data)
    data_tmp[data < seuil] = data[data < seuil]
    data = data_tmp
    
    return data , data_correct


def Normalisation(data):
    data_clean , data_correct = preprocessing(data)

    #indices à suprimer
    max = np.max(data_clean,axis = 1)
    indices = [i for i, e in enumerate(max) if e == 0]
    

    #suppression des lignes
    data_del = np.delete(data_clean,indices,axis = 0)
    data_clean = data_del

    #normalisation par ligne sans ecartype nul pour avoir des valeurs entre 0 et 1
    min = np.min(data_clean,axis = 1)
    max = np.max(data_clean,axis = 1)
    data_clean = (data_clean-min[:,np.newaxis])/max[:,np.newaxis]

    # retour des valeurs négatives

    #suppression des lignes
    data_del = np.delete(data_correct,indices,axis = 0)
    data_correct = data_del

    #indices des valeurs négatives
    indices_val_neg_i = np.where(data_correct < 0)
    
    data_clean[indices_val_neg_i] = -1*data_clean[indices_val_neg_i]

    return data_clean


