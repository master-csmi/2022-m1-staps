import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score,confusion_matrix,classification_report
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV

from sklearn.feature_selection import VarianceThreshold

from sklearn.metrics import recall_score,precision_score



from sklearn import metrics

#---------------------------------------------- 1) Modelisation et choix de l'algorithme de sélection --------------------

# Procédure d'évalution des modèles
def evaluation(model,X_train_3,y_train_3,X_test_3,y_test_3):
    model.fit(X_train_3,y_train_3)
    y_pred_3 = model.predict(X_test_3)
    # print(confusion_matrix(y_test_3 , y_pred_3))
    # print(classification_report(y_test_3 , y_pred_3))

    N, train_score ,val_score = learning_curve(model, X_train_3,y_train_3,
                            train_sizes = np.linspace(0.1,1.0,10),cv=5)

    plt.figure(figsize =(12,8))
    plt.plot(N,train_score.mean(axis = 1), label ='train score')
    plt.plot(N,val_score.mean(axis = 1), label ='validation score')
    plt.xlabel('amount of data')
    plt.ylabel('Performance of model')
    plt.legend()


# Optimisation des hyperparametres du modèle SVC et logistic
def optimiseur(model,parameters,X_train_3,y_train_3):
    grid = GridSearchCV(model,parameters)
    grid.fit(X_train_3,y_train_3)

    print("best parameters ",grid.best_params_)
    print("accuracy :",grid.best_score_)






#----------------------------------------------------- 2) features selections 



def selection_feature(X_test,X_train,seuil,col_total ):
    # élemination des colonnes à variances inferieur au seuil 0.8 ou 0.06 ou 0.04 ou 0.02
    selector = VarianceThreshold(threshold=seuil)
    selector.fit_transform(X_test)
    colonne_garder = np.array(X_test.columns)[selector.get_support()]

    colonne_suprimer = [i for i in col_total if not  i in colonne_garder ]

    
    print('colonne garder size = ',colonne_garder.shape)
    # print('colonne suprimer size = ',len(colonne_suprimer))
    print('colonne_garder = ',colonne_garder)


    # print('colonne_suprime = ',colonne_suprimer)

    # prediction avec uniquement les colonne garder
    X_test_keep = X_test[colonne_garder] 
    X_train_keep = X_train[colonne_garder] 

    return X_train_keep,X_test_keep


def evaluation_seuil(model,X_test,y_test_3,X_train,y_train_3,seuil):
    X_train_3,X_test_3 = selection_feature(X_test,X_train,seuil )
    fit_model(model,X_train_3,y_train_3)
    y_pred_3_log = prediction(model ,X_test_3)
    print_resulat(y_test_3,y_pred_3_log)
    # matrice_confusion(y_test_3, y_pred_3_log)



def give_precision ( y_test_3 ,y_pred_3_log) :
    precision = precision_score(y_true = y_test_3, y_pred = y_pred_3_log, average ='macro')
    return precision

def give_recall( y_test_3 ,y_pred_3_log) :
    recall = recall_score(y_true = y_test_3, y_pred = y_pred_3_log, average ='macro')
    return recall  

def give_f1score( y_test_3 ,y_pred_3_log) :
    f1score= f1_score(y_true = y_test_3, y_pred = y_pred_3_log , average ='macro')
    return f1score      


def give_list_precision(model,X_test,y_test_3,X_train,y_train_3,seuil,col_total ,list_precision ):
    X_train_3,X_test_3 = selection_feature(X_test,X_train,seuil,col_total  )
    fit_model(model,X_train_3,y_train_3)
    y_pred_3_log = prediction(model ,X_test_3)

    precision = give_precision (y_test_3 ,y_pred_3_log) 
    list_precision.append(precision) 

    return list_precision

def give_list_recall(model,X_test,y_test_3,X_train,y_train_3,seuil,col_total ,list_recall):
    X_train_3,X_test_3 = selection_feature(X_test,X_train,seuil,col_total  )
    fit_model(model,X_train_3,y_train_3)
    y_pred_3_log = prediction(model ,X_test_3)

    recall = give_recall( y_test_3 ,y_pred_3_log) 
    list_recall.append(recall)

    return list_recall

def give_list_f1score(model,X_test,y_test_3,X_train,y_train_3,seuil,col_total ,list_f1score):
    X_train_3,X_test_3 = selection_feature(X_test,X_train,seuil ,col_total )

    fit_model(model,X_train_3,y_train_3)
    y_pred_3_log = prediction(model ,X_test_3)

    f1score = give_f1score( y_test_3 ,y_pred_3_log) 
    list_f1score.append(f1score)

    return list_f1score




def col_selection(X_test,seuil,col_total):
    # élemination des colonnes à variances inferieur au seuil 0.8 ou 0.06 ou 0.04 ou 0.02
    selector = VarianceThreshold(threshold=seuil)
    selector.fit_transform(X_test)

    colonne_garder = np.array(X_test.columns)[selector.get_support()]
    colonne_suprimer = [i for i in col_total if not  i in colonne_garder ]

    N_keep = len(colonne_garder)
    N_suprim = len(colonne_suprimer)

    return N_suprim , N_keep



# ------------------------------------------------- 3) Tester avec le modèle final


def fit_model (model,X_train_3,y_train_3):
    model.fit(X_train_3,y_train_3)

def prediction ( model ,X_test_3):
    # prediction
    y_pred_3_log = model.predict(X_test_3)
    return y_pred_3_log
    
def print_resulat(y_test_3,y_pred_3_log):
    print( 'recall_score = ' ,recall_score(y_true = y_test_3, y_pred = y_pred_3_log, average ='macro'))
    print( 'f1-score = ' ,f1_score(y_true = y_test_3, y_pred = y_pred_3_log , average ='macro'))
    print( 'precision_score = ' ,precision_score(y_true = y_test_3, y_pred = y_pred_3_log, average ='macro'))
    
def matrice_confusion(y_test_3, y_pred_3_log):
    # Matrice de confusion
    confusion_3 = metrics.confusion_matrix(y_true = y_test_3,y_pred = y_pred_3_log)
    
    confusion = pd.DataFrame(confusion_3, index =['y_true: 0','y_true: 1','y_true: 2'] ,columns=['y_pred : 0','y_pred : 1','y_pred : 2'] )
    print('confusion matrix \n' ,confusion)

    plt.matshow(confusion_3, cmap=plt.cm.gray)
    plt.xlabel('y_true')
    plt.ylabel('y_pred')
    plt.show()
