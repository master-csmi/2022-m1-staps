import classification as classe 


from sklearn.feature_selection import SelectKBest ,f_classif
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC ,SVC
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd


# ---------------- Modelisation et choix de l'algorithme de sélection


# ---------------------------- Mise en place des données et quelques réglages 

df = pd.read_csv("/home/congo/Bureau/2022-m1-staps/motives_part/clustering.csv") 
#df = pd.read_csv("/home/congo/Bureau/2022-m1-staps/motives_part/boy.csv") 
#df = pd.read_csv("/home/congo/Bureau/2022-m1-staps/motives_part/girl.csv")



df.drop(df.columns[[0]], axis = 1, inplace = True) 
#print(df)


y_3 = df.loc[:,'cluster3']
# y_4 = df.loc[:,'cluster4']
# y_5 = df.loc[:,'cluster5']

X = df
X.drop(df.columns[[0,1,2]], axis = 1, inplace = True) 

col_total  = X.columns
# print(col_total)


train_ratio = 0.80
test_ratio = 0.20
validation_ratio = 0.10

X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(X, y_3, test_size=test_ratio)






# ---------------------------  evaluation model 
preprocessor = make_pipeline(PolynomialFeatures(2, include_bias = False) ,SelectKBest(f_classif , k=10))



SVC_3 = make_pipeline(preprocessor,SVC(random_state=0))
KNN_3 = make_pipeline(preprocessor,KNeighborsClassifier())
logreg_3 = make_pipeline(preprocessor,LogisticRegression())
LSVC_3 = make_pipeline(preprocessor,LinearSVC())


dict_of_models ={ 'KNN' :KNN_3,
                'logreg' :logreg_3,
                'LSVC' : LSVC_3,
               
                'SVC': SVC_3}

for names,model in dict_of_models.items() :
     print(names)
     classe.evaluation(model,X_train_3,y_train_3,X_test_3,y_test_3)
