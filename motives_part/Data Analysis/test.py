import pandas as pd


import preprocessing as prep



#                   ----------------- Initialisation -------------------

#Lecture des données
df = pd.read_excel("/home/congo/Bureau/2022-m1-staps/motives_part/data_motives_final.xlsx","haller") 

#stoker le nom des colonnes
column_name = [c for c in df]

# transformer boy en 0 et Girl en 1
df['Sexe'] = df['Sexe'].map({'Boy': 0 ,'Girl': 1}, na_action=None)
newdf = df.sort_values(by='Sexe')

# pont entre boy et girl
# print(newdf.loc[495,'Sexe'])
# print(newdf.loc[496,'Sexe'])


df_boy = newdf[:495]
# print(df_boy.loc[495,'Sexe'])
# print(df_boy.shape)


df_boy = newdf[:495]
df_girl = newdf[495:]
# print(df_girl.loc[:,'Sexe'])
# print(df_girl.shape)


# retour des nom des colonnes
newdf['Sexe'] = newdf['Sexe'].map({0 : 'Boy' ,
                             1 : 'Girl'
                             },
                             na_action=None)

# print(newdf.loc[:,'Sexe'])


# utilisation des fonctions sur l'initialisation !!
data = prep.imputation(df)

data_clean , data_correct = prep.preprocessing(data)

data_ready = prep.Normalisation(data)
