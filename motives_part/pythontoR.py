
import pandas as pd
import numpy as np


# ----------------- Transfert of data python to Rstudio for clustering


def transfert (data_ready):
    # mettre data en df
    df = pd.DataFrame(data_ready)
    #stoker le nom des colonnes
    column_name = [c for c in df]
    

    df_const = df.astype(float,errors='raise')
    # renommer les colonne
    df.columns = column_name[39:]
    df_const.columns = column_name[39:]
    #print(df.shape)

    # 'Confiance en soi' par 'Confiance_en_soi'
    # print(column_name[90])
    column_name[90] = 'Confiance_en_soi'
    # print(column_name[90])
    df = df.rename(columns={'Confiance en soi':'Confiance_en_soi'}) 

    # print(df['Confiance_en_soi'])
    # df_cluster.info()
    # probleme de type alors on le change
    df = df.astype(float,errors='raise')
    # df_cluster.info()    

    #renommer les lignes ( remplacer 0 par etudiant_0)
    nlin,ncol = data_ready.shape
    line_name = ['etudiant_' + str(i) for i in range(nlin)]
    # print(line_name)
    df.index = line_name
    df_const.index = line_name
    # print(df)

    # with pd.ExcelWriter('data_ready.xlsx') as writer:
    # df.to_excel(writer, freeze_panes=(1,1))

    df.to_csv('data_ready.csv')