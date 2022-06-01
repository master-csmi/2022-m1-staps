data <- read.csv("C:\\Users\\33783\\Desktop\\clustering.csv")

#------------------------------- lecture  des données 

# -----------------------------------

data <- read.csv("C:\\Users\\33783\\Desktop\\staps\\data_ready.csv",row.names = 1)

data_boy <- read.csv("C:\\Users\\33783\\Desktop\\data_ready_boy.csv",row.names = 1)
data_girl <- read.csv("C:\\Users\\33783\\Desktop\\data_ready_girl.csv",row.names = 1)

#data<- data_boy
#data <- data_girl


#head(data)

data_base <- data[,0:71] 
head(data_base)



#install.packages("FactoMineR")
#install.packages('https://cran.rstudio.com/bin/windows/contrib/4.0/FactoMineR_2.4.zip', lib='C:/Users/33783/Documents/R/win-library/4.0',repos = NULL)

library(FactoMineR)

library("factoextra")



#------------------------------------------------------ Partie PCA 

#------------------------------------------------------ 

res.pca <- PCA(data_base ,graph = TRUE)
print(res.pca)
summary(res.pca)


# Visualisation et interprétation

# Valeurs propres / Variances

eig.val <- get_eigenvalue(res.pca)
eig.val

fviz_eig(res.pca, addlabels = TRUE, ylim = c(0, 50))




# Resultas de la PCA 

# Contributions des variables aux PC
var <- get_pca_var(res.pca)
head(var$contrib,10)


# Contributions of variables to PC1,PC2,PC3,PC4,PC5 top = 35 
fviz_contrib(res.pca, choice = "var", axes = 1, top = 35) 
fviz_contrib(res.pca, choice = "var", axes = 2, top = 35) 
fviz_contrib(res.pca, choice = "var", axes = 3, top = 35) 
fviz_contrib(res.pca, choice = "var", axes = 4, top = 35) 
fviz_contrib(res.pca, choice = "var", axes = 5, top = 35) 



## les variables garder  lors de la PCA
donne = var$contrib
dim1  <- donne[,1]
dim2  <- donne[,2]
dim3  <- donne[,3]
dim4  <- donne[,4]
dim5  <- donne[,5]

print(dim1)
res <- dim1[dim1 > mean(dim1)]
print(names(res))
res1 <- names(res)


print(dim2)
res <- dim2[dim2 > mean(dim2)]
print(names(res))
res2 <- names(res)


print(dim3)
res <- dim3[dim3 > mean(dim3)]
print(names(res))
res3 <- names(res)


print(dim4)
res <- dim4[dim4 > mean(dim4)]
print(names(res))
res4 <- names(res)


print(dim5)
res <- dim5[dim5 > mean(dim5)]
print(names(res))
res4 <- names(res)


col_keep <- union(res1,res2)
col_keep <- union(col_keep,res3)
col_keep <- union(col_keep,res4)


col =colnames( data_base)
reste <- col %in% col_keep

print(col[col %in% col_keep == FALSE])



#------------------------------------------------------  Hierachical clustering 
res.pca<- PCA(data_base ,graph = FALSE, ncp = 5)
res.hcpc <- HCPC(res.pca,nb.clust=3,consol=FALSE,graph=TRUE)


library(factoextra)

# resultats  du dendogram avant de faire le cluster
fviz_dend(res.hcpc, 
          cex = 0.5,                    
          palette = "jco",               
          rect = TRUE, rect_fill = TRUE, 
          rect_border = "jco",           
          labels_track_height = 0.5 )


# mise en oeuvre du cluster pas trop différent du dendogram
fviz_cluster(res.hcpc,repel = TRUE,            
             show.clust.cent = TRUE, 
             palette = "jco",         
             ggtheme = theme_minimal(),
             main = "Factor map")


# Principal components + tree
plot(res.hcpc, choice = "3D.map")


# The original data with a column called clust 
# containing the partition
res.hcpc$data.clust

# Description of the clusters by the variables
s <- res.hcpc$desc.var
res.hcpc$desc.var


# Description of the clusters by the individuals
res.hcpc$desc.ind$para


# nombre de personne dans chaque cluster
library(plyr)
count(res.hcpc$data.clust,'clust')

#----------------------------------------------- exportation fichier pour la classification

# ------------------------------------------------  

# ajout d'un identifiant
index_name <- rownames(data)
index_name



# ajout de la colonne cluster
cluster <- res.hcpc$data.clust[72]
cluster


newdf <- cbind(data_base,cluster,index_name)
newdf


write.csv(x = newdf, file = "girl.csv")

write.csv(x = newdf, file = "boy.csv")

write.csv(x = newdf, file = "clusering.csv")
