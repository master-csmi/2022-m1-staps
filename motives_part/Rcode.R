install.packages("FactoMineR")
library(FactoMineR)
install.packages('https://cran.rstudio.com/bin/windows/contrib/4.0/FactoMineR_2.4.zip', lib='C:/Users/33783/Documents/R/win-library/4.0',repos = NULL)



data <- read.csv("C:\\Users\\33783\\Desktop\\clustering.csv")



data <- read.csv("C:\\Users\\33783\\Desktop\\test.csv",row.names = 1)

#head(data)

data_base <- data[,0:71] 
#head(data_base)

#summary(data_base)
library(FactoMineR)


############## Partie PCA ########

res.pca <- PCA(data_base ,graph = TRUE)
print(res.pca)
summary(res.pca)


# Visualisation et interprétation

# Valeurs propres / Variances
library("factoextra")
eig.val <- get_eigenvalue(res.pca)
eig.val

fviz_eig(res.pca, addlabels = TRUE, ylim = c(0, 50))

# D'après le graphique ci-dessus, nous pourrions vouloir nous 
# arrêter à la cinquième composante principale () car la variation est plus faible ).
# Cependant 39.79760% des informations (variances) contenues 
# dans les données sont retenues par les 5  premières composantes principales.


# Resultas de la PCA 
var <- get_pca_var(res.pca)
var
# Contributions des variables aux PC

head(var$contrib,10)


# Contributions of variables to PC1  #  top = 35 - 40
fviz_contrib(res.pca, choice = "var", axes = 1, top = 35) 

# Contributions of variables to PC2  #  top = 35 - 40
fviz_contrib(res.pca, choice = "var", axes = 2, top = 35) 

# Contributions of variables to PC3  #  top = 35 - 40
fviz_contrib(res.pca, choice = "var", axes = 3, top = 35) 

# Contributions of variables to PC4  #  top = 35 - 40
fviz_contrib(res.pca, choice = "var", axes = 4, top = 35) 

# Contributions of variables to PC5  #  top = 35 - 40
fviz_contrib(res.pca, choice = "var", axes = 5, top = 35) 


## les variables garder 
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


######## K-mean clustering #########
library(factoextra)
data_good <- data_base[,col_keep]
res.km <- kmeans(data_good, centers = 3)
print(res.km)

# Cluster size
res.km$size

# Cluster means
res.km$centers

fviz_cluster(res.km, data_good )





###### Hierachical clustering #######
res.pca<- PCA(data_base ,graph = FALSE, ncp = 5)
res.hcpc <- HCPC(res.pca,nb.clust=3,consol=FALSE,graph=TRUE)


library(factoextra)
fviz_dend(res.hcpc, 
          cex = 0.7,                    
          palette = "jco",               
          rect = TRUE, rect_fill = TRUE, 
          rect_border = "jco",           
          labels_track_height = 0.8 )


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
res.hcpc$desc.ind










install.packages("Factoshiny")
install.packages('https://cran.rstudio.com/bin/windows/contrib/4.0/Factoshiny_2.4.zip', lib='C:/Users/33783/Documents/R/win-library/4.0',repos = NULL)
library(Factoshiny)
Factoshiny(data_base)

res.pca <- PCA(data_base , ncp = 5 ,graph = TRUE)
hc <- HCPC(res.pca,nb.clust=3,consol=FALSE,graph=TRUE)

plot(hc,choice = "tree")
plot(hc,choice = "map", draw.tree = FALSE)
plot(hc,choice = "3D.map")
catdes(hc$data.clust,ncol(hc$data.clust))