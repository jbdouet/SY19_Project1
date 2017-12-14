source("regresseur_R.R")
source("classifieur_R.R")
dataset<-read.csv('data/tp3_clas_app.txt',sep=' ')
classifieur(dataset)
