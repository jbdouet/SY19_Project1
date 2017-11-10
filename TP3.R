################## REGRESSION LINEAIRE ############

data_reg<-read.csv('data/tp3_reg_app.txt',sep=' ')
n= nrow(data_reg)
ntrain=ceiling(n*2/3)
ntst=n-ntrain
train<-sample(1:n,ntrain)
data_reg.test<-data_reg[-train,]
data_reg.train<-data_reg[train,]

reg <- lm(y~., data = data_reg.train)

summary(reg)
confint(reg,level=0.95)

y <-fitted(reg,data_reg.test)
y
################## CLASSIFICATION ############

data_clas<-read.csv('data/tp3_clas_app.txt',sep=' ')

# pas de variables categoriques, pas de valeur manquante 

# Normalization 

# Separation train-test

n=nrow(data_clas)
ntrain=ceiling(n*2/3)
ntst=n-ntrain
train<-sample(1:n,ntrain)
data_clas.test<-data_clas[-train,]
data_clas.train<-data_clas[train,]


# Classification 
library(MASS)

########################### Model 1:LDA #####################

lda.clas<- lda(y~.,data=data_clas.train)
pred.clas<-predict(lda.clas,newdata=data_clas.test, type="response")
#pred.clas$class renvoie la classe predite 
table(data_clas.test$y,pred.clas$class) # compare les classes du test set avec les predictions
prop.table(table(data_clas.test$y,pred.clas$class))# transforme les valeurs en proba

cm= as.matrix(table(data_clas.test$y,pred.clas$class))
accuracy = sum(diag(cm)) / sum(cm) # = 0.7272727
precision = cm[1,1]/ apply(cm, 1, sum)[1]
recall =  cm[1,1] /  apply(cm, 2, sum)[1]
f1 = 2 * precision * recall / (precision + recall) 
library("pROC")
roc_curve<-roc(data_clas.test$y,as.vector(pred.clas$x))
plot(roc_curve)

library("ROCR")
pr <- prediction(pred.clas$x,data_clas.test$y )
auc <-performance(pr, measure='auc')
auc<- auc@y.values[[1]]
auc

######################## Model 2: Regression logistique ############

data_clas2.train  <- data_clas.train
data_clas2.test  <- data_clas.test
data_clas2.train$y <- data_clas.train$y-1
data_clas2.test$y <- data_clas.test$y-1

glm.fit<- glm(y~.,data=data_clas2.train,family=binomial)
summary(glm.fit)
# Pour regarder les coefficients significativement non nuls on selectionne ceux
# avec 2 ou 3 etoiles dans le summary 
pred.clas.glm<-predict(glm.fit,newdata=data_clas2.test, type = "response") # le type response est important pour avoir des probas !
# Contrairement au lda il n'y a pas d'argument $class, il faut donc fixer un threshold
table(data_clas2.test$y,pred.clas.glm>0.5) # on definit un threshold pour la classification
prop.table(table(data_clas2.test$y,pred.clas.glm>0.5))
logit<-predict(glm.fit,newdata=data_clas2.test,type='link') 
# le type link est le type par defaut et ne renvoie pas de proba, ils correspondent au log-odds
# voir lien proba/log-odds ici https://stats.idre.ucla.edu/other/mult-pkg/faq/general/faq-how-do-i-interpret-odds-ratios-in-logistic-regression/
roc_glm<-roc(data_clas2.test$y,pred.clas.glm)

plot(roc_glm,add=TRUE,col='red')

cm_glm= as.matrix(table(data_clas2.test$y,pred.clas.glm>0.5))
accuracy_glm = sum(diag(cm_glm)) / sum(cm_glm) 
precision_glm = cm_glm[1,1]/ apply(cm_glm, 1, sum)[1]
recall_glm =cm_glm[1,1] /  apply(cm_glm, 2, sum)[1]
f1_glm = 2 * precision_glm * recall_glm / (precision_glm + recall_glm) 

accuracy_glm # =0.6818182
f1_glm
hist(pred.clas.glm, breaks = 10) # montre comment sont réparties les valeurs sachant qu'on a mis le threshold de séparation des classe à 0.5 

#### Cross validation regression logistique 

K<-10
folds=sample(1:K,n,replace=TRUE)
CV<-rep(0,10)
for(i in (1:10)){
  for(k in (1:K)){ #### à modifier 
    glm.fit<- glm(y~.,data=data_clas2.train,family=binomial)
    cv_reg_glm<-predict(glm.fit,newdata=data_clas2.test)
    reg<-lm(Formula[[i]],data=pollution[folds!=k,])
    pred<-predict(reg,newdata=pollution[folds==k,])
    CV[i]<-CV[i]+ sum((pollution$Mortality[folds==k]-pred)^2)
  }
  CV[i]<-CV[i]/n
}
