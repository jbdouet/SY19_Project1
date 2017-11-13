set.seed(42)
library(MASS)
library(caret)
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

####   Pcr   ####

model_pcr <- caret::train(data_reg.train[,-51],data_reg.train$y,method='pcr')

#pour voir quels paramètres peuvent être "tuned":
modelLookup(model='pcr') # no parameters to tune

plot(model_pcr)# does not work if no parameters to tune

plot(varImp(object=model_pcr),main="PCR - Variable Importance")

predictions_pcr<-predict.train(object=model_pcr,data_reg.test[,-51],type="raw")


plot(predictions_pcr,data_reg.test$y)

################## CLASSIFICATION ############

data_clas<-read.csv('data/tp3_clas_app.txt',sep=' ')

# Separation train-test

n=nrow(data_clas)
ntrain=ceiling(n*2/3)
ntst=n-ntrain
train<-sample(1:n,ntrain)
data_clas.test<-data_clas[-train,]
data_clas.train<-data_clas[train,]

### Missing values 

sum(is.na(data_clas.train))
sum(is.na(data_clas.test))
# -> no missing values 

### Balance classes

dim(data_clas[data_clas$y ==1,])
dim(data_clas[data_clas$y ==2,])
# -> Imbalanced classes 

### Standardize data ###

# on met la colonne y comme type factor pour ne pas la standardizer puisque que les chiffres 1 et 2 representent des classes
data_clas.train$y=as.factor(data_clas.train$y) 
data_clas.test$y=as.factor(data_clas.test$y)

# On standardize et on remet les variables sous forme de data frames
data_clas_scaled.train<- lapply(data_clas.train, function(x) if(is.numeric(x)){
  scale(x, center=TRUE, scale=TRUE)
} else x)
data_clas_scaled.train=as.data.frame(data_clas_scaled.train)

data_clas_scaled.test<- lapply(data_clas.test, function(x) if(is.numeric(x)){
  scale(x, center=TRUE, scale=TRUE)
} else x)
data_clas_scaled.test=as.data.frame(data_clas_scaled.test)

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
# avec 2 ou 3 etoiles dans le summary # pas fait pour le moment
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

n_folds <- 10
folds_i <- sample(rep(1:n_folds, length.out = ntrain)) # !!! le ntrain doit correspondre à la taille du dataset que l'on utilisera dans la boucle de cross validation 
table(folds_i) # Pas le même nombre d'éléments 
CV<-rep(0,10)
for (k in 1:n_folds) {# we loop on the number of folds, to build k models
  test_i <- which(folds_i == k)
  # les datasets entre le fit et le predict doivent être les mêmes car c'est le même dataset que l'on divise en k-fold 
  # on peut utiliser le data set complet ou seulement le train et avoir une idée finale de la performance sur le test
  train_xy <- data_clas2.train[-test_i, ]
  test_xy <- data_clas2.train[test_i, ]
  print(k)
  glm.fit<- glm(y~.,data=train_xy,family=binomial)
  cv_pred<-predict(glm.fit,newdata=test_xy, type = "response") 
  CV[k]<- sum((test_xy$y-cv_pred)^2)
}
CVerror= sum(CV)/length(CV)

#### Modeles avec librairie caret ###

names(getModelInfo()) # donne le nom des modeles 

####   GLM    ####

model_glm <- train(data_clas_scaled.train[,-31],data_clas_scaled.train$y,method='glm',trControl= trainControl(
  method = "cv",
  number =10,
  verboseIter = TRUE))

#pour voir quels paramètres peuvent être "tuned":
modelLookup(model='glm') # no parameters to tune

plot(model_glm) # does not work if no parameters to tune

plot(varImp(object=model_glm),main="glm - Variable Importance")

predictions_glm<-predict.train(object=model_glm,data_clas_scaled.test[,-31],type="raw")
table(predictions_glm)

confusionMatrix(predictions_glm,data_clas_scaled.test$y)

####   rf - random forest    ####

model_rf <- train(data_clas_scaled.train[,-31],data_clas_scaled.train$y,method='rf',trControl= trainControl(
  method = "cv",
  number =10,
  verboseIter = TRUE))

#pour voir quels paramètres peuvent être "tuned":
modelLookup(model='rf') # no parameters to tune

plot(model_rf)# does not work if no parameters to tune

plot(varImp(object=model_rf),main="rf - Variable Importance")

predictions_rf<-predict.train(object=model_rf,data_clas_scaled.test[,-31],type="raw")
table(predictions_rf)

confusionMatrix(predictions_rf,data_clas_scaled.test$y)

####   nnet    ####

model_nnet <- train(data_clas.train[,-31],data_clas.train$y,method='nnet',trControl= trainControl(
  method = "cv",
  number =10,
  verboseIter = TRUE))

#pour voir quels paramètres peuvent être "tuned":
modelLookup(model='nnet') # no parameters to tune

plot(model_nnet)# does not work if no parameters to tune

plot(varImp(object=model_nnet),main="nnet - Variable Importance")

predictions_nnet<-predict.train(object=model_nnet,data_clas.test[,-31],type="raw")
table(predictions_nnet)

confusionMatrix(predictions_nnet,data_clas.test$y)

####   Naive Bayes    ####

model_naive_bayes <- train(data_clas_scaled.train[,-31],data_clas.train$y,method='naive_bayes',trControl= trainControl(
  method = "cv",
  number =10,
  verboseIter = TRUE))

#pour voir quels paramètres peuvent être "tuned":
modelLookup(model='naive_bayes') # no parameters to tune

plot(model_naive_bayes)# does not work if no parameters to tune

plot(varImp(object=model_naive_bayes),main="naive_bayes - Variable Importance")

predictions_naive_bayes<-predict.train(object=model_naive_bayes,data_clas_scaled.test[,-31],type="raw")
table(predictions_naive_bayes)

confusionMatrix(predictions_naive_bayes,data_clas.test$y) # accuracy =0.7424

####   SVM poly   ####

model_svmPoly <- train(data_clas_scaled.train[,-31],data_clas.train$y,method='svmPoly',trControl= trainControl(
  method = "cv",
  number =10,
  verboseIter = TRUE))

#pour voir quels paramètres peuvent être "tuned":
modelLookup(model='svmPoly') # no parameters to tune

plot(model_svmPoly)# does not work if no parameters to tune

plot(varImp(object=model_svmPoly),main="svmPoly - Variable Importance")

predictions_svmPoly<-predict.train(object=model_svmPoly,data_clas_scaled.test[,-31],type="raw")
table(predictions_svmPoly)

confusionMatrix(predictions_svmPoly,data_clas.test$y) 

####   mlpML   ####

model_mlpML <- train(data_clas_scaled.train[,-31],data_clas.train$y,method='mlpML',trControl= trainControl(
  method = "cv",
  number =10,
  verboseIter = TRUE))

#pour voir quels paramètres peuvent être "tuned":
modelLookup(model='mlpML') # no parameters to tune

plot(model_mlpML)# does not work if no parameters to tune

plot(varImp(object=model_mlpML),main="mlpML - Variable Importance")

predictions_mlpML<-predict.train(object=model_mlpML,data_clas_scaled.test[,-31],type="raw")
table(predictions_mlpML)

confusionMatrix(predictions_mlpML,data_clas.test$y) 


#### Regularization




### Principal component analysis 
