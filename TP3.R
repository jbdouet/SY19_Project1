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

yhat <-predict(reg,data_reg.test)
length(yhat)

mean((yhat-data_reg.test$y)^2)
plot(yhat,data_reg.test$y)
abline(0,1)

### Avec normalisation 

data_reg_scaled.train<- lapply(data_reg.train, function(x) if(is.numeric(x)){
  scale(x, center=TRUE, scale=TRUE)
} else x)
data_reg_scaled.train=as.data.frame(data_reg_scaled.train)

data_reg_scaled.test<- lapply(data_reg.test, function(x) if(is.numeric(x)){
  scale(x, center=TRUE, scale=TRUE)
} else x)
data_reg_scaled.test=as.data.frame(data_reg_scaled.test)

reg_scale <- lm(y~., data = data_reg_scaled.train)

summary(reg_scale)
confint(reg,level=0.95)

yhat_scale <-predict(reg_scale,data_reg_scaled.test)
length(yhat_scale)

mean((yhat_scale-data_reg_scaled.test$y)^2)
plot(yhat_scale,data_reg_scaled.test$y)
abline(0,1)

### Avec CV

n_folds <- 10
folds_i <- sample(rep(1:n_folds, length.out = n)) # !!! le ntrain doit correspondre à la taille du dataset que l'on utilisera dans la boucle de cross validation 
table(folds_i) # Pas le même nombre d'éléments 
CV<-rep(0,10)
for (k in 1:n_folds) {# we loop on the number of folds, to build k models
  test_i <- which(folds_i == k)
  # les datasets entre le fit et le predict doivent être les mêmes car c'est le même dataset que l'on divise en k-fold 
  # on peut utiliser le data set complet ou seulement le train et avoir une idée finale de la performance sur le test
  train_xy <- data_reg[-test_i, ]
  test_xy <- data_reg[test_i, ]
  print(k)
  fitControl <- trainControl(method = "cv",number = 10)
  model_lm <- caret::train(train_xy[,-51],train_xy$y,method='lm',trControl= fitControl)
  predictions_lm<-predict.train(object=model_lm,test_xy[,-51])
  CV[k]<- mean((test_xy$y-predictions_lm)^2)
}
CVerror= sum(CV)/length(CV)
CVerror
CV



####   Pcr   ####
fitControl <- trainControl(method = "cv",number = 10)
grid <- expand.grid(ncomp=c(3,5,10,20, 30,36,50))
model_pcr <- caret::train(data_reg.train[,-51],data_reg.train$y,method='pcr',trControl= fitControl,tuneGrid=grid)

model_pcr$bestTune
#pour voir quels paramètres peuvent être "tuned":
modelLookup(model='pcr') # no parameters to tune

plot(model_pcr)# does not work if no parameters to tune

plot(varImp(object=model_pcr),main="PCR - Variable Importance")

predictions_pcr<-predict.train(object=model_pcr,data_reg.test[,-51],type="raw")

mean((data_reg.test$y-predictions_pcr)^2)
plot(predictions_pcr,data_reg.test$y)
abline(0,1)

### Lasso  ###

fitControl <- trainControl(method = "cv",number = 10)
model_lasso <- caret::train(data_reg.train[,-51],data_reg.train$y,method='lasso',trControl= fitControl)

model_lasso$results
#pour voir quels paramètres peuvent être "tuned":
modelLookup(model='lasso') # no parameters to tune

plot(model_lasso)# does not work if no parameters to tune

plot(varImp(object=model_lasso),main="Lasso - Variable Importance")

predictions_lasso<-predict.train(object=model_lasso,data_reg.test[,-51])

mean((data_reg.test$y-predictions_lasso)^2)
plot(predictions_lasso,data_reg.test$y)
abline(0,1)  


n_folds <- 10
folds_i <- sample(rep(1:n_folds, length.out = n)) # !!! le ntrain doit correspondre à la taille du dataset que l'on utilisera dans la boucle de cross validation 
table(folds_i) # Pas le même nombre d'éléments 
CV<-rep(0,10)
for (k in 1:n_folds) {# we loop on the number of folds, to build k models
  test_i <- which(folds_i == k)
  # les datasets entre le fit et le predict doivent être les mêmes car c'est le même dataset que l'on divise en k-fold 
  # on peut utiliser le data set complet ou seulement le train et avoir une idée finale de la performance sur le test
  train_xy <- data_reg[-test_i, ]
  test_xy <- data_reg[test_i, ]
  print(k)
  fitControl <- trainControl(method = "cv",number = 10)
  model_lasso <- caret::train(train_xy[,-51],train_xy$y,method='lasso',trControl= fitControl)
  predictions_lasso<-predict.train(object=model_lasso,test_xy[,-51])
  CV[k]<- mean((test_xy$y-predictions_lasso)^2)
}
CVerror= sum(CV)/length(CV)
CVerror
CV

regresseur <- caret::train(data_reg[,-51],data_reg$y,method='lasso',trControl= fitControl)

#### randomGLM ###
fitControl <- trainControl(method = "cv",number = 10)
model_randomGLM <- caret::train(data_reg.train[,-51],data_reg.train$y,method='randomGLM',trControl= fitControl)

model_randomGLM$results
#pour voir quels paramètres peuvent être "tuned":
modelLookup(model='randomGLM') # no parameters to tune

plot(model_randomGLM)# does not work if no parameters to tune

plot(varImp(object=model_randomGLM),main="randomGLM - Variable Importance")

predictions_randomGLM<-predict.train(object=model_randomGLM,data_reg.test[,-51])

mean((data_reg.test$y-predictions_randomGLM)^2)
plot(predictions_randomGLM,data_reg.test$y)
abline(0,1)  


