#QDA classification, same algo than LDA

qda.clas<- qda(y~.,data=data_clas.train)
pred.clas2<-predict(qda.clas,newdata=data_clas.test, type="response")
table(data_clas.test$y,pred.clas2$class)
prop.table(table(data_clas.test$y,pred.clas2$class))
cm2= as.matrix(table(data_clas.test$y,pred.clas2$class))
accuracy2 = sum(diag(cm2)) / sum(cm2) #0.606

#Naive Bayes classification

library(e1071)
data_clas.train$y=as.factor(data_clas.train$y)
data_clas.test$y=as.factor(data_clas.test$y)
naive.clas<- naiveBayes(y~.,data=data_clas.train)
pred.clas.naive<-predict(naive.clas,newdata=data_clas.test)
perf.naive <-table(data_clas.test$y,pred.clas.naive)
cm_naive= as.matrix(table(data_clas.test$y,pred.clas.naive))
accuracy_naive = sum(diag(cm_naive)) / sum(cm_naive) #0.7575

#Regression linéaire

data_reg<-read.csv('data/tp3_reg_app.txt',sep=' ')
n= nrow(data_reg)
ntrain=ceiling(n*2/3)
ntst=n-ntrain
train<-sample(1:n,ntrain)
data_reg.test<-data_reg[-train,]
data_reg.train<-data_reg[train,]

#regression avec tous les prédicteurs

reg <- lm(y~., data = data_reg.train)
sm <-summary(reg)
quad_err <- mean(sm$residuals^2) #erreur quadratique 45.33 p-value=6.86e-16 R^2=0.8194

#regression lineaire en utilisant que les predicteurs les plus significatifs (** ou ***)on supprime X30 qui devient peu sign. dans ce nouveau domaine de prédicteurs

lm.model<- lm(y~X4+X12+X19+X22+X24+X27+X35+X39+X41, data = data_reg.train)
sm2<-summary(lm.model)
quad_err2<-mean(sm2$residuals^2)#98.30 p-value=2.2e-16 R^2=0.6083

library(leaps)

#Best subset (long à calculer)
lm.fit<-regsubsets(y~.,data=data_reg.train,method='exhaustive', nvmax=15, really.big = T)
plot(reg.fit,scale="r2")
lm_subset<- lm(y~X4+X6+X8+X12+X19+X20+X22+X24+X27+X30+X32+X35+X39+X41+X48, data = data_reg.train)
sm_subset<-summary(lm_subset)
quad_err_subset<-mean(sm_subset$residuals^2)#68.44 p-value=2.2e-16 R^2=0.7273


#Forward stepwise
lm.fit_forw<-regsubsets(y~.,data=data_reg.train,method='forward', nvmax=15)
#on obtient le même modèle que best subset


n_folds <- 10
folds_i <- sample(rep(1:n_folds, length.out = ntrain)) # !!! le ntrain doit correspondre à la taille du dataset que l'on utilisera dans la boucle de cross validation 
table(folds_i) # Pas le même nombre d'éléments 
CV<-rep(0,10)
for (k in 1:n_folds) {# we loop on the number of folds, to build k models
  test_i <- which(folds_i == k)
  # les datasets entre le fit et le predict doivent être les mêmes car c'est le même dataset que l'on divise en k-fold 
  # on peut utiliser le data set complet ou seulement le train et avoir une idée finale de la performance sur le test
  train_xy <- data_reg.train[-test_i, ]
  test_xy <- data_reg.train[test_i, ]
  print(k)
  lm.fit<- lm(y~.,data=train_xy)#ici pour comparer nos modèles on ajuste notre ensemble y~ .
  cv_pred<-predict(lm.fit,newdata=test_xy, type = "response") 
  CV[k]<- sum((test_xy$y-cv_pred)^2)
}
CVerror= sum(CV)/length(CV)
#Pour y~. on a CVerror = 1645.5
#Pour Best subset on a CVerror = 1201.5
#Pour les prédicteurs les plus significatifs on a CVerror = 1672.6

confint(reg,level=0.95)
y <-fitted(reg,data_reg.test)