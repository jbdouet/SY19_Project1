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

n= nrow(data_clas)
ntrain=ceiling(n*2/3)
ntst=n-ntrain
train<-sample(1:n,ntrain)
data_clas.test<-data_clas[-train,]
data_clas.train<-data_clas[train,]


# Classification 
library(MASS)

#### LDA ####

lda.clas<- lda(y~.,data=data_clas.train)
pred.clas<-predict(lda.clas,newdata=data_clas.test, type="response")
table(data_clas.test$y,pred.clas$class)
prop.table(table(data_clas.test$y,pred.clas$class))

cm= as.matrix(table(data_clas.test$y,pred.clas$class))
accuracy = sum(diag(cm)) / sum(cm) 
precision = sum(diag(cm)) / apply(cm, 2, sum) 
recall =  sum(diag(cm)) /  apply(cm, 1, sum)
f1 = 2 * precision * recall / (precision + recall) 
library("pROC")
roc_curve<-roc(data_clas.test$y,as.vector(pred.clas$x))
plot(roc_curve)

library("ROCR")
pr <- prediction(pred.clas$x,data_clas.test$y )
auc <-performance(pr, measure='auc')
auc<- auc@y.values[[1]]
auc

### Regression logistique ###
data_clas2.train  <- data_clas.train
data_clas2.test  <- data_clas.test
data_clas2.train$y <- data_clas.train$y-1
data_clas2.test$y <- data_clas.test$y-1

glm.fit<- glm(y~.,data=data_clas2.train,family=binomial)
summary(glm.fit)
# Pour regarder les coefficients significativement non nuls on s??lectionne ceux
# avec 2 ou 3 ??toiles dans le summary 
pred.clas.glm<-predict(glm.fit,newdata=data_clas2.test)
table(data_clas2.test$y,pred.clas.glm>0.5)
prop.table(table(data_clas2.test$y,pred.clas.glm>0.5))
logit<-predict(glm.fit,newdata=data_clas2.test,type='link')
roc_glm<-roc(data_clas2.test$y,pred.clas.glm)

plot(roc_glm,add=TRUE,col='red')

cm_glm= as.matrix(table(data_clas2.test$y,pred.clas.glm>0.4))
accuracy_glm = sum(diag(cm_glm)) / sum(cm_glm) 
precision_glm = sum(diag(cm_glm)) / apply(cm_glm, 2, sum) 
recall_glm =  sum(diag(cm_glm)) /  apply(cm_glm, 1, sum)
f1_glm = 2 * precision_glm * recall_glm / (precision_glm + recall_glm) 

accuracy_glm
hist(pred.clas.glm, breaks = 50)

