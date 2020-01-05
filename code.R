### Load the Dataset
hr <- read.csv(file.choose())
View(hr)
dim(hr)

hr$Attrition <- as.numeric(as.numeric(hr[,c(2)]))# Yes-2, No-1
hr$BusinessTravel <- as.numeric(as.numeric(hr[,c(3)]))# Travel-Rarely-3, Travel-Frequently-2, Non-Travel-1
hr$Department <- as.numeric(as.numeric(hr[,c(5)]))# Sales-3, Research & Development-2, Human Resources-1
hr$EducationField <- as.numeric(as.numeric(hr[,c(8)]))# Technical Degree-6, Other-5, Medical-4, Life Sciences-2, Human Resources-1 
hr$Gender <- as.numeric(as.numeric(hr[,c(12)]))# Male-2, Female-1
hr$JobRole <- as.numeric(as.numeric(hr[,c(16)]))
hr$MaritalStatus <- as.numeric(as.numeric(hr[,c(18)]))
hr$OverTime <- as.numeric(as.numeric(hr[,c(23)]))
hr$Over18 <- as.numeric(as.numeric(hr[,c(22)]))

###data processing
#####"EmployeeCount" and "EmployeeNumber" has no meaning for our analysis
hr <- hr[,-c(9,10)]
#####cleaning NAs
hr <- na.omit(hr)

#####upsample data to make it balance
library(caret)
table(hr$Attrition)
#######down sampling
set.seed(123)
hr$Attrition <- as.factor(hr$Attrition)
hr <- upSample(x=hr[,-ncol(hr)], y=hr$Attrition)
hr <- as.data.frame(hr)
table(hr$Attrition)


#####identify zero-variance column
which(apply(hr,2,var)==0)
#####remove zero variance columns from the dataset
hr_explore <- hr[,apply(hr,2,var)!=0]
#####remove "class" column from data
hr_explore <- hr_explore[,-c(31)]

######correlation
cor_matrix <- abs(cor(hr_explore[,-c(2)]))
diag(cor_matrix) <- 0
library(corrplot)
corrplot(cor_matrix,method = "square")

library(stats)
########################using PCA do reduce dimensions 
###data normalization
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
hr_explore[,-c(2)] <- as.data.frame(lapply(hr_explore[,-c(2)], normalize))

prComp <- prcomp(hr_explore[,-c(2)],scale. = TRUE)

std_dev <- prComp$sdev
pr_var <- std_dev^2
prop_varex <- pr_var/sum(pr_var)
sum(prop_varex[1:15])

plot(cumsum(prop_varex), xlab="Principal Component", ylab="Cumulative Proportion of Variance Explained", type = "b")
abline(h=0.766,col='red',v=15)

library(ggfortify)
autoplot(prComp, data = hr_explore[,-c(2)], loadings=TRUE , loadings.colour='blue', loadings.label=TRUE, loading.label.size=3)


############################# -comparing 4 classification - whoel data
library(caret)
######split data to test and training set
hr_train <- data.frame(attrition = hr_explore$Attrition, prComp$x)
hr_train <- hr_train[,1:16]
train <- sample(nrow(hr_train),nrow(hr_train)*0.8)
model_train <- hr_train[train,]
model_test <- hr_train[-train,]

####cross validation with 10 folds 
control <- trainControl(method = "cv", number = 10)

model_train$attrition <- as.factor(model_train$attrition)
model_test$attrition <- as.factor(model_test$attrition)
###lda
set.seed(123)
hr_lda <- train(attrition~., data=model_train, method="lda", trControl=control)
#######make predictions
prediction_lda <- predict(hr_lda, model_test)
#########prediction right rate
library(pROC)
roc_lda <- roc(model_test$attrition, as.numeric(prediction_lda))
auc(roc_lda)
confusionMatrix(prediction_lda,model_test$attrition)
roc1 <- plot(roc(model_test$attrition, as.numeric(prediction_lda)),print.auc = TRUE, col="blue")

###glm
set.seed(123)
hr_glm <- train(attrition~., data=model_train, method="glm", family=binomial, trControl=control)
########make predictions
prediction_glm <- predict(hr_glm, model_test)
#########prediction right rate
roc_glm <- roc(model_test$attrition, as.numeric(prediction_glm))
auc(roc_glm)
confusionMatrix(prediction_glm,model_test$attrition)
roc1 <- plot(roc(model_test$attrition, as.numeric(prediction_glm)),print.auc = TRUE, col="green", print.auc.y=0.2, add=TRUE)

###knn
set.seed(123)
hr_knn <- train(attrition~., data=model_train, method="knn", trControl=control)
########make predictions
prediction_knn <- predict(hr_knn, model_test)
#########prediction right rate
roc_knn <- roc(model_test$attrition, as.numeric(prediction_knn))
auc(roc_knn)
confusionMatrix(prediction_knn,model_test$attrition)
roc1 <- plot(roc(model_test$attrition, as.numeric(prediction_knn)),print.auc = TRUE, col="yellow", print.auc.y=0.4, add=TRUE)

####random forest
mtry <- sqrt(ncol(model_train))
tunegrid <- expand.grid(.mtry = mtry)
metric <- "Accuracy"
hr_rf <- train(attrition~.,data = model_train,method = "rf",metric=metric, tuneGrid=tunegrid, trControl=control)
########make predictions
prediction_rf <- predict(hr_rf, model_test)
#########prediction right rate
roc_rf <- roc(model_test$attrition, as.numeric(prediction_rf))
auc(roc_rf)
confusionMatrix(prediction_rf,model_test$attrition)
roc1 <- plot(roc(model_test$attrition, as.numeric(prediction_rf)),print.auc = TRUE, col="red", print.auc.y=0.6, add=TRUE)
legend("right", legend = c("ROC-lda", "ROC-glm","ROC-knn", "ROC-rf"), col = c("blue", "green","yellow","red"), lty = 1)
#################################################################################################
library(tidyverse)  # data manipulation
library(cluster)    # clustering algorithms
library(factoextra) # clustering algorithms & visualization
### "Clustering"
########distance matrix
library(factoextra)
distance <- get_dist(hr_train)
fviz_dist(distance, gradient = list(low = "#00AFBB", mid = "white", high = "#FC4E07"))

## Determine number of clusters

####function to compute average silhouette for k clusters
avg_sil <- function(k) {
  km.res <- kmeans(hr_train, centers = k, nstart = 25)
  ss <- silhouette(km.res$cluster,dist(hr_train))
  mean(ss[,3])
}

####compute and plot wss for k =2 to k = 15
k.values <- 2:15

####extract avg silhouette for 2-15 clusters
avg_sil_values <- map_dbl(k.values, avg_sil)

plot(k.values,avg_sil_values,
     type = "b", pch=19, frame=FALSE,
     xlab="Number of clusters K",
     ylab="Average Silhouettes")

fviz_nbclust(hr_train, kmeans, method = "silhouette")

## K-Means Cluster Analysis
fit <- kmeans(hr_train, 2, nstart = 25) #2 clusters


# Cluster sizes
sort(table(fit$clust))
clust <- names(sort(table(fit$clust)))

# First cluster
cluster1 <- hr_train[fit$clust==clust[1],]
# Second Cluster
cluster2 <- hr_train[fit$clust==clust[2],] 


##################################################################################
######cluster 1 - classification comparisons
######split data to test and training set
c1_sample_train <- sample(nrow(cluster1),nrow(cluster1)*0.8)
c1_train <- cluster1[c1_sample_train,]
c1_test <- cluster1[-c1_sample_train,]

c1_train$attrition <- as.factor(c1_train$attrition)
c1_test$attrition <- as.factor(c1_test$attrition)
###lda
set.seed(123)
c1_lda <- train(attrition~., data=c1_train, method="lda", trControl=control)
#######make predictions
prediction_c1_lda <- predict(c1_lda, c1_test)
#########prediction right rate
library(pROC)
roc_c1_lda <- roc(c1_test$attrition, as.numeric(prediction_c1_lda))
auc(roc_c1_lda)
confusionMatrix(prediction_c1_lda,c1_test$attrition)
#######plot roc
library(pROC)
roc_c1 <- plot(roc(c1_test$attrition, as.numeric(prediction_c1_lda)),print.auc = TRUE, col="blue")


###glm
set.seed(123)
c1_glm <- train(attrition~., data=c1_train, method="glm", family=binomial, trControl=control)
########make predictions
prediction_c1_glm <- predict(c1_glm, c1_test)
#########prediction right rate
roc_c1_glm <- roc(c1_test$attrition, as.numeric(prediction_c1_glm))
auc(roc_c1_glm)
confusionMatrix(prediction_c1_glm,c1_test$attrition)
########plot roc
roc_c1 <- plot(roc(c1_test$attrition, as.numeric(prediction_c1_glm)),print.auc = TRUE, col="green", print.auc.y=0.2, add=TRUE)


###knn
set.seed(123)
c1_knn <- train(attrition~., data=c1_train, method="knn", trControl=control)
########make predictions
prediction_c1_knn <- predict(c1_knn, c1_test)
#########prediction right rate
roc_c1_knn <- roc(c1_test$attrition, as.numeric(prediction_c1_knn))
auc(roc_c1_knn)
confusionMatrix(prediction_c1_knn, c1_test$attrition)
#######plot roc
roc_c1 <- plot(roc(c1_test$attrition, as.numeric(prediction_c1_knn)),print.auc = TRUE, col="yellow", print.auc.y=0.4, add=TRUE)

####random forest
mtry <- sqrt(ncol(c1_train))
tunegrid <- expand.grid(.mtry = mtry)
metric <- "Accuracy"
c1_rf <- train(attrition~.,data = c1_train, method = "rf", metric=metric, tuneGrid=tunegrid, trControl=control)
########make predictions
prediction_c1_rf <- predict(c1_rf, c1_test)
#########prediction right rate
roc_c1_rf <- roc(c1_test$attrition, as.numeric(prediction_c1_rf))
auc(roc_c1_rf)
confusionMatrix(prediction_c1_rf, c1_test$attrition)
##########plot roc
roc_c1 <- plot(roc(c1_test$attrition, as.numeric(prediction_c1_rf)),print.auc = TRUE, col="red", print.auc.y=0.6, add=TRUE)
legend("right", legend = c("ROC-lda", "ROC-glm","ROC-knn", "ROC-rf"), col = c("blue", "green","yellow","red"), lty = 1)

####################cluster 2 - classification comparisons
######split data to test and training set
c2_sample_train <- sample(nrow(cluster2),nrow(cluster2)*0.8)
c2_train <- cluster2[c2_sample_train,]
c2_test <- cluster2[-c2_sample_train,]

###lda
set.seed(123)
c2_lda <- train(attrition~., data=c2_train, method="lda", trControl=control)
#######make predictions
prediction_c2_lda <- predict(c2_lda, c2_test)
#########prediction right rate
library(pROC)
roc_c2_lda <- roc(c2_test$attrition, as.numeric(prediction_c2_lda))
auc(roc_c2_lda)
confusionMatrix(prediction_c2_lda, c2_test$attrition)
roc_c2 <- plot(roc(c2_test$attrition, as.numeric(prediction_c2_lda)),print.auc = TRUE, col="blue")


###glm
set.seed(123)
c2_glm <- train(attrition~., data=c2_train, method="glm", family=binomial, trControl=control)
########make predictions
prediction_c2_glm <- predict(c2_glm, c2_test)
#########prediction right rate
roc_c2_glm <- roc(c2_test$attrition, as.numeric(prediction_c2_glm))
auc(roc_c2_glm)
confusionMatrix(prediction_c2_glm, c2_test$attrition)
roc_c2 <- plot(roc(c2_test$attrition, as.numeric(prediction_c2_glm)),print.auc = TRUE, col="green", print.auc.y=0.2, add=TRUE)

###knn
set.seed(123)
c2_knn <- train(attrition~., data=c2_train, method="knn", trControl=control)
########make predictions
prediction_c2_knn <- predict(c2_knn, c2_test)
#########prediction right rate
roc_c2_knn <- roc(c2_test$attrition, as.numeric(prediction_c2_knn))
auc(roc_c2_knn)
confusionMatrix(prediction_c2_knn, c2_test$attrition)
roc_c2 <- plot(roc(c2_test$attrition, as.numeric(prediction_c2_knn)),print.auc = TRUE, col="yellow", print.auc.y=0.4, add=TRUE)

####random forest
mtry <- sqrt(ncol(c2_train))
tunegrid <- expand.grid(.mtry = mtry)
metric <- "Accuracy"
c2_rf <- train(attrition~.,data = c2_train, method = "rf", metric=metric, tuneGrid=tunegrid, trControl=control)
########make predictions
prediction_c2_rf <- predict(c2_rf, c2_test)
#########prediction right rate
roc_c2_rf <- roc(c2_test$attrition, as.numeric(prediction_c2_rf))
auc(roc_c2_rf)
confusionMatrix(prediction_c2_rf,c2_test$attrition)
roc_c2 <- plot(roc(c2_test$attrition, as.numeric(prediction_c2_rf)),print.auc = TRUE, col="red", print.auc.y=0.6, add=TRUE)
legend("right", legend = c("ROC-lda", "ROC-glm","ROC-knn", "ROC-rf"), col = c("blue", "green","yellow","red"), lty = 1)

