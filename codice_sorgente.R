# DATA IMPORT

# lettura file csv
apps = read.csv("apps-merged.csv", header = TRUE, sep = ";", encoding = "UTF-8")

# questa riga sistema il valore di dimensione di un record che superava i 4GB ed era quindi inconsistente
apps <- within(apps, sizeMB[sizeMB > 4000] <- 4000)

--------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------

# TRAINING SET
library(tidyverse)

apps_ml <- apps
# remove rating NA-----------------------------
# count records with rating = NA
na_rating_with_reviews <- apps_ml %>% filter(is.na(rating) & reviewsCount > 0)
dim(na_rating_with_reviews)
# delete records with rating = NA and reviewsCount > 0
apps_ml <- apps_ml[!(is.na(apps_ml$rating) & apps_ml$reviewsCount > 0),]
dim(apps_ml)
# set rating = 0 where reviewsCount = 0
apps_ml <- apps_ml %>% mutate(rating = (ifelse (reviewsCount == 0, 0, rating)))
summary(apps_ml)

# remove sizeMB NA------------------------------
# count records with sizeMB = NA
na_sizeMB <- apps_ml %>% filter(is.na(sizeMB))
dim(na_sizeMB)
# delete records with sizeMB = NA
apps_ml <- apps_ml[!(is.na(apps_ml$sizeMB)),]
dim(apps_ml)
summary(apps_ml)

# remove version NA-----------------------------
# count records with version = NA
na_version <- apps_ml %>% filter(is.na(version))
dim(na_version)
# delete records with sizeMB = NA
apps_ml <- apps_ml[!(is.na(apps_ml$version)),]
dim(apps_ml)
summary(apps_ml)

# check NA values
sum(is.na(apps_ml))

# remove id and name column 
apps_ml$id = NULL
apps_ml$name = NULL

--------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------

# RANDOM FOREST REGRESSION
library(rsample)      # data splitting 
library(randomForest) # basic implementation
library(ranger)       # a faster implementation of randomForest
library(caret)        # an aggregator package for performing many machine learning models

# for reproducibility 
set.seed(123)
# split for test and training
apps_ml_split <- initial_split(apps_ml, prop = .7)
# training set
apps_ml_train <- training(apps_ml_split)
# test set 
apps_ml_test  <- testing(apps_ml_split)

# for reproduciblity
set.seed(123)

# default RF model
rf_default <- randomForest(
  formula = rating ~ .,
  data    = apps_ml_train
)

rf_default
plot(rf_default, main="Default random forest Out-of-bag error")

# for reproduciblity
set.seed(123)
# further split training set into validation and training
validation_split <- initial_split(apps_ml_train, .8)

# training data
apps_ml_train_2 <- analysis(validation_split)

# validation data
apps_ml_validation <- assessment(validation_split)
x_test <- apps_ml_validation[setdiff(names(apps_ml_validation), "rating")]
y_test <- apps_ml_validation$rating

rf <- randomForest(
  formula = rating ~ .,
  data    = apps_ml_train_2,
  xtest   = x_test,
  ytest   = y_test
)

# extract OOB & validation errors
oobError <- sqrt(rf$mse)
validationError <- sqrt(rf$test$mse)

# compare error rates
tibble::tibble(
  `Out of Bag Error` = oobError,
  `Validation error` = validationError,
  ntrees = 1:rf$ntree
) %>%
  gather(Metric, RMSE, -ntrees) %>%
  ggplot(aes(ntrees, RMSE, color = Metric)) +
  geom_line() +
  scale_y_continuous() +
  xlab("Number of trees")


# 10-fold cross validation (caret package)
library(caret) # an aggregator package for performing many machine learning models

set.seed(123)

control <- trainControl(method = "cv", number = 10)

rf_ten_fold_cv <- train(rating~., 
                        data=apps_ml_train, 
                        method='rf', 
                        trControl=control)

print(rf_ten_fold_cv)

crossFoldValidationError <- sqrt(rf_ten_fold_cv$finalModel$mse)

# compare error rates
tibble::tibble(
  `Out of Bag error` = oobError,
  `Validation error` = validationError,
  `10-fold cross validation error` = crossFoldValidationError,
  ntrees = 1:rf$ntree
) %>%
  gather(Metric, RMSE, -ntrees) %>%
  ggplot(aes(ntrees, RMSE, color = Metric)) +
  geom_line() +
  scale_y_continuous() +
  xlab("Number of trees")

# prediction
# for reproduciblity
set.seed(123)

# RF model
rf_prediction <- randomForest(
  formula = rating ~ .,
  data    = apps_ml_train,
  importance= TRUE
)

prediction <- predict(rf_prediction, apps_ml_test)
head(prediction)


# feature importance
randomForest::importance(rf_prediction)

varImpPlot(rf_prediction, sort=TRUE, n.var=min(30, nrow(rf_prediction$importance)),
           type=NULL, class=NULL, scale=TRUE, 
           main="Feaure importance")

--------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------

# RANDOM FOREST CLASSIFICATION
library(tidyverse)
library(rsample)
library(caret)
library(e1071)

# copy dataset
apps_ml_bin <- apps_ml

# check rating column
summary(apps_ml_bin$rating)

median <- 4.2
dim(apps_ml_bin %>% filter(rating >= median))

apps_ml_bin <- apps_ml_bin %>% mutate(goodRating = (ifelse (rating >= median, 1, 0)))
apps_ml_bin$rating = NULL

dim(apps_ml_bin %>% filter(goodRating == 1))

apps_ml_bin <- apps_ml_bin %>% mutate(goodRating = factor(goodRating))
str(apps_ml_bin)


# for reproducibility 
set.seed(123)
# split for test and training
apps_ml_bin_split <- initial_split(apps_ml_bin, prop = .7)
# training set
apps_ml_bin_train <- training(apps_ml_bin_split)
# test set 
apps_ml_bin_test  <- testing(apps_ml_bin_split)


# prediction
randomForestClassifier <- train(goodRating ~ .,
                                data=apps_ml_bin_train,
                                method="rf",
                                prox=TRUE)

plot(randomForestClassifier)

prediction_bin <- predict(randomForestClassifier, apps_ml_bin_test, type="raw") # for class labels
head(prediction_bin)

# performance
confusionMatrix(prediction_bin, apps_ml_bin_test$goodRating, positive = '1')

# ROC and AUC
library(pROC)

# Prediction with prob
prediction_bin_prob <- predict(randomForestClassifier, apps_ml_bin_test, type="prob") 

# ROC curve
roc_curve <- roc(apps_ml_bin_test$goodRating, prediction_bin_prob$`1`)
plot(roc_curve, print.thres="best", print.thres.best.method="closest.topleft")

# AUC
auc(roc_curve)

--------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------
# SVM CLASSIFICATION
  
library(tidyverse)

apps_ml <- apps
# remove rating NA-----------------------------
# count records with rating = NA
na_rating_with_reviews <- apps_ml %>% filter(is.na(rating) & reviewsCount > 0)
dim(na_rating_with_reviews)
# delete records with rating = NA and reviewsCount > 0
apps_ml <- apps_ml[!(is.na(apps_ml$rating) & apps_ml$reviewsCount > 0),]
dim(apps_ml)
# set rating = 0 where reviewsCount = 0
apps_ml <- apps_ml %>% mutate(rating = (ifelse (reviewsCount == 0, 0, rating)))
summary(apps_ml)

# remove sizeMB NA------------------------------
# count records with sizeMB = NA
na_sizeMB <- apps_ml %>% filter(is.na(sizeMB))
dim(na_sizeMB)
# delete records with sizeMB = NA
apps_ml <- apps_ml[!(is.na(apps_ml$sizeMB)),]
dim(apps_ml)
summary(apps_ml)

# remove version NA-----------------------------
# count records with version = NA
na_version <- apps_ml %>% filter(is.na(version))
dim(na_version)
# delete records with sizeMB = NA
apps_ml <- apps_ml[!(is.na(apps_ml$version)),]
dim(apps_ml)
summary(apps_ml)

# check NA values
sum(is.na(apps_ml))

# remove name column
apps_ml$id = NULL
apps_ml$name = NULL

#carico libreria per SVM
install.packages("e1071")
library(e1071)

plot(apps$rating)

#Creo Train e Test Set
index <- sample(2,nrow(apps_ml),replace = TRUE, prob = c(0.7,0.3))
testset = apps_ml[index == 2,]
trainset = apps_ml[index == 1,]

#Istanzio e Creo il Modello
Model <- svm(rating~.,data = trainset, kernel = 'radial', cost = "25")

#Stampo il modello
print(Model)

#effettuo la previsione sul testset
svm.pred = predict(Model, testset)

#Visualizzo la prediction
svm.table = table(svm.pred, testset$rating)
svm.table

#Imposto Problema di Classificazione Binaria
library(tidyverse)
install.packages("rsample")
library(rsample)
library(caret)

# copy dataset
apps_ml_bin <- apps_ml

# check rating column
summary(apps_ml_bin$rating)

median <- 4.2
dim(apps_ml_bin %>% filter(rating >= median))

apps_ml_bin <- apps_ml_bin %>% mutate(goodRating = (ifelse (rating >= median, 1, 0)))
apps_ml_bin$rating = NULL

dim(apps_ml_bin %>% filter(goodRating == 1))

apps_ml_bin <- apps_ml_bin %>% mutate(goodRating = factor(goodRating))
str(apps_ml_bin)

#Creo Nuovi Test e Train Set

# for reproducibility 
set.seed(123)
# split for test and training
apps_ml_bin_split <- initial_split(apps_ml_bin, prop = .7)
# training set
apps_ml_bin_train <- training(apps_ml_bin_split)
# test set 
apps_ml_bin_test  <- testing(apps_ml_bin_split)

#analisi esplorativa training test
summary(apps_ml_bin_train)
head(apps_ml_bin_train)
tail(apps_ml_bin_train)
dim(apps_ml_bin_train)
plot(apps_ml_bin_train$goodRating)
#Creo Modello Classificazione Binaria                             
BModel <- svm(goodRating~.,data = apps_ml_bin_train, prob = TRUE, kernel = 'radial', cost ='25')

#Effettuo la previsione
print(BModel)
Bpred = predict(BModel, apps_ml_bin_test)
Bpred
Bpred_table = table(Bpred, apps_ml_bin_test$goodRating)
Bpred_table
confusionMatrix(Bpred, apps_ml_bin_test$goodRating,positive = '1')

#Calcolo Misure di Performance

# Precision
precision_0 = Bpred_table[1,1]/(Bpred_table[1,1]+Bpred_table[1,2])
precision_1 = Bpred_table[2,2]/(Bpred_table[2,2]+Bpred_table[2,1])
totalPrecision = (precision_0 + precision_1)/2

# Recall
recall_0 = Bpred_table[1,1]/(Bpred_table[1,1]+Bpred_table[2,1])
recall_1 = Bpred_table[2,2]/(Bpred_table[2,2]+Bpred_table[1,2])
totalRecall = (recall_0 + recall_1)/2

# F-Measure
fmeasure_0 = 2 / ((1/precision_0)+(1/recall_0))
fmeasure_1 = 2 / ((1/precision_1)+(1/recall_1))
totalFmeasure = (fmeasure_0 + fmeasure_1)/2

# Area Under Curve
install.packages("ROCR")
library(ROCR)
Bpred_ROC = predict(BModel, apps_ml_bin_test, probability = TRUE)
probs = attr(Bpred_ROC,"probabilities")[,2]

Bpred_ROC.rocr = prediction(probs, apps_ml_bin_test$goodRating)
perf.rocr = performance(Bpred_ROC.rocr,measure="auc",x.measure = "cutoff")
perf.tpr.rocr = performance(Bpred_ROC.rocr,"tpr","fpr")
plot(perf.tpr.rocr,colorize=T,main=paste("AUC:",(perf.rocr@y.values)))

perf.rocr
abline(a = 0, b = 1)

#10-fold Cross Validation
tuneControl <-tune.control(cross = 10)


