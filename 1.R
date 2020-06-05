#The caret package provides a consistent interface 
#into hundreds of machine learning algorithms 
#and provides useful convenience methods for 
#data visualization, data resampling, model tuning and model comparison etc.
library(caret)
library(ggplot2)

#loading the train and test data
test=read.csv(file="test.csv")
#viewing the dataset
View(test)
train=read.csv(file="train.csv")
View(train)

#create a list of 80% of the rows of the original dataset
#we can use for training
#we use set.seed() to obtain the same split every time
set.seed(123)
validation_set<-createDataPartition(train$Choice, p=0.80, list=FALSE)
View(validation_set)
#select 20% of the data for validation
validation<-train[-validation_set,]
View(validation)

#use the remaining 80% of data to train and test the models
train<-train[validation_set,]
View(train)

#dimensions of dataset
dim(train)

#types for each attribute
sapply(train,class)

#converting class label "Choice" to factor type from integer.
train$Choice=as.factor(train$Choice)

#checking the datatypes again
sapply(train,class)

#first 5 rows to get an idea of the data
head(train)

# as the label class "Choice" is categorical we check equal distribution
sum(train$Choice==0)
sum(train$Choice==1)
percentage <- prop.table(table(train$Choice) * 100)
cbind(freq=table(train$Choice), percentage=percentage)
p<-ggplot(train,aes(Choice)) 
p + geom_bar(fill="red")+ geom_text(stat='count',aes(label=..count..),vjust=-1)

#no changes are required as the number of 1's and 0's are almost equal.

#list the levels for the class label
levels(train$Choice)
#we can see there are two levels, making it a binary classification problem


#statistical summary
summary(train)

#Prediction algorithms before any preprocessing
#1.Set-up the test harness to use 10-fold. cross validation.
#2.Build 5 different models to predict.
#3.Select the best model.
#We will use 10-fold crossvalidation to estimate accuracy.
#This will split our dataset into 10 parts, train in 9 and test on 1 and release for all combinations of train-test splits. We will also repeat the process 3 times for each algorithm with different splits of the data into 10 groups, in an effort to get a more accurate estimate.

#Run algorithms using 10-fold cross validation
control<-trainControl(method="cv",number=10)
#We are using the metric of “Accuracy” to evaluate models. This is a ratio of the number of correctly predicted instances in divided by the total number of instances in the dataset multiplied by 100 to give a percentage (e.g. 95% accurate). We will be using the metric variable when we run build and evaluate each model next.
metric<-"Accuracy"

#Build models
#We reset the random number seed before reach run to ensure that the evaluation of each algorithm is performed using exactly the same data splits. It ensures the results are directly comparable.
#5 different algorithms

#a)Linear Algorithms
#Linear Discrimant Analysis(LDA)
set.seed(200)
fit.lda<-train(Choice~.,data=train,method="lda",metric=metric, trControl=control)

#b)Nonlinear Algorithms
#Classification and Regression Trees(Cart)
set.seed(200)
fit.cart<-train(Choice~.,data=train,method="rpart",metric=metric, trControl=control)

#k-Nearest Neighbors(kNN)
set.seed(200)
fir.knn<-train(Choice~.,data=train,method="knn",metric=metric, trControl=control)

#c) Advanced algorithms
#Support Vector Machines(SVM) with a linear kernel
set.seed(200)
fit.svm<-train(Choice~.,data=train,method="svmRadial",metric=metric, trControl=control)

#Random Forest
set.seed(200)
fit.rf<-train(Choice~.,data=train,method="rf",metric=metric, trControl=control)

#summarize accuracy