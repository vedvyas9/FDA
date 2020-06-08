#The caret package provides a consistent interface 
#into hundreds of machine learning algorithms 
#and provides useful convenience methods for 
#data visualization, data resampling, model tuning and model comparison etc.
install.packages('caret')
install.packages('e1071')
library(caret)
library(ggplot2)
library(kernlab)
library(tm)
library(randomForest)
library(dplyr)

#loading the train and test data
test=read.csv(file="test.csv")
#viewing the dataset
View(test)
train=read.csv(file="train.csv")
View(train
     )
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
#1.Set-up the test harness to use 10-fold  cross validation.
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
fit.knn<-train(Choice~.,data=train,method="knn",metric=metric, trControl=control)

#c) Advanced algorithms
#Support Vector Machines(SVM) with a linear kernel
set.seed(200)
fit.svm<-train(Choice~.,data=train,method="svmRadial",metric=metric, trControl=control)

#Random Forest
set.seed(200)
fit.rf<-train(Choice~.,data=train,method="rf",metric=metric, trControl=control)

#summarize accuracy of models
results<-resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)

#We can also create a plot of the model evaluation results and compare the spread and the mean accuracy of each model. There is a population of accuracy measures for each algorithm because each algorithm was evaluated 10 times (10 fold cross validation).
dotplot(results)

#We can clearly see that Random Forest has the most accuracy
#Summarize Best Model
print(fit.rf)

#Make Predictions
#RF was the most accurate algorithm.Now we want to get an idea of the accuracy of the model on our validation set.

#We can run the Rf model directly on the validation set and summarize the results in a confusion matrix.
validation$Choice<-as.factor(validation$Choice)
str(validation$Choice)
#estimate accuracy of RF on the validation dataset
predictions<-predict(fit.rf,validation)
confusionMatrix(predictions,validation$Choice)

#we can conclude that without pre-processing the accuracy with RF is 0.7746
#RF gives the best accuracy as this is a classification problem and not a regression or clustering problem

#Pre-Processing

#as there are many columns it is time consuming to plot a bell curve or histogram to check the skewness. Hence we use box-cox
#if the accuracy increases after applying box cox we will know that there was skewness
#we use box cox and not  Yeo-Johnson transformation as all the values are positive and non-zero.
#If a logarithmic transformation is applied to this distribution, the differences between smaller values will be expanded (because the slope of the logarithmic function is steeper when values are small) whereas the differences between larger values will be reduced (because of the very moderate slope of the log distribution for larger values).

#applying boxcox without removing outliers
train1=train
preprocessParams1<- preProcess(train1[,], method=c("BoxCox"))
print(preprocessParams1)
transformed1<- predict(preprocessParams1, train1[,])
summary(transformed1)

#Random Forest
set.seed(200)
fit.rf1<-train(Choice~.,data=transformed1,method="rf",metric=metric, trControl=control)

validation$Choice<-as.factor(validation$Choice)
str(validation$Choice)
predictions1<-predict(fit.rf1,validation)
confusionMatrix(predictions1,validation$Choice)
#accuracy=0.7627

#The value of center determines how column centering is performed. 
#The value of scale determines how column scaling is performed (after centering).
#applying center,scale on the train data without box cox applied and without removing outliers
train2=train
preprocessParams2 <- preProcess(train2[,], method=c("center","scale"))
print(preprocessParams2)
transformed2 <- predict(preprocessParams2, train2[,])
summary(transformed2)

#Random Forest
set.seed(200)
fit.rf2<-train(Choice~.,data=transformed2,method="rf",metric=metric, trControl=control)

validation$Choice<-as.factor(validation$Choice)
str(validation$Choice)
predictions2<-predict(fit.rf2,validation)
confusionMatrix(predictions2,validation$Choice)
#accuracy=0.5173

#applying centre,scale on the train data with box cox applied and without removing outliers
preprocessParams3 <- preProcess(train1[,], method=c("center","scale"))
print(preprocessParams3)
transformed3 <- predict(preprocessParams3, train1[,])
summary(transformed3)

#Random Forest
set.seed(200)
fit.rf3<-train(Choice~.,data=transformed3,method="rf",metric=metric, trControl=control)

validation$Choice<-as.factor(validation$Choice)
str(validation$Choice)
predictions3<-predict(fit.rf3,validation)
confusionMatrix(predictions3,validation$Choice)
#accuracy 0.5173

#Detecting outliers and eliminating them
train3=train
#1
Q1 <- quantile(train3$A_follower_count, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(train3$A_follower_count)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated<- subset(train3,train3$A_follower_count > (Q1[1] - 1.5*iqr1) & train3$A_follower_count< (Q1[2]+1.5*iqr1))
#2
Q1 <- quantile(eliminated$A_listed_count, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(eliminated$A_listed_count)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated<- subset(eliminated,eliminated$A_listed_count > (Q1[1] - 1.5*iqr1) & eliminated$A_listed_count< (Q1[2]+1.5*iqr1))
#3
Q1 <- quantile(eliminated$A_mentions_sent, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(eliminated$A_mentions_sent)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated<- subset(eliminated,eliminated$A_mentions_sent > (Q1[1] - 1.5*iqr1) & eliminated$A_mentions_sent< (Q1[2]+1.5*iqr1))
#4
Q1 <- quantile(eliminated$A_retweets_sent, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(eliminated$A_retweets_sent)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated<- subset(eliminated,eliminated$A_retweets_sent > (Q1[1] - 1.5*iqr1) & eliminated$A_retweets_sent< (Q1[2]+1.5*iqr1))
#5
Q1 <- quantile(eliminated$A_posts, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(eliminated$A_posts)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated<- subset(eliminated,eliminated$A_posts > (Q1[1] - 1.5*iqr1) & eliminated$A_posts< (Q1[2]+1.5*iqr1))
#6
Q1 <- quantile(eliminated$B_follower_count, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(eliminated$B_follower_count)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated<- subset(eliminated,eliminated$A_follower_count > (Q1[1] - 1.5*iqr1) & eliminated$A_follower_count< (Q1[2]+1.5*iqr1))
#7
Q1 <- quantile(eliminated$B_retweets_sent, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(eliminated$B_retweets_sent)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated<- subset(eliminated,eliminated$B_retweets_sent > (Q1[1] - 1.5*iqr1) & eliminated$B_retweets_sent< (Q1[2]+1.5*iqr1))
#8
Q1 <- quantile(eliminated$B_listed_count, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(eliminated$B_listed_count)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated<- subset(eliminated,eliminated$B_listed_count > (Q1[1] - 1.5*iqr1) & eliminated$B_listed_count< (Q1[2]+1.5*iqr1))
#9
Q1 <- quantile(eliminated$B_mentions_sent, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(eliminated$B_mentions_sent)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated<- subset(eliminated,eliminated$B_mentions_sent > (Q1[1] - 1.5*iqr1) & eliminated$B_mentions_sent< (Q1[2]+1.5*iqr1))
#10
Q1 <- quantile(eliminated$B_posts, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(eliminated$B_posts)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated<- subset(eliminated,eliminated$B_posts > (Q1[1] - 1.5*iqr1) & eliminated$B_posts< (Q1[2]+1.5*iqr1))
#11
Q1 <- quantile(eliminated$B_network_feature_1, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(eliminated$B_network_feature_1)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated<- subset(eliminated,eliminated$B_network_feature_1 > (Q1[1] - 1.5*iqr1) & eliminated$B_network_feature_1< (Q1[2]+1.5*iqr1))
#12
Q1 <- quantile(train3$A_following_count, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(train3$A_following_count)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated<- subset(train3,train3$A_following_count > (Q1[1] - 1.5*iqr1) & train3$A_following_count< (Q1[2]+1.5*iqr1))
#13
Q1 <- quantile(eliminated$A_mentions_received, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(eliminated$A_mentions_received)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated<- subset(eliminated,eliminated$A_mentions_received > (Q1[1] - 1.5*iqr1) & eliminated$A_mentions_received< (Q1[2]+1.5*iqr1))
#14
Q1 <- quantile(eliminated$A_retweets_received, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(eliminated$A_retweets_received)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated<- subset(eliminated,eliminated$A_retweets_received > (Q1[1] - 1.5*iqr1) & eliminated$A_retweets_received< (Q1[2]+1.5*iqr1))
#15
Q1 <- quantile(eliminated$A_network_feature_3, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(eliminated$A_network_feature_3)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated<- subset(eliminated,eliminated$A_network_feature_3 > (Q1[1] - 1.5*iqr1) & eliminated$A_network_feature_3< (Q1[2]+1.5*iqr1))
#16
Q1 <- quantile(eliminated$A_network_feature_2, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(eliminated$A_network_feature_2)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated<- subset(eliminated,eliminated$A_network_feature_2 > (Q1[1] - 1.5*iqr1) & eliminated$A_network_feature_2< (Q1[2]+1.5*iqr1))
#17
Q1 <- quantile(eliminated$A_network_feature_1, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(eliminated$A_network_feature_1)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated<- subset(eliminated,eliminated$A_network_feature_1 > (Q1[1] - 1.5*iqr1) & eliminated$A_network_feature_1< (Q1[2]+1.5*iqr1))
#18
Q1 <- quantile(eliminated$B_following_count, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(eliminated$B_following_count)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated<- subset(eliminated,eliminated$B_following_count > (Q1[1] - 1.5*iqr1) & eliminated$B_following_count< (Q1[2]+1.5*iqr1))
#19
Q1 <- quantile(eliminated$B_mentions_received, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(eliminated$B_mentions_received)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated<- subset(eliminated,eliminated$B_mentions_received > (Q1[1] - 1.5*iqr1) & eliminated$B_mentions_received< (Q1[2]+1.5*iqr1))
#20
Q1 <- quantile(eliminated$B_retweets_sent, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(eliminated$B_retweets_sent)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated<- subset(eliminated,eliminated$B_retweets_sent > (Q1[1] - 1.5*iqr1) & eliminated$B_retweets_sent< (Q1[2]+1.5*iqr1))
#21
Q1 <- quantile(eliminated$B_network_feature_3, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(eliminated$B_network_feature_3)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated<- subset(eliminated,eliminated$B_network_feature_3 > (Q1[1] - 1.5*iqr1) & eliminated$B_network_feature_3< (Q1[2]+1.5*iqr1))
#22
Q1 <- quantile(eliminated$B_network_feature_2, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(eliminated$B_network_feature_2)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated<- subset(eliminated,eliminated$B_network_feature_2 > (Q1[1] - 1.5*iqr1) & eliminated$B_network_feature_2< (Q1[2]+1.5*iqr1))

#applying BoxCox after removing outliers
preprocessParams4<- preProcess(eliminated[,], method=c("BoxCox"))
print(preprocessParams4)
transformed4 <- predict(preprocessParams4, eliminated[,])
summary(transformed4)

#Random Forest(outliers)
set.seed(200)
fit.rf4<-train(Choice~.,data=transformed4,method="rf",metric=metric, trControl=control)

validation$Choice<-as.factor(validation$Choice)
str(validation$Choice)
predictions4<-predict(fit.rf4,validation)
confusionMatrix(predictions4,validation$Choice)
#accuracy=0.6818

#applying center,scale after removing outliers
preprocessParams5<- preProcess(eliminated[,], method=c("center","scale"))
print(preprocessParams5)
transformed5 <- predict(preprocessParams5, eliminated[,])
summary(transformed5)

#Random Forest(outliers)
set.seed(200)
fit.rf5<-train(Choice~.,data=transformed5,method="rf",metric=metric, trControl=control)

validation$Choice<-as.factor(validation$Choice)
str(validation$Choice)
predictions5<-predict(fit.rf5,validation)
confusionMatrix(predictions5,validation$Choice)
#accuracy=0.5418



#Identifying the columns which effect the outcome most
train4=train
Choice<-as.numeric(train4$Choice)
#finding correlation

cor.test(train4$A_follower_count,Choice, method = "pearson")
cor.test(train4$A_listed_count,Choice, method = "pearson")
cor.test(train4$A_mentions_received,Choice,method = "pearson")
cor.test(train4$A_mentions_sent,Choice, method = "pearson")
cor.test(train4$A_retweets_received,Choice,method = "pearson")
cor.test(train4$A_retweets_sent,Choice, method = "pearson")
cor.test(train4$A_posts,Choice, method = "pearson")
cor.test(train4$A_network_feature_1,Choice, method = "pearson")
cor.test(train4$A_network_feature_2,Choice, method = "pearson")
cor.test(train4$A_network_feature_3,Choice, method = "pearson")


cor.test(train4$B_follower_count,Choice, method = "pearson")
cor.test(train4$B_listed_count,Choice, method = "pearson")
cor.test(train4$B_mentions_received,Choice,method = "pearson")
cor.test(train4$B_mentions_sent,Choice, method = "pearson")
cor.test(train4$B_retweets_received,Choice,method = "pearson")
cor.test(train4$B_retweets_sent,Choice, method = "pearson")
cor.test(train4$B_posts,Choice, method = "pearson")
cor.test(train4$B_network_feature_1,Choice, method = "pearson")
cor.test(train4B_network_feature_2,Choice, method = "pearson")
cor.test(train4$B_network_feature_3,Choice, method = "pearson")
cor.test(train4$B_following_count,Choice, method = "pearson")
cor(train4[, sapply(train4, class) != "factor"])

#summarize the correlation
cor(train4[, sapply(train4, class) != "factor"])

Choice<-as.factor(train4$Choice)
str(train4$Choice)



#appending train4 set with the most correlated values
train4<-train4[,c(-3,-6,-5,-10,-11,-12,-16,-17,-22,-23,-14)]
dim(train4)
colnames(train4)

#Random forest after appending train4 set with the most correlated values
set.seed(200)
fit.rfa<-train(Choice~.,data=train4,method="rf",metric=metric, trControl=control)

validation$Choice<-as.factor(validation$Choice)
str(validation$Choice)
predictionsa<-predict(fit.rfa,validation)
confusionMatrix(predictionsa,validation$Choice)
#accuracy 0.7682

#boxcox applied to the identified important columns
#without removing outliers
preprocessParams6<- preProcess(train4[,], method=c("BoxCox"))
print(preprocessParams6)
transformed6<- predict(preprocessParams6, train4[,])
summary(transformed6)


#Random Forest
set.seed(200)
fit.rf6<-train(Choice~.,data=transformed6,method="rf",metric=metric, trControl=control)

validation$Choice<-as.factor(validation$Choice)
str(validation$Choice)
predictions6<-predict(fit.rf6,validation)
confusionMatrix(predictions6,validation$Choice)
#Accuracy=0.7336

#applying center and scale on important columns
preprocessParams7<- preProcess(train4[,], method=c("center","scale"))
print(preprocessParams7)
transformed7 <- predict(preprocessParams7, train4[,])
summary(transformed7)

#Random Forest
set.seed(200)
fit.rf7<-train(Choice~.,data=transformed7,method="rf",metric=metric, trControl=control)

validation$Choice<-as.factor(validation$Choice)
str(validation$Choice)
predictions7<-predict(fit.rf7,validation)
confusionMatrix(predictions7,validation$Choice)
#Accuracy=0.52



#preprocessing for important columns along with removal of outliers
train5=train
train5<-train5[,c(-3,-6,-5,-10,-11,-12,-16,-17,-22,-23,-14)]
#1
Q1 <- quantile(train5$A_follower_count, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(train5$A_follower_count)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated1<- subset(train5,train5$A_follower_count > (Q1[1] - 1.5*iqr1) & train5$A_follower_count< (Q1[2]+1.5*iqr1))
#2
Q1 <- quantile(eliminated1$A_listed_count, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(eliminated1$A_listed_count)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated1<- subset(eliminated1,eliminated1$A_listed_count > (Q1[1] - 1.5*iqr1) & eliminated1$A_listed_count< (Q1[2]+1.5*iqr1))
#3
Q1 <- quantile(eliminated1$A_mentions_sent, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(eliminated1$A_mentions_sent)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated1<- subset(eliminated1,eliminated1$A_mentions_sent > (Q1[1] - 1.5*iqr1) & eliminated1$A_mentions_sent< (Q1[2]+1.5*iqr1))
#4
Q1 <- quantile(eliminated1$A_retweets_sent, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(eliminated1$A_retweets_sent)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated1<- subset(eliminated1,eliminated1$A_retweets_sent > (Q1[1] - 1.5*iqr1) & eliminated1$A_retweets_sent< (Q1[2]+1.5*iqr1))
#5
Q1 <- quantile(eliminated1$A_posts, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(eliminated1$A_posts)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated1<- subset(eliminated1,eliminated1$A_posts > (Q1[1] - 1.5*iqr1) & eliminated1$A_posts< (Q1[2]+1.5*iqr1))
#6
Q1 <- quantile(eliminated1$B_follower_count, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(eliminated1$B_follower_count)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated1<- subset(eliminated1,eliminated1$B_follower_count > (Q1[1] - 1.5*iqr1) & eliminated1$B_follower_count< (Q1[2]+1.5*iqr1))
#7
Q1 <- quantile(eliminated1$B_retweets_sent, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(eliminated1$B_retweets_sent)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated1<- subset(eliminated1,eliminated1$B_retweets_sent > (Q1[1] - 1.5*iqr1) & eliminated1$B_retweets_sent< (Q1[2]+1.5*iqr1))
#8
Q1 <- quantile(eliminated1$B_listed_count, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(eliminated1$B_listed_count)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated1<- subset(eliminated1,eliminated1$B_listed_count > (Q1[1] - 1.5*iqr1) & eliminated1$B_listed_count< (Q1[2]+1.5*iqr1))
#9
Q1 <- quantile(eliminated$B_mentions_sent, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(eliminated$B_mentions_sent)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated<- subset(eliminated,eliminated$B_mentions_sent > (Q1[1] - 1.5*iqr1) & eliminated$B_mentions_sent< (Q1[2]+1.5*iqr1))
#10
Q1 <- quantile(eliminated1$B_posts, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(eliminated1$B_posts)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated1<- subset(eliminated1,eliminated1$B_posts > (Q1[1] - 1.5*iqr1) & eliminated1$B_posts< (Q1[2]+1.5*iqr1))
#11
Q1 <- quantile(eliminated1$B_network_feature_1, probs=c(.25, .75), na.rm = FALSE)
iqr1 <- IQR(eliminated1$B_network_feature_1)
up <-  Q1[1]+1.5*iqr1 # Upper Range  
low<- Q1[1]-1.5*iqr1 # Lower Range
eliminated1<- subset(eliminated1,eliminated1$B_network_feature_1 > (Q1[1] - 1.5*iqr1) & eliminated1$B_network_feature_1< (Q1[2]+1.5*iqr1))


#applying BoxCox 
preprocessParams8<- preProcess(eliminated1[,], method=c("BoxCox"))
print(preprocessParams8)
transformed8 <- predict(preprocessParams8, eliminated1[,])
summary(transformed8)

#Random Forest(outliers)
set.seed(200)
fit.rf8<-train(Choice~.,data=eliminated1,method="rf",metric=metric, trControl=control)

validation$Choice<-as.factor(validation$Choice)
str(validation$Choice)
predictions8<-predict(fit.rf8,validation)
confusionMatrix(predictions8,validation$Choice)
#Accuracy=0.7455

#applying center,scale
preprocessParams9 <- preProcess(eliminated1[,], method=c("center","scale"))
print(preprocessParams9)
transformed9 <- predict(preprocessParams9, eliminated[,])
summary(transformed9)

#Random Forest(outliers)
set.seed(200)
fit.rf9<-train(Choice~.,data=transformed9,method="rf",metric=metric, trControl=control)

validation$Choice<-as.factor(validation$Choice)
str(validation$Choice)
predictions9<-predict(fit.rf9,validation)
confusionMatrix(predictions9,validation$Choice)
#accuracy=0.62


#summarize accuracy of models
results<-resamples(list(rf1=fit.rf1, rf2=fit.rf2, rf3=fit.rf3, rf4=fit.rf4, rf5=fit.rf5, rf6=fit.rf6, rf7=fit.rf7, rf8=fit.rf8, rf9=fit.rf9, rfa=fit.rfa))
summary(results)


#We can also create a plot of the model evaluation results and compare the spread and the mean accuracy of each model. There is a population of accuracy measures for each algorithm because each algorithm was evaluated 10 times (10 fold cross validation).
dotplot(results)

#We can clearly see that Random Forest2 and Random Forest3 have the most accuracy
#Summarize Best Models
print(fit.rf2)
print(fit.rf3)

#Make Predictions
#RF2(transformed2) and RF3(transformed3) were the most accurate algorithms.


#transformed2 and transformed3 sets gave the most accuracy
#we will train RF with those sets to predict values in the test set
test1=test
test2=test
test3=test
test4=test

model<-train(Choice~.,data=transformed2,method="rf",metric=metric, trControl=control)
model
predict_test<-predict(model,test1)
predict_test
summary(predict_test)

model1<-train(Choice~.,data=transformed3,method="rf",metric=metric, trControl=control)
model1
predict_test1<-predict(model1,test2)
predict_test1
summary(predict_test1)

model2<-train(Choice~.,data=transformed4,method="rf",metric=metric, trControl=control)
model2
predict_test2<-predict(model2,test3)
predict_test2
summary(predict_test2)

#original train dataset
model3<-train(Choice~.,data=train,method="rf",metric=metric, trControl=control)
model3
predict_test3<-predict(model3,test4)
predict_test3
summary(predict_test3)


#we can clearly see that model2 gives the most accurate predictions; it is clostest to the predictions made by the model using the original training set.
#model2 uses transformed4 set to train the RF model. i.e  the set we got applying BoxCox after removing outliers

#other ML Algos with transformed4
#a)Linear Algorithms
#Linear Discrimant Analysis(LDA)
set.seed(200)
fit.lda1<-train(Choice~.,data=transformed4,method="lda",metric=metric, trControl=control)

#b)Nonlinear Algorithms
#Classification and Regression Trees(Cart)
set.seed(200)
fit.cart1<-train(Choice~.,data=transformed4,method="rpart",metric=metric, trControl=control)

#k-Nearest Neighbors(kNN)
set.seed(200)
fit.knn1<-train(Choice~.,data=transformed4,method="knn",metric=metric, trControl=control)

#c) Advanced algorithms
#Support Vector Machines(SVM) with a linear kernel
set.seed(200)
fit.svm1<-train(Choice~.,data=transformed4,method="svmRadial",metric=metric, trControl=control)

#Random Forest
set.seed(200)
fit.rff<-train(Choice~.,data=transformed4,method="rf",metric=metric, trControl=control)

#summarize accuracy of models
results1<-resamples(list(lda=fit.lda1, cart=fit.cart1, knn=fit.knn1, svm=fit.svm1,rff=fit.rff))
summary(results1)

#We can also create a plot of the model evaluation results and compare the spread and the mean accuracy of each model. There is a population of accuracy measures for each algorithm because each algorithm was evaluated 10 times (10 fold cross validation).
dotplot(results1)
