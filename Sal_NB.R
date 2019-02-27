#####NAIVE_BAYES#####

#Prepare a classification model using Naive Bayes for salary data 

#loading dataset(training data)
library(DataExplorer)

sal_train<- read.csv(file.choose())
View(sal_train)
str(sal_train)
plot_str(sal_train)
plot_missing(sal_train)

sal_train$educationno <- as.factor(sal_train$educationno)
class(sal_train)
str(sal_train$educationno)

#loading dataset(testing data)
sal_test<- read.csv(file.choose())
View(sal_test)
str(sal_test)
plot_str(sal_test)
plot_missing(sal_test)

sal_test$educationno <- as.factor(sal_test$educationno)
class(sal_test)
str(sal_test$educationno)

#EDA and Visualizations
summary(sal_train)
#age: Mean= 38.44, Median= 37.00; As Mean>Median,it is skewed to the left.
#capitalgain: Mean= 1092, Median= 0.0; As Mean>Median,it is skewed to the left.
#capitalloss: Mean= 88.3, Median= 0.0; As Mean>Median,it is skewed to the left.
#hoursperweek: Mean= 40.93, Median= 40.00; As Mean>Median,it is skewed to the left.

summary(sal_test)
#age: Mean= 38.77, Median= 37.00; As Mean>Median,it is skewed to the left.
#capitalgain: Mean= 1120, Median= 0.0; As Mean>Median,it is skewed to the left.
#capitalloss: Mean= 89.04, Median= 0.0; As Mean>Median,it is skewed to the left.
#hoursperweek: Mean= 40.95, Median= 40.00; As Mean>Median,it is skewed to the left.

plot(sal_train$Salary, sal_train$age, xlab="Salary", ylab= "Age", col="red")
plot(sal_train$workclass,sal_train$Salary, xlab= "Work- Class", ylab="Salary")
plot(sal_train$education,sal_train$Salary, xlab= "Education", ylab="Salary")
plot(sal_train$educationno,sal_train$Salary, xlab= "Education number", ylab="Salary")
plot(sal_train$maritalstatus,sal_train$Salary, xlab= "Marital Status", ylab="Salary")
plot(sal_train$occupation,sal_train$Salary, xlab= "Occupation", ylab="Salary")
plot(sal_train$relationship,sal_train$Salary, xlab= "Relationship", ylab="Salary")
plot(sal_train$race,sal_train$Salary, xlab= "Race", ylab="Salary")
plot(sal_train$sex,sal_train$Salary, xlab= "Sex", ylab="Salary")
plot(sal_train$Salary,sal_train$capitalloss,ylab= "Capital gain", xlab="Salary")
plot(sal_train$Salary,sal_train$capitalloss, ylab= "Capital loss", xlab="Salary")
plot(sal_train$Salary,sal_train$hoursperweek, ylab= "Hours per week", xlab="Salary")
plot(sal_train$native,sal_train$Salary, xlab= "Native", ylab="Salary")

plot(sal_test$Salary, sal_test$age, xlab="Salary", ylab= "Age", col="red")
plot(sal_test$workclass,sal_test$Salary, xlab= "Work- Class", ylab="Salary")
plot(sal_test$education,sal_test$Salary, xlab= "Education", ylab="Salary")
plot(sal_test$educationno,sal_test$Salary, xlab= "Education number", ylab="Salary")
plot(sal_test$maritalstatus,sal_test$Salary, xlab= "Marital Status", ylab="Salary")
plot(sal_test$occupation,sal_test$Salary, xlab= "Occupation", ylab="Salary")
plot(sal_test$relationship,sal_test$Salary, xlab= "Relationship", ylab="Salary")
plot(sal_test$race,sal_test$Salary, xlab= "Race", ylab="Salary")
plot(sal_test$sex,sal_test$Salary, xlab= "Sex", ylab="Salary")
plot(sal_test$Salary,sal_test$capitalloss,ylab= "Capital gain", xlab="Salary")
plot(sal_test$Salary,sal_test$capitalloss, ylab= "Capital loss", xlab="Salary")
plot(sal_test$Salary,sal_test$hoursperweek, ylab= "Hours per week", xlab="Salary")
plot(sal_test$native,sal_test$Salary, xlab= "Native", ylab="Salary")

#Naive Bayes model
Model <- naiveBayes(sal_train$Salary ~ ., data = sal_train)
Model

Model_pred <- predict(Model,sal_test)
mean(Model_pred==sal_test$Salary)*100
# Model accuracy is 81.87%; Misclassification error is 18.13%

library(gmodels)
confusionMatrix(Model_pred,sal_test$Salary)

library(caret)
CrossTable(Model_pred,sal_test$Salary)
