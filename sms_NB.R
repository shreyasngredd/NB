#####NAIVE_BAYES#####

#Build a naive Bayes model on the data set for classifying the ham and spam

#loading dataset
sms<- read.csv(file.choose())
View(sms)

str(sms)
table(sms$type)

library(tm)
# Prepare corpus for the text data 
sms_cp<-Corpus(VectorSource(sms$text))

#cleaning data
corpus_clean<-tm_map(sms_cp,tolower)
corpus_clean<-tm_map(corpus_clean, removeNumbers)
corpus_clean<-tm_map(corpus_clean,removeWords, stopwords())
corpus_clean<-tm_map(corpus_clean,removePunctuation)
corpus_clean<-tm_map(corpus_clean,stripWhitespace)
class(corpus_clean)

#creating a document-term sparse matrix
sms_dtm <- DocumentTermMatrix(corpus_clean) 
class(sms_dtm)

#as.character(sms_dtm)

#creating training and test datasets
sms_raw_train <- sms[1:4169, ]
sms_raw_test  <- sms[4170:5559, ]

sms_dtm_train <- sms_dtm[1:4169, ]
sms_dtm_test  <- sms_dtm[4170:5559, ]

sms_corpus_train <- corpus_clean[1:4169]
sms_corpus_test  <- corpus_clean[4170:5559]

#checking if the proportion of spam is similar
prop.table(table(sms_raw_train$type))
prop.table(table(sms_raw_test$type))

#Indicator features for frequent words
sms_dict<-findFreqTerms(sms_dtm_train, 5)

sms_train <- DocumentTermMatrix(sms_corpus_train, list(dictionary = sms_dict))
sms_test  <- DocumentTermMatrix(sms_corpus_test, list(dictionary = sms_dict))
sms_dict

temp <- as.data.frame(as.matrix(sms_train))

View(temp)
dim(sms_train)
dim(sms_test)

#converting counts to a factor
convert_counts <- function(x) {
  x <- ifelse(x > 0, 1, 0)
  x <- factor(x, levels = c(0, 1), labels = c("No", "Yes"))
}

# applying convertion to columns of train/test data
sms_train <- apply(sms_train, MARGIN = 2, convert_counts)
sms_test  <- apply(sms_test, MARGIN = 2, convert_counts)
View(sms_train)
View(sms_test)

#running naive bayes
library(e1071)
sms_classifier <- naiveBayes(sms_train, sms_raw_train$type)
sms_classifier

#Evaluating model performance
sms_test_pred <- predict(sms_classifier, sms_test)

library(gmodels)
CrossTable(sms_test_pred, sms_raw_test$type,prop.chisq = FALSE, 
           prop.t = FALSE, prop.r = FALSE, dnn = c('predicted', 'actual'))

library(caret)
confusionMatrix(sms_test_pred,sms_raw_test$type)

mean(sms_test_pred==sms_raw_test$type)*100
# Model accuracy is 97.33%; Misclassification error is 2.67%
