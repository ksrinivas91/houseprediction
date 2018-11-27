#House Price Prediction using Random Forest
setwd("D:/analytics training/Important materials/Intern & Kaggle/house_price")
library(magrittr)
library(dplyr)			     # For data selection and filter
library(caret)			     # For parameter tuning
library(ggplot2)		     # For data plotting
library(randomForest)    # For Random Forest

df_train <- read.csv("train.csv",header=T)
df_test <- read.csv("test.csv",header=T)
attach(df_train)
attach(df_test)
apply(apply(df_train,2,is.na),2,sum);nrow(df_train)



#Visualise the dataset
# load libraries
library(Amelia)
library(mlbench)
# load dataset
df_train
# create a missing map
missmap(df_train, col=c("black", "red"), legend=TRUE)
#scatterplot
pairs(SalePrice~MSSubClass+LotFrontage+LotArea+GarageArea+GrLivArea,df_train,col=df_train$SalePrice)

# Correlation
cor(df_train[,c(5,27,63)])

#Explore Data
head(df_train)
head(df_test)
names(df_train)  
names(df_test)   
str(df_train)
str(df_test)

summary(df_train)
summary(df_test)

#Data Cleaning
# How many NA values are there in all
sum(is.na(df_train))
sum(is.na(df_test))

# How many values are missing and in which column
# Inner apply() reduces each value in column to TRUE or FALSE
# The outer apply(), sums up column wise and gives output

apply(apply(df_train,2,is.na),2,sum) ; nrow(df_train)  
apply(apply(df_test,2,is.na),2,sum) ; nrow(df_test)

df_train %<>% na.roughfix()
df_test %<>% na.roughfix()

#Check for any pending NA values
sum(is.na(df_train))
sum(is.na(df_test))

# Correlation1
library(corrplot)
df_train_cor <- cor(df_train[,c(5,27,63)])
corrplot(df_train_cor,method = "circle")

#Partition the data
trainindex<-createDataPartition(df_train$SalePrice,
                                p=0.8,
                                list=FALSE)

train <-df_train[trainindex,]
valid <-df_train[-trainindex,]
dim(train)
dim(valid)

#Build a model using random forest

rfModel <- randomForest(SalePrice ~. , data = train)
importance(rfModel)

rfPredict <- predict(rfModel, valid)
pred<-data.frame(rfPredict)

valid_verify <- cbind(valid$Id,valid$SalePrice,pred)
head(valid_verify)

SalePrice = predict(rfModel, valid,type = "response")

SalePrice<-data.frame(SalePrice)
head(SalePrice)
House_price <-cbind(data.frame("Id"=1461:(1460+nrow(valid)), SalePrice))
head(House_price)


##Training the model1
ctrl = trainControl(method="repeatedcv",
                    number=2,
                    repeats=1
)

# Create grid of parameters to be tuned
# In RF only one parameter can be tuned
tGrid <-  expand.grid(mtry = c(123))

# Model as also tune parameter(s)
system.time(
  rf_model<-train(SalePrice ~., # Standard formula notation
                  data=train[,-1],          # Except 'id'
                  method="rf",              # randomForest
                  nodesize= 10,              # 10 data-points/node. Speeds modeling
                  ntree =500,               # Default 500. Reduced to speed up modeling
                  do.trace= 10,             # Print output after every 10 trees
                  trControl=ctrl,           # cross-validation strategy
                  tuneGrid = tGrid
  )
)

#Results
rf_model$results             # Accuracy results
rf_model$bestTune            # Best parameter value
rf_model$metric              # Metric employed


#plot(randomForest(SalePrice ~ ., df_train)


#Prediction with RF model.
v_pred_prob <- predict(rf_model,
                       valid,
                       type = "raw")

pred = v_pred_prob
pred<-data.frame(pred)

valid_verify <- cbind(valid$Id,valid$SalePrice,pred)
head(valid_verify)

#Running the model on test data
SalePrice = predict(rf_model,
                    df_test,
                    type = "raw")


SalePrice<-data.frame(SalePrice)
head(SalePrice)
#Preparing prediction data as per submission template
# Add sequence number
y<-cbind(data.frame("Id"=1461:(1460+nrow(df_test)), SalePrice))
head(y)
# Save prediction to My_Submission.csv
write.csv(y,file="house_price.csv", quote = FALSE, row.names = FALSE)
