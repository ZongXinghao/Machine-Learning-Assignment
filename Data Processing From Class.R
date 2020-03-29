Sys.setenv(LANG = "en")

# Data processing library
library(data.table)       # Data manipulation
library(plyr)             # Data manipulation
library(stringr)          # String, text processing
library(vita)             # Quickly check variable importance
library(dataPreparation)  # Data preparation library
library(woeBinning)       # Decision tree-based binning for numerical and categorical variables
library(Boruta)           # Variable selection

# Machine learning library
library(mlr)          # Machine learning framework
library(caret)         # Data processing and machine learning framework
library(MASS)          # LDA
library(randomForest)  # RF
library(gbm)           # Boosting Tree
library(xgboost)       # XGboost

setwd("C:/Users/xzong/Desktop/MBD/Machine Learning/In-class Kaggle Competition/data/bank_mkt")


# Read train (full), test (holdout)
train_full <- read.csv('./bank_mkt_train.csv')  # Training dataset
test_holdout <- read.csv('./bank_mkt_test.csv')  # Holdout data set without response


# Print out to check the data type
str(train_full)

# Fix the value
train_full[, 'campaign'] <- train_full[, 'campaign'] - 1
test_holdout[, 'campaign'] <- test_holdout[, 'campaign'] - 1

# Quick check
min(train_full[, 'campaign'])  # Previously = 1
min(test_holdout[, 'campaign'])  # Previously = 1


# Check missing value
apply(is.na(train_full), 2, sum)

set.seed(1)

train_idx <- caret::createDataPartition(y=train_full[, 'subscribe'], p=.6, list=F)
train <- train_full[train_idx, ]  # Train 60%
valid_test <- train_full[-train_idx, ]  # Valid + Test 40%

valid_idx <- caret::createDataPartition(y=valid_test[, 'subscribe'], p=.5, list=F)
valid <- valid_test[valid_idx, ]  # Valid 20%
test <- valid_test[-valid_idx, ]  # Test 20%

train <-train_full
# By number
table(train$subscribe)
table(valid$subscribe)
table(test$subscribe)

# By percentage
table(train$subscribe) / nrow(train)
table(valid$subscribe) / nrow(valid)
table(test$subscribe) / nrow(test)


# PIMP-Algorithm For The Permutation Variable Importance Measure
# https://cran.r-project.org/web/packages/vita/vita.pdf
X <- train[, 2:(ncol(train)-1)]
y <- as.factor(train[, 'subscribe'])
rf_model <- randomForest(X, y, mtry=3, ntree=100, importance=T, seed=1)
pimp_varImp <- PIMP(X, y, rf_model, S=10, parallel=F, seed=123)


# Print out top most important variables
pimp_varImp$VarImp[order(pimp_varImp$VarImp[, 1], decreasing=T), ]


# Add new variable to train and test (holdout)
# Train, valid, test
train[, 'month_spring'] <- as.logical(train$month %in% c('mar', 'apr', 'may'))
valid[, 'month_spring'] <- as.logical(valid$month %in% c('mar', 'apr', 'may'))
test[, 'month_spring'] <- as.logical(test$month %in% c('mar', 'apr', 'may'))
# Test (holdout)
test_holdout[, 'month_spring'] <- as.logical(test_holdout$month %in% c('mar', 'apr', 'may'))


# Add new variable to train and test (holdout)
# Train, valid, test
train[, 'month_summer'] <- as.logical(train$month %in% c('jun', 'jul', 'aug'))
valid[, 'month_summer'] <- as.logical(valid$month %in% c('jun', 'jul', 'aug'))
test[, 'month_summer'] <- as.logical(test$month %in% c('jun', 'jul', 'aug'))
# Test (holdout)
test_holdout[, 'month_summer'] <- as.logical(test_holdout$month %in% c('jun', 'jul', 'aug'))

# Add new variable to train and test (holdout)
# Train, valid, test
train[, 'month_autumn'] <- as.logical(train$month %in% c('sep', 'oct', 'nov'))
valid[, 'month_autumn'] <- as.logical(valid$month %in% c('sep', 'oct', 'nov'))
test[, 'month_autumn'] <- as.logical(test$month %in% c('sep', 'oct', 'nov'))
# Test (holdout)
test_holdout[, 'month_autumn'] <- as.logical(test_holdout$month %in% c('sep', 'oct', 'nov'))


# Add new variable to train and test (holdout)
# Train, valid, test
train[, 'month_winter'] <- as.logical(train$month %in% c('dec', 'jan', 'feb'))
valid[, 'month_winter'] <- as.logical(valid$month %in% c('dec', 'jan', 'feb'))
test[, 'month_winter'] <- as.logical(test$month %in% c('dec', 'jan', 'feb'))
# Test (holdout)
test_holdout[, 'month_winter'] <- as.logical(test_holdout$month %in% c('dec', 'jan', 'feb'))



# Add new variable to train and test (holdout)
# Train, valid, test
train[, 'month_winter'] <- as.logical(train$month %in% c('dec', 'jan', 'feb'))
valid[, 'month_winter'] <- as.logical(valid$month %in% c('dec', 'jan', 'feb'))
test[, 'month_winter'] <- as.logical(test$month %in% c('dec', 'jan', 'feb'))
# Test (holdout)
test_holdout[, 'month_winter'] <- as.logical(test_holdout$month %in% c('dec', 'jan', 'feb'))

# Add new variable to train and test (holdout)
# pdays == 999 is a special value
# Train, valid, test
train[, 'pdays_999'] <- as.logical(train$pdays == 999)
valid[, 'pdays_999'] <- as.logical(valid$pdays == 999)
test[, 'pdays_999'] <- as.logical(test$pdays == 999)
# Test (holdout)
test_holdout[, 'pdays_999'] <- as.logical(test_holdout$pdays == 999)




# Get the IV and DV list name
# Dependent variable (DV)
dv_list <- c('subscribe')
# Independent variable (IV)
iv_list <- setdiff(colnames(train), dv_list)  # Exclude the target variable
iv_list <- setdiff(iv_list, 'client_id')  # Exclude the client_id




# Pick out categorical, boolean and numerical variable
iv_cat_list <- c()  # List to store categorical variable
iv_bool_list <- c()  # List to store boolean variable
iv_num_list <- c()  # List to store numerical variable
for (v in iv_list) {
  if (class(train[, v]) == 'factor') {  # Factor == categorical variable
    iv_cat_list <- c(iv_cat_list, v)
  } else if (class(train[, v]) == 'logical') {  # Logical == boolean variable
    iv_bool_list <- c(iv_bool_list, v)
  } else {  # Non-factor + Non-logical == numerical variable
    iv_num_list <- c(iv_num_list, v)
  }
}



# Grouping 12 categories in the variable job onto 3 groups using WOE
binning_cat <- woe.binning(train, 'subscribe', 'job')
binning_cat

View(train)

# Apply the binning to data
tmp <- woe.binning.deploy(train, binning_cat, add.woe.or.dum.var='woe')
head(tmp[, c('job', 'job.binned', 'woe.job.binned')])


# Loop through all categorical variables
for (v in iv_cat_list) {
  
  # Remapping categorical variable on train data
  binning_cat <- woe.binning(train, 'subscribe', v)
  
  # Apply the binning to the train, valid and test data
  train <- woe.binning.deploy(train, binning_cat, add.woe.or.dum.var='woe')
  valid <- woe.binning.deploy(valid, binning_cat, add.woe.or.dum.var='woe')
  test <- woe.binning.deploy(test, binning_cat, add.woe.or.dum.var='woe')
  
  # Apply the binning to the test (holdout) data
  test_holdout <- woe.binning.deploy(test_holdout, binning_cat, add.woe.or.dum.var='woe')
}



# Grouping the variable age onto 4 groups using WOE
binning_num <- woe.binning(train, 'subscribe', 'age')
binning_num



# Apply the binning to data
tmp <- woe.binning.deploy(train, binning_num, add.woe.or.dum.var='woe')
head(tmp[, c('age', 'age.binned', 'woe.age.binned')])



# Loop through all numerical variables
for (v in iv_num_list) {
  
  # Discretizing numerical variable on train data
  binning_num <- woe.binning(train, 'subscribe', v)
  
  # Apply the binning to the train, valid and test data
  train <- woe.binning.deploy(train, binning_num, add.woe.or.dum.var='woe')
  valid <- woe.binning.deploy(valid, binning_num, add.woe.or.dum.var='woe')
  test <- woe.binning.deploy(test, binning_num, add.woe.or.dum.var='woe')
  
  # Apply the binning to the test (holdout) data
  test_holdout <- woe.binning.deploy(test_holdout, binning_num, add.woe.or.dum.var='woe')
}


# Build the discretization
bins <- build_bins(dataSet=train, cols="age", n_bins=5, type="equal_freq", verbose=F)
####################???????????????????????

# Print out to check
bins


# Apply to the data
tmp <- fastDiscretization(dataSet=train, bins=bins, verbose=F)
setDF(tmp); setDF(train)  # Convert data.table to data.frame
head(tmp[, 'age'])


# Loop through all numerical variables
for (v in iv_num_list) {
  
  # Discretizing numerical variable on train data, n_bins=5
  bins <- build_bins(dataSet=train, cols=v, n_bins=5, type="equal_freq", verbose=F)
  
  # Apply the binning to the train, valid and test data
  tmp <- fastDiscretization(dataSet=train, bins=bins, verbose=F)
  setDF(tmp); setDF(train)  # Convert data.table to data.frame
  train[, paste0(v, '_freq_bin')] <- tmp[, v]  # Add new variable
  
  tmp <- fastDiscretization(dataSet=valid, bins=bins, verbose=F)
  setDF(tmp); setDF(valid)  # Convert data.table to data.frame
  valid[, paste0(v, '_freq_bin')] <- tmp[, v]  # Add new variable
  
  tmp <- fastDiscretization(dataSet=test, bins=bins, verbose=F)
  setDF(tmp); setDF(test)  # Convert data.table to data.frame
  test[, paste0(v, '_freq_bin')] <- tmp[, v]  # Add new variable
  
  # Apply the binning to the test (holdout) data
  tmp <- fastDiscretization(dataSet=test_holdout, bins=bins, verbose=F)
  setDF(tmp); setDF(test_holdout)  # Convert data.table to data.frame
  test_holdout[, paste0(v, '_freq_bin')] <- tmp[, v]  # Add new variable
}
####################fastDiscretization???????????????????????


# Build the discretization
bins <- build_bins(dataSet=train, cols="age", n_bins=5, type="equal_width", verbose=F)

# Print out to check
bins


# Apply to the data
tmp <- fastDiscretization(dataSet=train, bins=bins, verbose=F)
setDF(tmp); setDF(train)  # Convert data.table to data.frame
head(tmp[, 'age'])

# Loop through all numerical variables
for (v in iv_num_list) {
  
  # Discretizing numerical variable on train data, n_bins=5
  bins <- build_bins(dataSet=train, cols=v, n_bins=5, type="equal_width", verbose=F)
  
  # Apply the binning to the train, valid and test data
  tmp <- fastDiscretization(dataSet=train, bins=bins, verbose=F)
  setDF(tmp); setDF(train)  # Convert data.table to data.frame
  train[, paste0(v, '_width_bin')] <- tmp[, v]  # Add new variable
  
  tmp <- fastDiscretization(dataSet=valid, bins=bins, verbose=F)
  setDF(tmp); setDF(valid)  # Convert data.table to data.frame
  valid[, paste0(v, '_width_bin')] <- tmp[, v]  # Add new variable
  
  tmp <- fastDiscretization(dataSet=test, bins=bins, verbose=F)
  setDF(tmp); setDF(test)  # Convert data.table to data.frame
  test[, paste0(v, '_width_bin')] <- tmp[, v]  # Add new variable
  
  # Apply the binning to the test (holdout) data
  tmp <- fastDiscretization(dataSet=test_holdout, bins=bins, verbose=F)
  setDF(tmp); setDF(test_holdout)  # Convert data.table to data.frame
  test_holdout[, paste0(v, '_width_bin')] <- tmp[, v]  # Add new variable
}


# Get the IV and DV list name
# Dependent variable (DV)
dv_list <- c('subscribe')
# Independent variable (IV)
iv_list <- setdiff(colnames(train), dv_list)  # Exclude the target variable
iv_list <- setdiff(iv_list, 'client_id')  # Exclude the client_id



# Pick out categorical, boolean and numerical variable
iv_cat_list <- c()  # List to store categorical variable
iv_bool_list <- c()  # List to store boolean variable
iv_num_list <- c()  # List to store numerical variable
for (v in iv_list) {
  if (class(train[, v]) == 'factor') {  # Factor == categorical variable
    iv_cat_list <- c(iv_cat_list, v)
  } else if (class(train[, v]) == 'logical') {  # Logical == boolean variable
    iv_bool_list <- c(iv_bool_list, v)
  } else {  # Non-factor + Non-logical == numerical variable
    iv_num_list <- c(iv_num_list, v)
  }
}


# Build the dummy encoding
encoding <- build_encoding(dataSet=train, cols="job", verbose=F)


# Transform the categorical variable
tmp <- one_hot_encoder(dataSet=train, encoding=encoding, type='logical', drop=F, verbose=F)
setDF(tmp)
tmp <- tmp[, -ncol(tmp)]
head(tmp[, 84:ncol(tmp)])


# Loop through all categorical variables
for (v in iv_cat_list) {
  
  # Representing categorical variable on train data
  encoding <- build_encoding(dataSet=train, cols=v, verbose=F)
  
  # Apply the binning to the train, valid and test data
  train <- one_hot_encoder(dataSet=train, encoding=encoding, type='logical', drop=F, verbose=F)
  setDF(train)
  train <- train[, -ncol(train)]  # Drop the last dummy column
  
  valid <- one_hot_encoder(dataSet=valid, encoding=encoding, type='logical', drop=F, verbose=F)
  setDF(valid)
  valid <- valid[, -ncol(valid)]  # Drop the last dummy column
  
  test <- one_hot_encoder(dataSet=test, encoding=encoding, type='logical', drop=F, verbose=F)
  setDF(test)
  test <- test[, -ncol(test)]  # Drop the last dummy column
  
  # Apply the binning to the test (holdout) data
  test_holdout <- one_hot_encoder(dataSet=test_holdout, encoding=encoding, type='logical', drop=F, verbose=F)
  setDF(test_holdout)
  test_holdout <- test_holdout[, -ncol(test_holdout)]  # Drop the last dummy column
}



# Find the incidence rates per category of a variable
tb <- table(train$job, train$subscribe)
incidence_map <- data.frame('v1'=rownames(tb), 'v2'=tb[, '1'] / (tb[, '0'] + tb[, '1']))
colnames(incidence_map) <- c('job', 'job_incidence')
incidence_map



# Convert the categories with incidences
tmp <- plyr::join(x=train, y=incidence_map, by='job', type="left", match="all")  # Left join
head(tmp[, c('job', 'job_incidence')])



# Loop through all categorical variables
for (v in iv_cat_list) {
  
  # Find the incidence rates per category of a variable
  tb <- table(train[, v], train[, 'subscribe'])
  incidence_map <- data.frame('v1'=rownames(tb), 'v2'=tb[, '1'] / (tb[, '0'] + tb[, '1']))
  colnames(incidence_map) <- c(v, paste0(v, '_incidence'))  # Rename the columns to join
  
  # Apply the variable representation to the train, valid and test data
  train <- plyr::join(x=train, y=incidence_map, by=v, type="left", match="all")
  valid <- plyr::join(x=valid, y=incidence_map, by=v, type="left", match="all")
  test <- plyr::join(x=test, y=incidence_map, by=v, type="left", match="all")
  
  # Apply the binning to the test (holdout) data
  test_holdout <- plyr::join(x=test_holdout, y=incidence_map, by=v, type="left", match="all")
}


# Find the WOE per category of a variable
tb <- table(train$job, train$subscribe)
woe_map <- data.frame('v1'=rownames(tb), 'v2'=log(tb[, '1'] / tb[, '0']))
colnames(woe_map) <- c('job', 'job_woe')
woe_map


# Convert the categories with WOE
tmp <- plyr::join(x=train, y=woe_map, by='job', type="left", match="all")  # Left join
head(tmp[, c('job', 'job_woe')])


# Loop through all categorical variables
for (v in iv_cat_list) {
  
  # Find the incidence rates per category of a variable
  tb <- table(train[, v], train[, 'subscribe'])
  woe_map <- data.frame('v1'=rownames(tb), 'v2'=log(tb[, '1'] / tb[, '0']))
  colnames(woe_map) <- c(v, paste0(v, '_woe'))  # Rename the columns to join
  
  # Apply the variable representation to the train, valid and test data
  train <- plyr::join(x=train, y=woe_map, by=v, type="left", match="all")
  valid <- plyr::join(x=valid, y=woe_map, by=v, type="left", match="all")
  test <- plyr::join(x=test, y=woe_map, by=v, type="left", match="all")
  
  # Apply the binning to the test (holdout) data
  test_holdout <- plyr::join(x=test_holdout, y=woe_map, by=v, type="left", match="all")
}


# Transform the variable age on train and test (holdout)
# Train, valid, test
train[, 'age_log'] <- log(train[, 'age'])
valid[, 'age_log'] <- log(valid[, 'age'])
test[, 'age_log'] <- log(test[, 'age'])
# Test (holdout)
test_holdout[, 'age_log'] <- log(test_holdout[, 'age'])



# Standardize the variable age on train and test (holdout)
# Train, valid, test
train[, 'age_scaled'] <- scale(train[, 'age'], center=T, scale=T)  # sd = 1, mean = 0
valid[, 'age_scaled'] <- scale(valid[, 'age'], center=T, scale=T)  # sd = 1, mean = 0
test[, 'age_scaled'] <- scale(test[, 'age'], center=T, scale=T)  # sd = 1, mean = 0
# Test (holdout)
test_holdout[, 'age_scaled'] <- scale(test_holdout[, 'age'], center=T, scale=T)  # sd = 1, mean = 0



# Get the IV and DV list name
# Dependent variable (DV)
dv_list <- c('subscribe')
# Independent variable (IV)
iv_list <- setdiff(colnames(train), dv_list)  # Exclude the target variable
iv_list <- setdiff(iv_list, 'client_id')  # Exclude the client_id




# Pick out categorical, boolean and numerical variable
iv_cat_list <- c()  # List to store categorical variable
iv_bool_list <- c()  # List to store boolean variable
iv_num_list <- c()  # List to store numerical variable
for (v in iv_list) {
  if (class(train[, v]) == 'factor') {  # Factor == categorical variable
    iv_cat_list <- c(iv_cat_list, v)
  } else if (class(train[, v]) == 'logical') {  # Logical == boolean variable
    iv_bool_list <- c(iv_bool_list, v)
  } else {  # Non-factor + Non-logical == numerical variable
    iv_num_list <- c(iv_num_list, v)
  }
}



# Check missing value
# Train, valid, test
sum(apply(sapply(train, is.infinite), 2, sum))
sum(apply(sapply(valid, is.infinite), 2, sum))
sum(apply(sapply(test, is.infinite), 2, sum))
# Test (holdout)
sum(apply(sapply(test_holdout, is.infinite), 2, sum))




# Impute +/-Inf value by NA
# Train, valid, test
train[sapply(train, is.infinite)] <- NA
valid[sapply(valid, is.infinite)] <- NA
test[sapply(test, is.infinite)] <- NA
# Test (holdout)
test_holdout[sapply(test_holdout, is.infinite)] <- NA



# Check missing value
# Train, valid, test
sum(apply(is.na(train), 2, sum))
sum(apply(is.na(valid), 2, sum))
sum(apply(is.na(test), 2, sum))
# Test (holdout)
sum(apply(is.na(test_holdout), 2, sum))


# Impute missing value in numerical variable by mean
for (v in iv_num_list) {
  # Train, valid, test
  train[is.na(train[, v]), v] <- mean(train[, v], na.rm=T)
  valid[is.na(valid[, v]), v] <- mean(valid[, v], na.rm=T)
  test[is.na(test[, v]), v] <- mean(test[, v], na.rm=T)
  
  # Test (holdout)
  test_holdout[is.na(test_holdout[, v]), v] <- mean(test_holdout[, v], na.rm=T)
}



for (v in iv_cat_list) {
  # Train, valid, test
  train[, v] <- NULL
  valid[, v] <- NULL
  test[, v] <- NULL
  
  # Test (holdout)
  test_holdout[, v] <- NULL
}



# Convert boolean to int
for (v in iv_bool_list) {
  # Train, valid, test
  train[, v] <- as.integer(train[, v])
  valid[, v] <- as.integer(valid[, v])
  test[, v] <- as.integer(test[, v])
  
  # Test (holdout)
  test_holdout[, v] <- as.integer(test_holdout[, v])
}



# Find the constant variable
var_list <- c()
for (v in c(iv_num_list, iv_bool_list)) {
  var_list <- c(var_list, var(train[, v], na.rm=T))
}
constant_var <- c(iv_num_list, iv_bool_list)[var_list == 0]
constant_var



# Drop the constant variable
for (v in constant_var) {
  # Train, valid, test
  train[, v] <- NULL
  valid[, v] <- NULL
  test[, v] <- NULL
  
  # Test (holdout)
  test_holdout[, v] <- NULL
}
###################constant????????????????????????


FisherScore <- function(basetable, depvar, IV_list) {
  "
  This function calculate the Fisher score of a variable.
  
  Ref:
  ---
  Verbeke, W., Dejaeger, K., Martens, D., Hur, J., & Baesens, B. (2012). New insights into churn prediction in the telecommunication sector: A profit driven data mining approach. European Journal of Operational Research, 218(1), 211-229.
  "
  
  # Get the unique values of dependent variable
  DV <- unique(basetable[, depvar])
  
  IV_FisherScore <- c()
  
  for (v in IV_list) {
    fs <- abs((mean(basetable[which(basetable[, depvar]==DV[1]), v]) - mean(basetable[which(basetable[, depvar]==DV[2]), v]))) /
      sqrt((var(basetable[which(basetable[, depvar]==DV[1]), v]) + var(basetable[which(basetable[, depvar]==DV[2]), v])))
    IV_FisherScore <- c(IV_FisherScore, fs)
  }
  
  return(data.frame(IV=IV_list, fisher_score=IV_FisherScore))
}

varSelectionFisher <- function(basetable, depvar, IV_list, num_select=20) {
  "
  This function will calculate the Fisher score for all IVs and select the best
  top IVs.

  Assumption: all variables of input dataset are converted into numeric type.
  "
  
  fs <- FisherScore(basetable, depvar, IV_list)  # Calculate Fisher Score for all IVs
  num_select <- min(num_select, ncol(basetable))  # Top N IVs to be selected
  return(as.vector(fs[order(fs$fisher_score, decreasing=T), ][1:num_select, 'IV']))
}



# Calculate Fisher Score for all variable
# Get the IV and DV list
dv_list <- c('subscribe')  # DV list
iv_list <- setdiff(names(train), dv_list)  # IV list excluded DV
iv_list <- setdiff(iv_list, 'client_id')  # Excluded the client_id
fs <- FisherScore(train, dv_list, iv_list)
head(fs)




# Select top 20 variables according to the Fisher Score
best_fs_var <- varSelectionFisher(train, dv_list, iv_list, num_select=50)
head(best_fs_var, 10)



# Apply variable selection to the data
# Train
var_select <- names(train)[names(train) %in% best_fs_var]
train_processed <- train[, c('client_id', var_select, 'subscribe')]
# Valid
var_select <- names(valid)[names(valid) %in% best_fs_var]
valid_processed <- valid[, c('client_id', var_select, 'subscribe')]
# Test
var_select <- names(test)[names(test) %in% best_fs_var]
test_processed <- test[, c('client_id', var_select, 'subscribe')]
# Test (holdout)
var_select <- names(test_holdout)[names(test_holdout) %in% best_fs_var]
test_holdout_processed <- test_holdout[, c('client_id', var_select)]



# Check if train and test (holdout) have same variables
# Train, valid, test
dim(train_processed)
dim(valid_processed)
dim(test_processed)
# Test (holdout)
dim(test_holdout_processed)



# Rename the data columns
for (v in colnames(train_processed)) {
  
  # Fix the column name
  fix_name <- str_replace_all(v, "[^[:alnum:] ]", "_")
  fix_name <- gsub(' +', '', fix_name) 
  
  # Train, valid, test
  colnames(train_processed)[colnames(train_processed) == v] <- fix_name
  colnames(valid_processed)[colnames(valid_processed) == v] <- fix_name
  colnames(test_processed)[colnames(test_processed) == v] <- fix_name
  
  # Test (holdout)
  colnames(test_holdout_processed)[colnames(test_holdout_processed) == v] <- fix_name
}


write.csv(train_processed, file ="C:/Users/xzong/Desktop/MBD/Machine Learning/train_processed.csv" )
write.csv(test_holdout_processed, file ="C:/Users/xzong/Desktop/MBD/Machine Learning/test_holdout_processed.csv" )
