library(devtools)

# Failed with error:  'there is no package called 'lightgbm'
# Failed with error:  'there is no package called 'rpart.plot'
# Failed with error:  'there is no package called 'partykit'
# Failed with error:  'there is no package called 'tabplot'
# Failed with error:  'there is no package called 'rCharts'
# Failed with error:  'there is no package called 'plotly'
# Failed with error:  'there is no package called 'ggthemes'
# Failed with error:  'there is no package called 'plotluck'
# Failed with error:  'there is no package called 'CEoptim'
# Failed with error:  'there is no package called 'DT'
# Failed with error:  'there is no package called 'formattable'
# Failed with error:  'there is no package called 'shiny'
# Failed with error:  'there is no package called 'shinydashboard'
# Failed with error:  'there is no package called 'matrixStats'
# Failed with error:  'there is no package called 'R.utils'
# Failed with error:  'there is no package called 'Rtsne'
# Failed with error:  'there is no package called 'recommenderlab'
# Failed with error:  'there is no package called 'sparsity'
# Failed with error:  'there is no package called 'RcppArmadillo'
# Failed with error:  'there is no package called 'Deriv'
# Failed with error:  'there is no package called 'outliers'

# deps <- c('rpart.plot','partykit','tabplot','rCharts','plotly','ggthemes',
#           'plotluck','CEoptim','DT','formattable','shiny','shinydashboard','matrixStats',
#           'R.utils','Rtsne','recommenderlab','sparsity','RcppArmadillo','Deriv','outliers')
# for(i in deps){
#   if(!require(i, character.only = TRUE)){
#     install.packages(i)
#   }
# }
# 
# if(!require(rCharts)){
#   install.packages("RCurl")
# }
# 
# install.packages("jsonlite")
# install_github("Microsoft/LightGBM", subdir = "R-package")
# 
# install_github('ramnathv/rCharts')
# install_github("Laurae2/sparsity")
# install_github("Laurae2/Laurae")

require(Laurae)


# Load libraries
library(data.table)
library(Matrix)
library(xgboost)

# Create data
data(agaricus.train, package = "lightgbm")
data(agaricus.test, package = "lightgbm")
agaricus_data_train <- data.table(as.matrix(agaricus.train$data))
agaricus_data_test <- data.table(as.matrix(agaricus.test$data))
agaricus_label_train <- agaricus.train$label
agaricus_label_test <- agaricus.test$label
folds <- Laurae::kfold(agaricus_label_train, 5)

# Train a model (binary classification)
model <- CascadeForest(training_data = agaricus_data_train, # Training data
                       validation_data = agaricus_data_test, # Validation data
                       training_labels = agaricus_label_train, # Training labels
                       validation_labels = agaricus_label_test, # Validation labels
                       folds = folds, # Folds for cross-validation
                       boosting = FALSE, # Do not touch this unless you are expert
                       nthread = 1, # Change this to use more threads
                       cascade_lr = 1, # Do not touch this unless you are expert
                       training_start = NULL, # Do not touch this unless you are expert
                       validation_start = NULL, # Do not touch this unless you are expert
                       cascade_forests = rep(4, 5), # Number of forest models
                       cascade_trees = 10, # Number of trees per forest
                       cascade_rf = 2, # Number of Random Forest in models
                       cascade_seeds = 1:5, # Seed per layer
                       objective = "binary:logistic",
                       eval_metric = Laurae::df_logloss,
                       multi_class = 2, # Modify this for multiclass problems
                       early_stopping = 2, # stop after 2 bad combos of forests
                       maximize = FALSE, # not a maximization task
                       verbose = TRUE, # print information during training
                       low_memory = FALSE)

# Attempt to perform fake multiclass problem
agaricus_label_train[1:100] <- 2

# Train a model (multiclass classification)
model <- CascadeForest(training_data = agaricus_data_train, # Training data
                       validation_data = agaricus_data_test, # Validation data
                       training_labels = agaricus_label_train, # Training labels
                       validation_labels = agaricus_label_test, # Validation labels
                       folds = folds, # Folds for cross-validation
                       boosting = FALSE, # Do not touch this unless you are expert
                       nthread = 1, # Change this to use more threads
                       cascade_lr = 1, # Do not touch this unless you are expert
                       training_start = NULL, # Do not touch this unless you are expert
                       validation_start = NULL, # Do not touch this unless you are expert
                       cascade_forests = rep(4, 5), # Number of forest models
                       cascade_trees = 10, # Number of trees per forest
                       cascade_rf = 2, # Number of Random Forest in models
                       cascade_seeds = 1:5, # Seed per layer
                       objective = "multi:softprob",
                       eval_metric = Laurae::df_logloss,
                       multi_class = 3, # Modify this for multiclass problems
                       early_stopping = 2, # stop after 2 bad combos of forests
                       maximize = FALSE, # not a maximization task
                       verbose = TRUE, # print information during training
                       low_memory = FALSE)


## End(Not run)
