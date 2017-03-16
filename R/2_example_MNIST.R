# Load libraries
library(Laurae) #devtools::install_github("Laurae2/Laurae@99ad3c2")
library(data.table)
library(Matrix)
# install.packages("xgboost")
library(xgboost) # COMPILED FROM SOURCE ONLY

# Set working directory accordingly
# setwd("D:/")
# Download MNIST data here: https://pjreddie.com/projects/mnist-in-csv/
train <- fread("R/mnist_train.csv")
test <- fread("R/mnist_test.csv")

label_train <- train$V1
label_test <- test$V1
train$V1 <- NULL
test$V1 <- NULL

# debug xgboost, mandatory
for (i in 1:784) {
  train[[i]] <- as.numeric(train[[i]])
  test[[i]] <- as.numeric(test[[i]])
}

# SUBSAMPLE DATA - 2000 SAMPLES ONLY - COMMENT TO NOT SUBSAMPLE
valid <- train[2001:60000, ]
train <- train[1:2000, ]
label_valid <- label_train[2001:60000]
label_train <- label_train[1:2000]

# Create folds - set it to larger like 5 but it gets slower
folds <- kfold(label_train, k = 3)

# Do Cascade Forest using xgboost Random Forest / Complete-Tree Random Forest behind the wheels ------
# 0.899+
model <- CascadeForest(training_data = train,
                       validation_data = test,
                       training_labels = label_train,
                       validation_labels = label_test,
                       folds = folds,
                       boosting = FALSE,
                       nthread = 4, # More threads if you feel so
                       cascade_lr = 1,
                       training_start = NULL,
                       validation_start = NULL,
                       cascade_forests = c(rep(4, 2), 0), # c(rep(4, 2), 0) should be enough
                       # cascade_forests = c(rep(8, 4), 0), # c(rep(4, 2), 0) should be enough
                       cascade_trees = 100, # Set this to much higher like 1000 (cf official paper)
                       cascade_rf = 4, # If you changed cascade_forests, change this value accordingly
                       objective = "multi:softprob",
                       eval_metric = Laurae::df_acc,
                       multi_class = 10,
                       early_stopping = 4, # Keep it otherwise if you make it longer it will take forever to stop
                       maximize = TRUE,
                       verbose = TRUE,
                       low_memory = FALSE,
                       essentials = TRUE,
                       garbage = TRUE)

# Now compare to xgboost ------
dtrain <- xgb.DMatrix(data = Laurae::DT2mat(train), label = label_train)
dtest <- xgb.DMatrix(data = Laurae::DT2mat(test), label = label_test)
gc()

# [250]	train-merror:0.000000	test-merror:0.094700
# 0.905300 accuracy
gc()
set.seed(11111)
model2 <- xgb.train(params = list(nthread = 4, # More threads if you feel so
                                  eta = 0.10,
                                  max_depth = 6,
                                  booster = "gbtree",
                                  tree_method = "hist",
                                  grow_policy = "depthwise"),
                    objective = "multi:softprob",
                    num_class = 10,
                    eval_metric = "merror",
                    nrounds = 1000000,
                    early_stopping_rounds = 50,
                    data = dtrain,
                    watchlist = list(train = dtrain, test = dtest),
                    verbose = 1)


# Try with Multi-Grained Scanning for gcForest ------
library(plyr)
create_progress_bar(name = "win") # Get rid of progress bar if you don't like, or set to text
new_train <- plyr::alply(train, 1, function(x) {matrix(as.numeric(x), nrow = 28, ncol = 28)}, .progress = "win")
new_test <- plyr::alply(test, 1, function(x) {matrix(as.numeric(x), nrow = 28, ncol = 28)}, .progress = "win")

# Run Multi-Grained Scanning ------
new_model <- MGScanning(data = new_train,
                        labels = label_train,
                        folds = folds,
                        dimensions = 2,
                        depth = 10, # Default official implementation
                        stride = 2, # Set this to 1 if you want maximum performance, but it is VERY SLOW + adds many features
                        nthread = 4, # More threads if you feel so
                        n_forest = 2, # Following official implementation
                        n_trees = 30, # Following official implementation
                        random_forest = 1, # Following official implementation
                        seed = 0,
                        objective = "multi:softprob",
                        eval_metric = df_acc,
                        multi_class = 10,
                        garbage = TRUE)

# Predict on train datag ------
new_train2 <- MGScanning_pred(model = new_model,
                              data = new_train,
                              folds = folds,
                              dimensions = 2,
                              multi_class = 10)

# Predict on test data ------
new_test2 <- MGScanning_pred(model = new_model,
                             data = new_test,
                             folds = NULL,
                             dimensions = 2,
                             multi_class = 10)

# Create new datasets ------
new_train2 <- Laurae::DTcbind(new_train2, train)
new_test2 <- Laurae::DTcbind(new_test2, test)

# Do Deep Forest / gcForest ------
# 0.9112+
model <- CascadeForest(training_data = new_train2,
                       validation_data = new_test2,
                       training_labels = label_train,
                       validation_labels = label_test,
                       folds = folds,
                       boosting = FALSE,
                       nthread = 1, # More threads if you feel so
                       cascade_lr = 1,
                       training_start = NULL,
                       validation_start = NULL,
                       cascade_forests = c(rep(8, 4), 0), # c(rep(4, 2), 0) should be enough
                       cascade_trees = 100, # Set this to much higher like 1000 (cf official paper)
                       cascade_rf = 4, # If you changed cascade_forests, change this value accordingly
                       objective = "multi:softprob",
                       eval_metric = Laurae::df_acc,
                       multi_class = 10,
                       early_stopping = 2,
                       maximize = TRUE,
                       verbose = TRUE,
                       low_memory = FALSE,
                       essentials = TRUE,
                       garbage = TRUE)


# Try with xgboost as final booster instead of Cascade Forest ------
dtrain2 <- xgb.DMatrix(data = Laurae::DT2mat(new_train2), label = label_train)
dtest2 <- xgb.DMatrix(data = Laurae::DT2mat(new_test2), label = label_test)
gc()

# [223]	train-merror:0.000000	test-merror:0.072300 
# 0.927700 accuracy, not converged
gc()
set.seed(11111)
model2 <- xgb.train(params = list(nthread = 1, # More threads if you feel so
                                  eta = 0.10,
                                  max_depth = 6,
                                  booster = "gbtree",
                                  tree_method = "hist",
                                  grow_policy = "depthwise"),
                    objective = "multi:softprob",
                    num_class = 10,
                    eval_metric = "merror",
                    nrounds = 1000000,
                    early_stopping_rounds = 50,
                    data = dtrain2,
                    watchlist = list(train = dtrain2, test = dtest2),
                    verbose = 1)

# end ---------------------------------------------------------------------
