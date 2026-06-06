#install.packages("EMP")
#install.packages('data.tree')
#install.packages('partykit')
#install.packages("C:/Users/sellp/Downloads/prof.tree_0.1.0.tar.gz", repos = NULL, type = "source")
#install.packages("C:/Users/sellp/Downloads/proftree_1.0.0.tar.gz", repos = NULL, type = "source")
#install.packages("remotes")
#remotes::install_github("SebastiaanHoppner/ProfTree/proftree")
#install.packages("devtools")
#devtools::install_github("SebastiaanHoppner/ProfTree/proftree")
#install.packages("caret")


library(proftree)
library(EMP)
library(devtools)
library(caret)

####### data set preparation ##############
status_summary <- as.data.frame(table(CHURNDATAPREPROCESSED$STATUS, useNA = "ifany"))
colnames(status_summary) <- c("STATUS", "COUNT")
status_summary$PERCENTAGE <- round(
  status_summary$COUNT / sum(status_summary$COUNT), 2
)
status_1_percentage <- status_summary$PERCENTAGE[status_summary$STATUS == 1]


churndata <- CHURNDATAPREPROCESSED
churndata$CLV = churndata$SIX_MONTHS_TOTAL_REVENUE_ORIG* (1 / status_1_percentage)
churndata <- churndata[, !(names(churndata) %in% c("TENURE_MONTHS_orig","SIX_MONTHS_TOTAL_REVENUE_orig"))]
churndata$STATUS <- as.factor(churndata$STATUS)
levels(churndata$STATUS)
mean(churndata$CLV)
set.seed(123)

########### train test split############
n <- nrow(churndata)
train_idx <- sample(1:n, size = 0.7 * n)
Lambda <- 0.01
train_data <- churndata[train_idx, ]
test_data  <- churndata[-train_idx, ]
#getwd()
#write.csv(train_data, "train_data.csv", row.names = FALSE)
#write.csv(test_data, "test_data.csv", row.names = FALSE)

#####################################################
######### initial model fit - proftree ##############
#####################################################

churndata <- churndata[, !(names(churndata) %in% c("CLV"))]
ProfTree <- proftree( STATUS ~ ., 
                       data = train_data, 
                       control = proftree.control(maxdepth = 9L,niterations = 10000L,ntrees = 100L,clv = 2280.744, d = 1000, f=10,lambda = 0.01) )

##evaluate for train set
scores_train <- predict(ProfTree, newdata = train_data, type = "prob")
scores_positive_train <- scores_train[,2]
EMP_train <- empChurn(scores = scores_positive_train, classes = train_data$STATUS,clv = 2280.744, d = 1000, f = 10)$EMP
print(EMP_train)

EMP_train_frac <- empChurn(scores = scores_positive_train, classes = train_data$STATUS,clv = 2280.744, d = 1000, f = 10)$EMPfrac
print(EMP_train_frac)

predicted_class <- ifelse(scores_positive_train > 0.5, 1, 0)
accuracy <- mean(predicted_class == train_data$STATUS)
print(accuracy)

cm <- confusionMatrix(
  as.factor(predicted_class),
  as.factor(train_data$STATUS),
  positive = "1"
)
cm$byClass["Recall"]
cm$byClass["F1"]


##evaluate for test set
scores <- predict(ProfTree, newdata = test_data, type = "prob")
scores_positive <- scores[,2]

EMP <- empChurn(scores = scores_positive, classes = test_data$STATUS,clv = 2280.744, d = 1000, f = 10)$EMP
print(EMP)
EMP_frac <- empChurn(scores = scores_positive, classes = test_data$STATUS,clv = 2280.744, d = 1000, f = 10)$EMPfrac
print(EMP_frac)
print(EMP - Lambda * width(ProfTree))
print(ProfTree$info$evalfun)

predicted_class_test <- ifelse(scores_positive > 0.5, 1, 0)
accuracy <- mean(predicted_class_test == test_data$STATUS)
print(accuracy)

cm <- confusionMatrix(
  as.factor(predicted_class_test),
  as.factor(test_data$STATUS),
  positive = "1"
)
cm$byClass["Recall"]
cm$byClass["F1"]

#####################################################
######## model fit - decission tree #################
#####################################################

  library(rpart)
  model_dt <- rpart(STATUS ~ .,  data = train_data,method = "class"   )
  y_pred <- predict(model_dt, test_data, type = "class")
  y_prob <- predict(model_dt, test_data, type = "prob")
  y_prob_churn <- y_prob[,2]   
  
  ##evaluate for train set
  y_prob_train <- predict(model_dt, train_data, type = "prob")
  y_prob_churn_train <- y_prob_train[,2]   
  EMP_DT <- empChurn(scores = y_prob_churn_train, classes = train_data$STATUS,clv = 2280.744, d = 1000, f = 10)$EMP
  print(EMP_DT)
  
  EMP_DT_frac <- empChurn(scores = y_prob_churn_train, classes = train_data$STATUS,clv = 2280.744, d = 1000, f = 10)$EMPfrac
  print(EMP_DT_frac)



  predicted_class <- ifelse(y_prob_churn_train > 0.5, 1, 0)
  accuracy <- mean(predicted_class == train_data$STATUS)
  print(accuracy)
  
  cm <- confusionMatrix(
    as.factor(predicted_class),
    as.factor(train_data$STATUS),
    positive = "1"
  )
  cm$byClass["Recall"]
  cm$byClass["F1"]
  
  ##evaluate for test set
  y_prob_test <- predict(model_dt, test_data, type = "prob")
  y_prob_churn_test <- y_prob_test[,2]   
  EMP_DT <- empChurn(scores = y_prob_churn_test, classes = test_data$STATUS,clv = 2280.744, d = 1000, f = 10)$EMP
  print(EMP_DT)
  EMP_DT_frac <- empChurn(scores = y_prob_churn_test, classes = test_data$STATUS,clv = 2280.744, d = 1000, f = 10)$EMPfrac
  print(EMP_DT_frac)
  
  predicted_class_test <- ifelse(y_prob_churn_test > 0.5, 1, 0)
  accuracy <- mean(predicted_class_test == test_data$STATUS)
  print(accuracy)

  cm <- confusionMatrix(
    as.factor(predicted_class_test),
    as.factor(test_data$STATUS),
    positive = "1"
  )
  cm$byClass["Recall"]
  cm$byClass["F1"]
  
####################################################
########Cross Validation - proftree ############
####################################################

lambda_grid <- c(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1)

# Store all results
all_results <- data.frame(
  lambda = numeric(),
  repetition = integer(),
  fold = character(),
  empc = numeric()
)

# ProfTree control defaults (fixed across all runs)
clv_val <- 2280.744
d_val <- 1000
f_val <- 10

for (lam in lambda_grid) {
  cat("\n========== Lambda =", lam, "==========\n")
  
  for (rep in 1:5) {
    cat("  Repetition", rep, "... ")
    
    # Random 50/50 split
    n <- nrow(train_data)
    idx <- sample(1:n, size = n / 2)
    fold_a <- train_data[idx, ]
    fold_b <- train_data[-idx, ]
    
    # --- Round A: Train on fold_a, test on fold_b ---
    model_a <- proftree(
      STATUS ~ ., 
      data = fold_a,
      control = proftree.control(
        maxdepth = 9L,
        niterations = 10000L,
        ntrees = 100L,
        clv = clv_val,
        d = d_val,
        f = f_val,
        lambda = lam
      )
    )
    pred_a <- predict(model_a, newdata = fold_b, type = "prob")
    empc_a <- empChurn(classes = fold_b$STATUS,scores = pred_a[,2], clv = clv_val, d = d_val, f = f_val)$EMP
    
    
    all_results <- rbind(all_results, data.frame(
      lambda = lam, repetition = rep, fold = "A", empc = empc_a
    ))
    
    # --- Round B: Train on fold_b, test on fold_a ---
    model_b <- proftree(
      STATUS ~ ., 
      data = fold_b,
      control = proftree.control(
        maxdepth = 9L,
        niterations = 10000L,
        ntrees = 100L,
        clv = clv_val,
        d = d_val,
        f = f_val,
        lambda = lam
      )
    )
    pred_b <- predict(model_b, newdata = fold_a, type = "prob")
    empc_b <- empChurn(classes = fold_a$STATUS, scores = pred_b[,2], clv = clv_val, d = d_val, f = f_val)$EMP
    
    all_results <- rbind(all_results, data.frame(
      lambda = lam, repetition = rep, fold = "B", empc = empc_b
    ))
    
    cat("Fold A EMPC =", round(empc_a, 2), 
        "| Fold B EMPC =", round(empc_b, 2), "\n")
  }
}


# =============================================
# Summary Table
# =============================================
summary_table <- aggregate(
  empc ~ lambda, 
  data = all_results, 
  FUN = function(x) c(
    mean = round(mean(x), 2),
    sd = round(sd(x), 2),
    median = round(median(x), 2),
    min = round(min(x), 2),
    max = round(max(x), 2)
  )
)
summary_table <- do.call(data.frame, summary_table)
colnames(summary_table) <- c("Lambda", "Mean_EMPC", "SD", "Median", "Min", "Max")

cat("\n========== SUMMARY ==========\n")
print(summary_table)

# Find optimal lambda
best_lambda <- summary_table$Lambda[which.max(summary_table$Mean_EMPC)]
cat("\nOptimal Lambda:", best_lambda, "\n")

# =============================================
# Box Plot 
# =============================================
all_results$lambda_label <- paste0("λ = ", all_results$lambda)

# Order labels
label_order <- paste0("λ = ", lambda_grid)
all_results$lambda_label <- factor(all_results$lambda_label, levels = label_order)

# Calculate means for red dots
means <- aggregate(empc ~ lambda_label, data = all_results, FUN = mean)

# Plot
png("proftree_lambda_cv.png", width = 800, height = 500)
boxplot(
  empc ~ lambda_label, 
  data = all_results,
  xlab = "", 
  ylab = "EMPC (LKR)",
  main = "Average EMPC for various values of ProfTree's λ",
  col = "white",
  outline = TRUE,
  las = 1
)
points(1:length(lambda_grid), means$empc, col = "red", pch = 16, cex = 1.3)
lines(1:length(lambda_grid), means$empc, col = "red", lwd = 1.5)
dev.off()

##with selected lambda
ProfTree <- proftree( STATUS ~ ., 
                      data = train_data, 
                      control = proftree.control(maxdepth = 9L,niterations = 10000L,ntrees = 100L,clv = 2280.744, d = 1000, f=10,lambda = 1) )



##test with selected lambda
scores <- predict(ProfTree, newdata = test_data, type = "prob")
EMP <- empChurn(scores = scores[,2], classes = test_data$STATUS,clv = 2280.744, d = 1000, f = 10)$EMP
print(EMP)
EMP_frac <- empChurn(scores = scores[,2], classes = test_data$STATUS,clv = 2280.744, d = 1000, f = 10)$EMPfrac
print(EMP_frac)
print(EMP - Lambda * width(ProfTree))
print(ProfTree$info$evalfun)

predicted_class_test <- ifelse(scores[,2] > 0.5, 1, 0)
accuracy <- mean(predicted_class_test == test_data$STATUS)
print(accuracy)

cm <- confusionMatrix(
  as.factor(predicted_class_test),
  as.factor(test_data$STATUS),
  positive = "1"
)
cm$byClass["Recall"]
cm$byClass["F1"]
