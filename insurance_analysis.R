################################################################################
# MA478 Homework 2: Auto Insurance Predictive Modeling
# This script builds logistic and linear regression models to predict:
#   1) Probability of car crash (TARGET_FLAG)
#   2) Cost of claim given crash (TARGET_AMT)
################################################################################

# Load required packages
library(tidyverse)
library(MASS)
library(car)
library(caret)
library(pROC)
library(corrplot)

set.seed(478)

################################################################################
# SECTION 1: DATA EXPLORATION
################################################################################

# Load data
train_raw <- read.csv("insurance_training_data.csv", stringsAsFactors = TRUE)
eval_raw <- read.csv("insurance_evaluation_data.csv", stringsAsFactors = TRUE)

cat("Training data:", nrow(train_raw), "rows,", ncol(train_raw), "columns\n")
cat("Evaluation data:", nrow(eval_raw), "rows,", ncol(eval_raw), "columns\n")

# Function to convert currency strings to numeric
clean_currency <- function(x) {
  if(is.factor(x) | is.character(x)) {
    x <- gsub("[$,]", "", as.character(x))
    return(as.numeric(x))
  }
  return(x)
}

# Clean currency columns in both datasets
currency_cols <- c("INCOME", "HOME_VAL", "BLUEBOOK", "OLDCLAIM")
for(col in currency_cols) {
  train_raw[[col]] <- clean_currency(train_raw[[col]])
  eval_raw[[col]] <- clean_currency(eval_raw[[col]])
}

# Examine response variable distributions
cat("\nTARGET_FLAG distribution:\n")
print(table(train_raw$TARGET_FLAG))
cat("Crash rate:", round(mean(train_raw$TARGET_FLAG), 4), "\n")

# Summary statistics for numeric variables
numeric_vars <- c("KIDSDRIV", "AGE", "HOMEKIDS", "YOJ", "INCOME", "HOME_VAL",
                  "TRAVTIME", "BLUEBOOK", "TIF", "OLDCLAIM", "CLM_FREQ", 
                  "MVR_PTS", "CAR_AGE")

cat("\nNumeric variable summary:\n")
summary(train_raw[, numeric_vars])

# Check missing values
missing_counts <- colSums(is.na(train_raw))
cat("\nMissing values:\n")
print(missing_counts[missing_counts > 0])

# Correlation with TARGET_FLAG (numeric vars only, removing NAs)
cor_data <- train_raw[, c(numeric_vars, "TARGET_FLAG")]
cor_complete <- cor_data[complete.cases(cor_data), ]
correlations <- cor(cor_complete)[, "TARGET_FLAG"]
cat("\nCorrelations with TARGET_FLAG:\n")
print(sort(correlations[-length(correlations)], decreasing = TRUE))

################################################################################
# SECTION 2: DATA PREPARATION
################################################################################

# Create clean copies
train <- train_raw
eval <- eval_raw

# Store median values from training data for imputation
age_med <- median(train$AGE, na.rm = TRUE)
yoj_med <- median(train$YOJ, na.rm = TRUE)
income_med <- median(train$INCOME, na.rm = TRUE)
homeval_med <- median(train$HOME_VAL, na.rm = TRUE)
carage_med <- median(train$CAR_AGE[train$CAR_AGE >= 0], na.rm = TRUE)

# Create missing value indicator flags before imputation
train$AGE_MISS <- as.integer(is.na(train$AGE))
train$YOJ_MISS <- as.integer(is.na(train$YOJ))
train$INCOME_MISS <- as.integer(is.na(train$INCOME))
train$HOME_VAL_MISS <- as.integer(is.na(train$HOME_VAL))
train$CAR_AGE_MISS <- as.integer(is.na(train$CAR_AGE) | train$CAR_AGE < 0)

eval$AGE_MISS <- as.integer(is.na(eval$AGE))
eval$YOJ_MISS <- as.integer(is.na(eval$YOJ))
eval$INCOME_MISS <- as.integer(is.na(eval$INCOME))
eval$HOME_VAL_MISS <- as.integer(is.na(eval$HOME_VAL))
eval$CAR_AGE_MISS <- as.integer(is.na(eval$CAR_AGE) | eval$CAR_AGE < 0)

# Impute missing values with training medians
train$AGE[is.na(train$AGE)] <- age_med
train$YOJ[is.na(train$YOJ)] <- yoj_med
train$INCOME[is.na(train$INCOME)] <- income_med
train$HOME_VAL[is.na(train$HOME_VAL)] <- homeval_med
train$CAR_AGE[is.na(train$CAR_AGE) | train$CAR_AGE < 0] <- carage_med

eval$AGE[is.na(eval$AGE)] <- age_med
eval$YOJ[is.na(eval$YOJ)] <- yoj_med
eval$INCOME[is.na(eval$INCOME)] <- income_med
eval$HOME_VAL[is.na(eval$HOME_VAL)] <- homeval_med
eval$CAR_AGE[is.na(eval$CAR_AGE) | eval$CAR_AGE < 0] <- carage_med

# Log transformations for right-skewed monetary variables
train$LOG_INCOME <- log1p(train$INCOME)
train$LOG_HOME_VAL <- log1p(train$HOME_VAL)
train$LOG_BLUEBOOK <- log1p(train$BLUEBOOK)
train$LOG_OLDCLAIM <- log1p(train$OLDCLAIM)

eval$LOG_INCOME <- log1p(eval$INCOME)
eval$LOG_HOME_VAL <- log1p(eval$HOME_VAL)
eval$LOG_BLUEBOOK <- log1p(eval$BLUEBOOK)
eval$LOG_OLDCLAIM <- log1p(eval$OLDCLAIM)

# Square root for count-like variables
train$SQRT_TRAVTIME <- sqrt(train$TRAVTIME)
train$SQRT_MVR_PTS <- sqrt(train$MVR_PTS)

eval$SQRT_TRAVTIME <- sqrt(eval$TRAVTIME)
eval$SQRT_MVR_PTS <- sqrt(eval$MVR_PTS)

# Quadratic age term to capture U-shaped risk
train$AGE_SQ <- train$AGE^2
eval$AGE_SQ <- eval$AGE^2

# Binary indicators for key categorical variables
train$IS_URBAN <- as.integer(train$URBANICITY == "Highly Urban/ Urban")
train$IS_REVOKED <- as.integer(train$REVOKED == "Yes")
train$IS_COMMERCIAL <- as.integer(train$CAR_USE == "Commercial")
train$IS_MARRIED <- as.integer(train$MSTATUS == "Yes")
train$IS_MALE <- as.integer(train$SEX == "M")
train$HAS_KIDS_DRIVE <- as.integer(train$KIDSDRIV > 0)

eval$IS_URBAN <- as.integer(eval$URBANICITY == "Highly Urban/ Urban")
eval$IS_REVOKED <- as.integer(eval$REVOKED == "Yes")
eval$IS_COMMERCIAL <- as.integer(eval$CAR_USE == "Commercial")
eval$IS_MARRIED <- as.integer(eval$MSTATUS == "Yes")
eval$IS_MALE <- as.integer(eval$SEX == "M")
eval$HAS_KIDS_DRIVE <- as.integer(eval$KIDSDRIV > 0)

# Create data subset for linear regression (crashes only)
train_crash <- train[train$TARGET_FLAG == 1, ]
train_crash$LOG_TARGET_AMT <- log1p(train_crash$TARGET_AMT)

cat("\nData preparation complete.")
cat("\nCrash observations for linear regression:", nrow(train_crash), "\n")

################################################################################
# SECTION 3: BUILD MODELS
################################################################################

#-------------------------------------------------------------------------------
# LOGISTIC REGRESSION MODELS (Predicting TARGET_FLAG)
#-------------------------------------------------------------------------------

# Model 1: Full model with original and transformed variables
logit1 <- glm(TARGET_FLAG ~ KIDSDRIV + AGE + HOMEKIDS + YOJ + LOG_INCOME + 
                PARENT1 + LOG_HOME_VAL + MSTATUS + SEX + EDUCATION + 
                TRAVTIME + CAR_USE + LOG_BLUEBOOK + TIF + CAR_TYPE + 
                RED_CAR + LOG_OLDCLAIM + CLM_FREQ + REVOKED + MVR_PTS + 
                CAR_AGE + URBANICITY,
              data = train, family = binomial(link = "logit"))

cat("\n========== LOGISTIC MODEL 1: Full Model ==========\n")
print(summary(logit1))

# Model 2: Stepwise selection starting from full model with transformations
logit_full <- glm(TARGET_FLAG ~ KIDSDRIV + AGE + AGE_SQ + HOMEKIDS + YOJ + 
                    LOG_INCOME + PARENT1 + LOG_HOME_VAL + MSTATUS + SEX + 
                    EDUCATION + SQRT_TRAVTIME + CAR_USE + LOG_BLUEBOOK + TIF + 
                    CAR_TYPE + RED_CAR + LOG_OLDCLAIM + CLM_FREQ + REVOKED + 
                    SQRT_MVR_PTS + CAR_AGE + IS_URBAN + 
                    INCOME_MISS + HOME_VAL_MISS + YOJ_MISS,
                  data = train, family = binomial(link = "logit"))

logit2 <- stepAIC(logit_full, direction = "both", trace = 0)

cat("\n========== LOGISTIC MODEL 2: Stepwise Selection ==========\n")
print(summary(logit2))

# Model 3: Parsimonious theory-driven model
logit3 <- glm(TARGET_FLAG ~ CLM_FREQ + MVR_PTS + IS_REVOKED + IS_COMMERCIAL + 
                IS_URBAN + KIDSDRIV + TIF + LOG_HOME_VAL + IS_MARRIED,
              data = train, family = binomial(link = "logit"))

cat("\n========== LOGISTIC MODEL 3: Parsimonious Model ==========\n")
print(summary(logit3))

# Coefficient interpretation for Model 3
cat("\nOdds Ratios for Model 3:\n")
print(round(exp(coef(logit3)), 4))

#-------------------------------------------------------------------------------
# LINEAR REGRESSION MODELS (Predicting TARGET_AMT for crashes)
#-------------------------------------------------------------------------------

# Model 1: Full linear model
lm1 <- lm(LOG_TARGET_AMT ~ KIDSDRIV + AGE + HOMEKIDS + YOJ + LOG_INCOME + 
            LOG_HOME_VAL + MSTATUS + SEX + EDUCATION + TRAVTIME + CAR_USE + 
            LOG_BLUEBOOK + TIF + CAR_TYPE + RED_CAR + LOG_OLDCLAIM + CLM_FREQ + 
            MVR_PTS + CAR_AGE + IS_URBAN,
          data = train_crash)

cat("\n========== LINEAR MODEL 1: Full Model ==========\n")
print(summary(lm1))

# Model 2: Stepwise selection
lm_full <- lm(LOG_TARGET_AMT ~ KIDSDRIV + AGE + AGE_SQ + HOMEKIDS + YOJ + 
                LOG_INCOME + LOG_HOME_VAL + MSTATUS + SEX + EDUCATION + 
                SQRT_TRAVTIME + CAR_USE + LOG_BLUEBOOK + TIF + CAR_TYPE + 
                RED_CAR + LOG_OLDCLAIM + CLM_FREQ + SQRT_MVR_PTS + CAR_AGE + 
                IS_URBAN + INCOME_MISS + HOME_VAL_MISS,
              data = train_crash)

lm2 <- stepAIC(lm_full, direction = "both", trace = 0)

cat("\n========== LINEAR MODEL 2: Stepwise Selection ==========\n")
print(summary(lm2))

# Model 3: Parsimonious model focused on vehicle characteristics
lm3 <- lm(LOG_TARGET_AMT ~ LOG_BLUEBOOK + CAR_TYPE + CAR_AGE + IS_COMMERCIAL + 
            MVR_PTS + AGE,
          data = train_crash)

cat("\n========== LINEAR MODEL 3: Parsimonious Model ==========\n")
print(summary(lm3))

# Check VIF for multicollinearity
cat("\nVIF for Linear Model 2:\n")
print(vif(lm2))

################################################################################
# SECTION 4: MODEL SELECTION AND EVALUATION
################################################################################

#-------------------------------------------------------------------------------
# Logistic Model Performance Metrics
#-------------------------------------------------------------------------------

calc_logit_metrics <- function(model, data, name) {
  probs <- predict(model, newdata = data, type = "response")
  preds <- ifelse(probs >= 0.5, 1, 0)
  actual <- data$TARGET_FLAG
  
  cm <- table(Predicted = preds, Actual = actual)
  TP <- ifelse("1" %in% rownames(cm) & "1" %in% colnames(cm), cm["1","1"], 0)
  TN <- ifelse("0" %in% rownames(cm) & "0" %in% colnames(cm), cm["0","0"], 0)
  FP <- ifelse("1" %in% rownames(cm) & "0" %in% colnames(cm), cm["1","0"], 0)
  FN <- ifelse("0" %in% rownames(cm) & "1" %in% colnames(cm), cm["0","1"], 0)
  
  accuracy <- (TP + TN) / (TP + TN + FP + FN)
  precision <- ifelse((TP + FP) > 0, TP / (TP + FP), 0)
  sensitivity <- ifelse((TP + FN) > 0, TP / (TP + FN), 0)
  specificity <- ifelse((TN + FP) > 0, TN / (TN + FP), 0)
  f1 <- ifelse((precision + sensitivity) > 0, 
               2 * precision * sensitivity / (precision + sensitivity), 0)
  
  roc_obj <- roc(actual, probs, quiet = TRUE)
  auc_val <- as.numeric(auc(roc_obj))
  
  list(
    name = name,
    aic = AIC(model),
    accuracy = accuracy,
    error_rate = 1 - accuracy,
    precision = precision,
    sensitivity = sensitivity,
    specificity = specificity,
    f1 = f1,
    auc = auc_val,
    confusion = cm,
    roc = roc_obj
  )
}

m1_metrics <- calc_logit_metrics(logit1, train, "Model 1: Full")
m2_metrics <- calc_logit_metrics(logit2, train, "Model 2: Stepwise")
m3_metrics <- calc_logit_metrics(logit3, train, "Model 3: Parsimonious")

cat("\n========== LOGISTIC MODEL COMPARISON ==========\n")
logit_comparison <- data.frame(
  Model = c(m1_metrics$name, m2_metrics$name, m3_metrics$name),
  AIC = round(c(m1_metrics$aic, m2_metrics$aic, m3_metrics$aic), 1),
  Accuracy = round(c(m1_metrics$accuracy, m2_metrics$accuracy, m3_metrics$accuracy), 4),
  Precision = round(c(m1_metrics$precision, m2_metrics$precision, m3_metrics$precision), 4),
  Sensitivity = round(c(m1_metrics$sensitivity, m2_metrics$sensitivity, m3_metrics$sensitivity), 4),
  Specificity = round(c(m1_metrics$specificity, m2_metrics$specificity, m3_metrics$specificity), 4),
  F1 = round(c(m1_metrics$f1, m2_metrics$f1, m3_metrics$f1), 4),
  AUC = round(c(m1_metrics$auc, m2_metrics$auc, m3_metrics$auc), 4)
)
print(logit_comparison)

cat("\nConfusion Matrices:\n")
cat("\nModel 1:\n"); print(m1_metrics$confusion)
cat("\nModel 2:\n"); print(m2_metrics$confusion)
cat("\nModel 3:\n"); print(m3_metrics$confusion)

#-------------------------------------------------------------------------------
# Linear Model Performance Metrics
#-------------------------------------------------------------------------------

calc_lm_metrics <- function(model, data, name) {
  pred <- predict(model, newdata = data)
  actual <- data$LOG_TARGET_AMT
  
  residuals <- actual - pred
  mse <- mean(residuals^2)
  rmse <- sqrt(mse)
  r2 <- summary(model)$r.squared
  adj_r2 <- summary(model)$adj.r.squared
  f_stat <- summary(model)$fstatistic[1]
  
  # Back-transform to dollars
  pred_dollars <- expm1(pred)
  actual_dollars <- expm1(actual)
  rmse_dollars <- sqrt(mean((actual_dollars - pred_dollars)^2))
  
  list(
    name = name,
    mse = mse,
    rmse = rmse,
    rmse_dollars = rmse_dollars,
    r2 = r2,
    adj_r2 = adj_r2,
    f_stat = f_stat,
    n_pred = length(coef(model)) - 1
  )
}

lm1_metrics <- calc_lm_metrics(lm1, train_crash, "Model 1: Full")
lm2_metrics <- calc_lm_metrics(lm2, train_crash, "Model 2: Stepwise")
lm3_metrics <- calc_lm_metrics(lm3, train_crash, "Model 3: Parsimonious")

cat("\n========== LINEAR MODEL COMPARISON ==========\n")
lm_comparison <- data.frame(
  Model = c(lm1_metrics$name, lm2_metrics$name, lm3_metrics$name),
  R2 = round(c(lm1_metrics$r2, lm2_metrics$r2, lm3_metrics$r2), 4),
  Adj_R2 = round(c(lm1_metrics$adj_r2, lm2_metrics$adj_r2, lm3_metrics$adj_r2), 4),
  MSE = round(c(lm1_metrics$mse, lm2_metrics$mse, lm3_metrics$mse), 4),
  RMSE = round(c(lm1_metrics$rmse, lm2_metrics$rmse, lm3_metrics$rmse), 4),
  RMSE_Dollars = round(c(lm1_metrics$rmse_dollars, lm2_metrics$rmse_dollars, lm3_metrics$rmse_dollars), 2),
  F_Stat = round(c(lm1_metrics$f_stat, lm2_metrics$f_stat, lm3_metrics$f_stat), 2),
  Predictors = c(lm1_metrics$n_pred, lm2_metrics$n_pred, lm3_metrics$n_pred)
)
print(lm_comparison)

#-------------------------------------------------------------------------------
# Final Model Selection
#-------------------------------------------------------------------------------

# Select Model 2 (Stepwise) for both logistic and linear regression
final_logit <- logit2
final_lm <- lm2

cat("\n========== FINAL MODEL SELECTIONS ==========\n")
cat("\nFinal Logistic Model Formula:\n")
print(formula(final_logit))
cat("\nFinal Linear Model Formula:\n")
print(formula(final_lm))

################################################################################
# PREDICTIONS ON EVALUATION DATA
################################################################################

# Predict crash probability
eval$P_CRASH <- predict(final_logit, newdata = eval, type = "response")

# Classify using 0.5 threshold
eval$CRASH_PRED <- ifelse(eval$P_CRASH >= 0.5, 1, 0)

# Predict claim amount (log scale, then back-transform)
eval$LOG_AMT_PRED <- predict(final_lm, newdata = eval)
eval$AMT_PRED <- expm1(eval$LOG_AMT_PRED)

# Expected cost = P(crash) * E[cost | crash]
eval$EXPECTED_COST <- eval$P_CRASH * eval$AMT_PRED

# Summary of predictions
cat("\n========== EVALUATION DATA PREDICTIONS SUMMARY ==========\n")
cat("Predicted crashes:", sum(eval$CRASH_PRED), "out of", nrow(eval), "\n")
cat("Predicted crash rate:", round(mean(eval$P_CRASH), 4), "\n")
cat("Mean predicted amount (given crash): $", round(mean(eval$AMT_PRED), 2), "\n")
cat("Mean expected cost per policy: $", round(mean(eval$EXPECTED_COST), 2), "\n")
cat("Total expected claims: $", format(round(sum(eval$EXPECTED_COST)), big.mark = ","), "\n")

# Create output dataframe for appendix
predictions <- data.frame(
  INDEX = eval$INDEX,
  P_CRASH = round(eval$P_CRASH, 4),
  CRASH_CLASS = eval$CRASH_PRED,
  PRED_COST = round(eval$AMT_PRED, 2),
  EXPECTED_COST = round(eval$EXPECTED_COST, 2)
)

# Save predictions to CSV
write.csv(predictions, "evaluation_predictions.csv", row.names = FALSE)
cat("\nPredictions saved to 'evaluation_predictions.csv'\n")

# Display first 20 predictions
cat("\nFirst 20 predictions:\n")
print(head(predictions, 20))

################################################################################
# GENERATE PLOTS FOR REPORT
################################################################################

# ROC curves comparison
pdf("roc_curves.pdf", width = 8, height = 6)
plot(m1_metrics$roc, col = "blue", main = "ROC Curves: Logistic Regression Models")
plot(m2_metrics$roc, col = "red", add = TRUE)
plot(m3_metrics$roc, col = "darkgreen", add = TRUE)
legend("bottomright", 
       legend = c(paste("Model 1 (AUC =", round(m1_metrics$auc, 3), ")"),
                  paste("Model 2 (AUC =", round(m2_metrics$auc, 3), ")"),
                  paste("Model 3 (AUC =", round(m3_metrics$auc, 3), ")")),
       col = c("blue", "red", "darkgreen"), lwd = 2)
dev.off()

# Residual plots for final linear model
pdf("residual_plots.pdf", width = 10, height = 8)
par(mfrow = c(2, 2))
plot(final_lm)
par(mfrow = c(1, 1))
dev.off()

# Correlation plot
pdf("correlation_plot.pdf", width = 10, height = 10)
cor_vars <- c("TARGET_FLAG", "KIDSDRIV", "AGE", "YOJ", "INCOME", "HOME_VAL",
              "TRAVTIME", "BLUEBOOK", "TIF", "OLDCLAIM", "CLM_FREQ", "MVR_PTS", "CAR_AGE")
cor_subset <- train[, cor_vars]
cor_subset <- cor_subset[complete.cases(cor_subset), ]
cor_matrix <- cor(cor_subset)
corrplot(cor_matrix, method = "color", type = "lower", tl.cex = 0.8,
         addCoef.col = "black", number.cex = 0.6)
dev.off()

# TARGET_FLAG distribution
pdf("target_distribution.pdf", width = 10, height = 5)
par(mfrow = c(1, 2))
barplot(table(train$TARGET_FLAG), col = c("steelblue", "coral"),
        main = "Distribution of Crash Occurrence",
        xlab = "Crash (0=No, 1=Yes)", ylab = "Count",
        names.arg = c("No Crash", "Crash"))
hist(train_crash$TARGET_AMT, breaks = 50, col = "coral",
     main = "Claim Amount Distribution (Crashes Only)",
     xlab = "Claim Amount ($)")
par(mfrow = c(1, 1))
dev.off()

cat("\nPlots saved to PDF files.\n")

################################################################################
# GENERATE LATEX-READY OUTPUT TABLES
################################################################################

# Function to print LaTeX table
cat("\n========== LATEX-READY TABLES ==========\n")

# Logistic model comparison table
cat("\n% Logistic Model Comparison Table\n")
cat("\\begin{table}[H]\n\\centering\n")
cat("\\caption{Logistic Regression Model Performance Comparison}\n")
cat("\\begin{tabular}{lrrrrrrr}\n\\toprule\n")
cat("Model & AIC & Accuracy & Precision & Sensitivity & Specificity & F1 & AUC \\\\\n\\midrule\n")
for(i in 1:nrow(logit_comparison)) {
  cat(sprintf("%s & %.0f & %.4f & %.4f & %.4f & %.4f & %.4f & %.4f \\\\\n",
              logit_comparison$Model[i], logit_comparison$AIC[i],
              logit_comparison$Accuracy[i], logit_comparison$Precision[i],
              logit_comparison$Sensitivity[i], logit_comparison$Specificity[i],
              logit_comparison$F1[i], logit_comparison$AUC[i]))
}
cat("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# Linear model comparison table
cat("\n% Linear Model Comparison Table\n")
cat("\\begin{table}[H]\n\\centering\n")
cat("\\caption{Linear Regression Model Performance Comparison}\n")
cat("\\begin{tabular}{lrrrrrrr}\n\\toprule\n")
cat("Model & $R^2$ & Adj. $R^2$ & MSE & RMSE & RMSE (\\$) & F-stat & Pred \\\\\n\\midrule\n")
for(i in 1:nrow(lm_comparison)) {
  cat(sprintf("%s & %.4f & %.4f & %.4f & %.4f & %.2f & %.2f & %d \\\\\n",
              lm_comparison$Model[i], lm_comparison$R2[i],
              lm_comparison$Adj_R2[i], lm_comparison$MSE[i],
              lm_comparison$RMSE[i], lm_comparison$RMSE_Dollars[i],
              lm_comparison$F_Stat[i], lm_comparison$Predictors[i]))
}
cat("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# Predictions table (first 30)
cat("\n% Evaluation Predictions Table (First 30)\n")
cat("\\begin{longtable}{rrrrr}\n")
cat("\\caption{Evaluation Data Predictions} \\\\\n\\toprule\n")
cat("INDEX & P(Crash) & Classification & Pred. Cost (\\$) & Expected Cost (\\$) \\\\\n\\midrule\n")
cat("\\endfirsthead\n\\toprule\n")
cat("INDEX & P(Crash) & Classification & Pred. Cost (\\$) & Expected Cost (\\$) \\\\\n\\midrule\n")
cat("\\endhead\n\\bottomrule\n\\endfoot\n")
for(i in 1:min(30, nrow(predictions))) {
  cat(sprintf("%d & %.4f & %d & %.2f & %.2f \\\\\n",
              predictions$INDEX[i], predictions$P_CRASH[i],
              predictions$CRASH_CLASS[i], predictions$PRED_COST[i],
              predictions$EXPECTED_COST[i]))
}
cat("\\multicolumn{5}{c}{\\textit{(Complete data in CSV file)}} \\\\\n")
cat("\\end{longtable}\n")

cat("\n========== ANALYSIS COMPLETE ==========\n")
