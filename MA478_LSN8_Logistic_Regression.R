# ==============================================================
# Logistic Regression Example: Pima Diabetes (faraway)
# ==============================================================

library(tidyverse)
library(faraway)
library(pROC)

set.seed(478)   # reproducibility

# ==============================================================
# 1. Load + quick exploration
# ==============================================================

data(pima)
glimpse(pima)
help(pima)

# This data is missing all the entries in the test column, so we'll 
# get it from somewhere else (Kaggle).  The columns have slightly 
# different names, but their meaning is the same.
pima <- read_csv("pima.CSV") %>% 
  mutate(Outcome = as.factor(Outcome))

# Outcome is 'Outcome' instead of 'test' (0/1)
table(pima$Outcome)

# quick numeric summaries
summary(pima)

# visualize separation for a couple predictors
pima %>%
  ggplot(aes(Glucose, fill = Outcome, group = Outcome)) +
  geom_density(alpha = .4)

pima %>%
  ggplot(aes(BMI, fill = Outcome, group = Outcome)) +
  geom_density(alpha = .4)

# ==============================================================
# 2. Train / Test split
# ==============================================================

n <- nrow(pima)
train_id <- sample(1:n, size = 0.7*n)

train <- pima[train_id, ]
test  <- pima[-train_id, ]

# ==============================================================
# 3. Fit logistic models
# ==============================================================

# Full Model
fit_full <- glm(Outcome ~ ., data = train, family = binomial)

summary(fit_full)


# Smaller Model for Comparison (example)
fit_small <- glm(Outcome ~ Glucose + BMI + Age, 
                 data = train, family = binomial)

summary(fit_small)

# ==============================================================
# 4. Interpret coefficients (odds ratios)
# ==============================================================

coef_table <- broom::tidy(fit_full)

odds_ratios <- coef_table %>%
  mutate(
    OR = exp(estimate),
    OR_low  = exp(estimate - 1.96*std.error),
    OR_high = exp(estimate + 1.96*std.error)
  )

odds_ratios

# classroom talking point:
# exp(beta_j) = multiplicative change in odds for 1-unit increase in x_j

exp(confint(fit_full)) # using profile likelihood
exp(confint.default(fit_full)) # using Wald

# ==============================================================
# 5. Model Comparison
# ==============================================================

# Likelihood Ratio Test via Deviance
anova(fit_small, fit_full, test = "Chisq")

AIC(fit_small, fit_full)

# ==============================================================
# 6. Prediction
# ==============================================================

prob_hat <- predict(fit_full, newdata = test, type = "response")
# What is the response variable?  It's not y_i, it's p_i.

# default 0.5 threshold
class_hat <- ifelse(prob_hat > 0.5, "pos", "neg") %>%
  factor(levels = c("neg","pos"))

# ==============================================================
# 7. Confusion matrix + accuracy
# ==============================================================

cm <- table(Predicted = class_hat, Actual = test$Outcome)
cm

accuracy <- sum(diag(cm)) / sum(cm)
accuracy

# additional metrics (often useful)
sensitivity <- cm[2,2] / sum(cm[,2])
specificity <- cm[1,1] / sum(cm[,1])

c(accuracy = accuracy,
  sensitivity = sensitivity,
  specificity = specificity)

# There's also precision, recall, PPV, NPV, F1, ...

# ==============================================================
# 8. ROC curve + AUC
# ==============================================================

roc_obj <- roc(test$Outcome, prob_hat)

auc(roc_obj)

plot(roc_obj, print.auc = TRUE)

# ==============================================================
# 9. (Optional) Threshold-Tuning Demonstration
# ==============================================================

thresholds <- seq(0,1,by=.01)

acc_vec <- map_dbl(thresholds, function(t){
  preds <- factor(ifelse(prob_hat > t, 1, 0),
                  levels = c(0, 1))
  mean(preds == test$Outcome)
})

tibble(threshold = thresholds,
       accuracy = acc_vec) %>%
  ggplot(aes(threshold, accuracy)) +
  geom_line()
