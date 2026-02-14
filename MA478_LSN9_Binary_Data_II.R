# ============================================================
# MA478 GLM Review: Logistic Regression with Pima Dataset
# Focus: model building, deviance tests, interpretation, metrics,
#        diagnostics, calibration
# Data: faraway::pima (response = test; 1 = diabetes)
# ============================================================

# ---------------------------
# 0) Setup
# ---------------------------
library(tidyverse)
library(faraway)

# Modeling helpers
library(broom)             # tidy() / glance()
library(pROC)              # ROC / AUC
library(ResourceSelection) # Hosmer-Lemeshow (optional but handy)
library(car)               # vif (optional)

set.seed(478)              # reproducibility

# ---------------------------
# 1) Load + quick audit
# ---------------------------
data(pima, package = "faraway")

glimpse(pima)
summary(pima)

# Response should be 0/1 (often it is in faraway)
# Make sure it's coded as integer 0/1, and also keep a factor copy for plots.
pima <- pima %>%
  mutate(
    test = as.integer(test),
    test_f = factor(test, levels = c(0, 1), labels = c("No diabetes", "Diabetes"))
  )

# ---------------------------
# 2) Data cleaning: "0" sometimes represents missing for physiologic measures
# Common choice in Pima-style datasets:
# glucose, diastolic, triceps, insulin, bmi cannot realistically be 0.
# (pregnant and diabetes pedigree can be 0; age cannot be 0 but usually isn't.)
# ---------------------------
pima <- pima %>%
  mutate(across(c(glucose, diastolic, triceps, insulin, bmi),
                ~ na_if(., 0)))

# Check missingness after recode
pima %>%
  summarise(across(everything(), ~ sum(is.na(.))))

# Option A (simplest for class): drop incomplete cases
pima_cc <- pima %>% drop_na(glucose, diastolic, triceps, insulin, 
                            bmi, age, diabetes, pregnant)

# Option B (simple median impute) — uncomment if you prefer keeping rows
# median_impute <- function(x) { replace(x, is.na(x), median(x, na.rm = TRUE)) }
# pima_cc <- pima %>%
#   mutate(across(where(is.numeric), median_impute))

# ---------------------------
# 3) Train/test split
# ---------------------------
n <- nrow(pima_cc)
idx_train <- sample(seq_len(n), size = floor(0.75 * n))
train <- pima_cc[idx_train, ]
test  <- pima_cc[-idx_train, ]

# Do not touch test for model decisions beyond final evaluation
glimpse(train)

# ---------------------------
# 4) Quick EDA (training only)
# ---------------------------
train %>%
  ggplot(aes(x = test_f)) +
  geom_bar() +
  labs(title = "Class balance (training set)", x = NULL, y = "Count")

# A few predictor distributions
train %>%
  ggplot(aes(x = glucose)) +
  geom_histogram(bins = 30) +
  labs(title = "Glucose distribution (training)", x = "glucose")

train %>%
  ggplot(aes(x = glucose, fill = test_f)) +
  geom_histogram(bins = 30, position = "identity", alpha = 0.5) +
  labs(title = "Glucose by outcome (training)", x = "glucose", fill = NULL)

# Pairwise feel (optional quick)
GGally::ggpairs(train) # Click the zoom magnifying glass to pop out and enlarge the plot.

# ---------------------------
# 5) Fit logistic regression models (GLM)
# ---------------------------
# Baseline (intercept-only)
m0 <- glm(test ~ 1, data = train, family = binomial(link = "logit"))

# Simple model
m1 <- glm(test ~ glucose + bmi + age, data = train, family = binomial())

# Expanded model (reasonable set)
m2 <- glm(test ~ pregnant + glucose + diastolic + triceps + insulin + bmi + diabetes + age,
          data = train, family = binomial())

# A “leaner” alternative (example of model selection conversation)
m3 <- glm(test ~ pregnant + glucose + bmi + diabetes + age,
          data = train, family = binomial())

# Summaries
summary(m1)
summary(m2)
summary(m3)

# Compare via AIC (smaller is better)
AIC(m0, m1, m2, m3)

# ---------------------------
# 6) Deviance-based hypothesis tests (nested models)
# ---------------------------
# Likelihood ratio test (difference in deviance) for nested models:
anova(m0, m1, test = "Chisq")
anova(m1, m3, test = "Chisq")
anova(m3, m2, test = "Chisq")   # m3 nested in m2? (yes, if m3 predictors are subset of m2)

# NOTE:
# - For logistic regression, these are large-sample chi-square approximations.
# - They answer: "Does the larger model fit significantly better?"

# ---------------------------
# 7) Interpret coefficients, odds ratios, confidence intervals
# ---------------------------
tidy(m3) %>%
  mutate(
    odds_ratio = exp(estimate),
    OR_low = exp(estimate - 1.96 * std.error),
    OR_high = exp(estimate + 1.96 * std.error)
  )

# Prefer profile likelihood intervals when possible (can be slower)
# confint() in glm does profile likelihood by default.
or_table <- broom::tidy(m3) %>%
  mutate(
    OR = exp(estimate)
  )

ci_prof <- confint(m3) # profile likelihood CIs
ci_prof_OR <- exp(ci_prof)

cbind(or_table, ci_prof_OR[rownames(ci_prof_OR), ]) %>%
  as_tibble() %>%
  rename(OR_low = `2.5 %`, OR_high = `97.5 %`)

# Interpretation tip:
# exp(beta_glucose) = multiplicative change in odds for a 1-unit increase in glucose,
# holding other predictors fixed. 

# ---------------------------
# 8) Predicted probabilities + classification
# ---------------------------
# Helper: confusion matrix + common metrics at a given threshold
metrics_at_threshold <- function(p_hat, y, thr = 0.5) {
  yhat <- ifelse(p_hat >= thr, 1, 0)
  TP <- sum(yhat == 1 & y == 1)
  TN <- sum(yhat == 0 & y == 0)
  FP <- sum(yhat == 1 & y == 0)
  FN <- sum(yhat == 0 & y == 1)
  
  acc  <- (TP + TN) / (TP + TN + FP + FN)
  sens <- ifelse((TP + FN) == 0, NA, TP / (TP + FN))   # aka recall or TPR
  spec <- ifelse((TN + FP) == 0, NA, TN / (TN + FP))   # TNR
  prec <- ifelse((TP + FP) == 0, NA, TP / (TP + FP))
  ppv  <- ifelse((TP + FP) == 0, NA, TP / (TP + FP))
  npv  <- ifelse((TN + FN) == 0, NA, TN / (TN + FN))
  f1   <- ifelse(is.na(prec) | is.na(sens) | (prec + sens) == 0, NA, 2 * prec * sens / (prec + sens))
  
  tibble(
    threshold = thr,
    TP = TP, TN = TN, FP = FP, FN = FN,
    accuracy = acc,
    sensitivity = sens,
    specificity = spec,
    precision = prec,
    recall = sens,
    ppv = ppv,
    npv = npv,
    F1 = f1
  )
}

# Choose a “final” model for evaluation (use m3 here)
test$p_hat <- predict(m3, newdata = test, type = "response")

# Confusion matrix at 0.5
metrics_at_threshold(test$p_hat, test$test, thr = 0.5)

# Look across thresholds (Is something other than 0.5 desirable?)
thr_grid <- seq(0.05, 0.95, by = 0.05)
metric_curve <- map_dfr(thr_grid, ~ metrics_at_threshold(test$p_hat, test$test, thr = .x))

metric_curve %>%
  pivot_longer(cols = c(accuracy, sensitivity, specificity, precision, F1),
               names_to = "metric", values_to = "value") %>%
  ggplot(aes(x = threshold, y = value)) +
  geom_line() +
  facet_wrap(~ metric, scales = "free_y") +
  labs(title = "Performance vs threshold (test set)")

# Pick threshold by Youden's J (sensitivity + specificity - 1) on TEST (for demo only)
# In practice pick on training or via CV, then evaluate once on test.
metric_curve %>%
  mutate(youdenJ = sensitivity + specificity - 1) %>%
  arrange(desc(youdenJ)) %>%
  slice(1)

# ---------------------------
# 9) ROC curve + AUC
# ---------------------------
roc_obj <- roc(response = test$test, predictor = test$p_hat, quiet = TRUE)
auc(roc_obj)

plot(roc_obj, main = "ROC Curve (test set)")

# Add AUC in a caption-like way
cat("AUC =", as.numeric(auc(roc_obj)), "\n")

# ---------------------------
# 10) Calibration diagnostics 
# ---------------------------

# Calibration is about "probability honesty." A model is well calibrated if:
# - Among observations where the model predicts ~30% risk, about 30% actually experience the event;
# - Among those predicted at ~70%, about 70% actually experience the event; etc.

# 10a) Calibration plot (bin predicted probs, compare mean predicted vs observed)
cal <- test %>%
  mutate(bin = ntile(p_hat, 10)) %>%
  group_by(bin) %>%
  summarise(
    p_mean = mean(p_hat),
    y_mean = mean(test),
    n = n(),
    .groups = "drop"
  )

ggplot(cal, aes(x = p_mean, y = y_mean)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1) +
  labs(
    title = "Calibration plot (test set, deciles)",
    x = "Mean predicted probability",
    y = "Observed event rate"
  )

# 10b) Brier score (mean squared error for probabilities)
# Two models could be equal in terms of classification accuracy but very
# very different in terms of how far off their estimated probabilities are. 
brier <- mean((test$test - test$p_hat)^2)
brier

# 10c) Hosmer-Lemeshow test (commonly taught; interpret cautiously)
# (Need numeric 0/1 response)
# Groups observations by predicted probability (again, usually deciles)
# Compares observed vs expected counts in each group
# Computes a Pearson-style chi-square statistic
# Use cautiously!  This test is sensitive to sample size, depends on number of bins,
# often rejects “good” models in large samples, and often fails to reject bad models 
# in small samples
hoslem.test(test$test, test$p_hat, g = 10)

# ---------------------------
# 11) Core GLM diagnostics (training fit)
# ---------------------------
# Fitted probabilities and linear predictor
train$p_hat <- predict(m3, type = "response")
train$eta   <- predict(m3, type = "link")     # linear predictor Xβ

# Residual types
train$resid_dev <- residuals(m3, type = "deviance")
train$resid_pear <- residuals(m3, type = "pearson")

# 11a) Residuals vs linear predictor (eta), deviance residuals
# Why isn't this very helpful?
ggplot(train, aes(x = eta, y = resid_dev)) +
  geom_point(alpha = 0.7) +
  geom_hline(yintercept = 0) +
  labs(title = "Deviance residuals vs linear predictor (training)",
       x = expression(eta == X*beta), y = "Deviance residual")

# 11b) Residuals vs fitted probability
# Why isn't this very helpful?
ggplot(train, aes(x = p_hat, y = resid_dev)) +
  geom_point(alpha = 0.7) +
  geom_hline(yintercept = 0) +
  labs(title = "Deviance residuals vs fitted probability (training)",
       x = expression(hat(p)), y = "Deviance residual")

# 11c) “Binned residual plot” (Gelman-Hill style quick version)
# Bin fitted probabilities, then look at average residual per bin
# Think about how this is related to some other diagnostics we've considered.
binned_resid <- function(p_hat, y, bins = 20) {
  df <- tibble(p_hat = p_hat, y = y) %>%
    mutate(bin = ntile(p_hat, bins)) %>%
    group_by(bin) %>%
    summarise(
      p_mean = mean(p_hat),
      resid_mean = mean(y - p_hat),
      n = n(),
      se = sqrt(mean(p_hat * (1 - p_hat)) / n), # rough scale for reference
      .groups = "drop"
    )
  df
}

br <- binned_resid(train$p_hat, train$test, bins = 20)

ggplot(br, aes(x = p_mean, y = resid_mean)) +
  geom_point() +
  geom_hline(yintercept = 0) +
  geom_errorbar(aes(ymin = resid_mean - 2 * se, ymax = resid_mean + 2 * se)) +
  labs(title = "Binned residual plot (training)",
       x = "Mean fitted probability in bin",
       y = expression(paste("Mean (y - ", hat(p), ") in bin")))

# ---------------------------
# 12) Influence / leverage / outliers
# ---------------------------
# Hat values (leverage), Cook's distance, DFBETAs
train$hat <- hatvalues(m3)
train$cooks <- cooks.distance(m3)

# DFBETAs: n x p matrix, one column per coefficient
dfb <- as_tibble(dfbetas(m3))
names(dfb) <- paste0("DFB_", names(coef(m3)))
train <- bind_cols(train, dfb)

# 12a) Leverage vs residual magnitude
ggplot(train, aes(x = hat, y = abs(resid_dev))) +
  geom_point(alpha = 0.7) +
  labs(title = "Leverage vs |deviance residual| (training)",
       x = "hat value (leverage)", y = "|deviance residual|")

# 12b) Cook's distance
ggplot(train, aes(x = seq_along(cooks), y = cooks)) +
  geom_point() +
  labs(title = "Cook's distance by observation (training)",
       x = "Observation index", y = "Cook's D")

# Flag top influential points
train %>%
  mutate(row_id = row_number()) %>%
  arrange(desc(cooks)) %>%
  select(row_id, cooks, hat, resid_dev, pregnant, glucose, bmi, diabetes, age, test) %>%
  slice(1:10)

# ---------------------------
# 13) Collinearity check (optional)
# ---------------------------
# VIF is not “required” for GLMs but is a useful stability check.
# Interpretation: How much larger is the variance of \hat{\beta}_j than it would
# be if X_j were orthogonal to the other predictors?
# Multicollinearity does NOT bias coefficients. It inflates standard errors.
# Inflated SEs ⇒ 
# - unstable estimates
# - wide confidence intervals
# - “significance” that flips with small data changes
# So VIF is a coefficient stability diagnostic, not a prediction diagnostic.
# ~1 is good (essentially no collinearity), 1-2 is mild (usually okay), 2-5 is
# moderate (watch out), 5-10 coefficients are unstable, >10 red flag
vif(m3)

# ---------------------------
# 14) Link function reminders (quick demo)
# ---------------------------
# Same linear predictor, different link:
m3_probit <- glm(test ~ pregnant + glucose + bmi + diabetes + age,
                 data = train, family = binomial(link = "probit"))

m3_cloglog <- glm(test ~ pregnant + glucose + bmi + diabetes + age,
                  data = train, family = binomial(link = "cloglog"))

AIC(m3, m3_probit, m3_cloglog)

# Compare predicted probabilities on the test set
test_df<- test
test %>%
  transmute(
    p_logit   = predict(m3,        newdata = test_df, type = "response"),
    p_probit  = predict(m3_probit, newdata = test_df, type = "response"),
    p_cloglog = predict(m3_cloglog,newdata = test_df, type = "response")
  ) %>%
  summarise(across(everything(), list(mean = mean, sd = sd)))

# ---------------------------
# 15) Wrap-up: one “report” tibble for the test set
# ---------------------------
final_eval <- tibble(
  model = "m3 (logit)",
  AUC = as.numeric(auc(roc(response = test$test, predictor = test$p_hat, quiet = TRUE))),
  Brier = mean((test$test - test$p_hat)^2)
) %>%
  bind_cols(metrics_at_threshold(test$p_hat, test$test, thr = 0.5) %>% select(-threshold))

final_eval
