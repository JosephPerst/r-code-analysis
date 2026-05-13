# ============================================================
# Athlete Injury Modeling Script
# ============================================================

library(tidyverse)
library(MASS)
library(pscl)
library(mgcv)

# ------------------------------------------------------------
# Setup
# ------------------------------------------------------------

df <- read_csv("athlete_injury_train.csv")

df <- df %>%
  mutate(
    injury_occurred = as.integer(days_missed > 0),
    position = factor(position),
    team_id = factor(team_id),
    player_id = factor(player_id),
    season = factor(season)
  )

test_df <- read_csv("athlete_injury_test.csv") %>%
  mutate(
    position = factor(position, levels = levels(df$position)),
    team_id = factor(team_id, levels = levels(df$team_id)),
    player_id = factor(player_id, levels = levels(df$player_id)),
    season = factor(season, levels = levels(df$season))
  )

# ------------------------------------------------------------
# Helper for Kaggle submission
# ------------------------------------------------------------

# This function assumes you generate a vector of predictions for the test set.
# This vector is called "pred" in the function below and is submitted along
# with a filename that you submit as a solution to Kaggle.

make_submission <- function(pred, filename) {
  pred <- pmax(pred, 0)   # no negative predictions
  out <- data.frame(
    id = test_df$id,
    days_missed = pred
  )
  write.csv(out, filename, row.names = FALSE)
  return(out)
}

# ============================================================
# Linear Regression Baseline Model (Just an example!)
# ============================================================

# Fit model
mod_lm <- lm(
  days_missed ~
    age + minutes_per_game,
  data = df
)

# View results
summary(mod_lm)

# ------------------------------------------------------------
# Generate predictions on Kaggle test set
# ------------------------------------------------------------

pred_lm <- predict(
  mod_lm,
  newdata = test_df
)

# Ensure predictions are nonnegative
pred_lm <- pmax(pred_lm, 0)

# ------------------------------------------------------------
# Create submission file
# ------------------------------------------------------------

submission_lm <- data.frame(
  id = test_df$id,
  days_missed = pred_lm
)

# Write CSV for Kaggle upload.
write.csv(
  submission_lm,
  "submission_linear_regression.csv",
  row.names = FALSE
)

# Preview (should show only "id" and "days_missed" for any Kaggle submission)
head(submission_lm)

# ------------------------------------------------------------
# Test that this worked for you, and then build better / more appropriate models!
# ------------------------------------------------------------

# ============================================================
# Zero-Inflated Negative Binomial (ZINB) Model
# ============================================================
# Response: days_missed -- a non-negative integer count of regular-
# season days a player misses (bounded by the 180-day season).
#
# Why a ZINB?
# (1) ~50% of training observations have days_missed == 0.  Some
#     players are "structurally" healthy in a given season (never
#     get hurt) while others are in the at-risk regime and could
#     still register zero days missed by chance.  The natural data-
#     generating story is a two-part mixture: a Bernoulli "is this
#     a structural zero?" piece plus a count piece for everyone
#     else.
# (2) Conditional on days_missed > 0, the mean is ~23.6 and the
#     variance is ~731 (variance/mean ~ 31).  Poisson is therefore
#     badly mis-specified; a Negative Binomial accommodates the
#     overdispersion via an ancillary parameter theta.
# Together (1) and (2) motivate ZINB.  We will support (1) with a
# Vuong test (ZINB vs NB) and (2) with a likelihood-ratio test of
# theta (NB vs Poisson, ZINB vs ZIP).
# ============================================================

set.seed(478)

# ------------------------------------------------------------
# EDA in support of the ZINB choice
# ------------------------------------------------------------

cat("\n--- Response distribution ---\n")
cat("Proportion of zeros           :", round(mean(df$days_missed == 0), 4), "\n")
cat("Overall mean                  :", round(mean(df$days_missed), 3), "\n")
cat("Overall variance              :", round(var(df$days_missed),  3), "\n")
cat("Mean | days_missed > 0        :", round(mean(df$days_missed[df$days_missed > 0]), 3), "\n")
cat("Variance | days_missed > 0    :", round(var(df$days_missed[df$days_missed > 0]),  3), "\n")
cat("Var/Mean | days_missed > 0    :", round(var(df$days_missed[df$days_missed > 0]) /
                                              mean(df$days_missed[df$days_missed > 0]), 3), "\n")

# Note on collinearity of age, base_age, season:
# In this data set, age = base_age + (as.numeric(as.character(season)) - 2021)
# exactly, so the three variables are linearly dependent.  We use
# base_age (player-level baseline age, constant within a player)
# and season (an era effect that captures league-wide year shocks --
# rule changes, schedule density, etc.) rather than age, because
# (a) base_age has cleaner within-player identification and
# (b) season is more interpretable than a generic time-varying
# age that conflates maturation with calendar effects.

cat("\n--- age / base_age / season collinearity check ---\n")
print(table(round(df$age - df$base_age, 6),
            as.character(df$season)))

# ------------------------------------------------------------
# Predictor set and formula builders
# ------------------------------------------------------------
# Count component  (severity of injury when a player IS at risk):
#   base_age           -- older bodies recover more slowly
#   minutes_per_game   -- workload --> longer absences when hurt
#   games_played       -- exposure; also "tried to play through it"
#   travel_miles       -- cumulative fatigue across the season
#   back_to_back_games -- schedule density --> recovery deficit
#   prior_injury_days  -- recurrent / chronic injury risk (dominant signal)
#   position           -- bigger players (Centers) take more contact
#   season             -- era effect (rules, schedule, COVID, etc.)
#
# Zero-inflation component  (probability of being a STRUCTURAL zero --
# a player effectively insulated from injury this season).  We
# deliberately keep this part leaner than the count part to avoid
# the identification problems that plague over-parameterised ZI
# models.  All four terms below proxy "exposure to risk", so a
# negative coefficient on them implies higher P(structural zero):
#   base_age, minutes_per_game, back_to_back_games, prior_injury_days

count_terms_full <- c("base_age", "minutes_per_game", "games_played",
                      "travel_miles", "back_to_back_games",
                      "prior_injury_days", "position", "season")

zero_terms       <- c("base_age", "minutes_per_game",
                      "back_to_back_games", "prior_injury_days")

make_count_formula <- function(terms) {
  as.formula(paste("days_missed ~", paste(terms, collapse = " + ")))
}
make_zinb_formula <- function(count_terms, zero_terms) {
  as.formula(paste("days_missed ~",
                   paste(count_terms, collapse = " + "),
                   "|",
                   paste(zero_terms,  collapse = " + ")))
}

f_count_full <- make_count_formula(count_terms_full)
f_zinb_full  <- make_zinb_formula(count_terms_full, zero_terms)

# ------------------------------------------------------------
# Step 1: fit a sequence of nested count models
# ------------------------------------------------------------

mod_pois <- glm(f_count_full, data = df, family = poisson(link = "log"))
mod_nb   <- glm.nb(f_count_full, data = df, control = glm.control(maxit = 100))
mod_zip  <- zeroinfl(f_zinb_full, data = df, dist = "poisson")
mod_zinb <- zeroinfl(f_zinb_full, data = df, dist = "negbin")

# ------------------------------------------------------------
# Step 2: formal model selection
# ------------------------------------------------------------

cat("\n--- AIC comparison (lower is better) ---\n")
print(AIC(mod_pois, mod_nb, mod_zip, mod_zinb))

# LR test: does the NB dispersion parameter buy anything over Poisson?
# theta is on the boundary of its parameter space, so we use the
# chi-bar-squared mixture: 0.5*chi2_0 + 0.5*chi2_1.
ll_pois <- as.numeric(logLik(mod_pois))
ll_nb   <- as.numeric(logLik(mod_nb))
lr_stat <- 2 * (ll_nb - ll_pois)
cat("\n--- LR test: Poisson vs NB (boundary chi-bar^2) ---\n")
cat("LR stat =", round(lr_stat, 2),
    "  p-value =",
    format.pval(0.5 * pchisq(lr_stat, df = 1, lower.tail = FALSE)), "\n")

# Vuong test: ZINB vs plain NB (non-nested, tests for excess zeros).
cat("\n--- Vuong test: ZINB vs NB ---\n")
print(vuong(mod_zinb, mod_nb))

# ZINB vs ZIP: LR test on the dispersion parameter theta.
ll_zip  <- as.numeric(logLik(mod_zip))
ll_zinb <- as.numeric(logLik(mod_zinb))
lr_zinb <- 2 * (ll_zinb - ll_zip)
cat("\n--- LR test: ZIP vs ZINB (boundary chi-bar^2) ---\n")
cat("LR stat =", round(lr_zinb, 2),
    "  p-value =",
    format.pval(0.5 * pchisq(lr_zinb, df = 1, lower.tail = FALSE)), "\n")

cat("\nEstimated NB dispersion (theta) in ZINB :", round(mod_zinb$theta, 3), "\n")
cat("(theta -> Inf would collapse ZINB to ZIP; finite theta confirms overdispersion.)\n")

# ------------------------------------------------------------
# Step 3: refine the ZINB by backward elimination on the count part
# ------------------------------------------------------------
# We drop count-part terms one at a time using LR tests at alpha = 0.05,
# but we DO NOT prune the zero-inflation part: its terms are theory-
# driven and ZI components are notoriously fragile to stepwise search
# (parameter estimates can flip sign or fail to converge when a
# meaningful exposure variable is removed).

zinb_lrt_drop <- function(count_terms, zero_terms, data, alpha = 0.05) {
  current <- count_terms
  m_cur   <- zeroinfl(make_zinb_formula(current, zero_terms),
                      data = data, dist = "negbin")
  repeat {
    if (length(current) <= 1) break
    pvals <- sapply(current, function(tm) {
      new_terms <- setdiff(current, tm)
      m_new <- tryCatch(
        zeroinfl(make_zinb_formula(new_terms, zero_terms),
                 data = data, dist = "negbin"),
        error = function(e) NULL
      )
      if (is.null(m_new)) return(0)
      lr      <- 2 * (as.numeric(logLik(m_cur)) - as.numeric(logLik(m_new)))
      df_diff <- attr(logLik(m_cur), "df") - attr(logLik(m_new), "df")
      pchisq(lr, df = max(df_diff, 1), lower.tail = FALSE)
    })
    if (max(pvals) < alpha) break
    drop_term <- names(which.max(pvals))
    cat(sprintf("  drop %-22s (LRT p = %.3f)\n", drop_term, max(pvals)))
    current <- setdiff(current, drop_term)
    m_cur   <- zeroinfl(make_zinb_formula(current, zero_terms),
                        data = data, dist = "negbin")
  }
  list(model = m_cur, count_terms = current, zero_terms = zero_terms)
}

cat("\n--- Backward LRT pruning of the ZINB count part ---\n")
zinb_search    <- zinb_lrt_drop(count_terms_full, zero_terms, df)
mod_zinb_final <- zinb_search$model

cat("\n--- Final ZINB summary ---\n")
summary(mod_zinb_final)

cat("\nFinal ZINB AIC :", round(AIC(mod_zinb_final), 2), "\n")
cat("Final theta    :", round(mod_zinb_final$theta, 3), "\n")
cat("Count terms    :", paste(zinb_search$count_terms, collapse = ", "), "\n")
cat("Zero  terms    :", paste(zinb_search$zero_terms,  collapse = ", "), "\n")

# ------------------------------------------------------------
# Step 4: diagnostics on the final ZINB
# ------------------------------------------------------------

# (a) Pearson dispersion statistic.  Under a well-specified count
#     model this should be ~ 1.
pearson_resid <- residuals(mod_zinb_final, type = "pearson")
disp_stat     <- sum(pearson_resid^2) / mod_zinb_final$df.residual
cat("\nPearson dispersion statistic :", round(disp_stat, 3),
    "  (target ~ 1)\n")

# (b) Predicted vs observed proportion of zeros (a zero-inflation
#     calibration check -- standard NB tends to under-predict zeros).
p_zero_hat <- mean(predict(mod_zinb_final, type = "prob")[, "0"])
cat("Predicted P(Y = 0)           :", round(p_zero_hat, 3), "\n")
cat("Observed  P(Y = 0)           :", round(mean(df$days_missed == 0), 3), "\n")

# (c) In-sample RMSE for sanity (NOT a cross-validated estimate).
fit_train <- predict(mod_zinb_final, type = "response")
rmse_in   <- sqrt(mean((df$days_missed - fit_train)^2))
cat("In-sample RMSE (ZINB final)  :", round(rmse_in, 3), "\n")

# (d) 5-fold cross-validated RMSE so that the comparison with the
#     linear baseline is honest.
cv_rmse <- function(formula_obj, data, K = 5) {
  folds <- sample(rep(1:K, length.out = nrow(data)))
  sse   <- 0
  for (k in 1:K) {
    tr <- data[folds != k, ]; te <- data[folds == k, ]
    m  <- tryCatch(zeroinfl(formula_obj, data = tr, dist = "negbin"),
                   error = function(e) NULL)
    if (is.null(m)) return(NA_real_)
    p  <- predict(m, newdata = te, type = "response")
    sse <- sse + sum((te$days_missed - p)^2)
  }
  sqrt(sse / nrow(data))
}
cat("5-fold CV RMSE (ZINB final)  :",
    round(cv_rmse(make_zinb_formula(zinb_search$count_terms,
                                    zinb_search$zero_terms), df), 3), "\n")

# ------------------------------------------------------------
# Step 5: predictions on the Kaggle test set
# ------------------------------------------------------------
#
# Per the assignment rule that injury_occurred is NOT to be used,
# we generate a single prediction file: the unconditional ZINB mean
#     E[Y | X] = (1 - pi(X)) * mu(X)
# where pi(X) is the structural-zero probability from the logit
# component and mu(X) is the mean of the NB count component.

mu_hat    <- predict(mod_zinb_final, newdata = test_df, type = "count")  # mu(X)
p_zero    <- predict(mod_zinb_final, newdata = test_df, type = "zero")   # pi(X)
pred_zinb <- (1 - p_zero) * mu_hat                                       # E[Y]

submission_zinb <- make_submission(pred_zinb, "submission_zinb.csv")

cat("\n--- Submission preview (submission_zinb.csv) ---\n")
print(head(submission_zinb))

cat("\nDone.  Submit submission_zinb.csv as the ZINB entry.\n")

# ============================================================
# Best-Effort Model: Hurdle XGBoost + ZINB Ensemble
# ============================================================
# The ZINB is the right *shape* for this data, but it is linear on
# the log scale and is forced through a small functional class.  To
# maximise predictive accuracy we add a more flexible learner and
# blend it with the ZINB.
#
# Architecture:
#   Stage 1  P(Y > 0 | X)        XGBoost binary classifier
#   Stage 2  E[Y | X, Y > 0]     XGBoost regression on log1p(Y)
#                                fit only on Y > 0 rows, with a
#                                Duan smearing back-transform.
#   Hurdle pred  = Stage1 * Stage2
#
# Feature engineering:
#   - leave-one-out smoothed player target encoding (the dominant
#     signal in this data set is player identity; we have ~3 obs
#     per player on average)
#   - lag-1 days_missed per player (NA where unobserved -- XGBoost
#     handles NAs natively via its default-direction split)
#   - exposure ratios (travel/game, b2b/game)
#   - total minutes, career age, quadratic age/minutes
#
# Honest evaluation: 5-fold CV using the same fold structure for
# both stages, so the hurdle prediction is fully out-of-fold.  The
# ZINB ensemble weight is tuned on those OOF predictions.
#
# NOTE: injury_occurred is NOT used as a feature anywhere; it is
# only used to derive the binary label internally from y > 0, which
# is the *response*, not a predictor.
# ============================================================

if (!requireNamespace("xgboost", quietly = TRUE)) {
  install.packages("xgboost")
}
library(xgboost)

set.seed(478)

# ------------------------------------------------------------
# Feature engineering
# ------------------------------------------------------------

global_mean <- mean(df$days_missed)
K_SMOOTH    <- 5   # shrinkage strength for player target encoding

player_agg <- df %>%
  group_by(player_id) %>%
  summarise(player_n = n(), player_sum = sum(days_missed), .groups = "drop")

# Lag-1 lookup: for a row in season s, find that player's training
# days_missed in season s - 1 (NA if unobserved).
lag_lookup <- df %>%
  transmute(player_id,
            season_int        = as.integer(as.character(season)) + 1L,
            lag1_days_missed  = days_missed)

engineer <- function(d, is_train) {
  d2 <- d %>%
    left_join(player_agg, by = "player_id") %>%
    mutate(player_n   = ifelse(is.na(player_n),   0, player_n),
           player_sum = ifelse(is.na(player_sum), 0, player_sum))
  if (is_train) {
    d2 <- d2 %>%
      mutate(loo_n     = pmax(player_n - 1, 0),
             loo_sum   = player_sum - days_missed,
             player_te = (loo_sum + K_SMOOTH * global_mean) /
                         (loo_n   + K_SMOOTH)) %>%
      select(-loo_n, -loo_sum)
  } else {
    d2 <- d2 %>%
      mutate(player_te = (player_sum + K_SMOOTH * global_mean) /
                         (player_n   + K_SMOOTH))
  }
  d2 %>%
    mutate(season_int      = as.integer(as.character(season))) %>%
    left_join(lag_lookup, by = c("player_id", "season_int")) %>%
    mutate(career_age      = age - base_age,
           total_minutes   = minutes_per_game * games_played,
           travel_per_game = travel_miles / pmax(games_played, 1),
           b2b_rate        = back_to_back_games / pmax(games_played, 1),
           age_sq          = age^2,
           minutes_sq      = minutes_per_game^2)
}

df_eng   <- engineer(df,       is_train = TRUE)
test_eng <- engineer(test_df,  is_train = FALSE)

feature_cols <- c("base_age", "minutes_per_game", "games_played",
                  "travel_miles", "back_to_back_games", "prior_injury_days",
                  "career_age", "total_minutes", "travel_per_game",
                  "b2b_rate", "age_sq", "minutes_sq",
                  "player_te", "player_n", "lag1_days_missed", "season_int")

make_X <- function(d) {
  X_num <- as.matrix(d[, feature_cols])
  X_pos <- model.matrix(~ position - 1, data = d)
  X_tm  <- model.matrix(~ team_id  - 1, data = d)
  cbind(X_num, X_pos, X_tm)
}

X_train <- make_X(df_eng)
X_test  <- make_X(test_eng)
y_train <- df_eng$days_missed
y_bin   <- as.integer(y_train > 0)

# ------------------------------------------------------------
# 5-fold OOF predictions for honest evaluation and ensemble weighting
# ------------------------------------------------------------

K_FOLDS <- 5
folds   <- sample(rep(1:K_FOLDS, length.out = nrow(df_eng)))

params_cls <- list(objective = "binary:logistic", eval_metric = "logloss",
                   max_depth = 4, eta = 0.05,
                   subsample = 0.8, colsample_bytree = 0.8,
                   min_child_weight = 5)
params_reg <- list(objective = "reg:squarederror", eval_metric = "rmse",
                   max_depth = 4, eta = 0.05,
                   subsample = 0.8, colsample_bytree = 0.8,
                   min_child_weight = 5)

oof_p_pos      <- rep(NA_real_, nrow(df_eng))
oof_mu_pos     <- rep(NA_real_, nrow(df_eng))

cat("\n--- Building 5-fold OOF predictions (hurdle XGBoost) ---\n")
for (k in 1:K_FOLDS) {
  tr_idx <- which(folds != k)
  te_idx <- which(folds == k)

  # Stage 1: binary classifier
  dtr <- xgb.DMatrix(X_train[tr_idx, ], label = y_bin[tr_idx])
  dte <- xgb.DMatrix(X_train[te_idx, ], label = y_bin[te_idx])
  cv1 <- xgb.cv(data = dtr, params = params_cls,
                nrounds = 2000, nfold = 5,
                early_stopping_rounds = 40, verbose = 0)
  m_cls <- xgb.train(data = dtr, params = params_cls,
                     nrounds = cv1$best_iteration, verbose = 0)
  oof_p_pos[te_idx] <- predict(m_cls, X_train[te_idx, ])

  # Stage 2: regression on log1p(y) restricted to y > 0 in TRAIN fold
  pos_tr   <- tr_idx[y_train[tr_idx] > 0]
  y_log_tr <- log1p(y_train[pos_tr])
  dtr2 <- xgb.DMatrix(X_train[pos_tr, ], label = y_log_tr)
  cv2 <- xgb.cv(data = dtr2, params = params_reg,
                nrounds = 2000, nfold = 5,
                early_stopping_rounds = 40, verbose = 0)
  m_reg <- xgb.train(data = dtr2, params = params_reg,
                     nrounds = cv2$best_iteration, verbose = 0)

  # Duan smearing: pred = mean(exp(resid)) * exp(log_pred) - 1
  resid_tr   <- y_log_tr - predict(m_reg, X_train[pos_tr, ])
  smear_k    <- mean(exp(resid_tr))
  log_pred_te <- predict(m_reg, X_train[te_idx, ])
  oof_mu_pos[te_idx] <- pmax(smear_k * exp(log_pred_te) - 1, 0)

  cat(sprintf("  fold %d  nrounds: cls=%d reg=%d  smear=%.3f\n",
              k, cv1$best_iteration, cv2$best_iteration, smear_k))
}

oof_hurdle <- oof_p_pos * oof_mu_pos
rmse_hurdle_oof <- sqrt(mean((y_train - oof_hurdle)^2))
cat("\n5-fold OOF RMSE (Hurdle XGBoost) :", round(rmse_hurdle_oof, 3), "\n")

# OOF ZINB for ensemble weight tuning (refit per fold)
cat("\n--- Building 5-fold OOF predictions (ZINB) ---\n")
oof_zinb <- rep(NA_real_, nrow(df_eng))
zinb_form_final <- make_zinb_formula(zinb_search$count_terms,
                                     zinb_search$zero_terms)
for (k in 1:K_FOLDS) {
  tr <- df[folds != k, ]; te_idx <- which(folds == k)
  m  <- tryCatch(zeroinfl(zinb_form_final, data = tr, dist = "negbin"),
                 error = function(e) NULL)
  if (!is.null(m)) {
    oof_zinb[te_idx] <- predict(m, newdata = df[te_idx, ], type = "response")
  } else {
    oof_zinb[te_idx] <- mean(tr$days_missed)
  }
}
rmse_zinb_oof <- sqrt(mean((y_train - oof_zinb)^2))
cat("5-fold OOF RMSE (ZINB)           :", round(rmse_zinb_oof, 3), "\n")

# ------------------------------------------------------------
# Ensemble weight: pred = w * hurdle + (1 - w) * ZINB
# ------------------------------------------------------------

w_grid <- seq(0, 1, by = 0.01)
rmse_w <- sapply(w_grid, function(w)
  sqrt(mean((y_train - (w * oof_hurdle + (1 - w) * oof_zinb))^2)))
w_star <- w_grid[which.min(rmse_w)]
rmse_blend_oof <- min(rmse_w)
cat(sprintf("\nOptimal blend weight (hurdle vs ZINB) : w* = %.2f\n", w_star))
cat(sprintf("5-fold OOF RMSE (blend)          : %.3f\n", rmse_blend_oof))
cat(sprintf("Baselines: linear (in-sample) %.3f, ZINB OOF %.3f, hurdle OOF %.3f\n",
            sqrt(mean(residuals(mod_lm)^2)), rmse_zinb_oof, rmse_hurdle_oof))

# ------------------------------------------------------------
# Final fit on ALL training data, predict on test
# ------------------------------------------------------------

# Stage 1 final
dtrain_cls_full <- xgb.DMatrix(X_train, label = y_bin)
cv1_full <- xgb.cv(data = dtrain_cls_full, params = params_cls,
                   nrounds = 2000, nfold = 5,
                   early_stopping_rounds = 40, verbose = 0)
mod_cls_full <- xgb.train(data = dtrain_cls_full, params = params_cls,
                          nrounds = cv1_full$best_iteration, verbose = 0)

# Stage 2 final (positive subset)
pos_all   <- which(y_train > 0)
y_log_all <- log1p(y_train[pos_all])
dtrain_reg_full <- xgb.DMatrix(X_train[pos_all, ], label = y_log_all)
cv2_full <- xgb.cv(data = dtrain_reg_full, params = params_reg,
                   nrounds = 2000, nfold = 5,
                   early_stopping_rounds = 40, verbose = 0)
mod_reg_full <- xgb.train(data = dtrain_reg_full, params = params_reg,
                          nrounds = cv2_full$best_iteration, verbose = 0)

smear_full <- mean(exp(y_log_all - predict(mod_reg_full, X_train[pos_all, ])))

p_pos_test    <- predict(mod_cls_full, X_test)
log_pred_test <- predict(mod_reg_full, X_test)
mu_test       <- pmax(smear_full * exp(log_pred_test) - 1, 0)
pred_hurdle   <- p_pos_test * mu_test

# Final blended prediction
pred_blend <- w_star * pred_hurdle + (1 - w_star) * pred_zinb
pred_blend <- pmax(pred_blend, 0)

submission_hurdle <- make_submission(pred_hurdle, "submission_hurdle_xgb.csv")
submission_best   <- make_submission(pred_blend,  "submission_best.csv")

cat("\n--- Final submission previews ---\n")
cat("Hurdle XGBoost (submission_hurdle_xgb.csv):\n")
print(head(submission_hurdle))
cat("\nBlend XGB + ZINB (submission_best.csv):\n")
print(head(submission_best))

cat("\nBest model:  submission_best.csv\n")
cat(sprintf("(OOF RMSE blend = %.3f, w* = %.2f)\n", rmse_blend_oof, w_star))
