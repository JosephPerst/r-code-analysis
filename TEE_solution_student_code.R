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
