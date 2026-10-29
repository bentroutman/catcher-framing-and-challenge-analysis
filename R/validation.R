# Catcher Framing & Automated Challenge System Analysis: Model Validation

# Checks how the strike probability GAMs and the challenge XGBoost model do on
# games they were not trained on, and how much the framing results move.
# Input files (mlb_data_prompt.csv, milb_data_prompt.csv, re288_2023_prompt.csv) are not included in this repository

library(dplyr)
library(tidyr)
library(mgcv)
library(xgboost)

# Runs the main analysis first so the data and fitted models exist
if (!exists("adjusted_mlb_data_summary")) {
  source("catcher_framing_analysis.R")
}

set.seed(0)
num_folds <- 5

# Rank-based AUC
auc <- function(y, p) {
  ranks <- rank(p)
  n1 <- as.numeric(sum(y == 1))
  n0 <- as.numeric(sum(y == 0))
  (sum(ranks[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

# AUC, log loss, Brier score, and accuracy for one set of predictions
evaluate <- function(y, p, name) {
  p_clipped <- pmin(pmax(p, 1e-6), 1 - 1e-6)
  data.frame(Model = name,
    AUC = auc(y, p),
    LogLoss = -mean(y * log(p_clipped) + (1 - y) * log(1 - p_clipped)),
    Brier = mean((y - p)^2),
    Accuracy = mean((p > 0.5) == y))
}

# ----------------------------------------------------------------------
# Part 1: Strike probability GAMs, 5-fold cross validation by game
# ----------------------------------------------------------------------

# Each game goes to one fold, so no game is in both training and test data
cv_data <- mlb_data |>
  filter(!is.na(BatterSide), !is.na(z_center), !is.na(RelSpeed),
    !is.na(InducedVertBreak), !is.na(HorzBreak), !is.na(PitcherThrows))
games <- unique(cv_data$game_pk)
game_fold <- sample(rep(1:num_folds, length.out = length(games)))
cv_data$fold <- game_fold[match(cv_data$game_pk, games)]
cv_data$HeldOutProb <- NA
cv_data$BaselineProb <- NA

# Fit on 4 folds and test on the one left out, separately for each batter side
for (k in 1:num_folds) {
  for (side in c("Left", "Right")) {
    train <- cv_data |> filter(fold != k, BatterSide == side)
    test_rows <- which(cv_data$fold == k & cv_data$BatterSide == side)
    gam_fit <- gam(
      strike ~ s(sz_plate_x, z_center, k = 55) + PitcherThrows + count +
        RelSpeed + InducedVertBreak + HorzBreak + Runners + ScoreDiff + outs,
      family = binomial, data = train, method = "REML")
    baseline_fit <- glm(strike ~ count, family = binomial, data = train)
    cv_data$HeldOutProb[test_rows] <- predict(gam_fit, cv_data[test_rows, ],
      type = "response")
    cv_data$BaselineProb[test_rows] <- predict(baseline_fit,
      cv_data[test_rows, ], type = "response")
  }
}

# In-sample predictions come from the full-data GAMs in the main script
in_sample <- cv_data |> filter(!is.na(StrikeProb))
strike <- as.integer(cv_data$strike)
gam_results <- bind_rows(
  evaluate(as.integer(in_sample$strike), in_sample$StrikeProb,
    "GAM in-sample"),
  evaluate(strike, cv_data$HeldOutProb, "GAM held-out games"),
  evaluate(strike, cv_data$BaselineProb, "Count-only model held-out games"))
print(gam_results)

# Predicted vs actual strike rate by decile of held-out probability
calibration <- cv_data |>
  mutate(Decile = ntile(HeldOutProb, 10)) |>
  group_by(Decile) |>
  summarize(Pitches = n(),
    MeanPredicted = mean(HeldOutProb),
    ActualRate = mean(strike))
print(calibration)

# ----------------------------------------------------------------------
# Part 2: Is framing value stable? Compare odd and even games
# ----------------------------------------------------------------------

# Framing value rebuilt from held-out predictions, split by game number
cv_data <- cv_data |>
  mutate(HeldOutResidual = strike - HeldOutProb,
    Half = ifelse(game_pk %% 2 == 0, "Even", "Odd"))
split_half <- cv_data |>
  group_by(catcher_id, Half) |>
  summarize(n_pitches = n(),
    RunsPer7500 = sum(HeldOutResidual) * 0.125 / n_pitches * 7500,
    .groups = "drop") |>
  pivot_wider(names_from = Half, values_from = c(n_pitches, RunsPer7500)) |>
  filter(n_pitches_Even >= 1500, n_pitches_Odd >= 1500)
split_half_cor <- cor(split_half$RunsPer7500_Even, split_half$RunsPer7500_Odd)
cat("Catchers:", nrow(split_half), "\n")
cat("Odd vs even game correlation:", split_half_cor, "\n")
# Spearman-Brown correction for the full sample
cat("Reliability of full sample:", 2 * split_half_cor / (1 + split_half_cor),
  "\n")

# ----------------------------------------------------------------------
# Part 3: Challenge model, 5-fold cross validation by game on Triple-A data
# ----------------------------------------------------------------------

params <- list(objective = "binary:logistic", eval_metric = "auc",
  max_depth = 6, eta = 0.1, subsample = 0.8, colsample_bytree = 0.8)

# Same features as the main script
challenge_data <- milb_data |> filter(ChallengesLeft > 0)
challenge_features <- challenge_data |>
  select(sz_plate_x, z_center, RelSpeed, InducedVertBreak, HorzBreak,
    Runners, ScoreDiff, outs, inning, ChallengesLeft, DeltaRE,
    PitcherThrows, count, TrueStrike, CalledStrikeInitially) |>
  mutate(PitcherThrows = as.numeric(factor(PitcherThrows)),
    count = as.numeric(factor(count)),
    across(everything(), as.numeric))
challenge_label <- challenge_data$challenged
challenge_games <- unique(challenge_data$game_pk)
challenge_game_fold <- sample(rep(1:num_folds, length.out =
    length(challenge_games)))
challenge_fold <- challenge_game_fold[match(challenge_data$game_pk,
  challenge_games)]
challenge_held_out <- rep(NA, nrow(challenge_data))
for (k in 1:num_folds) {
  dtrain <- xgb.DMatrix(as.matrix(challenge_features[challenge_fold != k, ]),
    label = challenge_label[challenge_fold != k])
  fit <- xgb.train(params, dtrain, nrounds = 200)
  challenge_held_out[challenge_fold == k] <- predict(fit,
    as.matrix(challenge_features[challenge_fold == k, ]))
}
challenge_in_sample <- predict(xgb_model, as.matrix(challenge_features))

# Base rate row predicts the overall rate for every pitch
challenge_results <- bind_rows(
  evaluate(challenge_label, challenge_in_sample, "Challenge in-sample"),
  evaluate(challenge_label, challenge_held_out, "Challenge held-out games"),
  evaluate(challenge_label, rep(mean(challenge_label),
      length(challenge_label)), "Challenge base rate"))
print(challenge_results)

# A challenge can only succeed if the call did not match the true zone. This
# is how often that rule matches the Triple-A outcome
rule_check <- milb_data |> filter(challenged == 1, !is.na(Error))
cat("Challenges:", nrow(rule_check), "\n")
cat("Rule matches outcome:", mean(rule_check$challenge_successful ==
    rule_check$Error), "\n")

# Overturn rate per pitch in the Triple-A data, actual vs predicted
challenge_wrong_call <- !is.na(challenge_data$Error) &
  challenge_data$Error == 1
predicted_overturn <- challenge_held_out * challenge_wrong_call
actual_overturn <- challenge_data$challenged == 1 &
  challenge_data$challenge_successful == 1
cat("Actual overturn rate per pitch:", mean(actual_overturn), "\n")
cat("Predicted overturn rate per pitch:", mean(predicted_overturn), "\n")

# ----------------------------------------------------------------------
# Part 4: How much does the change in framing value depend on the assumptions?
# ----------------------------------------------------------------------

# Percent change in framing value for the 10 best and 10 worst catchers
# (at least 3,000 taken pitches, residual-based, before the challenge system)
catchers <- adjusted_mlb_data_summary |> filter(n_pitches >= 3000)
best <- catchers |> arrange(desc(WithoutChallengesPer7500)) |> head(10)
worst <- catchers |> arrange(WithoutChallengesPer7500) |> head(10)
cat("Best 10 catchers:",
  100 * (mean(best$RunsPer7500) / mean(best$WithoutChallengesPer7500) - 1),
  "% change\n")
cat("Worst 10 catchers:",
  100 * (mean(worst$RunsPer7500) / mean(worst$WithoutChallengesPer7500) - 1),
  "% change\n")

# Spread across catchers before and after the challenge system
cat("SD of runs per 7,500 before:", sd(catchers$WithoutChallengesPer7500),
  "after:", sd(catchers$RunsPer7500), "\n")

# The main script assumes the top 10% of challenge probability gets challenged
# and that challenges left is a sample from Triple-A pitches in the same
# inning. XGBoost also uses random subsamples, so the challenge model changes a
# little with the seed. Here I repeat the simulation with different thresholds
# and seeds, refitting the challenge model each time.
sensitivity <- data.frame(Threshold = numeric(), Seed = integer(),
  BestPctChange = numeric(), WorstPctChange = numeric())
for (threshold in c(0.8, 0.9, 0.95)) {
  for (seed in 1:5) {
    set.seed(seed)
    sim_model <- xgb.train(params, xgb.DMatrix(as.matrix(challenge_features),
        label = challenge_label), nrounds = 200)
    milb_challenge_prob <- rep(0, nrow(milb_data))
    milb_challenge_prob[milb_data$ChallengesLeft > 0] <- predict(sim_model,
      as.matrix(challenge_features))
    sim <- mlb_data
    set.seed(seed)
    sim$ChallengesLeft <- NA
    for (i in unique(sim$inning)) {
      pool <- milb_data$ChallengesLeft[
        milb_data$inning == min(i, max(milb_data$inning))]
      rows <- sim$inning == i
      sim$ChallengesLeft[rows] <- sample(pool, sum(rows), replace = TRUE)
    }
    sim$ChallengeProb <- 0
    has_challenges <- sim$ChallengesLeft > 0
    sim_features <- sim |>
      filter(has_challenges) |>
      select(sz_plate_x, z_center, RelSpeed, InducedVertBreak, HorzBreak,
        Runners, ScoreDiff, outs, inning, ChallengesLeft, DeltaRE,
        PitcherThrows, count, TrueStrike, CalledStrikeInitially = strike) |>
      mutate(PitcherThrows = as.numeric(factor(PitcherThrows)),
        count = as.numeric(factor(count)))
    sim$ChallengeProb[has_challenges] <- predict(sim_model,
      as.matrix(sim_features))
    sim$Challengeable <- sim$ChallengeProb >
      quantile(milb_challenge_prob, threshold)
    sim_wrong_call <- !is.na(sim$Error) & sim$Error == 1
    sim$ChallengeSuccessProb <- 0
    sim$ChallengeSuccessProb[sim$Challengeable & sim_wrong_call] <- 1
    sim_summary <- sim |>
      filter(!is.na(ExtraRuns)) |>
      mutate(AdjustedRuns = ExtraRuns *
          (1 - ChallengeSuccessProb * ChallengeProb)) |>
      group_by(catcher_id) |>
      summarize(n_pitches = n(),
        RunsPer7500 = sum(AdjustedRuns) / n_pitches * 7500,
        WithoutChallengesPer7500 = sum(ExtraRuns) / n_pitches * 7500) |>
      filter(n_pitches >= 3000)
    sim_best <- sim_summary |> arrange(desc(WithoutChallengesPer7500)) |>
      head(10)
    sim_worst <- sim_summary |> arrange(WithoutChallengesPer7500) |> head(10)
    row <- nrow(sensitivity) + 1
    sensitivity[row, "Threshold"] <- threshold
    sensitivity[row, "Seed"] <- seed
    sensitivity[row, "BestPctChange"] <- 100 *
      (mean(sim_best$RunsPer7500) / mean(sim_best$WithoutChallengesPer7500) - 1)
    sensitivity[row, "WorstPctChange"] <- 100 *
      (mean(sim_worst$RunsPer7500) /
          mean(sim_worst$WithoutChallengesPer7500) - 1)
  }
}
print(sensitivity)
sensitivity |>
  group_by(Threshold) |>
  summarize(BestMin = min(BestPctChange), BestMax = max(BestPctChange),
    WorstMin = min(WorstPctChange), WorstMax = max(WorstPctChange)) |>
  print()
cat("Mean over all runs, best 10:", mean(sensitivity$BestPctChange),
  "worst 10:", mean(sensitivity$WorstPctChange), "\n")

# Bootstrap whole games to see how much the result depends on which games are
# in the sample. The challenge models are not refit, so this only covers the
# sampling of MLB games
set.seed(0)
mlb_games <- unique(mlb_data$game_pk)
game_rows <- split(seq_len(nrow(mlb_data)), mlb_data$game_pk)
boot <- data.frame(BestPctChange = numeric(200), WorstPctChange = numeric(200))
for (b in 1:200) {
  sampled_games <- sample(mlb_games, replace = TRUE)
  boot_data <- mlb_data[unlist(game_rows[as.character(sampled_games)]), ]
  boot_summary <- boot_data |>
    filter(!is.na(AdjustedRuns)) |>
    group_by(catcher_id) |>
    summarize(n_pitches = n(),
      RunsPer7500 = sum(AdjustedRuns) / n_pitches * 7500,
      WithoutChallengesPer7500 = sum(ExtraRuns) / n_pitches * 7500) |>
    filter(n_pitches >= 3000)
  boot_best <- boot_summary |> arrange(desc(WithoutChallengesPer7500)) |>
    head(10)
  boot_worst <- boot_summary |> arrange(WithoutChallengesPer7500) |> head(10)
  boot$BestPctChange[b] <- 100 *
    (mean(boot_best$RunsPer7500) / mean(boot_best$WithoutChallengesPer7500) - 1)
  boot$WorstPctChange[b] <- 100 *
    (mean(boot_worst$RunsPer7500) /
        mean(boot_worst$WithoutChallengesPer7500) - 1)
}
cat("95% interval, best 10 catchers:",
  quantile(boot$BestPctChange, c(0.025, 0.975)), "\n")
cat("95% interval, worst 10 catchers:",
  quantile(boot$WorstPctChange, c(0.025, 0.975)), "\n")
