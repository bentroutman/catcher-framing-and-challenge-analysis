# Catcher Framing & Automated Challenge System Analysis

# Input files (mlb_data_prompt.csv, milb_data_prompt.csv, re288_2023_prompt.csv) are not included in this repository

library(dplyr)
library(tidyr)
library(splines)
library(mgcv)
library(ggplot2)
library(xgboost)
library(Matrix)

# ----------------------------------------------------------------------
# Part 1: Framing value under current rules (residual-based and context-specific)
# ----------------------------------------------------------------------

# Load MLB Data, prepare for strike probability model
mlb_data <- read.csv("mlb_data_prompt.csv") |>
  mutate(strike = event %in% c("called_strike", "strikeout"),
    z_center = (sz_plate_z - (sz_top - sz_bottom)) /
      (sz_top - sz_bottom) - 0.5,
    count = factor(paste(balls, strikes, sep = "-")),
    BatterSide = factor(BatterSide),
    PitcherThrows = factor(PitcherThrows),
    Runners = runner_on_1 + runner_on_2 + runner_on_3,
    ScoreDiff = abs(away_score - home_score))

# Split batters into left and right handed for GAM
lhh_index <- !is.na(mlb_data$BatterSide) & mlb_data$BatterSide == "Left"
rhh_index <- !is.na(mlb_data$BatterSide) & mlb_data$BatterSide == "Right"

# Predict strike probability
gam_L <- gam(
  strike ~ s(sz_plate_x, z_center, k = 55) + PitcherThrows + count +
    RelSpeed + InducedVertBreak + HorzBreak + Runners + ScoreDiff + outs,
  family = binomial,
  data = subset(mlb_data, BatterSide == "Left"),
  method = "REML")
gam_R <- gam(
  strike ~ s(sz_plate_x, z_center, k = 55) + PitcherThrows + count +
    RelSpeed + InducedVertBreak + HorzBreak + Runners + ScoreDiff + outs,
  family = binomial,
  data = subset(mlb_data, BatterSide == "Right"),
  method = "REML")
mlb_data$StrikeProb[lhh_index] <- predict(
  gam_L,
  newdata = mlb_data[lhh_index, ],
  type = "response")
mlb_data$StrikeProb[rhh_index] <- predict(
  gam_R,
  newdata = mlb_data[rhh_index, ],
  type = "response")

# Calculate extra strikes and runs
mlb_data <- mlb_data |>
  mutate(Residual = strike - StrikeProb,
    ExtraRuns = Residual * 0.125)
mlb_data_summary <- mlb_data |>
  drop_na(StrikeProb) |>
  group_by(catcher_id) |>
  summarize(n_pitches = n(),
    ExtraStrikes = sum(Residual),
    ExtraStrikesPerPitch = ExtraStrikes / n_pitches,
    ExtraRuns = ExtraStrikes * 0.125,
    RunsPer7500 = ExtraRuns / n_pitches * 7500)

# Function for graphing distributions
plot_histogram <- function(data, title, subtitle) {
  ggplot(data, aes(x = RunsPer7500)) +
    geom_histogram(aes(y = after_stat(density)), bins = 15, fill = "gray70",
    color = "black") +
    geom_density(linewidth = 1, color = "blue") +
    geom_vline(xintercept = mean(data$RunsPer7500), linetype = "dashed",
    color = "red") +
    labs(title = title, subtitle = subtitle,
    x = "Runs per 7,500 Taken Pitches", y = "Density") +
    theme_minimal()
}

# Plot Residual-based data
plot_histogram(data.frame(RunsPer7500 =
      mlb_data_summary$RunsPer7500[mlb_data_summary$n_pitches >= 3000]),
  "Distribution of Catcher Framing Value (MLB)",
  "Catchers with at least 3,000 taken pitches"
)

# Display summary table
mlb_data_summary |>
  filter(n_pitches >= 3000) |>
  summarize(Count = n(),
    Mean = mean(RunsPer7500),
    Median = median(RunsPer7500),
    SD = sd(RunsPer7500),
    Min = min(RunsPer7500),
    Max = max(RunsPer7500))

# Load Run Expectancy Data

# I changed the formatting of the data in Excel to prevent the count from converting to a date
run_expectancy_data <- read.csv("re288_2023_prompt.csv")

# Determine "true" strikes using coordinates
mlb_data <- mlb_data |>
  mutate(TrueStrike = ifelse(sz_plate_x >= -0.83 & sz_plate_x <= 0.83 &
        sz_plate_z + 0.14 >= sz_bottom & sz_plate_z - 0.14 <= sz_top, 1, 0))

# Calculate run expectancy changes
mlb_data <- mlb_data |>
  filter(balls <= 3) |> # Remove errant entry with 4 balls
  mutate(Error = as.integer(strike != TrueStrike),
    BallsAfterStrike = ifelse(strikes != 2, balls, 0),
    StrikesAfterStrike = ifelse(strikes != 2, strikes + 1, 0),
    OutsAfterStrike = ifelse(strikes == 2, outs + 1, outs),
    BallsAfterBall = ifelse(balls != 3, balls + 1, 0),
    StrikesAfterBall = ifelse(balls != 3, strikes, 0),
    RunsAfterBall = ifelse(balls == 3 & runner_on_1 & runner_on_2 &
        runner_on_3, 1, 0),
    Runner3AfterBall = ifelse(balls == 3 & runner_on_2 & runner_on_1,
      1, runner_on_3),
    Runner2AfterBall = ifelse(balls == 3 & runner_on_1, 1, runner_on_2),
    Runner1AfterBall = ifelse(balls == 3, 1, runner_on_1)) |>
  left_join(run_expectancy_data, by = c(
      "runner_on_1", "runner_on_2", "runner_on_3", "OutsAfterStrike" = "outs",
      "BallsAfterStrike" = "balls", "StrikesAfterStrike" = "strikes")) |>
  rename(RE_AfterStrike = RE) |>
  left_join(run_expectancy_data, by = c(
      "Runner1AfterBall" = "runner_on_1", "Runner2AfterBall" = "runner_on_2",
      "Runner3AfterBall" = "runner_on_3", "outs",
      "BallsAfterBall" = "balls", "StrikesAfterBall" = "strikes")) |>
  mutate(RE = ifelse(RunsAfterBall, RE + 1, RE),
    RE_AfterStrike = ifelse(OutsAfterStrike == 3, 0, RE_AfterStrike)) |>
  rename(RE_AfterBall = RE) |>
  mutate(DeltaRE = RE_AfterBall - RE_AfterStrike,
    ContextSpecificRuns = Residual * DeltaRE)
mlb_data_summary_contextualized <- mlb_data |>
  drop_na(StrikeProb) |>
  group_by(catcher_id) |>
  summarize(n_pitches = n(),
    ContextSpecificRuns = sum(ContextSpecificRuns),
    RunsPer7500 = ContextSpecificRuns / n_pitches * 7500)

# Plot Context-Specific Data
plot_histogram(data.frame(RunsPer7500 =
      mlb_data_summary_contextualized$RunsPer7500[
      mlb_data_summary_contextualized$n_pitches >= 3000]),
  "Context-Specific Distribution of Catcher Framing Value (MLB)",
  "Catchers with at least 3,000 taken pitches"
)
mlb_data_summary_contextualized |>
  filter(n_pitches >= 3000) |>
  summarize(Count = n(),
    Mean = mean(RunsPer7500),
    Median = median(RunsPer7500),
    SD = sd(RunsPer7500),
    Min = min(RunsPer7500),
    Max = max(RunsPer7500))

# ----------------------------------------------------------------------
# Part 2: Framing value with a 2-challenge ABS system (models trained on Triple-A data)
# ----------------------------------------------------------------------

# Load MiLB Data
milb_data <- read.csv("milb_data_prompt.csv")

# Prepare data for strike probability model
milb_data <- milb_data |>
  filter(balls <= 3, strikes <= 2) |> # Remove errant entries
  mutate(strike = event %in% c("called_strike", "strikeout"),
    z_center = (sz_plate_z - (sz_top - sz_bottom)) /
      (sz_top - sz_bottom) - 0.5,
    count = factor(paste(balls, strikes, sep = "-")),
    BatterSide = factor(BatterSide),
    PitcherThrows = factor(PitcherThrows),
    Runners = runner_on_1 + runner_on_2 + runner_on_3,
    ScoreDiff = abs(away_score - home_score),
    challenge_successful = replace_na(challenge_successful, 0),
    CalledStrikeInitially = ifelse((strike & !challenge_successful) |
        (!strike & challenge_successful), TRUE, FALSE))
gam_L_milb <- gam(
  strike ~ s(sz_plate_x, z_center, k = 55) + PitcherThrows + count +
    RelSpeed + InducedVertBreak + HorzBreak + Runners + ScoreDiff + outs,
  family = binomial,
  data = subset(milb_data, BatterSide == "Left"),
  method = "REML")
gam_R_milb <- gam(
  strike ~ s(sz_plate_x, z_center, k = 55) + PitcherThrows + count +
    RelSpeed + InducedVertBreak + HorzBreak + Runners + ScoreDiff + outs,
  family = binomial,
  data = subset(milb_data, BatterSide == "Right"),
  method = "REML")
milb_lhh_index <- !is.na(milb_data$BatterSide) & milb_data$BatterSide == "Left"
milb_rhh_index <- !is.na(milb_data$BatterSide) &
  milb_data$BatterSide == "Right"
milb_data$StrikeProb[milb_lhh_index] <- predict(
  gam_L_milb,
  newdata = milb_data[milb_lhh_index, ],
  type = "response")
milb_data$StrikeProb[milb_rhh_index] <- predict(
  gam_R_milb,
  newdata = milb_data[milb_rhh_index, ],
  type = "response")
milb_data$Residual <- milb_data$CalledStrikeInitially - milb_data$StrikeProb
milb_data$TrueStrike <- ifelse(milb_data$sz_plate_x >= -0.83 &
    milb_data$sz_plate_x <= 0.83 & milb_data$sz_plate_z + 0.14 >=
    milb_data$sz_bottom & milb_data$sz_plate_z - 0.14 <= milb_data$sz_top, 1, 0)

# Calculate RE changes
milb_data <- milb_data |>
  drop_na(z_center, BatterSide, RelSpeed) |>
  mutate(Error = as.integer(CalledStrikeInitially != TrueStrike),
    BallsAfterStrike = ifelse(strikes != 2, balls, 0),
    StrikesAfterStrike = ifelse(strikes != 2, strikes + 1, 0),
    OutsAfterStrike = ifelse(strikes == 2, outs + 1, outs),
    BallsAfterBall = ifelse(balls != 3, balls + 1, 0),
    StrikesAfterBall = ifelse(balls != 3, strikes, 0),
    RunsAfterBall = ifelse(balls == 3 & runner_on_1 & runner_on_2 &
        runner_on_3, 1, 0),
    Runner3AfterBall = ifelse(balls == 3 & runner_on_2 & runner_on_1,
      1, runner_on_3),
    Runner2AfterBall = ifelse(balls == 3 & runner_on_1, 1, runner_on_2),
    Runner1AfterBall = ifelse(balls == 3, 1, runner_on_1)) |>
  left_join(run_expectancy_data, by = c(
      "runner_on_1", "runner_on_2", "runner_on_3", "OutsAfterStrike" = "outs",
      "BallsAfterStrike" = "balls", "StrikesAfterStrike" = "strikes")) |>
  rename(RE_AfterStrike = RE) |>
  left_join(run_expectancy_data, by = c(
      "Runner1AfterBall" = "runner_on_1", "Runner2AfterBall" = "runner_on_2",
      "Runner3AfterBall" = "runner_on_3", "outs",
      "BallsAfterBall" = "balls", "StrikesAfterBall" = "strikes")) |>
  mutate(RE = ifelse(RunsAfterBall, RE + 1, RE),
    RE_AfterStrike = ifelse(OutsAfterStrike == 3, 0, RE_AfterStrike)) |>
  rename(RE_AfterBall = RE) |>
  mutate(DeltaRE = RE_AfterBall - RE_AfterStrike)

# Calculate number of challenges left
milb_data <- milb_data |>
  group_by(game_pk) |>
  fill(challenge_teamid, .direction = "down") |>
  fill(challenge_teamid, .direction = "up") |>
  mutate(FailedChallenge =
      ifelse(challenged == 1 & challenge_successful == 0, 1, 0),
    ChallengesUsedAfter = lag(cumsum(FailedChallenge), default = 0),
    ChallengesLeft = pmax(2 - ChallengesUsedAfter, 0)) |>
  ungroup()

# There are several instances in the data set where teams shouldn't be able to
# challenge, but still used one. I assume these are games where 3 challenges are
# allowed instead of 2.

# Challenge probability model
data_filtered <- milb_data |> filter(ChallengesLeft > 0)
features <- data_filtered |>
  select(sz_plate_x, z_center, RelSpeed, InducedVertBreak, HorzBreak,
    Runners, ScoreDiff, outs, inning, ChallengesLeft, DeltaRE,
    PitcherThrows, count, TrueStrike, CalledStrikeInitially)
features <- features |>
  mutate(PitcherThrows = as.numeric(factor(PitcherThrows)),
    count = as.numeric(factor(count)),
    across(everything(), as.numeric))
label <- data_filtered$challenged
dtrain <- xgb.DMatrix(data = as.matrix(features), label = label)
params <- list(objective = "binary:logistic", eval_metric = "auc",
  max_depth = 6, eta = 0.1, subsample = 0.8,
  colsample_bytree = 0.8)
xgb_model <- xgb.train(params = params, data = dtrain, nrounds = 200,
  verbose = 1)

# Predict MiLB challenge probability (used for determining challengable opportunities in MLB)
features <- as.matrix(features)
preds <- as.numeric(predict(xgb_model, features))
milb_data$ChallengeProb[milb_data$ChallengesLeft > 0] <- preds
milb_data[milb_data$ChallengesLeft == 0, ]$ChallengeProb <- 0

# Challenge success model
data_success <- milb_data |> filter(challenged == 1)
features <- data_success |>
  select(sz_plate_x, z_center, RelSpeed, InducedVertBreak, HorzBreak,
    Runners, ScoreDiff, outs, inning, Error, count, PitcherThrows,
    CalledStrikeInitially, TrueStrike) |>
  mutate(PitcherThrows = as.numeric(factor(PitcherThrows)),
    count = as.numeric(factor(count)))
label <- data_success$challenge_successful
dtrain <- xgb.DMatrix(as.matrix(features), label = label)
params <- list(objective = "binary:logistic", eval_metric = "auc",
  max_depth = 6, eta = 0.1, subsample = 0.8,
  colsample_bytree = 0.8)
xgb_success <- xgb.train(params, data = dtrain, nrounds = 200)

# Created a random sample of challenges left, as there is currently no value for it.
# Over a large sample, this is representative.
set.seed(1)
mlb_data$ChallengesLeft <- sample(
  milb_data$ChallengesLeft,
  size = nrow(mlb_data),
  replace = TRUE)

# Predict challenge use for MLB using AAA data
features <- mlb_data |>
  filter(ChallengesLeft > 0) |>
  select(sz_plate_x, z_center, RelSpeed, InducedVertBreak, HorzBreak, Runners,
    ScoreDiff, outs, inning, ChallengesLeft, DeltaRE,
    PitcherThrows, count, TrueStrike, CalledStrikeInitially = strike)
features <- features |>
  mutate(PitcherThrows = as.numeric(factor(PitcherThrows)),
    count = as.numeric(factor(count)))
mlb_data$ChallengeProb[mlb_data$ChallengesLeft > 0] <-
  as.numeric(predict(xgb_model, as.matrix(features)))
mlb_data$ChallengeProb[mlb_data$ChallengesLeft == 0] <- 0

# Predict challenge probability, limiting to realistic situations
mlb_data <- mlb_data |>
  mutate(Challengeable =
      ChallengeProb > quantile(milb_data$ChallengeProb, 0.9))

# Predict challenge effectiveness
features_all <- mlb_data |>
  select(sz_plate_x, z_center, RelSpeed, InducedVertBreak, HorzBreak,
    Runners, ScoreDiff, outs, inning, Error, count, PitcherThrows,
    CalledStrikeInitially = strike, TrueStrike) |>
  mutate(PitcherThrows = as.numeric(factor(PitcherThrows)),
    count = as.numeric(factor(count)))
mlb_data$ChallengeSuccessProb <- 0
mlb_data$ChallengeSuccessProb[mlb_data$Challengeable] <-
  predict(xgb_success,
    as.matrix(features_all[mlb_data$Challengeable, ]))

# Summarize data
mlb_data <- mlb_data |>
  mutate(OverturnProb = ChallengeSuccessProb * ChallengeProb,
    AdjustedRuns = ExtraRuns * (1 - OverturnProb),
    RunChange = ExtraRuns - AdjustedRuns)
mlb_data_contextualized <- mlb_data |>
  filter(!is.na(ContextSpecificRuns)) |>
  mutate(OverturnProb = ChallengeSuccessProb * ChallengeProb,
    AdjustedRuns = ContextSpecificRuns * (1 - OverturnProb),
    RunChange = ContextSpecificRuns - AdjustedRuns)
adjusted_mlb_data_summary <- mlb_data |>
  filter(!is.na(AdjustedRuns)) |>
  group_by(catcher_id) |>
  summarize(n_pitches = n(),
    AdjustedRuns = sum(AdjustedRuns),
    RunsPer7500 = AdjustedRuns / n_pitches * 7500,
    RunsWithoutChallenges = sum(ExtraRuns),
    WithoutChallengesPer7500 = RunsWithoutChallenges /
      n_pitches * 7500,
    RunChange = sum(RunChange),
    RunChangePer7500 = RunsPer7500 - WithoutChallengesPer7500)
adjusted_mlb_data_summary_contextualized <- mlb_data_contextualized |>
  group_by(catcher_id) |>
  summarize(n_pitches = n(),
    AdjustedRuns = sum(AdjustedRuns),
    RunsPer7500 = AdjustedRuns / n_pitches * 7500,
    RunsWithoutChallenges = sum(ContextSpecificRuns),
    WithoutChallengesPer7500 = RunsWithoutChallenges /
      n_pitches * 7500,
    RunChange = sum(RunChange),
    RunChangePer7500 = RunsPer7500 - WithoutChallengesPer7500)

# Plot Challenge-Adjusted Residual-Based Data
plot_histogram(data.frame(RunsPer7500 =
      adjusted_mlb_data_summary$RunsPer7500[
      adjusted_mlb_data_summary$n_pitches >= 3000]),
  "Challenge-Adjusted Distribution of Catcher Framing (MLB)",
  "Catchers with at least 3,000 taken pitches"
)
adjusted_mlb_data_summary |>
  filter(n_pitches >= 3000) |>
  summarize(Count = n(),
    Mean = mean(RunsPer7500),
    Median = median(RunsPer7500),
    SD = sd(RunsPer7500),
    Min = min(RunsPer7500),
    Max = max(RunsPer7500))

# Plot Challenge-Adjusted Context-Specific Data
plot_histogram(data.frame(RunsPer7500 =
      adjusted_mlb_data_summary_contextualized$RunsPer7500[
      adjusted_mlb_data_summary_contextualized$n_pitches >= 3000]),
  "Context-Specific Challenge-Adjusted Distribution of Framing Value (MLB)",
  "Catchers with at least 3,000 taken pitches"
)
adjusted_mlb_data_summary_contextualized |>
  filter(n_pitches >= 3000) |>
  summarize(Count = n(),
    Mean = mean(RunsPer7500),
    Median = median(RunsPer7500),
    SD = sd(RunsPer7500),
    Min = min(RunsPer7500),
    Max = max(RunsPer7500))

# ----------------------------------------------------------------------
# Part 3: Catchers who gain or lose the most under the challenge system
# ----------------------------------------------------------------------

# Residual-based biggest fallers
adjusted_mlb_data_summary |> arrange(RunChangePer7500) |>
  filter(n_pitches >= 3000) |> head(10) |> select(catcher_id, RunsPer7500,
    WithoutChallengesPer7500, RunChangePer7500)

# Residual-based biggest risers
adjusted_mlb_data_summary |> arrange(desc(RunChangePer7500)) |>
  filter(n_pitches >= 3000) |> head(10) |> select(catcher_id, RunsPer7500,
    WithoutChallengesPer7500, RunChangePer7500)

# Context-specific biggest fallers
adjusted_mlb_data_summary_contextualized |> arrange(RunChangePer7500) |>
  filter(n_pitches >= 3000) |> head(10) |> select(catcher_id, RunsPer7500,
    WithoutChallengesPer7500, RunChangePer7500)

# Context-specific biggest risers
adjusted_mlb_data_summary_contextualized |> arrange(desc(RunChangePer7500)) |>
  filter(n_pitches >= 3000) |> head(10) |> select(catcher_id, RunsPer7500,
    WithoutChallengesPer7500, RunChangePer7500)
