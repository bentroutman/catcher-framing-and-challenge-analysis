# Catcher Framing & Automated Challenge System Analysis

This project models catcher framing value in Major League Baseball and evaluates how the implementation of a ball–strike challenge system would affect framing value across the league.

The analysis combines pitch-level strike probability modeling with machine learning–based estimates of challenge usage and success, using both MLB and Triple-A data.

---

## Overview

The project is structured around three main components:

1. **Strike Probability Modeling**
   - Estimated pitch-level strike probabilities using Generalized Additive Models (GAMs)
   - Models are split by batter handedness to capture asymmetric strike zones
   - No catcher-specific information is included, producing a league-average baseline

2. **Framing Value Estimation**
   - Framing value calculated using residuals:
     ```
     (Observed strike – Expected strike probability)
     ```
   - Residuals are aggregated by catcher to estimate “extra” strikes gained or lost
   - Converted to runs using:
     - A context-neutral method (constant runs per strike)
     - A context-specific method based on run expectancy (count, baserunners, outs)

3. **Challenge System Modeling**
   - Trained an XGBoost model on Triple-A data to estimate the probability a pitch is challenged
   - Used the Triple-A success rate of a challenge, which depends on whether the umpire's call matched the true zone
   - Combined to estimate pitch-level overturn probability (~1%)
   - Applied to MLB data to simulate changes in framing value distributions

---

## Key Findings

- Catcher framing value remains significant under a challenge system, though its impact is reduced
- The 10 best framers lose about 9% of their framing value (between 7% and 11% across the assumptions I tested, see Validation)
- The 10 worst framers see their negative value shrink by about 15% (between 12% and 18%) due to overturned egregious misses
- The overall distribution compresses toward zero, and the standard deviation across catchers falls from 8.6 to 7.7 runs per 7,500 pitches
- Context-specific framing exhibits higher variance and better captures in-game impact, while context-neutral framing is more suitable for skill evaluation

---

## Methodology Highlights

- **GAMs** were chosen to capture smooth, non-linear spatial effects of pitch location
- **XGBoost** was used for challenge modeling due to its ability to handle interactions and nonlinear decision boundaries
- A challenge can only succeed if the umpire's call did not match the true zone, so success uses the Triple-A rate for each case (96% when the call was wrong, 4% when it was right) instead of a second model
- MLB data has no challenges left value, so it is sampled from Triple-A pitches in the same inning, since teams run out of challenges as the game goes on
- Basis dimension (`k = 55`) selected to balance bias and variance, validated using k-index diagnostics
- Framing value normalized to runs per 7,500 taken pitches to approximate a full season workload
- All models were checked on games they were not trained on (see Validation)

---

## Validation

`R/validation.R` checks the models on held-out games, so no game is in both the training and test data. Every number below comes from running that script on the provided data.

### Strike probability GAMs

5-fold cross validation by game on the MLB data.

| Model | AUC | Log loss | Accuracy |
|---|---|---|---|
| GAM, in-sample | 0.980 | 0.170 | 92.5% |
| GAM, held-out games | 0.980 | 0.170 | 92.5% |
| Count-only model, held-out games | 0.682 | 0.580 | 67.8% |

The held-out results match the in-sample results, so the GAMs are not overfitting. Calibration also holds up. In each decile of predicted strike probability, the actual strike rate is within a percentage point or so of the prediction (for example 0.392 predicted and 0.394 actual in the seventh decile).

### Framing value stability

I rebuilt framing value from the held-out predictions and split each catcher's pitches into odd and even games. Across 47 catchers with at least 1,500 pitches in each half, the correlation is 0.72 (about 0.83 after a Spearman-Brown correction for using the full sample), so the metric is a fairly stable measure of the catcher.

### Challenge model

5-fold cross validation by game on the Triple-A data.

| Model | AUC | Log loss |
|---|---|---|
| Challenge, in-sample | 0.961 | 0.057 |
| Challenge, held-out games | 0.937 | 0.066 |
| Challenge, base rate only | 0.500 | 0.102 |

Challenges are rare (about 2% of pitches), so accuracy is not useful and I use AUC and log loss. The model loses a little on new games, but it still beats the base rate by a wide margin.

An earlier version used a second XGBoost model for challenge success. It reached an AUC of 0.999, but only because it was given whether the call matched the true zone, and in the MLB data it gave an average 26% overturn probability to calls that were correct. A correct call cannot be overturned, so I replaced it with the Triple-A success rates. A challenge succeeded in 96.2% of the cases where the call was wrong and 4.0% of the cases where it was right, and the rule "succeeds only if the call was wrong" matches the outcome in 96.1% of the 6,927 challenges.

As a check on the overturn probability, the predicted rate of overturned calls in the Triple-A data is 1.14% per pitch (held-out challenge probability times the success rate), and the actual rate is 1.13%.

### How much the result depends on assumptions

The main script assumes that only pitches in the top 10% of challenge probability get challenged, and that challenges left in MLB is a sample of the Triple-A values in the same inning. XGBoost also uses random subsamples, so the challenge model changes a little with the seed. I repeated the simulation with a top 20%, 10%, and 5% threshold and five seeds each, refitting the challenge model every time.

| Challenge threshold | Best 10 catchers | Worst 10 catchers |
|---|---|---|
| Top 20% | -9.2% to -11.1% | -12.3% to -17.0% |
| Top 10% | -8.9% to -10.6% | -12.4% to -17.2% |
| Top 5% | -6.8% to -9.1% | -13.0% to -17.6% |

The seed alone moves the result by a few percentage points, which is why I report ranges. Averaged over all 15 runs, the best 10 catchers lose 9.3% and the worst 10 catchers shrink by 14.9%. Resampling whole games 200 times gives a 95% interval of -12.1% to -5.6% for the best 10 catchers and -16.8% to -11.9% for the worst 10. The challenge model is not refit in the bootstrap, so this only covers which MLB games are in the sample.

### Limits

The challenge model is trained on Triple-A and applied to MLB, and no MLB challenge data exists to test that step. The challenges left sampling and the 10% challenge threshold are also not tested against real MLB behavior.

---

## Tools & Technologies

- R (mgcv, xgboost, tidyverse)
- Generalized Additive Models (GAM)
- Gradient Boosted Decision Trees (XGBoost)
- Pitch-level MLB and Triple-A data

---

## Repository Structure
```
R/
  catcher_framing_analysis.R   full analysis: strike probability GAMs, framing value, challenge models
  validation.R                 held-out validation, stability checks, and sensitivity of the challenge results
docs/
  code-walkthrough.pdf         annotated code with outputs and charts
  presentation.pptx            summary of methods and findings
```

---

## Running the Code
```r
install.packages(c("dplyr", "tidyr", "mgcv", "ggplot2", "xgboost", "Matrix"))
```
Run `R/catcher_framing_analysis.R` from a folder containing the three input files (`mlb_data_prompt.csv`, `milb_data_prompt.csv`, `re288_2023_prompt.csv`). These were provided with the project prompt and are not included in this repository. The run expectancy file needs the columns `outs`, `runner_on_1`, `runner_on_2`, `runner_on_3`, `balls`, `strikes`, and `RE`. See the [code walkthrough](docs/code-walkthrough.pdf) for outputs and charts.

To reproduce the validation, run `R/validation.R` from the same folder. It runs the main script first if the results are not already in memory, and takes about 20 minutes in total.
