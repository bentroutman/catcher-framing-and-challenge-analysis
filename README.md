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
   - Trained XGBoost models on Triple-A data to estimate:
     - Probability a pitch is challenged
     - Probability a challenge is successful, given a challenge
   - Combined to estimate pitch-level overturn probability (~1%)
   - Applied to MLB data to simulate changes in framing value distributions

---

## Key Findings

- Catcher framing value remains significant under a challenge system, though its impact is reduced
- The 10 best framers lose about 21% of their framing value, and the result stays between 17% and 22% under the different assumptions I tested (see Validation)
- The 10 worst framers see their negative value shrink by only about 4%
- The overall distribution compresses toward zero, and the standard deviation across catchers falls from 8.6 to 7.6 runs per 7,500 pitches
- Context-specific framing exhibits higher variance and better captures in-game impact, while context-neutral framing is more suitable for skill evaluation

---

## Methodology Highlights

- **GAMs** were chosen to capture smooth, non-linear spatial effects of pitch location
- **XGBoost** was used for challenge modeling due to its ability to handle interactions and nonlinear decision boundaries
- Basis dimension (`k = 55`) selected to balance bias and variance, validated using k-index diagnostics
- Framing value normalized to runs per 7,500 taken pitches to approximate a full season workload
- All models were checked on games they were not trained on (see Validation)

---

## Validation

`R/validation.R` checks the models on held-out games, so no game is in both the training and test data. Every number below comes from running that script on the provided data.

**Strike probability GAMs** (5-fold cross validation by game, MLB data)

| Model | AUC | Log loss | Accuracy |
|---|---|---|---|
| GAM, in-sample | 0.980 | 0.170 | 92.5% |
| GAM, held-out games | 0.980 | 0.170 | 92.5% |
| Count-only model, held-out games | 0.682 | 0.580 | 67.8% |

The held-out results match the in-sample results, so the GAMs are not overfitting. Calibration also holds up. In each decile of predicted strike probability, the actual strike rate is within a percentage point or so of the prediction (for example 0.392 predicted and 0.394 actual in the seventh decile).

**Framing value stability.** I rebuilt framing value from the held-out predictions and split each catcher's pitches into odd and even games. Across 47 catchers with at least 1,500 pitches in each half, the correlation is 0.72 (about 0.83 after a Spearman-Brown correction for using the full sample), so the metric is a fairly stable measure of the catcher.

**Challenge models** (5-fold cross validation by game, Triple-A data)

| Model | AUC | Log loss |
|---|---|---|
| Challenge, in-sample | 0.962 | 0.056 |
| Challenge, held-out games | 0.937 | 0.066 |
| Challenge, base rate only | 0.500 | 0.102 |
| Success, in-sample | 1.000 | 0.004 |
| Success, held-out games | 0.999 | 0.040 |
| Success, base rate only | 0.500 | 0.691 |

Challenges are rare (about 2% of pitches), so accuracy is not useful for the challenge model and I use AUC and log loss. The challenge model loses a little on new games, but it still beats the base rate by a wide margin. The success model is almost given the answer, since a challenge succeeds when the umpire's call did not match the true zone and the model has that as a feature. Its AUC is high by construction and should not be read as strong evidence on its own.

As a check on the overturn probability, the predicted rate of overturned calls in the Triple-A data is 1.12% per pitch (held-out challenge probability times success probability), and the actual rate is 1.13%.

**How much the result depends on assumptions.** The main script assumes that only pitches in the top 10% of challenge probability get challenged, and that challenges left in MLB is a random sample of the Triple-A values. I repeated the simulation with a top 20%, 10%, and 5% threshold and three random seeds each.

| Challenge threshold | Best 10 catchers | Worst 10 catchers |
|---|---|---|
| Top 20% | -21.9% to -22.1% | -1.7% to -2.8% |
| Top 10% | -21.1% to -21.3% | -2.5% to -3.6% |
| Top 5% | -17.1% to -17.4% | -5.5% to -6.5% |

Resampling whole games 200 times gives a 95% interval of -24.3% to -17.8% for the best 10 catchers and -7.0% to -0.5% for the worst 10. The challenge models are not refit in the bootstrap, so this only covers which MLB games are in the sample.

**Limits.** The challenge models are trained on Triple-A and applied to MLB, and no MLB challenge data exists to test that step. The random challenges-left assumption and the 10% challenge threshold are also not tested against real MLB behavior. Under the context-specific version, the worst catchers do not improve at all, so the worst-framer result depends on which version is used.

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
