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
- Elite framers lose approximately 10–20% of their value
- Poor framers see reduced negative impact due to overturned egregious misses
- The overall distribution compresses toward zero, reducing league-wide variance
- Context-specific framing exhibits higher variance and better captures in-game impact, while context-neutral framing is more suitable for skill evaluation

---

## Methodology Highlights

- **GAMs** were chosen to capture smooth, non-linear spatial effects of pitch location
- **XGBoost** was used for challenge modeling due to its ability to handle interactions and nonlinear decision boundaries
- Basis dimension (`k = 55`) selected to balance bias and variance, validated using k-index diagnostics
- Framing value normalized to runs per 7,500 taken pitches to approximate a full season workload

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
docs/
  code-walkthrough.pdf         annotated code with outputs and charts
  presentation.pptx            summary of methods and findings
```

---

## Running the Code
```r
install.packages(c("dplyr", "tidyr", "mgcv", "ggplot2", "xgboost", "Matrix"))
```
Run `R/catcher_framing_analysis.R` from a folder containing the three input files (`mlb_data_prompt.csv`, `milb_data_prompt.csv`, `re288_2023_prompt.csv`). These were provided with the project prompt and are not included in this repository; see the [code walkthrough](docs/code-walkthrough.pdf) for outputs and charts.
