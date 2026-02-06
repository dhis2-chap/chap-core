# Interaction Effects

## What Are Interaction Effects?

An interaction effect occurs when the relationship between a predictor and
the outcome depends on another variable. In disease forecasting, a common
example is that rainfall's effect on disease may differ between regions --
heavy rainfall might strongly increase cases in lowland areas but have
little effect in highlands.

In a linear regression without interactions, adding region indicators only
shifts the baseline (intercept) per region. The slope of each predictor is
the same everywhere:

    cases = offset_region + b1 * rainfall + b2 * temperature

With an interaction term, the slope itself varies by region:

    cases = offset_region + b1_region * rainfall + b2 * temperature

This is achieved by multiplying the region indicator with the predictor,
creating a new feature: `region_A * rainfall`, `region_B * rainfall`, etc.

## Regression Models Need Explicit Interactions

In linear regression (and generalized linear models), the model can only
learn relationships that are explicitly encoded in the features. If the
effect of rainfall differs between regions but the model has no interaction
term, it is forced to find a single average effect -- which may be wrong
for all regions.

Common interaction patterns in disease forecasting:

- **Location x climate**: different climate sensitivities per region
- **Location x season**: different seasonal patterns per region (as shown
  in the multi-region walkthrough)
- **Season x climate**: climate effects that vary by time of year

The downside of interactions is that they multiply the number of
parameters. With 10 regions and 12 months, location x season creates
120 interaction features. With limited data, this can lead to overfitting.

## Flexible ML Models Learn Interactions Implicitly

Tree-based models (Random Forest, XGBoost) and deep learning models
can learn interaction effects without being told to look for them.

A decision tree naturally creates interactions through its branching
structure. A split on "region = A" followed by a split on "rainfall > 100"
effectively learns a region-specific rainfall threshold -- an interaction
between region and rainfall.

Deep learning models learn interactions through their hidden layers.
Nonlinear activation functions allow the model to combine inputs in
complex ways without explicit feature engineering.

## When to Use Explicit Interactions

**Use explicit interactions when:**

- Using linear regression or GLMs
- You have domain knowledge about which interactions matter
- The dataset is small enough that implicit learning would overfit

**Rely on implicit interactions when:**

- Using tree-based or deep learning models
- You have many potential interactions and want the model to discover
  which matter
- The dataset is large enough to support learning complex patterns

In practice, even with flexible models, adding known important interactions
as explicit features can help -- it makes it easier for the model to find
patterns that domain knowledge suggests should exist.
