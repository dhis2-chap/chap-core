# Tuesday Afternoon - 24 Feb

## Design Document: CLIM-442 - Prepare Tuesday Afternoon Session

### Context

This session follows Tuesday morning (which covers evaluation) and Monday afternoon
(evaluation walkthrough). By this point, participants have built a minimalist model,
gotten it running in CHAP, and evaluated it. The afternoon session introduces two
key extensions: handling multiple regions and lagged effects.

### Session Goals

- Introduce how to handle multiple regions and the rationale for borrowing information across regions
- Introduce lagged effects and basic ways to add them to models
- Have participants extend their own model with multiple regions and lags

### Existing Material

The `learn_modelling.md` page already contains substantial written material on:

- **Multiple regions** (section "Multiple regions"): separate model per region,
  borrowing strength, bias-variance tradeoff
- **Lags and autoregressive effects** (section "Lags and autoregressive effect"):
  lag concept, data alignment, smoothing across lags
- Links to external tutorial repos: `minimalist_multiregion`, `minimalist_example_lag`
  (Python and R versions)

What is missing: doctested Python walkthroughs that participants can follow
interactively within the docs, similar to the evaluation walkthrough.

### Plan

#### 1. Doctested walkthrough: linear regression, seasonal effects, lagged effects (CLIM-477)

Create a new documentation page with executable code blocks showing:

- Simple linear regression on disease data (single region, no lags)
- Adding a seasonal effect (e.g. month-of-year or Fourier terms as predictors)
- Adding lagged climate predictors and observing the effect on predictions
- Evaluating each variant using the existing `Evaluation.create` workflow

Use the same `exec="on" session="..." source="above"` pattern from the evaluation
walkthrough. Use `example_data/laos_subset.csv` or similar existing example data.

#### 2. Doctested walkthrough: multi-region modeling strategies (CLIM-478)

Create a documentation page with executable code blocks covering:

- Separate model per region (independent fits)
- Single global model (all regions pooled, ignoring region identity)
- Model with shared fixed effect but separate seasonal effect per region
- Partial pooling / hierarchical model (if supported by existing estimators)

Each approach should be evaluated side-by-side so participants can compare
prediction skill across strategies.

#### 3. Overview of modeling classes supporting semi-global models (CLIM-479)

Create a reference table or page documenting which modeling classes/estimators in
chap-core support:

- Separate model per region
- Global model (shared parameters)
- Semi-global / partial pooling (shared some effects, separate others)

This requires auditing the existing model classes in `chap_core/models/` and
`chap_core/predictor/` to determine their capabilities.

#### 4. Interaction effects explanation (CLIM-480)

Write a conceptual section explaining:

- How location-specific effects can be achieved through interaction terms
  (e.g. region x climate variable)
- The difference between regression models (which need explicit interaction terms)
  and flexible ML models (which can learn interactions implicitly)
- Practical examples showing when explicit interactions are needed vs not

### Open Questions

- Should seasonal effects be introduced in this session or earlier (Tuesday morning)?
  Comment on CLIM-442 suggests seasonal effects make a clearer case for borrowing
  strength across regions.
- Which existing estimators in chap-core can serve as examples for the
  multi-region strategies? Need to audit `chap_core/predictor/` and
  `chap_core/models/`.
- Are the external tutorial repos (`minimalist_multiregion`, `minimalist_example_lag`)
  sufficient, or should the doctested walkthroughs replace them?
- What example dataset to use -- `laos_subset.csv` has multiple regions and is
  already used in the evaluation walkthrough.

### Dependencies

- The evaluation walkthrough (CLIM-474, merged) provides the pattern for executable
  doc pages
- External tutorial repos on GitHub provide reference implementations
- `learn_modelling.md` provides the conceptual text that walkthroughs should complement
