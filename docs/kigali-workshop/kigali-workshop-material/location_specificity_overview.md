# Location Specificity in Common Model Classes

Different model classes vary in how naturally they support region-specific
effects versus sharing information across regions. This overview summarizes
the main approaches.

## Reference Table

| Model class | Separate per region | Global (shared) | Semi-global / partial pooling |
|---|---|---|---|
| Linear regression | Yes (fit per region) | Yes (pooled) | Via mixed effects (e.g. `lme4`) |
| ARIMA | Yes (standard use) | Not typical | Not typical |
| ETS | Yes (standard use) | Not typical | Not typical |
| Hierarchical Bayesian | Yes | Yes | Yes (primary use case) |
| Random Forest / XGBoost | Yes | Yes (with region feature) | Via region feature |
| Deep learning (LSTM, etc.) | Yes | Yes (with embedding) | Via learned embeddings |

## Key Observations

**Traditional time series models (ARIMA, ETS)** are inherently
single-series models. They fit one model per region by default and have no
built-in mechanism for sharing information across regions.

**Linear regression** is flexible: it can be fit per region, pooled across
regions, or extended with mixed effects (random intercepts/slopes per
region) to partially pool. The walkthroughs in this session demonstrate
these approaches using indicator variables and interaction terms.

**Hierarchical Bayesian models** (e.g. using PyMC, Stan, or INLA) are
designed for partial pooling. Each region's parameters are drawn from a
shared distribution, with the amount of shrinkage toward the group mean
learned from data. This makes them well-suited for settings with many
regions and limited data per region.

**Tree-based models (Random Forest, XGBoost)** can handle multiple regions
by including a region identifier as a feature. The tree splits can then
learn different patterns for different regions. This provides implicit
partial pooling -- regions with similar patterns share tree structure.

**Deep learning models** typically handle multiple regions by learning
region embeddings (dense vector representations). These embeddings allow
the model to learn similarities between regions and share information
accordingly, similar in spirit to partial pooling.

## Which Approach Naturally Supports Borrowing Strength?

Borrowing strength (using data from all regions to improve estimates for
each individual region) is most natural in:

- **Hierarchical Bayesian models** -- this is their primary design goal
- **Tree-based models** -- implicit through shared tree structure
- **Deep learning** -- through learned embeddings

It requires explicit setup in:

- **Linear regression** -- needs mixed-effects formulation or
  regularization
- **ARIMA / ETS** -- not naturally supported; requires external mechanisms
  like meta-learning or forecast combination
