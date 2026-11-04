# Vocabulary (domain specific terms used in the code)

## Signals and threshold policies

Chap records a **signal** when a forecast meets a configured threshold level. It
is unverified: WHO EWAR reserves **alert** for an event that has been verified,
risk assessed and requires an intervention. See the glossary in the
[2014 EWAR guide](https://www.who.int/publications/i/item/WHO-HSE-GCR-LYO-2014.4)
and the glossary and signal-event-alert-response framework in
[EWAR in emergencies (2022)](https://www.who.int/publications/i/item/9789240063587)
(published on WHO's website in January 2023).

For forecasts, WHO/TDR dengue EWARS uses **alarm signal** and **alarm threshold**
for the probability cutoff. See section 2.3.8 of the
[2020 second edition](https://www.who.int/publications/i/item/9789240003750),
which updates the 2018 guide. Chap's `Signal` represents this forecast alarm
signal; it does not establish that an outbreak has occurred.

A `ThresholdPolicy` contains ordered `ThresholdLevel` entries. Each level pairs
an outbreak threshold (`threshold_params`) with an `alarm_threshold`, a probability
between 0 and 1. These policy and level names are Chap configuration terms, not
WHO workflow stages. Level names such as `level_1` and `level_2` are user-defined,
ordered from least to most severe. A prediction records at most one signal per
org unit and period, at the most severe level reached.

Chap retains its inclusive probability comparison: the fraction of forecast
samples strictly above the outbreak threshold must be **at least**
`alarm_threshold`. The dengue guide describes a strict `>` probability comparison;
this terminology change does not change Chap's calculation or claim to reproduce
the complete dengue EWARS algorithm.

`SignalReleaseStatus` controls dissemination: `pending` awaits review, `released`
is cleared for dissemination, and `withheld` records a decision not to release.
`reviewed_by` and `reviewed_at` record that decision. Release does not perform
epidemiological verification or risk assessment and does not turn a signal into
a WHO alert.

### API changes from alert terminology

| Previous name | Current name |
| --- | --- |
| `/v1/crud/alerts` | `/v1/crud/signals` |
| `/v1/crud/alert-policies` | `/v1/crud/threshold-policies` |
| `/v1/crud/alerts/$approve` | `/v1/crud/signals/$set-release-status` |
| `alertPolicyId`, `alertPolicy` | `thresholdPolicyId`, `thresholdPolicy` |
| `alerts`, `alertIds` | `signals`, `signalIds` |
| `exceedanceThreshold` | `alarmThreshold` |
| `noAlertOutputCode` | `noSignalOutputCode` |
| `approved`, `approvedBy`, `approvedAt` | `releaseStatus`, `reviewedBy`, `reviewedAt` |
| `approved` / `declined` states | `released` / `withheld` states |

These are breaking changes, including the policy query parameter on the outbreak
metrics endpoint. The review queue is `GET /v1/crud/signals?releaseStatus=pending`.
The migration preserves IDs, policy links, level names, output codes and review
metadata. Existing clients must use the new paths and fields; there are no legacy
aliases. Python names use the corresponding snake case fields and the
`signal_tables`, `signal_service` and `threshold_policy_service` modules.


### Model template

A model template is a flexible "model" which can be configured. A model template typically presents various options (hyperparameters, possible covariates, etc) which are open for configuration. 


### Configured model

A configured model can be made from a model template by applying chocies to the options presented by the model template. Only a configured model can actually be trained on a given dataset (as opposed to a model template, since a model template does not necessarily have enough information about how to train or predict).

### ExternalModel

ExternalModel is a wrapper around an external model (that can e.g. be an R model) to make it compatible with the interface of ConfiguredModel. This means that ExternalModel has train/predict similarily to ConfiguredModel, but these methods are wrappers that runs the train/predict of external models.

## Some other terms we use

Backtest: Is the same as evaluation for now (used as a term in the REST API)


## Runner

A runner is something that can run commands. ExternalModels (not ConfiguredModels) have a Runner object attached to them. When train/predict is called, the runner is handling how to j6o05..,m.


## Backtest and Evaluation

Backtest and evaluation are used interchangeably for now. They refer to the process of evaluating a model on a dataset by splitting the dataset into training and test sets multiple times (e.g. using cross-validation) and measuring performance on the test sets.

Evaluations is a common term in the machine learning domain. Backtest is often used more specifically in time series forecasting, where the model is tested on past data to see how well it would have performed.


