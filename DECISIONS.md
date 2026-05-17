# Decisions

- Added `ready_for_follow_up` on backtests to track which evaluations are marked for follow-up.
- Require a saved configuration `name` so configured models with data source can be distinguished.
- Link jobs to prediction setups by storing prediction setup IDs in job metadata so the dashboard can find running prediction jobs for a setup.
