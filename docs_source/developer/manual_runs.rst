Manual runs
===========

This page contains a list of manual tests/runs/demos 
that we routinely run to either show or test the functionality of CHAP.

Some of these we plan to create automated tests for in the future.


1. Running chap evaluate for an external model
----------------------------------------------

Note: This functionality is automaticaly tested on every push to Github using Github Actions.


The command below should work when chap is installed and Docker is available on the system.

```bash
chap evaluate --model-name https://github.com/dhis2-chap/chap_auto_ewars --dataset-name ISIMIP_dengue_harmonized --dataset-country brazil
```

2. Using the Prediction app with chap-core
------------------------------------------

...