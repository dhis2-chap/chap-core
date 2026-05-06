# Examples of chap eval commands

The following are examples of running various chap-integrated models on various datasets:

* minimalist_example_r: 
`chap eval --model-name https://github.com/dhis2-chap/minimalist_example_r --dataset-csv https://raw.githubusercontent.com/dhis2/climate-health-data/refs/heads/main/lao/chap_LAO_admin1_monthly.csv --output-file eval.nc --plot --run-config.debug --backtest-params.n-splits 2`
* minimalist_multiregion_r: 
`chap eval --model-name https://github.com/dhis2-chap/minimalist_multiregion_r --dataset-csv https://raw.githubusercontent.com/dhis2/climate-health-data/refs/heads/main/lao/chap_LAO_admin1_monthly.csv --output-file eval.nc --plot --run-config.debug --backtest-params.n-splits 2`
* minimalist_example_lag_r: 
`chap eval --model-name https://github.com/dhis2-chap/minimalist_example_lag_r --dataset-csv https://raw.githubusercontent.com/dhis2/climate-health-data/refs/heads/main/lao/chap_LAO_admin1_monthly.csv --output-file eval.nc --plot --run-config.debug --backtest-params.n-splits 2`
* Madagascar_ARIMA: 
`chap eval --model-name https://github.com/dhis2-chap/Madagascar_ARIMA --dataset-csv https://raw.githubusercontent.com/dhis2/climate-health-data/refs/heads/main/lao/chap_LAO_admin1_monthly.csv --output-file eval.nc --plot --run-config.debug --backtest-params.n-splits 2`
* Epidemiar: `chap eval --model-name https://github.com/dhis2-chap/epidemiar_example_model --dataset-csv ../epidemiar_example_data/input/laos_test_data.csv --output-file eval.nc --plot --run-config.debug --backtest-params.n-splits 2`
* chap_auto_ewars_weekly: `chap eval --model-name https://github.com/dhis2-chap/chap_auto_ewars_weekly --dataset-csv https://raw.githubusercontent.com/dhis2/climate-health-data/refs/heads/main/lao/chap_LAO_admin1_monthly.csv --output-file eval.nc --plot --run-config.debug --backtest-params.n-splits 1`
* chap_auto_ewars: `chap eval --model-name https://github.com/dhis2-chap/chap_auto_ewars --dataset-csv https://raw.githubusercontent.com/dhis2/climate-health-data/refs/heads/main/lao/chap_LAO_admin1_monthly.csv --output-file eval.nc --plot --run-config.debug --backtest-params.n-splits 1`

Note that the Epidemiar command uses a local file path for the supplied dataset as it requires weekly data. The command above works when cloning the epidemiar_example_model locally and if the command is run from the folder chap-core, then it assumes that the cloned repository is in the same folder as chap-core, and we use the relative file path. You can also simply download the csv file laos_test_data.csv from the github folder and reference the path to the local file.

chap_auto_ewars only accepts monthly data while chap_auto_ewars_weekly can use both weekly and monthly data, should be combined together soon. Additionally there is a version of chap_auto_ewars which uses spatial smoothing and a geojson file which can be ran as
* `chap eval --model-name https://github.com/Halvardgithub/chap_auto_ewars --dataset-csv ../chap_auto_ewars/example_data_Viet/historic_data.csv --output-file eval.nc --plot --run-config.debug --backtest-params.n-splits 1`
The above uses local files and their relative paths, the files are available at the github url.

## For Windows users
Windows users might have issues with the commands above. The solution is to clone the repositories for the external models and add the optional command `--run-config.run-directory-type use_existing`. An example is shown below.
* minimalist_example_r: `chap eval --model-name /mnt/c/Users/NAME/Documents/GitHub/minimalist_example_r/ --dataset-csv https://raw.githubusercontent.com/dhis2/climate-health-data/refs/heads/main/lao/chap_LAO_admin1_monthly.csv --output-file eval.nc --plot --run-config.debug --backtest-params.n-splits 2 --run-config.run-directory-type use_existing`

Note that you need to use your own local file path, and if you are using WSL and ubuntu this might be with `mnt` from linux, even on a Windows system. 

### Warnings
When running the command with a local file path for the model folder you can in theory run the command from any folder, not just from chap-core. However, running `chap eval` with `--run-config.run-directory-type use_existing` from the same folder as you are using as the `--model-name` will cause an infinite copying loop. Sometimes, if a command fails, it might be necessary to exit and open the folder again, for example run `cd ../chap-core` to go one folder up and then back to chap-core. Additionally, having an active VPN can also confuse Chap and cause the commands to fail.