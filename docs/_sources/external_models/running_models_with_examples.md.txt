# Running models with examples
When running external models from the command line interface of a local CHAP setup, chap evaluate needs a few arguments. The following commands can be used to run some implemented external models on datasets internally in CHAP.  Note that this might fail for Windows users. The general syntax is to state the argument name(which starts with "--"), and then the actual value aften an empty space, for instance --dataset-name followed by ISIMIP_dengue_harmonized. If you want a different name for the produced report simply change report.pdf. All the code for the models can be found at the url referenced in the commands.
* minimalist_example_r: 
`chap evaluate --model-name https://github.com/dhis2-chap/minimalist_example_r --dataset-name ISIMIP_dengue_harmonized --dataset-country vietnam --report-filename report.pdf --debug --n-splits=2`
* minimalist_multiregion_r: 
`chap evaluate --model-name https://github.com/dhis2-chap/minimalist_multiregion_r --dataset-name ISIMIP_dengue_harmonized --dataset-country vietnam --report-filename report.pdf --debug --n-splits=2`
* minimalist_example_lag_r: 
`chap evaluate --model-name https://github.com/dhis2-chap/minimalist_example_lag_r --dataset-name ISIMIP_dengue_harmonized --dataset-country vietnam --report-filename report.pdf --debug --n-splits=2`
* Madagascar_ARIMA: 
`chap evaluate --model-name https://github.com/dhis2-chap/Madagascar_ARIMA --dataset-name ISIMIP_dengue_harmonized --dataset-country vietnam --report-filename report.pdf --debug --n-splits=2`
* Epidemiar: `chap evaluate --model-name https://github.com/dhis2-chap/epidemiar_example_model --dataset-csv ../epidemiar_example_data/input/laos_test_data.csv --report-filename report.pdf --debug --n-splits=2`
* chap_auto_ewars_weekly: `chap evaluate --model-name https://github.com/dhis2-chap/chap_auto_ewars_weekly --dataset-name ISIMIP_dengue_harmonized --dataset-country vietnam --report-filename report.pdf --debug --n-splits=1`
* chap_auto_ewars: `chap evaluate --model-name https://github.com/dhis2-chap/chap_auto_ewars --dataset-name ISIMIP_dengue_harmonized --dataset-country vietnam --report-filename report.pdf --debug --n-splits=1`

Note that the Epidemiar command uses a local file path for the supplied dataset as it requires weekly data, which is not currently available in CHAP's internal datasets. The command above works when cloning the epidemiar_example_model locally and if the command is run from the folder chap-core, then it assumes that the cloned repository is in the same folder as chap-core, and we use the relative file path. You can also simply dowload the csv file laos_test_data.csv from the github folder and reference the path to the local file.

chap_auto_ewars only accepts mothly data while chap_auto_ewars_weekly can use both weekly and monthly data, should be combined together soon. Additionaly there is a version of chap_auto_ewars which uses spatial smoothing and a geojson file which can be ran as
* `chap evaluate --model-name https://github.com/Halvardgithub/chap_auto_ewars --dataset-csv ../chap_auto_ewars/example_data_Viet/historic_data.csv --polygons-json ../chap_auto_ewars/example_data_Viet/vietnam.json --report-filename report.pdf --debug --n-splits=1 --polygons-id-field VARNAME_1`
The above uses local files and their relative paths, the files are available at the github url.

## For Windows users
Windows users might have issues with the commands above. The solution is to clone the repositories for the external models and add the optional command `--run-directory-type use_existing`. An example is shown below.
* minimalist_example_r: `chap evaluate --model-name /mnt/c/Users/NAME/Documents/GitHub/minimalist_example_r/ --dataset-name ISIMIP_dengue_harmonized --dataset-country vietnam --report-filename report.pdf --debug --n-splits=2 --run-directory-type use_existing`

Note that you need to use your own local file path, and if you are using WSL and ubuntu this might be with `mnt` from linux, even on a Windows system. 

### Warnings
When running the command with a local file path for the model folder you can in theory run the command from any folder, not just from chap-core. However, running `chap evaluate` with `--run-directory-type use_existing` from the same folder as you are using as the `--model-name` will cause an inifnite copying loop. Sometimes, if a command fails, it might be neccessary to exit and open the folder again, for example run `cd ../chap-core` to go one folder up and then back to chap-core. Additionaly, having an active VPN can aslo confuse CHAP and cause the commands to fail.