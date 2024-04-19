## ############################################################################
##
## LONDON SCHOOL OF HYGIENE AND TROPICAL MEDICINE (LSHTM)
##
## ############################################################################
##
## DISCLAIMER:
## This script has been developed for research purposes only.
## The script is provided without any warranty of any kind, either express or
## implied. The entire risk arising out of the use or performance of the sample
## script and documentation remains with you.
## In no event shall LSHTM, its author, or anyone else involved in the
## creation, production, or delivery of the script be liable for any damages
## whatsoever (including, without limitation, damages for loss of business
## profits, business interruption, loss of business information, or other
## pecuniary loss) arising out of the use of or inability to use the sample
## scripts or documentation, even if LSHTM has been advised of the
## possibility of such damages.
##
## ############################################################################
##
## DESCRIPTION
## Runs all routines
##
## Version control: GitLab
## Initially created on 05 May 2019
##
##
## Written by: Felipe J Colon-Gonzalez
## For any problems with this code, please contact Felipe.Colon@lshtm.ac.uk
##
## ############################################################################
## INSTRUCTIONS TO RUN ON NON-INTERACTIVE MODE:
##
## Rscript 05_Model_outputs.R "working_dir" "input_dir" "scripts_dir" "output_dir" \
##                    "livedata_dir"
## ############################################################################

# clear
rm(list=ls())

j <-  "20110101"

source(file.path("scripts/00_Functions.R"))
source(file.path("scripts/01_Load_packages.R"))

# Read the csv
d <- read_csv("../../example_data/hydro_met_subset.csv")
print(d)

# get the rainfaill column from the d dataframe
rainfall <- d$rainfall

print(rainfall)

