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

myDir  <- "./"
input   <- file.path(myDir, j, "observed")
scripts <- file.path(myDir, "scripts")
output  <- file.path(myDir, "output")
live    <- file.path(myDir, j, "forecast")
setwd(myDir)

# Odir
if(!dir.exists(file.path(output))) {
  dir.create(file.path(output))
}

# Load dependencies
source(file.path(scripts, "05_Model_outputs.R"))

# -------------
# Eof
# -------------

