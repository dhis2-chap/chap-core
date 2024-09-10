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
## Sub-routine to install and load required packages
##
## Version control: GitLab
## Initially created on 15 Jan 2019
##
## Written by: Felipe J Colon-Gonzalez
## For any problems with this code, please contact felipe.colon@lshtm.ac.uk
##
## ############################################################################
r = getOption("repos")
r["CRAN"] = "http://cran.us.r-project.org"
options(repos = r)


print(R.version.string)
version <- R.Version()
major   <- as.numeric(version$major)
minor   <- as.numeric(version$minor)

stopifnot(major>3 || (major==3 && minor>=2))

pkglist <- c("zoo", "Hmisc", "lubridate", "hydroGOF", "plyr", "dplyr",
             "raster", "tidyr", "data.table", "xtable", "MASS",
             "magrittr", "rgdal", "tidyverse", "spdep", "ModelMetrics",
             "forecast", "parallel")

new.packages <- pkglist[!(pkglist %in% installed.packages()[,"Package"])]

"%ni%" <- Negate("%in%")
if("INLA" %ni% installed.packages()) {
    install.packages("INLA",
                     repos=c(
                         getOption("repos"),
                         INLA="https://inla.r-inla-download.org/R/stable"),
                     dep=TRUE)
}

if(length(new.packages)){
     install.packages(new.packages,
                      repos="http://cran.ma.imperial.ac.uk/")
     }

lapply(pkglist, require, character.only=TRUE)
require(INLA)

if(major>3 || (major==3 && minor>=2)==TRUE){
       message('R packages successfully loaded')
}

rm(pkglist, new.packages, version)


# ----------------
# Eof
# ----------------

