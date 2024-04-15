
# install INLA
r = getOption("repos")
r["CRAN"] = "http://cran.us.r-project.org"
options(repos = r)
#install.packages("INLA", repos=c(getOption("repos"), INLA="https://inla.r-inla-download.org/R/testing"), dep=TRUE)

# load INLA
#library(INLA)

#  select other packages
packages <- c("data.table", "tidyverse", "sf", "sp", "spdep",
              "dlnm", "tsModel", "hydroGOF","RColorBrewer",
              #"geofacet",
              "ggpubr", "ggthemes")

# install.packages
lapply(packages, install.packages, character.only = TRUE)
