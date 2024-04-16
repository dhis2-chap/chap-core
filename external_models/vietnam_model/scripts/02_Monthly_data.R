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
## Sub-routine to get monthly records from annual data
##
## Version control: GitLab
## Initially created on 13 Feb 2019
##
##
## Written by: Felipe J Colon-Gonzalez
## For any problems with this code, please contact Felipe.Colon@lshtm.ac.uk
## 
## ############################################################################

# pkgs
source(file.path(scripts, "00_Functions.R"))
source(file.path(scripts, "01_Load_packages.R"))

# Annual to monthly
annual <- obsFun("^observed.*?\\.csv" )
  
# tsd
x <- tsdate("dengue")

# set pars
pu <- a2m(annual, "periurban_landcover") %>%
      dplyr::select(parametername, tsdatetime, areaid, tsvalue)
ur <- a2m(annual, "urban_landcover")  %>%
      dplyr::select(parametername, tsdatetime, areaid, tsvalue)
ru <- a2m(annual, "rural_landcover")  %>%
      dplyr::select(parametername, tsdatetime, areaid, tsvalue)
po <- a2m(annual, "population")  %>%
      dplyr::select(parametername, tsdatetime, areaid, tsvalue)

# rm
rm(x)

# ofiles
write_csv(pu, file.path(input, 
                     paste0("observed_periurban_landcover_04_19_",
                            Sys.Date(), ".csv")))
write_csv(ur, file.path(input,
                     paste0("observed_urban_landcover_04_19_",
                            Sys.Date(), ".csv")))
write_csv(ru, file.path(input,
                     paste0("observed_rural_landcover_04_19_",
                            Sys.Date(), ".csv")))
write_csv(po, file.path(input,
                     paste0("observed_population_landcover_04_19_",
                            Sys.Date(), ".csv")))

annual2 <- annual %>% spread(parametername, tsvalue)

  
# ------------------------
# Eof
# ------------------------



