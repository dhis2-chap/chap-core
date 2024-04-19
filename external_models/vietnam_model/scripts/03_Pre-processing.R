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
## Sub-routine to pre-process data for modelling
##
## Version control: GitLab
## Initially created on 15 Jan 2019
##
##
## Written by: Felipe J Colon-Gonzalez
## For any problems with this code, please contact Felipe.Colon@lshtm.ac.uk
## 
## ############################################################################

# pkgs
source(file.path(scripts, "02_Monthly_data.R"))

# input
den <- read_obs("dengue")
d1  <- min(ymd(den$tsdatetime))
d2  <- max(den$tsdatetime)
rm(den)

myData <- hist_data("^observed.*?\\.csv")

# forec
fore <- fore_data("^forecast.*?\\.csv")

# unlink
unlink(paste0(file.path(input), "/",
              list.files(path=file.path(input),
                         pattern="population|rural|urban")))

# Correct tsdatetime
myData$tsdatetime <- ceiling_date(ymd(paste(year(myData$tsdatetime),
                           month(myData$tsdatetime),
                           "01", sep="-")), 'month') - days(1)
fore$tsdatetime <- ceiling_date(ymd(paste(year(fore$tsdatetime),
                                          month(fore$tsdatetime),
                                          "01", sep="-")), 'month') - days(1)

# Ensure unique dates
myData %<>% group_by(areaid, tsdatetime) %>%
  dplyr::summarise_all(mean, na.rm=TRUE) 

fore %<>% group_by(areaid, tsdatetime, ensmember) %>%
  dplyr::summarise_all(mean, na.rm=TRUE)

# geog
myMap <- rgdal::readOGR(file.path(input), "province")

# wrangling
minDate           <- min(myData$tsdatetime)
names(myData)     <- gsub(" ", "_", names(myData))
myData$year       <- factor(lubridate::year(myData$tsdatetime))
myData$month      <- factor(lubridate::month(myData$tsdatetime))
myData$dtr        <- myData$maximum_temperature - myData$minimum_temperature
fore$year         <- factor(lubridate::year(fore$tsdatetime))
fore$month        <- factor(lubridate::month(fore$tsdatetime))
fore$dtr          <- fore$maximum_temperature - fore$minimum_temperature

myData$incidence  <- (myData$dengue_cases / myData$population) * 1e5

myData            <- inner_join(myData, myMap@data) 

# times
d1 <- min(fore$tsdatetime)
d2 <- max(fore$tsdatetime)

# add land to forecast
df1         <- sel_land(annual2)  
fore$areaid <- factor(fore$areaid)
fore        <- inner_join(fore, df1)

# sel
newData <- myData %>%
  ungroup() %>%
  dplyr::select(areaid, tsdatetime,
                year, month, population,
                minimum_temperature, 
                maximum_temperature,
                precipitation_amount_per_day, 
                nino34_anomaly,
                specific_surface_humidity, dtr,
                wind_speed, periurban_landcover, 
                urban_landcover, rural_landcover, 
                dengue_cases) %>%
  dplyr::mutate(ensmember="tsvalue_ensemble_00") %>%
  data.table()

newFore <- fore %>%
  dplyr::mutate(dengue_cases=NA) %>%
  dplyr::select(areaid, tsdatetime,
                year, month, population, 
                minimum_temperature, 
                maximum_temperature,
                precipitation_amount_per_day, 
                nino34_anomaly,
                specific_surface_humidity, dtr,
                wind_speed, periurban_landcover, 
                urban_landcover, rural_landcover, 
                dengue_cases, ensmember) %>%
  data.table()

newData <- rbind(newData, newFore)

# rollm

copytmin <- function(x){
  x[is.na(x)] <- 0
  x <- x + mintemp$baseline
}

copytmax <- function(x){
  x[is.na(x)] <- 0
  x <- x + maxtemp$baseline
}

copypre <- function(x){
  x[is.na(x)] <- 0
  x <- x + meanpre$baseline
}

copyshum <- function(x){
  x[is.na(x)] <- 0
  x <- x + meanshum$baseline
}

copydtr <- function(x){
  x[is.na(x)] <- 0
  x <- x + meandtr$baseline
}

copynino <- function(x){
  x[is.na(x)] <- 0
  x <- x + meananom$baseline
}

movav <- function(x) {
  rollapply(x, width=3, FUN=mean,
          fill=NA, align="right")
}

movav2 <- function(x) {
  rollapply(x, width=4, FUN=mean,
            fill=NA, align="right")
  
}

mintemp <- dplyr::select(newData, areaid, tsdatetime, minimum_temperature,
                          ensmember) %>%
  tidyr::spread(ensmember, minimum_temperature) %>%
  dplyr::rename(baseline=tsvalue_ensemble_00) %>%
  dplyr::mutate(baseline=replace_na(baseline, 0)) 

maxtemp <- dplyr::select(newData, areaid, tsdatetime, maximum_temperature,
                          ensmember) %>%
  tidyr::spread(ensmember, maximum_temperature) %>%
  dplyr::rename(baseline=tsvalue_ensemble_00) %>%
  dplyr::mutate(baseline=replace_na(baseline, 0)) 

meanpre <- dplyr::select(newData, areaid, tsdatetime, 
                         precipitation_amount_per_day, ensmember) %>%
  tidyr::spread(ensmember, precipitation_amount_per_day) %>%
  dplyr::rename(baseline=tsvalue_ensemble_00) %>%
  dplyr::mutate(baseline=replace_na(baseline, 0)) 

meanshum <- dplyr::select(newData, areaid, tsdatetime,
                          specific_surface_humidity, ensmember) %>%
  tidyr::spread(ensmember, specific_surface_humidity) %>%
  dplyr::rename(baseline=tsvalue_ensemble_00) %>%
  dplyr::mutate(baseline=replace_na(baseline, 0)) 

meandtr <- dplyr::select(newData, areaid, tsdatetime, dtr,
                          ensmember) %>%
  tidyr::spread(ensmember, dtr) %>%
  dplyr::rename(baseline=tsvalue_ensemble_00) %>%
  dplyr::mutate(baseline=replace_na(baseline, 0)) 

meananom <- dplyr::select(newData, areaid, tsdatetime, nino34_anomaly,
                         ensmember) %>%
  tidyr::spread(ensmember, nino34_anomaly) %>%
  dplyr::rename(baseline=tsvalue_ensemble_00) %>%
  dplyr::mutate(baseline=replace_na(baseline, 0)) 

mintemp %<>%
  mutate_at(vars(matches('tsvalue_ensemble_')),
            copytmin) %>%
  mutate_at(vars(matches('tsvalue_ensemble_')),
            movav) %>%
  dplyr::filter(tsdatetime >=d1) %>%
  tidyr::gather(ensmember, tmin02, -(areaid:baseline)) %>%
  dplyr::select(-baseline) %>%
  data.table()

maxtemp %<>%
  mutate_at(vars(matches('tsvalue_ensemble_')),
            copytmax) %>%
  mutate_at(vars(matches('tsvalue_ensemble_')),
            movav) %>%
  dplyr::filter(tsdatetime >=d1) %>%
  tidyr::gather(ensmember, tmax02, -(areaid:baseline)) %>%
  dplyr::select(-baseline) %>%
  data.table()

meanpre %<>%
  mutate_at(vars(matches('tsvalue_ensemble_')),
            copypre) %>%
  mutate_at(vars(matches('tsvalue_ensemble_')),
            movav) %>%
  dplyr::filter(tsdatetime >=d1) %>%
  tidyr::gather(ensmember, pre02, -(areaid:baseline)) %>%
  dplyr::select(-baseline) %>%
  data.table()

meanshum %<>%
  mutate_at(vars(matches('tsvalue_ensemble_')),
            copyshum) %>%
  mutate_at(vars(matches('tsvalue_ensemble_')),
            movav) %>%
  dplyr::filter(tsdatetime >=d1) %>%
  tidyr::gather(ensmember, shum02, -(areaid:baseline)) %>%
  dplyr::select(-baseline) %>%
  data.table()

meandtr %<>%
  mutate_at(vars(matches('tsvalue_ensemble_')),
            copydtr) %>%
  mutate_at(vars(matches('tsvalue_ensemble_')),
            movav) %>%
  dplyr::filter(tsdatetime >=d1) %>%
  tidyr::gather(ensmember, dtr02, -(areaid:baseline)) %>%
  dplyr::select(-baseline) %>%
  data.table()

meananom %<>%
  mutate_at(vars(matches('tsvalue_ensemble_')),
            copynino) %>%
  mutate_at(vars(matches('tsvalue_ensemble_')),
            movav2) %>%
  dplyr::filter(tsdatetime >=d1) %>%
  tidyr::gather(ensmember, nino3403, -(areaid:baseline)) %>%
  dplyr::select(-baseline) %>%
  data.table()

newData %<>% group_by(areaid, ensmember) %>%
  dplyr::mutate(
    tmin02=rollapply(minimum_temperature, width=3, FUN=mean,
                     fill=NA, align="right"),
    tmax02=rollapply(maximum_temperature, width=3, FUN=mean,
                     fill=NA, align="right"),
    pre02=rollapply(precipitation_amount_per_day, width=3, FUN=mean,
                    fill=NA, align="right"),
    shum02=rollapply(specific_surface_humidity, width=3, FUN=mean,
                     fill=NA, align="right"),
    dtr02=rollapply(dtr, width=3, FUN=mean,
                    fill=NA, align="right"),
    nino3403=rollapply(nino34_anomaly, width=4, FUN=mean,
                    fill=NA, align="right")) 
newData$tmin02[newData$tsdatetime >= d1] <- NA
newData$tmax02[newData$tsdatetime >= d1] <- NA
newData$pre02[newData$tsdatetime >= d1] <- NA
newData$shum02[newData$tsdatetime >= d1] <- NA
newData$dtr02[newData$tsdatetime >= d1] <- NA

newData$areaid  <- as.numeric(as.character(newData$areaid))
mintemp$areaid  <- as.numeric(as.character(mintemp$areaid))
maxtemp$areaid  <- as.numeric(as.character(maxtemp$areaid))
meanpre$areaid  <- as.numeric(as.character(meanpre$areaid))
meanshum$areaid <- as.numeric(as.character(meanshum$areaid))
meandtr$areaid  <- as.numeric(as.character(meandtr$areaid))
meananom$areaid <- as.numeric(as.character(meananom$areaid))

newData %<>% left_join(mintemp, by=c('areaid', 'tsdatetime', 'ensmember')) %>%
  dplyr::mutate(tmin02=coalesce(tmin02.x, tmin02.y)) %>%
  dplyr::select(-tmin02.x, -tmin02.y)
newData %<>% left_join(maxtemp, by=c('areaid', 'tsdatetime', 'ensmember')) %>%
  dplyr::mutate(tmax02=coalesce(tmax02.x, tmax02.y)) %>%
  dplyr::select(-tmax02.x, -tmax02.y)
newData %<>% left_join(meanpre, by=c('areaid', 'tsdatetime', 'ensmember')) %>%
  dplyr::mutate(pre02=coalesce(pre02.x, pre02.y)) %>%
  dplyr::select(-pre02.x, -pre02.y)
newData %<>% left_join(meanshum, by=c('areaid', 'tsdatetime', 'ensmember')) %>%
  dplyr::mutate(shum02=coalesce(shum02.x, shum02.y)) %>%
  dplyr::select(-shum02.x, -shum02.y)
newData %<>% left_join(meandtr, by=c('areaid', 'tsdatetime', 'ensmember')) %>%
  dplyr::mutate(dtr02=coalesce(dtr02.x, dtr02.y)) %>%
  dplyr::select(-dtr02.x, -dtr02.y)
newData %<>% left_join(meananom, by=c('areaid', 'tsdatetime', 'ensmember')) %>%
  dplyr::mutate(nino3403=coalesce(nino3403.x, nino3403.y)) %>%
  dplyr::select(-nino3403.x, -nino3403.y)

# New season
newData %<>% dplyr::mutate(
  date2=tsdatetime %m-% months(6),
  month2=month(date2),
  year2=year(date2)
)

# IDs
newData$areaid       <- factor(newData$areaid)
newData$ID.area      <- as.numeric(newData$areaid)
newData$ID.area1     <- as.numeric(newData$areaid)
newData$ID.area2     <- as.numeric(newData$areaid)
newData$ID.year      <- as.numeric(as.character(newData$year2))
newData$ID.year1     <- as.numeric(as.character(newData$year2))
newData$ID.month     <- as.numeric(as.character(newData$month2))
newData$ID.month1    <- as.numeric(as.character(newData$month2))

# Lagged obs
newData %<>% group_by(areaid) %>%
  dplyr::mutate(dengueL1=lag(dengue_cases, 1))

rm(myData, fore)


# ----------------
# Eof
# ----------------


