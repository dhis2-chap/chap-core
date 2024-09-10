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
## Fitting model superensemble using Bayesian Model Averaging
##
## Version control: GitLab
## Initially created on 12 Nov 2019
##
##
## Written by: Felipe J Colon-Gonzalez
## For any problems with this code, please contact felipe.colon@lshtm.ac.uk
## 
## ############################################################################

# depend
source(file.path(scripts, "03_Pre-processing.R"))

# geog
temp    <- poly2nb(myMap, queen=FALSE)
nb2INLA("vnm_graph", temp)
vnm.adj <- paste(getwd(), "/vnm_graph", sep="")

# Models
f0 <- dengue_cases ~ loglag +
  f(ID.area, model='bym', graph=vnm.adj, 
    adjust.for.con.comp=FALSE, constr=TRUE, 
    scale.model=TRUE, 
    hyper = list(prec.unstruct=list(prior='pc.prec',param=c(3, 0.01)),
                 prec.spatial=list(prior='pc.prec', param=c(3, 0.01)))) +
  f(ID.year, model='iid', 
    hyper=list(prec = list(prior='pc.prec',param = c(3, 0.01))),
    group=ID.area2, 
    control.group=list(model='iid',hyper = list(
      prec = list(prior='pc.prec',param=c(3, 0.01))))) + 
  f(ID.month1, model='ar1', 
    hyper = list(prec=list(prior='pc.prec',param=c(3, 0.01)),
                 rho = list(prior='pc.cor1', param = c(0.5, 0.75))),
    group=ID.area1, 
    control.group = list(model='iid',
                         hyper=list(prec=list(prior='pc.prec',
                                              param=c(3, 0.01)))))

f1 <- dengue_cases ~ loglag +
  # Spatial random effect
  f(ID.area, model='bym', graph=vnm.adj, 
    adjust.for.con.comp=FALSE, constr=TRUE, 
    scale.model=TRUE, 
    # Precision of unstructure random effects
    hyper = list(prec.unstruct=list(prior='pc.prec',param=c(3, 0.01)),
                 prec.spatial=list(prior='pc.prec', param=c(3, 0.01)))) +
  # Year random effect
  f(ID.year, model='iid', 
    hyper=list(prec = list(prior='pc.prec',param = c(3, 0.01))),
    group=ID.area2, 
    control.group=list(model='iid',hyper = list(
      prec = list(prior='pc.prec',param=c(3, 0.01))))) + 
  f(ID.month1, model='ar1', 
    hyper = list(prec=list(prior='pc.prec',param=c(3, 0.01)),
                 # Autocorrelation
                 rho = list(prior='pc.cor1', param = c(0.5, 0.75))),
    group=ID.area1, 
    control.group = list(model='iid',
                         hyper=list(prec=list(prior='pc.prec',
                                              param=c(3, 0.01))))) +
  # Remaing fixed effects
  periurban_landcover + urban_landcover +
  shum02 + wind_speed + dtr02 + nino3403 


f2 <- dengue_cases ~ loglag +
  f(ID.area, model='bym', graph=vnm.adj, 
    adjust.for.con.comp=FALSE, constr=TRUE, 
    scale.model=TRUE, 
    hyper = list(prec.unstruct=list(prior='pc.prec',param=c(3, 0.01)),
                 prec.spatial=list(prior='pc.prec', param=c(3, 0.01)))) +
  f(ID.year, model='iid', 
    hyper=list(prec = list(prior='pc.prec',param = c(3, 0.01))),
    group=ID.area2, 
    control.group=list(model='iid',hyper = list(
      prec = list(prior='pc.prec',param=c(3, 0.01))))) + 
  f(ID.month1, model='ar1', 
    hyper = list(prec=list(prior='pc.prec',param=c(3, 0.01)),
                 rho = list(prior='pc.cor1', param = c(0.5, 0.75))),
    group=ID.area1, 
    control.group = list(model='iid',
                         hyper=list(prec=list(prior='pc.prec',
                                              param=c(3, 0.01))))) + 
  # Remaing fixed effects
  periurban_landcover + urban_landcover + 
  shum02 + dtr02 

f3 <- dengue_cases ~ loglag +
  f(ID.area, model='bym', graph=vnm.adj, 
    adjust.for.con.comp=FALSE, constr=TRUE, 
    scale.model=TRUE, 
    hyper = list(prec.unstruct=list(prior='pc.prec',param=c(3, 0.01)),
                 prec.spatial=list(prior='pc.prec', param=c(3, 0.01)))) +
  f(ID.year, model='iid', 
    hyper=list(prec = list(prior='pc.prec',param = c(3, 0.01))),
    group=ID.area2, 
    control.group=list(model='iid',hyper = list(
      prec = list(prior='pc.prec',param=c(3, 0.01))))) + 
  f(ID.month1, model='ar1', 
    hyper = list(prec=list(prior='pc.prec',param=c(3, 0.01)),
                 rho = list(prior='pc.cor1', param = c(0.5, 0.75))),
    group=ID.area1, 
    control.group = list(model='iid',
                         hyper=list(prec=list(prior='pc.prec',
                                              param=c(3, 0.01))))) + 
  # Remaing fixed effects
  periurban_landcover + urban_landcover + 
  tmin02 + dtr02 + nino3403

f4 <- dengue_cases ~ loglag +
  f(ID.area, model='bym', graph=vnm.adj, 
    adjust.for.con.comp=FALSE, constr=TRUE, 
    scale.model=TRUE, 
    hyper = list(prec.unstruct=list(prior='pc.prec',param=c(3, 0.01)),
                 prec.spatial=list(prior='pc.prec', param=c(3, 0.01)))) +
  f(ID.year, model='iid', 
    hyper=list(prec = list(prior='pc.prec',param = c(3, 0.01))),
    group=ID.area2, 
    control.group=list(model='iid',hyper = list(
      prec = list(prior='pc.prec',param=c(3, 0.01))))) + 
  f(ID.month1, model='ar1', 
    hyper = list(prec=list(prior='pc.prec',param=c(3, 0.01)),
                 rho = list(prior='pc.cor1', param = c(0.5, 0.75))),
    group=ID.area1, 
    control.group = list(model='iid',
                         hyper=list(prec=list(prior='pc.prec',
                                              param=c(3, 0.01))))) + 
  # Remaing fixed effects
  periurban_landcover + urban_landcover + 
  tmin02 + tmax02

f5 <- dengue_cases ~ loglag +
  f(ID.area, model='bym', graph=vnm.adj, 
    adjust.for.con.comp=FALSE, constr=TRUE, 
    scale.model=TRUE, 
    hyper = list(prec.unstruct=list(prior='pc.prec',param=c(3, 0.01)),
                 prec.spatial=list(prior='pc.prec', param=c(3, 0.01)))) +
  f(ID.year, model='iid', 
    hyper=list(prec = list(prior='pc.prec',param = c(3, 0.01))),
    group=ID.area2, 
    control.group=list(model='iid',hyper = list(
      prec = list(prior='pc.prec',param=c(3, 0.01))))) + 
  f(ID.month1, model='ar1', 
    hyper = list(prec=list(prior='pc.prec',param=c(3, 0.01)),
                 rho = list(prior='pc.cor1', param = c(0.5, 0.75))),
    group=ID.area1, 
    control.group = list(model='iid',
                         hyper=list(prec=list(prior='pc.prec',
                                              param=c(3, 0.01))))) + 
  # Remaing fixed effects
  periurban_landcover + urban_landcover + 
  wind_speed

# Create lag1 of dengue cases
newData %<>% group_by(areaid) %>%
  dplyr::mutate(lag1=lag(dengue_cases, 1))

# Dataset + lead1
lead1 <- dplyr::filter(newData, tsdatetime <= d1)

# Dengue data for lag1 -> last obs for lag0
pred0 <- lead1 %>%
  dplyr::select(areaid, dengue_cases, tsdatetime) %>%
  dplyr::filter(tsdatetime == lubridate::rollback(d1)) %>%
  dplyr::rename(lag1=dengue_cases)
pred0$tsdatetime <- max(lead1$tsdatetime)

lead1 %<>% full_join(pred0, by=c('areaid', 'tsdatetime')) %>%
  dplyr::mutate(lag1 = coalesce(lag1.x, lag1.y)) %>% 
  dplyr::select(-lag1.x, -lag1.y) 

# Compute log of lag1
lead1$loglag <- log1p(lead1$lag1)

# fit
mod1.1  <- inla4bma(f0, lead1) 

# mean fitted
lead1$mu <- mod1.1$summary.fitted.values[,'mean']

rm(mod1.1)

# Separate lead2
lead2 <- dplyr::filter(newData, tsdatetime <= ceiling_date(
  d1 %m+% months(1), 'month') - days(1))

# Dengue data for lag2
pred1 <- lead1 %>%
  dplyr::select(areaid, mu, tsdatetime) %>%
  dplyr::filter(tsdatetime == d1) %>%
  dplyr::group_by(areaid, tsdatetime) %>%
  dplyr::summarise(mu=mean(mu)) %>%
  dplyr::rename(lag1=mu)
pred1$tsdatetime <- max(lead2$tsdatetime)

lead2 %<>% full_join(pred0, by=c('areaid', 'tsdatetime')) %>%
  dplyr::mutate(lag1 = coalesce(lag1.x, lag1.y)) %>% 
  dplyr::select(-lag1.x, -lag1.y) %>%
  full_join(pred1, by=c('areaid', 'tsdatetime')) %>%
  dplyr::mutate(lag1 = coalesce(lag1.x, lag1.y)) %>% 
  dplyr::select(-lag1.x, -lag1.y)

# Compute log of lag2
lead2$loglag <- log1p(lead2$lag1)

# fit
mod2.1  <- inla4bma(f0, lead2) 

# mean fitted
lead2$mu <- mod2.1$summary.fitted.values[,'mean']

rm(mod2.1)

# Separate lead3
lead3 <- dplyr::filter(newData, tsdatetime <= ceiling_date(
  d1 %m+% months(2), 'month') - days(1))

# Dengue data for lag3
pred2 <- lead2 %>%
  dplyr::select(areaid, mu, tsdatetime) %>%
  dplyr::filter(tsdatetime ==  ceiling_date(d1 %m+% months(1), 'month') - 
                  days(1)) %>%
  dplyr::group_by(areaid, tsdatetime) %>%
  dplyr::summarise(mu=mean(mu)) %>%
  dplyr::rename(lag1=mu)
pred2$tsdatetime <- max(lead3$tsdatetime)

lead3 %<>%
  full_join(pred0, by=c('areaid', 'tsdatetime')) %>%
  dplyr::mutate(lag1 = coalesce(lag1.x, lag1.y)) %>% 
  dplyr::select(-lag1.x, -lag1.y) %>%
  full_join(pred1, by=c('areaid', 'tsdatetime')) %>%
  dplyr::mutate(lag1 = coalesce(lag1.x, lag1.y)) %>% 
  dplyr::select(-lag1.x, -lag1.y) %>%
  full_join(pred2, by=c('areaid', 'tsdatetime')) %>%
  dplyr::mutate(lag1 = dplyr::coalesce(lag1.x, lag1.y)) %>% 
  dplyr::select(-lag1.x, -lag1.y)

# Compute log of lag
lead3$loglag <- log1p(lead3$lag1)

# fit
mod3.1 <- inla4bma(f0, lead3) 

# mean fitted
lead3$mu <- mod3.1$summary.fitted.values[,'mean']

rm(mod3.1)

lead4 <- dplyr::filter(newData, tsdatetime <= ceiling_date(
  d1 %m+% months(3), 'month') - days(1))

# Dengue data for lag3
pred3 <- lead3 %>%
  dplyr::select(areaid, mu, tsdatetime) %>%
  dplyr::filter(tsdatetime ==  ceiling_date(d1 %m+% months(2), 'month') - 
                  days(1)) %>%
  dplyr::group_by(areaid, tsdatetime) %>%
  dplyr::summarise(mu=mean(mu)) %>%
  dplyr::rename(lag1=mu)
pred3$tsdatetime <- max(lead4$tsdatetime)

lead4 %<>% full_join(pred0, by=c('areaid', 'tsdatetime')) %>%
  dplyr::mutate(lag1 = coalesce(lag1.x, lag1.y)) %>% 
  dplyr::select(-lag1.x, -lag1.y) %>%
  full_join(pred1, by=c('areaid', 'tsdatetime')) %>%
  dplyr::mutate(lag1 = coalesce(lag1.x, lag1.y)) %>% 
  dplyr::select(-lag1.x, -lag1.y) %>%
  full_join(pred2, by=c('areaid', 'tsdatetime')) %>%
  dplyr::mutate(lag1 = coalesce(lag1.x, lag1.y)) %>% 
  dplyr::select(-lag1.x, -lag1.y) %>%
  full_join(pred3, by=c('areaid', 'tsdatetime')) %>%
  dplyr::mutate(lag1 = coalesce(lag1.x, lag1.y)) %>% 
  dplyr::select(-lag1.x, -lag1.y)

# Compute log of lag4
lead4$loglag <- log1p(lead4$lag1)

# fit
mod4.1  <- inla4bma(f0, lead4) 

# mean fitted
lead4$mu <- mod4.1$summary.fitted.values[,'mean']

rm(mod4.1)

# Separate lead5
lead5 <- dplyr::filter(newData, tsdatetime <= ceiling_date(
  d1 %m+% months(4), 'month') - days(1))

# Dengue data for lag3
pred4 <- lead4 %>%
  dplyr::select(areaid, mu, tsdatetime) %>%
  dplyr::filter(tsdatetime == ceiling_date(
    d1 %m+% months(3), 'month') - days(1)) %>%
  dplyr::group_by(areaid, tsdatetime) %>%
  dplyr::summarise(mu=mean(mu)) %>%
  dplyr::rename(lag1=mu)
pred4$tsdatetime <- max(lead5$tsdatetime)

lead5 %<>% full_join(pred0, by=c('areaid', 'tsdatetime')) %>%
  dplyr::mutate(lag1 = coalesce(lag1.x, lag1.y)) %>% 
  dplyr::select(-lag1.x, -lag1.y) %>%
  full_join(pred1, by=c('areaid', 'tsdatetime')) %>%
  dplyr::mutate(lag1 = coalesce(lag1.x, lag1.y)) %>% 
  dplyr::select(-lag1.x, -lag1.y) %>%
  full_join(pred2, by=c('areaid', 'tsdatetime')) %>%
  dplyr::mutate(lag1 = coalesce(lag1.x, lag1.y)) %>% 
  dplyr::select(-lag1.x, -lag1.y) %>%
  full_join(pred3, by=c('areaid', 'tsdatetime')) %>%
  dplyr::mutate(lag1 = coalesce(lag1.x, lag1.y)) %>% 
  dplyr::select(-lag1.x, -lag1.y) %>%
  full_join(pred4, by=c('areaid', 'tsdatetime')) %>%
  dplyr::mutate(lag1 = coalesce(lag1.x, lag1.y)) %>% 
  dplyr::select(-lag1.x, -lag1.y)

# Compute log of lag2
lead5$loglag <- log1p(lead5$lag1)

# fit
mod5.1  <- inla4bma(f0, lead5) 

# mean fitted
lead5$mu <- mod5.1$summary.fitted.values[,'mean']

rm(mod5.1)

# Separate lead6
lead6 <- dplyr::filter(newData, tsdatetime <= ceiling_date(
  d1 %m+% months(5), 'month') - days(1))

# Dengue data for lag3
pred5 <- lead5 %>%
  dplyr::select(areaid, mu, tsdatetime) %>%
  dplyr::filter(tsdatetime == ceiling_date(
    d1 %m+% months(4), 'month') - days(1)) %>%
  dplyr::group_by(areaid, tsdatetime) %>%
  dplyr::summarise(mu=mean(mu)) %>%
  dplyr::rename(lag1=mu)
pred5$tsdatetime <- max(lead6$tsdatetime)

lead6 %<>% full_join(pred0, by=c('areaid', 'tsdatetime')) %>%
  dplyr::mutate(lag1 = coalesce(lag1.x, lag1.y)) %>% 
  dplyr::select(-lag1.x, -lag1.y) %>%
  full_join(pred1, by=c('areaid', 'tsdatetime')) %>%
  dplyr::mutate(lag1 = coalesce(lag1.x, lag1.y)) %>% 
  dplyr::select(-lag1.x, -lag1.y) %>%
  full_join(pred2, by=c('areaid', 'tsdatetime')) %>%
  dplyr::mutate(lag1 = coalesce(lag1.x, lag1.y)) %>% 
  dplyr::select(-lag1.x, -lag1.y) %>%
  full_join(pred3, by=c('areaid', 'tsdatetime')) %>%
  dplyr::mutate(lag1 = coalesce(lag1.x, lag1.y)) %>% 
  dplyr::select(-lag1.x, -lag1.y) %>%
  full_join(pred4, by=c('areaid', 'tsdatetime')) %>%
  dplyr::mutate(lag1 = coalesce(lag1.x, lag1.y)) %>% 
  dplyr::select(-lag1.x, -lag1.y) %>%
  full_join(pred5, by=c('areaid', 'tsdatetime')) %>%
  dplyr::mutate(lag1 = coalesce(lag1.x, lag1.y)) %>% 
  dplyr::select(-lag1.x, -lag1.y)

# Compute log of lag2
lead6$loglag <- log1p(lead6$lag1)

# fit
mod6.1  <- inla4bma(f1, lead6) 
mod6.2  <- inla4bma(f2, lead6) 
mod6.3  <- inla4bma(f3, lead6) 
mod6.4  <- inla4bma(f4, lead6) 
mod6.5  <- inla4bma(f5, lead6) 

###########################################################
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# NOT TO RUN 
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
###########################################################
lead6$mu1  <- mod6.1$summary.fitted.values[,'mean']
lead6$mu2  <- mod6.2$summary.fitted.values[,'mean']
lead6$mu3  <- mod6.3$summary.fitted.values[,'mean']
lead6$mu4  <- mod6.4$summary.fitted.values[,'mean']
lead6$mu5  <- mod6.5$summary.fitted.values[,'mean']

dfr <- lead6 %>%
  dplyr::select(areaid, tsdatetime, dengue_cases, 
                mu1, mu2, mu3, mu4, mu5)

fwrite(dfr, file.path(output,
                      paste0("fitted_values_individual_models_",
                             d1, ".csv")))
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
###########################################################

# ---------------------
# BMA
# ---------------------
myModels <- list(mod6.1, mod6.2, mod6.3, mod6.4, mod6.5)

mliks   <- get.mliks(myModels)
dics    <- get.dics(myModels)
w1      <- reweight(mliks) * 0.5
w2      <- reweight(dics)  * 0.5
weights <- w1 + w2

# Create model superensemble with BMA
ensemble <- fitmargBMA2(myModels, ws=weights,
                        item='marginals.linear.predictor')

f1 <- lead6$mu1 * weights[1]
f2 <- lead6$mu2 * weights[2]
f3 <- lead6$mu3 * weights[3]
f4 <- lead6$mu4 * weights[4]
f5 <- lead6$mu5 * weights[5]
fitted.mean <- f1 + f2 + f3 + f4 + f5

lb1 <- mod6.1$summary.fitted.values[,"0.025quant"] * weights[1]
lb2 <- mod6.2$summary.fitted.values[,"0.025quant"] * weights[2]
lb3 <- mod6.3$summary.fitted.values[,"0.025quant"] * weights[3]
lb4 <- mod6.4$summary.fitted.values[,"0.025quant"] * weights[4]
lb5 <- mod6.5$summary.fitted.values[,"0.025quant"] * weights[5]
fitted.lbci <- lb1 + lb2 + lb3 + lb4 + lb5

ub1 <- mod6.1$summary.fitted.values[,"0.975quant"] * weights[1]
ub2 <- mod6.2$summary.fitted.values[,"0.975quant"] * weights[2]
ub3 <- mod6.3$summary.fitted.values[,"0.975quant"] * weights[3]
ub4 <- mod6.4$summary.fitted.values[,"0.975quant"] * weights[4]
ub5 <- mod6.5$summary.fitted.values[,"0.975quant"] * weights[5]
fitted.ubci <- ub1 + ub2 + ub3 + ub4 + ub5

sd1 <- mod6.1$summary.fitted.values[,"sd"] * weights[1]
sd2 <- mod6.2$summary.fitted.values[,"sd"] * weights[2]
sd3 <- mod6.3$summary.fitted.values[,"sd"] * weights[3]
sd4 <- mod6.4$summary.fitted.values[,"sd"] * weights[4]
sd5 <- mod6.5$summary.fitted.values[,"sd"] * weights[5]
fitted.sdev <- sd1 + sd2 + sd3 + sd4 + sd5

lead6$fitted.mean <- fitted.mean
lead6$fitted.lbci <- fitted.lbci
lead6$fitted.ubci <- fitted.ubci
lead6$fitted.sdev <- fitted.sdev
lead6$pi          <- (lead6$fitted.mean / lead6$population) * 1e5
lead6$piuu        <- (lead6$fitted.ubci / lead6$population) * 1e5
lead6$pill        <- (lead6$fitted.lbci / lead6$population) * 1e5
lead6$si          <- (lead6$fitted.sdev / lead6$population) * 1e5

mu1 <- lead6$fitted.mean
E1 <- (lead6$dengue_cases - mu1) / sqrt(mu1)
N <- nrow(lead6)
p <- 6 # Max number coef
dispersion <- sum(E1^2, na.rm=TRUE) / (N - p)
dispersion <- data.table(date=j, model="ensmeble", disper=dispersion)

# residuals
png(file.path(output, paste0("residuals_", j, ".png")),
    width=5, height=3.5, units="in", res=150)
par(mar=c(5,5,3,3))
plot(lead6$fitted.mean, E1, main=paste0("Residuals - ", j),
     xlab="Fitted", yla="Normalised residuals", cex.lab=1.2,
     cex.main=1.5, pch=20, col="gray50", cex=0.5)
dev.off()

# margs
marg.fitted <- lapply(ensemble, function(x) {
  inla.tmarginal(exp, x)
})

fdate <- ymd(max(lead6$tsdatetime)-days(1800))

thr1 <- lead6 %>%
  dplyr::filter(tsdatetime < d1) %>%
  dplyr::filter(tsdatetime >= fdate) %>%
  group_by(areaid, month) %>%
  dplyr::summarise(mean_cases=mean(dengue_cases, na.rm=TRUE),
                   sd_cases=sd(dengue_cases, na.rm=TRUE)) %>%
  dplyr::mutate(epi.1d=mean_cases + sd_cases,
                epi.2d=mean_cases + 2*sd_cases)

# If threshold is to low, replace accordingly
thr1$epi.1d[thr1$epi.1d < 3] <- 3
thr1$epi.2d[thr1$epi.2d < 5] <- 5

lead6 <- inner_join(lead6, thr1)

thr2 <- lead6 %>%
  dplyr::filter(tsdatetime < d1) %>%
  group_by(areaid, month) %>%
  dplyr::summarise(epi.q25=quantile(dengue_cases, probs=0.25, na.rm=TRUE),
                   epi.q50=quantile(dengue_cases, probs=0.50, na.rm=TRUE),
                   epi.q75=quantile(dengue_cases, probs=0.75, na.rm=TRUE),
                   epi.q95=quantile(dengue_cases, probs=0.95, na.rm=TRUE))

# If threshold is to low, replace accordingly
thr2$epi.q25[thr2$epi.q25 < 3]  <- 3
thr2$epi.q50[thr2$epi.q50 < 5]  <- 5
thr2$epi.q75[thr2$epi.q75 < 7]  <- 7
thr2$epi.q95[thr2$epi.q95 < 10] <- 10

lead6 <- inner_join(lead6, thr2)

thresholds <- lead6 %>%
  dplyr::select(areaid, tsdatetime, month,
                mean_cases, sd_cases,
                epi.1d, epi.2d, epi.q25,
                epi.q50, epi.q75, epi.q95) %>%
  dplyr::filter(tsdatetime >= d1)

fwrite(thresholds, file.path(output,
                             paste0("epidemic_thresholds",
                                    d1, ".csv")),
       row.names=FALSE)

lead6$prob.1sd <- NA
lead6$prob.2sd <- NA
lead6$prob.q75 <- NA
lead6$prob.q95 <- NA

start <- min(which(lead6$ensmember!='tsvalue_ensemble_00'))
end   <- max(which(lead6$ensmember!='tsvalue_ensemble_00'))
index <- seq(start, end, by=1)

for(i in index){
  
  marg <- marg.fitted[[i]]
  
  lead6$prob.1sd[i] <- 1 - inla.pmarginal(
    q = lead6$epi.1d[[i]],
    marginal = marg)
  lead6$prob.2sd[i] <- 1 - inla.pmarginal(
    q = lead6$epi.2d[[i]],
    marginal = marg)
  lead6$prob.q75[i] <- 1 - inla.pmarginal(
    q = lead6$epi.q75[[i]],
    marginal = marg)
  lead6$prob.q95[i] <- 1 - inla.pmarginal(
    q = lead6$epi.q95[[i]],
    marginal = marg)
  
}

# -------------
# Eof
# -------------


