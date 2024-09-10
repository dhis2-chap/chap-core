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
## Compute output files
##
## Version control: GitLab
## Initially created on 06 Dec 2019
##
##
## Written by: Felipe J Colon-Gonzalez
## For any problems with this code, please contact felipe.colon@lshtm.ac.uk
## 
## ############################################################################


# depend
source(file.path(scripts, "04_Fit_models.R"))

# sel
prd <- dplyr::filter(lead6, tsdatetime>=d1, tsdatetime <=d2, 
                     ensmember != "tsvalue_ensemble_00") %>%
  data.table()


# tsstd
prd$ensstd <- gsub("tsvalue", "tsstd", prd$ensmember)

o1 <- prd %>%
  dplyr::mutate(parametername="predicted_dengue_cases") %>%
  dplyr::select(parametername, tsdatetime, areaid, ensmember, fitted.mean) %>%
  tidyr::spread(ensmember, fitted.mean) %>%
  dplyr::arrange(tsdatetime) %>%
  data.table()

o1ll <- prd %>%
  dplyr::mutate(parametername="predicted_dengue_cases_lower_bound") %>%
  dplyr::select(parametername, tsdatetime, areaid, ensmember, fitted.lbci) %>%
  tidyr::spread(ensmember, fitted.lbci) %>%
  dplyr::arrange(tsdatetime) %>%
  data.table()

o1uu <- prd %>%
  dplyr::mutate(parametername="predicted_dengue_cases_upper_bound") %>%
  dplyr::select(parametername, tsdatetime, areaid, ensmember, fitted.ubci) %>%
  tidyr::spread(ensmember, fitted.ubci) %>%
  dplyr::arrange(tsdatetime) %>%
  data.table()

o2 <- prd %>%
  dplyr::mutate(parametername="predicted_dengue_cases") %>%
  dplyr::select(parametername, tsdatetime, areaid, ensstd, fitted.sdev) %>%
  tidyr::spread(ensstd, fitted.sdev) %>%
  dplyr::arrange(tsdatetime) %>%
  data.table()

o3 <- prd %>%
  dplyr::mutate(parametername="predicted_dengue_incidence") %>%
  dplyr::select(parametername, tsdatetime, areaid, ensmember, pi) %>%
  tidyr::spread(ensmember, pi) %>%
  dplyr::arrange(tsdatetime) %>%
  data.table()

o3ll <- prd %>%
  dplyr::mutate(parametername="predicted_dengue_incidence_lower_bound") %>%
  dplyr::select(parametername, tsdatetime, areaid, ensmember, pill) %>%
  tidyr::spread(ensmember, pill) %>%
  dplyr::arrange(tsdatetime) %>%
  data.table()

o3uu <- prd %>%
  dplyr::mutate(parametername="predicted_dengue_incidence_lower_bound") %>%
  dplyr::select(parametername, tsdatetime, areaid, ensmember, piuu) %>%
  tidyr::spread(ensmember, piuu) %>%
  dplyr::arrange(tsdatetime) %>%
  data.table()

o4 <- prd %>%
  dplyr::mutate(parametername="predicted_dengue_incidence") %>%
  dplyr::select(parametername, tsdatetime, areaid, ensstd, si) %>%
  tidyr::spread(ensstd, si) %>%
  dplyr::arrange(tsdatetime) %>%
  data.table()

o1b <- full_join(o1, o2) %>%
  gather(ensmember, value, -(parametername:areaid)) %>%
  separate(ensmember, c("key1", "key2", "key3"), "_")   %>%
  arrange(tsdatetime, key3) %>%
  mutate(ensmember=paste(key1, key2, key3, sep="_")) %>%
  dplyr::select(-one_of( c("key1", "key2", "key3"))) %>%
  data.table()

o2b <- full_join(o3, o4) %>%
  gather(ensmember, value, -(parametername:areaid)) %>%
  separate(ensmember, c("key1", "key2", "key3"), "_")   %>%
  arrange(tsdatetime, key3) %>%
  mutate(ensmember=paste(key1, key2, key3, sep="_")) %>%
  dplyr::select(-one_of( c("key1", "key2", "key3"))) %>%
  data.table()

# order key
key1 <- c(names(o1)[1:3], t(unique(o1b$ensmember))) 
key2 <- c(names(o1)[1:3], t(unique(o2b$ensmember))) 

o1b %<>% 
  spread(ensmember, value, convert=TRUE) %>%
  data.table()
o1b <- o1b[,key1, with=FALSE]

o2b %<>% 
  spread(ensmember, value, convert=TRUE) %>%
  data.table()
o2b <- o2b[,key2, with=FALSE]

o5 <- lead6 %>%
  group_by(areaid, tsdatetime) %>%
  dplyr::filter(tsdatetime >= d1) %>%
  dplyr::summarise(prob1sd=mean(prob.1sd, na.rm=TRUE),
                   ll1sd=min(prob.1sd, na.rm=TRUE),
                   uu1sd=max(prob.1sd, na.rm=TRUE),
                   prob2sd=max(prob.2sd, na.rm=TRUE),
                   ll2sd=min(prob.2sd, na.rm=TRUE),
                   uu2sd=max(prob.2sd, na.rm=TRUE),
                   probq75=mean(prob.q75, na.rm=TRUE),
                   llq75=min(prob.q75, na.rm=TRUE),
                   uuq75=max(prob.q75, na.rm=TRUE),
                   probq95=mean(prob.q95, na.rm=TRUE),
                   llq95=min(prob.q95, na.rm=TRUE),
                   uuq95=max(prob.q95, na.rm=TRUE)) %>%
  ungroup() %>%
  dplyr::mutate(areaid=as.numeric(as.character(areaid))) %>%
  arrange(tsdatetime, areaid)

# ofiles
write.csv(thresholds, file.path(output, 
                         paste0("epidemic_thresholds",
                                min(prd$tsdatetime), ".csv")),
          row.names=FALSE)

write.csv(o1b, file.path(output, 
                         paste0("predicted_dengue_cases_",
                                min(prd$tsdatetime), ".csv")),
          row.names=FALSE)

write.csv(o1ll, file.path(output, 
                          paste0("predicted_dengue_cases_lower_bound_",
                                 min(prd$tsdatetime), ".csv")),
          row.names=FALSE)

write.csv(o1uu, file.path(output, 
                          paste0("predicted_dengue_cases_upper_bound_",
                                 min(prd$tsdatetime), ".csv")),
          row.names=FALSE)

write.csv(o2b, file.path(output, 
                         paste0("predicted_dengue_incidence_",
                                min(prd$tsdatetime), ".csv")),
          row.names=FALSE)

write.csv(o3ll, file.path(output, 
                          paste0("predicted_dengue_incidence_lower_bound_",
                                 min(prd$tsdatetime), ".csv")),
          row.names=FALSE)

write.csv(o3uu, file.path(output, 
                          paste0("predicted_dengue_incidence_upper_bound_",
                                 min(prd$tsdatetime), ".csv")),
          row.names=FALSE)

write.csv(o5, file.path(output, 
                        paste0("probability_exceeding_thresholds_",
                               min(prd$tsdatetime), ".csv")),
          row.names=FALSE)

# -------------
# Eof
# -------------


