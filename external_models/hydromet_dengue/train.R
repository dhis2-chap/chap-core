args = commandArgs(trailingOnly=TRUE)
data_filename = args[1]
output_model_filename = args[2]
map_graph_file = args[3]
data_filename = 'training_data_til_2005.csv'
output_model_filename = "output/model1_config.RData"
source("external_models/hydromet_dengue/common_things.R")

inla.setOption(num.threads = "4:1")
formula <- Y ~ 1 +
  f(T1, replicate = S2, model = "rw1", scale.model = TRUE, cyclic = TRUE, constr = TRUE, hyper = precision.prior) +
  f(S1, model = "bym2", replicate = T2,
   #graph = "output/map.graph",
   graph = map_graph_file,
    scale.model = TRUE, hyper = precision.prior) +
  basis_tmin + basis_pdsi + urban_basis1_pdsi + Vu

model1 <- inla(formula, data = df, family = "nbinomial", offset = log(E),
               control.inla = list(strategy = 'adaptive'),
               control.compute = list(dic = TRUE, config = TRUE,
                                      cpo = TRUE, return.marginals = FALSE),
               control.fixed = list(correlation.matrix = TRUE,
                                        prec.intercept = 1, prec = 1),
                   control.predictor = list(link = 1, compute = TRUE),
                   verbose = FALSE)
model1 <- inla.rerun(model1)
save(model1, file = output_model_filename)
# model <- model1
#
# # define number of samples
# s <- 1000
#
# casestopred <- data$dengue_cases # response variable
# idx.pred <- which(data$year_index == yyear & data$month == mmonth)
# casestopred[idx.pred] <- NA # replace cases in year and month of interest to NA
# mpred <- length(idx.pred)
# df$Y <- casestopred
#
#     xx <- inla.posterior.sample(s, model)
#     xx.s <- inla.posterior.sample.eval(function(...) c(theta[1], Predictor[idx.pred]), xx)
#     y.pred <- matrix(NA, mpred, s)
#     for(s.idx in 1:s) {
#         xx.sample <- xx.s[, s.idx]
#         y.pred[, s.idx] <- rnbinom(mpred, mu = exp(xx.sample[-1]), size = xx.sample[1])
#     }
#     preds <- list(year = 2000 + yyear, month = mmonth, idx.pred = idx.pred,
#                   mean = apply(y.pred, 1, mean), median = apply(y.pred, 1, median),
#                   lci = apply(y.pred, 1, quantile, probs = c(0.025)),
#                   uci = apply(y.pred, 1, quantile, probs = c(0.975)))
#     save(preds, file = paste0("output/preds_",2000 + yyear, "_", mmonth, ".RData"))
# #}
