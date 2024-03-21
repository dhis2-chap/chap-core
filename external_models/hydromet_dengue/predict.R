# !/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)
data_filename = args[1]
model_filename = args[2]
# model_filename = "output/model1_config.RData"
#data_filename = 'masked_data_til_2005.csv'

source("common_things.R")
# Get rows with NA in dengue_cases
masked_data = data[is.na(data$dengue_cases),]
#mask = data$dengue_cases == NA
# print(mask)
#masked_data = data[mask]
unique.years = unique(masked_data$year)
print(unique.years)
unique.months = unique(masked_data$month)
print(unique.months)
stopifnot(length(unique.years) == 1)
stopifnot(length(unique.months) == 1)
yyear = unique.years[1]
mmonth = unique.months[1]
inla.setOption(num.threads = "4:1")

load(file = model_filename)
model <- model1
s <- 1000
casestopred <- data$dengue_cases # response variable
idx.pred <- which(casestopred == NA)
mpred <- length(idx.pred)
df$Y <- casestopred

    xx <- inla.posterior.sample(s, model)
    xx.s <- inla.posterior.sample.eval(function(...) c(theta[1], Predictor[idx.pred]), xx)
    y.pred <- matrix(NA, mpred, s)
    for(s.idx in 1:s) {
        xx.sample <- xx.s[, s.idx]
        y.pred[, s.idx] <- rnbinom(mpred, mu = exp(xx.sample[-1]), size = xx.sample[1])
    }
    preds <- list(year = 2000 + yyear, month = mmonth, idx.pred = idx.pred,
                  mean = apply(y.pred, 1, mean), median = apply(y.pred, 1, median),
                  lci = apply(y.pred, 1, quantile, probs = c(0.025)),
                  uci = apply(y.pred, 1, quantile, probs = c(0.975)))
    save(preds, file = paste0("output/preds_",2000 + yyear, "_", mmonth, ".RData"))

