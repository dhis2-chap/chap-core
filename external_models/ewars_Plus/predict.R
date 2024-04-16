# !/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)
data_filename = args[1]
model_filename = args[2]
load(file = model_filename)
s <- 1000
inla.posterior.sample(s, model)


casestopred <- data$dengue_cases # response variable
idx.pred <- which(casestopred == NA)
mpred <- length(idx.pred)
df$Y <- casestopred
xx <- inla.posterior.sample(s, model) #This samples parameters of the model
xx.s <- inla.posterior.sample.eval(function(...) c(theta[1], Predictor[idx.pred]), xx) # This extracts the expected value and hyperparameters from the samples
y.pred <- matrix(NA, mpred, s)
s.idx = -1 # predict the last one
xx.sample <- xx.s[, s.idx]
y.pred[, s.idx] <- rnbinom(mpred,  mu = exp(xx.sample[-1]), size = xx.sample[1])
    }
    preds <- list(year = 2000 + yyear, month = mmonth, idx.pred = idx.pred,
                  mean = apply(y.pred, 1, mean), median = apply(y.pred, 1, median),
                  lci = apply(y.pred, 1, quantile, probs = c(0.025)),
                  uci = apply(y.pred, 1, quantile, probs = c(0.975)))
    save(preds, file = paste0("output/preds_",2000 + yyear, "_", mmonth, ".RData"))

