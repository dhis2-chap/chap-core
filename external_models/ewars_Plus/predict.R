# !/usr/bin/env Rscript
library(INLA)
args = commandArgs(trailingOnly=TRUE)
data_filename = '/tmp/tmptet26o2v'
model_filename = 'ewars_Plus.model'
out_filename = 'tmp.csv'
load(file = model_filename)
s <- 2
ss = inla.posterior.sample(s, model)
df <- read.table(data_filename, sep=',', header=TRUE)
df = df[1:5,]
casestopred <- df$Y # response variable
idx.pred <- which(is.na(casestopred))
print(idx.pred)
mpred <- length(idx.pred)
xx <- inla.posterior.sample(s, model)  # This samples parameters of the model

xx.s <- inla.posterior.sample.eval(
function(...) c(theta[1], Predictor[idx.pred]), xx) # This extracts the expected value and hyperparameters from the samples
print(xx.s)
y.pred <- matrix(NA, mpred, s)
for (s.idx in 1:s){
  xx.sample <- xx.s[, s.idx]
  for (p.idx in 1:mpred){
    y.pred[p.idx, s.idx] <- rnbinom(1,  mu = exp(xx.sample[1+p.idx]), size = xx.sample[1])
  }
}
predictions = y.pred[, 1]
df$Y[idx.pred] = predictions
write.csv(df, out_filename)
