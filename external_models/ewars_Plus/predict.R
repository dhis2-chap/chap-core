# !/usr/bin/env Rscript
library(INLA)
args = commandArgs(trailingOnly=TRUE)
#model_filename = 'ewars_Plus.model'
#model_filename = args[1]
#data_filename = '/home/knut/Sources/climate_health/example_data/ewars_predict.csv'#args[2]#
#out_filename = 'tmp.csv'# args[3]# 'tmp.csv'
model_filename = args[1]
data_filename = args[2]
out_filename = args[3]

load(file = model_filename)
s <- 100
df = read.csv(data_filename)
ss = inla.posterior.sample(s, model)
df <- read.table(data_filename, sep=',', header=TRUE)
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

new.df = df[idx.pred,]
new.df$mean = rowMeans(y.pred)

new.df$std = apply(y.pred, 1, sd)
new.df$max = apply(y.pred, 1, max)
new.df$min = apply(y.pred, 1, min)

new.df$quantile_low = apply(y.pred, 1, function(row) quantile(row, 0.1))
new.df$median = apply(y.pred, 1, function(row) quantile(row, 0.5))
new.df$quantile_high = apply(y.pred, 1, function(row) quantile(row, 0.9))
print(new.df)
#predictions = y.pred[, 1]
#df
#df$Y[idx.pred] = predictions
write.csv(new.df, out_filename)
