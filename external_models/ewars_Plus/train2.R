# Read in command line args filenames
args = commandArgs(trailingOnly=TRUE)
#data_filename = '/home/knut/Sources/climate_health/example_data/ewars_train_data.csv'
data_filename = args[1]
output_model_filename = args[2]

#install.packages("INLA")
library(INLA)

# our variables= 'rainsum' 'meantemperature'

mymodel <- function(formula, data = df, family = "nbinomial", config = FALSE)
{
  model <- inla(formula = formula, data = data, family = family, offset = log(E),
                control.inla = list(strategy = 'adaptive'),
                control.compute = list(dic = TRUE, config = config, cpo = TRUE, return.marginals = FALSE),
                control.fixed = list(correlation.matrix = TRUE, prec.intercept = 1, prec = 1),
                control.predictor = list(link = 1, compute = TRUE),
                verbose = F)
 # model <- inla.rerun(model)
  return(model)
}
# Y = number of cases
# E = pop.var.dat
# T1 = week
# T2 = year
# S1 = district

precision.prior <<- list(prec = list(prior = "pc.prec", param = c(0.5, 0.01)))
alarm_vars = c('rainsum', 'meantemperature')
baseformula <- Y ~ 1 + f(T1 , replicate = S1, model = "rw1", cyclic = TRUE, constr = TRUE, scale.model = TRUE,  hyper = precision.prior) +
f(S1 , replicate=T2, model = "iid")



formula0.1 <- eval(parse(text=paste0("update.formula(baseformula, ~. +",paste(alarm_vars, collapse ='+'),')')))

# TODO: add
df = read.table(data_filename, sep=',', header=TRUE)



model = mymodel(formula0.1, df, config = TRUE)
#model = inla.rerun(model)
save(model, file = output_model_filename)


# formula0.2 <- eval(parse(text=paste0("update.formula(baseformula, ~. +",paste(basis_var_n,collapse ='+'),')')))