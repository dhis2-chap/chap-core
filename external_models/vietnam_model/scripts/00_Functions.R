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
## User-defined functions
##
## Version control: GitLab
## Initially created on 10 Sep 2019
##
## Depends on: none
##
## Written by: Felipe J Colon-Gonzalez
## For any problems with this code, please contact Felipe.Colon@lshtm.ac.uk
## 
## ############################################################################


obsFun <- function(x){
  paste0(file.path(input, "annual"), "/",
         list.files(path=file.path(input, "annual"),
                    pattern=x)) %>% 
    purrr::map(read_csv) %>% 
    purrr::map(slice, rep(1:n(), each=12)) %>%
    purrr::map(arrange, parametername, areaid, tsdatetime) %>%
    purrr::map(mutate, 
               y1=min(year(tsdatetime)),
               y2=max(year(tsdatetime)),
               tsdatetime=rep(seq.Date(
                 as.Date(paste(unique(y1), "02-01", sep="-")),
                 as.Date(paste(unique(y2), "12-31", sep="-"))+1,
                 by="months")-1,
                 times=length(unique(areaid)))) %>%
    purrr::reduce(rbind) %>%
    dplyr::distinct() %>%
    dplyr::select(-y1, -y2) %>%
    dplyr::arrange(parametername, areaid, tsdatetime) %>%
    dplyr::mutate(year=year(tsdatetime),
                  month=month(tsdatetime),
                  tsd=paste(year, sprintf("%02d", month), 
                            "01", sep="-"))
}


hist_data <- function(y){
  paste0(file.path(input), "/",
         list.files(path=file.path(input),
                    pattern=y)) %>% 
    purrr::map(read_csv) %>% 
    purrr::reduce(rbind) %>%
    dplyr::distinct() %>%
    dplyr::select(areaid, parametername, tsdatetime, tsvalue) %>%
    tidyr::spread(parametername, tsvalue) %>%
    dplyr::arrange(areaid, tsdatetime) %>%
    dplyr::filter(tsdatetime >=d1 & tsdatetime <= d2)
}

leads <- paste0('lead', 1:6)

fore_data <- function(j){
  paste0(file.path(live), "/",
         list.files(path=file.path(live),
                    pattern=j)) %>% 
    purrr::map(read_csv) %>% 
    purrr::map(distinct) %>%
    purrr::map(function(x) dplyr::select(x, -contains('tsstd'))) %>%
    purrr::map(function(x) dplyr::mutate(
      x,
      month=month(ymd(tsdatetime)),
      lead=paste0("lead", 
                  as.numeric(factor(tsdatetime)))) ) %>%
    purrr::reduce(rbind) %>%
    dplyr::filter(lead %in% leads) %>%
    dplyr::select(areaid, parametername, tsdatetime, contains('tsvalue')) %>%
    tidyr::gather(ensmember, tsvalue, -(areaid:tsdatetime)) %>%
    tidyr::spread(parametername, tsvalue) %>%
    dplyr::arrange(areaid, ensmember, tsdatetime) 
}

tsdate <- function(x){ 
  paste0(file.path(input), "/",
         list.files(path=file.path(input), pattern=x)) %>% 
    read_csv() %>%
    dplyr::distinct() %>%
    arrange(areaid, tsdatetime)  %>%
    dplyr::mutate(year=year(tsdatetime),
                  month=month(tsdatetime),
                  tsd=paste(year, sprintf("%02d", month), 
                            "01", sep="-")) %>%
    dplyr::select(areaid, tsdatetime, tsd)
}

a2m <- function(i, y){
  dplyr::filter(i, parametername==y & tsd %in% x$tsd) %>%
    dplyr::mutate(tsdatetime=x$tsdatetime) %>%
    dplyr::select(-c(year, month, tsd)) %>%
    dplyr::arrange(areaid, tsdatetime)
}


read_obs <- function(x){
  paste0(file.path(input), "/",
         list.files(path=file.path(input),
                    pattern=x)) %>% 
    read_csv()
}

sel_land <- function(x) {
  x %>% 
    dplyr::filter(tsdatetime >= d1 & tsdatetime <= d2) %>%
    dplyr::select(areaid, tsdatetime, periurban_landcover, 
                  rural_landcover, urban_landcover, population) %>%
    dplyr::mutate(areaid=factor(areaid)) %>%
    arrange(areaid, tsdatetime)
}

# ---------------
# BMA funs
# ---------------
get.mliks <- function(objlist) {
  as.vector(unlist(mclapply(objlist, function(X) {
    res <- try(X$mlik[1, 1]) 
    if(!is.numeric(res))
      res <- NA
    res
  })))
}

get.dics <- function(objlist) {
  as.vector(unlist(mclapply(objlist, function(X) {
    res <- try(X$dic$dic) 
    if(!is.numeric(res))
      res <- NA
    res
  })))
}


reweight <- function(x) {
  w <- exp(x - max(x, na.rm = TRUE))
  w <- w / sum(w, na.rm = TRUE)
  w
}

#Version of splinefun to return 0 outside x-range
mysplinefun <- function(
  x, y = NULL,
  method = c("fmm", "periodic", "natural", "monoH.FC")[1],
  ties = mean)
{
  xmin<-min(x)
  xmax<-max(x)
  
  ff <- splinefun(x=x, y=y, method=method, ties=ties)
  
  fff<-function(x, deriv)
  {
    
    sapply(x, function(X){
      if(X<xmin |X>xmax)
        return(0)
      else
        return(ff(X))
    })
  }
  
}

# Compute and re-scale marginal using splines
fitmarg<-function(x, logy, logp=0, usenormal=FALSE)
{
  if(!usenormal)
  {
    #func = splinefun(x, logy-max(logy))
    #post = exp(logp+func(x))
    
    logpost<-logy-max(logy)+logp-max(logp)
    post<-exp(logpost)
    
    post.func = splinefun(x, post)
    
    ## normalize
    xx = seq(min(x), max(x),  len = 1000)
    #z = sum(post.func(xx)) * diff(xx)[1]
    z=integrate(post.func, min(x), max(x))$value
    post.func = mysplinefun(x=xx, y=(post.func(xx) / z) )
    
  }
  else
  {
    
    xx = seq(min(x), max(x),  len = 1000)
    meany=sum(x*exp(logp+logy))/sum(exp(logp+logy))
    sdy=sqrt(   sum(((x-meany)^2)*exp(logp+logy))/sum(exp(logp+logy))  )
    
    post.func = function(x){dnorm(x, mean=meany, sd=sdy)}
    
  }
  
  return(post.func)
} 


# Fits marginal using BMA
fitmargBMA<-function(margs, ws, len=100)
{
  
  ws<-ws/sum(ws)
  
  xmin <- quantile((unlist(lapply(margs, function(X){min(X[,1])}))), 0.25)
  xmax <- quantile(unlist(lapply(margs, function(X){max(X[,1])})), 0.75)
  
  
  xx<-seq(xmin, xmax, len=len)
  
  
  margsws<-lapply(1:length(margs), function(i){
    func<-fitmarg(margs[[i]][,1], log(margs[[i]][,2]))
    ws[i]*func(xx)
  })
  
  margsws<-do.call(cbind, margsws)
  
  d<-data.frame(x=xx, y=apply(margsws, 1, sum))
  names(d)<-c("x", "y")
  
  return(d)
}

# Weighted sum of summary results in matrices
fitmatrixBMA<-function(models, ws, item)
{
  lmatrix<-lapply(models, function(X){X[[item]]})
  
  auxbma<-ws[1]*lmatrix[[1]]
  for(i in 2:length(lmatrix)){auxbma<-auxbma+ws[i]*lmatrix[[i]]}
  
  return(auxbma)
}


# Weighted sum of summary results in lists
fitlistBMA<-function(models, ws, item)
{
  nlist<-names(models[[1]][[item]])
  auxlist<-as.list(rep(NA, length(nlist)))
  names(auxlist)<-nlist
  
  for(ele in nlist)
  {
    lmatrix<-lapply(models, function(X){X[[item]][[ele]]})
    
    auxbma<-ws[1]*lmatrix[[1]]
    for(i in 2:length(lmatrix)){auxbma<-auxbma+ws[i]*lmatrix[[i]]}
    
    auxlist[[ele]]<-auxbma
  }
  return(auxlist)
}


# Weighted sum of marginals 
# The elements to handle are lists with different marginals (in matrix form)
fitmargBMA2<-function(models, ws, item)
{
  
  if(is.null(models[[1]][[item]]))
    return(NULL)
  
  nlist<-names(models[[1]][[item]])
  auxlist<-as.list(rep(NA, length(nlist)))
  names(auxlist)<-nlist
  
  for(ele in nlist)
  {
    lmatrix<-lapply(models, function(X){X[[item]][[ele]]})
    #xxr<-range(unlist(lapply(lmatrix, function(X){X[,1]})))
    
    #Compute range using a weighted sum of min and max
    xxr<-c(NA, NA)
    xxr[1]<-sum(ws*unlist(lapply(lmatrix,function(X){min(X[,1])})))
    xxr[2]<-sum(ws*unlist(lapply(lmatrix,function(X){max(X[,1])})))
    
    xx<-seq(xxr[1], xxr[2], length.out=81)
    
    auxbma<-rep(0, length(xx))
    for(i in 1:length(lmatrix)){
      auxspl<- mysplinefun(x=lmatrix[[i]][,1], y=lmatrix[[i]][,2])
      auxbma<-auxbma+ws[i]*auxspl(xx)
    }
    
    auxlist[[ele]]<-cbind(xx, auxbma)
  }
  return(auxlist)
}


#Returns a summary of the fitted.values using BMA
BMArho<-function(models, rho, logrhoprior=rep(1, length(rho)) )
{
  mlik<-unlist(lapply(models, function(X){X$mlik[1]}))
  
  post.func <- fitmarg(rho, mlik, logrhoprior)
  
  #	mlik.func = splinefun(rho, mlik - max(mlik))
  #
  #	post = exp(logrhoprior) * exp(mlik.func(rho))
  #	post.func = splinefun(rho, post)
  #
  #	## normalize
  #	rrho = seq(min(rho), max(rho),  len = 1000)
  #	z = sum(post.func(rrho)) * diff(rrho)[1]
  #	post.func = splinefun(rrho, post.func(rrho) / z)
  
  #Weights for BMA
  ws<-(post.func(rho))
  ws<-ws/sum(ws)
  
  #Fitted values with BMA
  
  #Compute BMA
  fvalues<-mclapply(1:length(models), function(X){
    ws[X]*models[[X]]$summary.fitted.values[,1]})
  fvalues<-data.frame(fvalues)
  fvalues<-apply(fvalues, 1, sum)
  
  return(fvalues)
}

#Performs BMA on a number of things
INLABMA<-function(models, rho, logrhoprior=rep(1, length(rho)), 
                  impacts=FALSE, usenormal=FALSE )
{
  
  #require(INLA)
  
  mlik<-unlist(lapply(models, function(X){X$mlik[1]}))
  post.func <- fitmarg(rho, mlik, logrhoprior, usenormal)
  
  #Weights for BMA
  ws<-(post.func(rho))
  ws<-ws/sum(ws)
  
  mfit<-list(rho=list())
  mfit$rho$marginal<-data.frame(x=seq(min(rho), max(rho), len=100))
  mfit$rho$marginal$y<-post.func(mfit$rho$marginal$x)
  mfit$rho$marginal<-as.matrix(mfit$rho$marginal)
  
  #Summary statistics
  margsum <- INLA::inla.zmarginal(mfit$rho$marginal, TRUE)
  mfit$rho$mean<-margsum$mean
  mfit$rho$sd<-margsum$sd
  mfit$rho$quantiles<-unlist(margsum[-c(1:2)])
  
  
  
  #Results which are stored in matrices
  mateff<-c("summary.fixed", "summary.lincomb", 
            #"summary.random", 
            "summary.linear.predictor", "summary.fitted.values")
  # summary.fixed
  # mfit$summary.fixed<-fitmatrixBMA(lapply(models,
  # function(X){X$summary.fixed}), ws)
  
  lmat<-mclapply(mateff, function(X){fitmatrixBMA(models, ws, X)})
  names(lmat)<-mateff
  
  mfit<-c(mfit, lmat)
  
  #Results stored in lists
  listeff<-c("dic", "cpo")
  
  leff<-mclapply(listeff, function(X){fitlistBMA(models, ws, X)})
  names(leff)<-listeff
  
  mfit<-c(mfit, leff)
  
  #Results are marginals
  listmarg<-c("marginals.fixed", "marginals.lincomb",
              "marginals.lincomb.derived", #"marginals.random",
              "marginals.linear.predictor", #"marginals.fitted.values",
              "marginals.hyperpar", "marginals.spde2.blc")
  
  margeff<-mclapply(listmarg, function(X){fitmargBMA2(models, ws, X)})
  names(margeff)<-listmarg
  
  mfit<-c(mfit, margeff)
  
  
  #Impacts
  mfit$impacts<-FALSE
  if(impacts)
  {
    mfit$impacts<-TRUE
    
    summimp<-c("summary.total.impacts", "summary.direct.impacts", 
               "summary.indirect.impacts")
    matsummimp<-mclapply(summimp, function(X){fitmatrixBMA(models, ws, X)})
    names(matsummimp)<-summimp
    mfit<-c(mfit, matsummimp)
    
    margimp<-c("marginals.total.impacts","marginals.direct.impacts",
               "marginals.indirect.impacts")
    lmargimp<-mclapply(margimp, function(X){fitmargBMA2(models, ws, X)})
    names(lmargimp)<-margimp
    
    mfit<-c(mfit, lmargimp)
    
    #Recompute impacts summaries
    mfit<-recompute.impacts(mfit)
    
  }
  
  return(mfit)
}

# inla 4 bma fun
inla4bma <- function(x, y) {
  inla(as.formula(x), 
       family="nbinomial", data=y,
       offset=log(population),
       control.predictor=list(compute=TRUE, link=1), 
       control.compute=list(dic=TRUE, waic=TRUE,
                            config=TRUE),
       control.inla=list(strategy="gaussian",
                         int.strategy="eb")) 
}
# EoF

