
#nlag<-12

#alarm_var<-"rainsum"
#dat<-"data_augmented"

get_cross_basis<-function(alarm_var,dat){
  dat.N<-get(dat)[,c('district',alarm_var)]
  names(dat.N)[2]<-"alarm_var"
  
  if(str_detect(alarm_var,"rain|prec")){
    lag_knots<-equalknots(0:nlag, 3)
  }else{
    lag_knots<-nlag/2
  }
  
  lag_var<- tsModel::Lag(dat.N$alarm_var, group = dat.N$district, k = 0:nlag)
  
  basis_var <- crossbasis(lag_var,
                            argvar = list(fun = "ns", knots = equalknots(dat.N$alarm_var, 2)),
                            arglag = list(fun = "ns", knots = lag_knots))
  
  colnames(basis_var) = paste0("basis_",alarm_var,'.', colnames(basis_var))
  basis.var<-list(basis_var)
  names(basis.var)<-paste0("basis_",alarm_var)
  basis.var
}


precision.prior <<- list(prec = list(prior = "pc.prec", param = c(0.5, 0.01)))


mymodel <- function(formula, data = df, family = "nbinomial", config = FALSE)
  
{
  model <- inla(formula = formula, data = data, family = family, offset = log(E),
                control.inla = list(strategy = 'adaptive'), 
                control.compute = list(dic = TRUE, config = config, 
                                       cpo = TRUE, return.marginals = FALSE),
                control.fixed = list(correlation.matrix = TRUE, 
                                     prec.intercept = 1, prec = 1),
                control.predictor = list(link = 1, compute = TRUE), 
                verbose = F)
 # model <- inla.rerun(model)
  return(model)
}

mymodel2 <- function(formula, data = df, family = "nbinomial", config = FALSE)
  
{
  model <- inla(formula = formula, data = data, family = family, offset = log(E),
                control.inla = list(strategy = 'adaptive'), 
                control.compute = list(dic = TRUE, config = config, 
                                       cpo = TRUE, return.marginals = T),
                control.fixed = list(correlation.matrix = TRUE, 
                                     prec.intercept = 1, prec = 1),
                control.predictor = list(link = 1, compute = TRUE), 
                verbose = F,
                num.threads="8:1")
  #model <- inla.rerun(model)
  model
}

mymodel3 <- function(formula, data = df, family = "nbinomial", config = FALSE)
  
{
  model <- inla(formula = formula, data = data, family = family, offset = log(E),
                control.inla = list(strategy = 'adaptive'), 
                control.compute = list(dic = TRUE, config = config, 
                                       cpo = TRUE, return.marginals = T),
                control.fixed = list(correlation.matrix = TRUE, 
                                     prec.intercept = 1, prec = 1),
                control.predictor = list(link = 1, compute = TRUE), 
                control.mode =list(restart=T,theta=theta_beg),
                verbose = F)
                #num.threads="6:1")
  #model <- inla.rerun(model)
  model
}
names(inla.models()$likelihood)

mymodelZIF <- function(formula, data = df, family = "zeroinflatednbinomial0", config = FALSE)
  
{
  model <- inla(formula = formula, data = data, family = family, offset = log(E),
                control.inla = list(strategy = 'adaptive'), 
                control.compute = list(dic = TRUE, config = T, 
                                       cpo = TRUE, return.marginals = F),
                control.fixed = list(correlation.matrix = TRUE, 
                                     prec.intercept = 1, prec = 1),
                control.predictor = list(link = 1, compute = TRUE), 
                verbose = T,
                num.threads="8:1")
  # model <- inla.rerun(model)
  return(model)
}


## UI ouput function

get_UI_update_d<-function(obj.plots,panel_Update,shinyOutputType,cols,
                          out_ref){
  
  dat_cont_UI<-data.frame(var=obj.plots,num=1:length(obj.plots))
  
  #length(covar_to_Plot)/2
  
  max_groups<-ceiling(length(obj.plots)/cols)
  
  dat_cont_UI$group=expand.grid(a=1:cols,b=1:max_groups)[,2][1:length(obj.plots)]
  
  
  dat_cont_UI$plot_Output<-paste0(out_ref,"_plot_",dat_cont_UI$num)
  
  if(cols==1){
    column_size<-12
  }else{
    column_size<-6 
  }
  #t<-1
  get_Fluid<-function(t){
    dat_For_Ui1<-dat_cont_UI %>% dplyr::filter(group==t)
    
    if(cols==2){
      
      dd<-paste("fluidRow(",
                paste0('column(',column_size,',',shinyOutputType,'("',dat_For_Ui1$plot_Output[1],'")),'),
                paste0('column(',column_size,',offset =0,',shinyOutputType,'("',dat_For_Ui1$plot_Output[2],'")))')
      )
    }else{
      dd<-paste("fluidRow(",
                paste0('column(',column_size,',',shinyOutputType,'("',dat_For_Ui1$plot_Output[1],'")))')
      )
    }
    #DT::dataTableOutput("Uploaded_data")
    dd
  }
  
  all_out<-foreach(a=1:max_groups,.combine =c)%do% get_Fluid(a)
  
  
  par_text0<-parse(text=
                     paste0('tabPanel("',panel_Update,'",',
                            paste(all_out,collapse =',')
                            , ')'))
  
  par_text0
  
}

## update log plots UI

get_slider_input_lag<-function(var,pp){
  if(pp==1){
    offset_val<-0
  }else{
    offset_val<-4
  }
  
  aa<-tribble(~var,
              "column(12,",
              paste0('offset=',offset_val,","),
              paste0('sliderInput(inputId = "',var,'_',pp,'",'),
              paste0('label = "',"Var_",pp,'",'),
              paste0('min=',min,","),
              paste0('max=',max,","),
              paste0('value=',vals_beg[pp],","),
              paste0('step=',step.val,')'),
              
  )
  paste(aa$var,collapse =' ')
}
#all_basis_vars$basis_rainsum

get_fluid_slice_Output<-function(p,var.Obj){
  var<-get(var.Obj)[p]
    #dat_slider<-var_names_New_model()$dat
  min<-round(min(dat_slider[,var],na.rm=T),0)
  min.1<-min(dat_slider[,var],na.rm=T)
  max<-round(max(dat_slider[,var],na.rm=T),0)
  max.1<-max(dat_slider[,var],na.rm=T)
  
  val.slid<-pretty(min:max,50)
  id.rm<-which(val.slid<min.1|val.slid>max.1)
  if(length(id.rm)>0){
    val.sliders<-val.slid[-which(val.slid<min.1|val.slid>max.1)]
    
  }else{
    val.sliders<-val.slid
  }
  

  min.slid<-min(val.sliders,na.rm=T)
  max.slid<-max(val.sliders,na.rm=T)
  
  if(str_detect(paste0(val.sliders[2]),'[.]')){
    dec_points<-str_length(str_extract(val.sliders[2],'[.][:number:]+'))-1
  
  step.val<-round(unique(diff(val.sliders))[1],dec_points)
}else{
  step.val<-unique(diff(val.sliders))[1]
}  
  values_length<-1:length(val.sliders)
  
  sel_idx<-c(as.integer(quantile(values_length,0.25)),
             as.integer(quantile(values_length,0.35)),
             as.integer(quantile(values_length,0.95)))
  
  vals_beg<-val.sliders[sel_idx]
  
  get_slider_input_lag<-function(var,pp){
    if(pp==1){
      offset_val<-0
    }else{
      offset_val<-4
    }
    
    aa<-tribble(~var,
                "column(12,",
                paste0('offset=',offset_val,","),
                paste0('sliderInput(inputId = "',var,'_',pp,'",'),
                paste0('label = "'," ",'",'),
                paste0('min=',min.slid,","),
                paste0('max=',max.slid,","),
                paste0('value=',vals_beg[pp],","),
                paste0('step=',step.val,'))'),
                
    )
    paste(aa$var,collapse =' ')
  }
                    
  
  all_slider<-foreach(a=1:3)%do% get_slider_input_lag(var,a)
  cmd_str<-paste0('tabPanel("',var,'_slice",',
                  paste('inputPanel(',paste(unlist(all_slider),collapse =','),'),'),
                  paste0('plotOutput("',var,'_Slice_plot"))'))
  #eval(parse(text=cmd_str))
  cmd_str
}


##slider update finction

update_slider_vals<-function(tt,var.Obj){
  Var<-get(var.Obj)[tt]
  #dat_slider<-var_names_New_model()$dat
  min1<-min(dat_slider[,Var],na.rm=T)
  max1<-max(dat_slider[,Var],na.rm=T)
  
  val.slid<-round(seq(min1,max1+5,length=200),2)
  
  min<-min(val.slid,na.rm=T)
  max<-max(val.slid,na.rm=T)
  
  
  sel_idx<-c(as.integer(quantile(1:200,0.25)),
             as.integer(quantile(1:200,0.75)),
             as.integer(quantile(1:200,0.95)))
  
  vals_beg<-val.slid[sel_idx]
  
  step.val<-max(diff(seq(min,max+5,length=200)))
  
  slider_up_lag<-function(var,pp){
    var1<-paste0(var,'_',1:3)
    aa<-tribble(~var,
                paste0('updateSliderInput(session,'),
                paste0('"',var1[pp],'",'),      
                paste0('min=',min,","),
                paste0('max=',max,","),
                paste0('value=',vals_beg[pp],","),
                paste0('step=',step.val,')'),
                
    )
    paste(aa$var,collapse =' ')
  }
  
  
  all_slider<-foreach(a=1:3)%do% slider_up_lag(Var,a)
  
  cmd_str<-unlist(all_slider)
  #eval(parse(text=cmd_str))
  cmd_str
}

## plot previous years

## weekly preictions for model validation

get_weekly_prediction<-function(pp){
  
  #cat(paste0("\nYear_Week  ",run_grid[pp,]$YR,'><',run_grid[pp,]$week),'\n')
  df_pred<<-df1 
  idx.pred<<-which(df_pred$T2== run_grid[pp,]$YR&df_pred$T1== run_grid[pp,]$week)
  df_pred$Y[idx.pred]<-NA
  
  #if(exists("pred_one")){
    #rm(pred_one)
  #}
  pred_one<<-mymodel3(formula0.2a,df_pred,config =T)
  
  set.seed(4500)
  s <- 1000
  
  xx <<- inla.posterior.sample(s,pred_one)
  xx.s <<- inla.posterior.sample.eval(function(...) c(theta[1], Predictor[idx.pred]), xx)
  #dim(xx.s)
  mpred<-length(idx.pred)
  y.pred <- matrix(NA,mpred, s)
  
  #s.idx<-2
  for(s.idx in 1:s) {
    xx.sample <- xx.s[, s.idx]
    y.pred[, s.idx] <- rnbinom(mpred, mu = exp(xx.sample[-1]), size = xx.sample[1])
  }
  
  mus<-data.frame(mu_p025=    exp(apply(xx.s[-1,],1, quantile,probs=c(0.025))),
                  mu_mean=    exp(apply(xx.s[-1,],1, mean)),
                  mu_p975=    exp(apply(xx.s[-1,],1, quantile,probs=c(0.975))),
                  size_p025=    quantile(xx.s[1,],0.025),
                  size_mean=    mean(xx.s[1,]),
                  size_p975=    quantile(xx.s[1,],0.975)
  )
  YR<-run_grid[pp,]$YR
  pred_vals<-data.frame(year=year_eval,
                        district=data_augmented$district[idx.pred],
                        week=df_pred$T1[idx.pred],
                        mus,
                        observed=df1$Y[idx.pred],
                        predicted=apply(y.pred,1,mean),
                        p25=apply(y.pred,1,quantile,probs=c(0.025)),
                        p50=apply(y.pred,1,quantile,probs=c(0.5)),
                        p975=apply(y.pred,1,quantile,probs=c(0.975)),
                        fitted=pred_one$summary.fitted.values$mean[idx.pred],
                        fittedp25=pred_one$summary.fitted.values$`0.025quant`[idx.pred],
                        fitted975=pred_one$summary.fitted.values$`0.975quant`[idx.pred],
                        
                        index=idx.pred,
                        row.names =NULL
  )
  
  #cor(pred_vals$fitted,pred_vals$prcedicted)
  #pp<-6
  cat(paste0('done week ',run_grid[pp,]$week,'  of 52 \n'))
  cat(paste('Total running time:',round(pred_one$cpu.used[4],2)),'s\n\n')
  
  pred_vals
}

#pp<-1

get_weekly_prop_pred<-function(pp){
  pros_week_remove<-pros_week[-pp,]
  
  
  id.remove1<-which(paste0(data_Combined_model$year,'_',data_Combined_model$week) %in% 
                      paste0(pros_week_remove$year,'_',pros_week_remove$week) &
                      data_Combined_model$source=="predict")
  
  all_basis_vars_now<<-lapply(all_basis_vars_pros, FUN=function(x)  x[-id.remove1,])
  df_pred<<-df_pros[-id.remove1,]
  dat_Now<<-data_Combined_model[-id.remove1,]
  idx.pred<<-which(dat_Now$source=="predict")
  
  #dim(df_pred)
  #dim(all_basis_vars_now$basis_meantemperature)
  basis_var_n<-paste0('all_basis_vars_now$',names(all_basis_vars_now))
  
  if(length(add.var)>0){
    formula0.2a <- eval(parse(text=paste0("update.formula(baseformula, ~. +",paste(add.var,collapse ='+'),'+',paste(basis_var_n,collapse ='+'),')')))
  }else{
    formula0.2a <- eval(parse(text=paste0("update.formula(baseformula, ~. +",paste(basis_var_n,collapse ='+'),')')))
    
  }
  
  pred_one<<-mymodel2(formula0.2a,df_pred,config =T)
  
  set.seed(4500)
  s <- 1000
  
  xx <<- inla.posterior.sample(s,pred_one)
  xx.s <<- inla.posterior.sample.eval(function(...) c(theta[1], Predictor[idx.pred]), xx)
  #dim(xx.s)
  mpred<-length(idx.pred)
  y.pred <- matrix(NA,mpred, s)
  
  #s.idx<-2
  for(s.idx in 1:s) {
    xx.sample <- xx.s[, s.idx]
    y.pred[, s.idx] <- rnbinom(mpred, mu = exp(xx.sample[-1]), size = xx.sample[1])
  }
  
  mus<<-data.frame(mu_p025=    exp(apply(xx.s[-1,],1, quantile,probs=c(0.025))),
                  mu_mean=    exp(apply(xx.s[-1,],1, mean)),
                  mu_p975=    exp(apply(xx.s[-1,],1, quantile,probs=c(0.975))),
                  size_p025=    quantile(xx.s[1,],0.025),
                  size_mean=    mean(xx.s[1,]),
                  size_p975=    quantile(xx.s[1,],0.975)
  )
  
  pred_vals<-data.frame(year=unique(pros_week$year),
                        district=dat_Now$district[idx.pred],
                        week=df_pred$T1[idx.pred],
                        pop=dat_Now$pop[idx.pred],
                        mus,
                        observed=df_pred$Y[idx.pred],
                        predicted=apply(y.pred,1,mean),
                        p25=apply(y.pred,1,quantile,probs=c(0.025)),
                        p50=apply(y.pred,1,quantile,probs=c(0.5)),
                        p975=apply(y.pred,1,quantile,probs=c(0.975)),
                        fitted=pred_one$summary.fitted.values$mean[idx.pred],
                        fittedp25=pred_one$summary.fitted.values$`0.025quant`[idx.pred],
                        fitted975=pred_one$summary.fitted.values$`0.975quant`[idx.pred],
                        
                        index=idx.pred,
                        row.names =NULL
  )
  
  #cor(pred_vals$fitted,pred_vals$predicted)
  
  pred_vals
}

output_dist_new_model<-selectInput(inputId = 'district_new',
                                   label = 'District',
                                   choices = c(3:20),
                                   selected =20,
                                   selectize =T,
                                   multiple =F)
var<-'district_new'
#dist<-20:25


create_input_UI_district<-function(var){
  dTT1<<-sort(unique(dat_slider$district))
  dTT<<-dTT1[dTT1%in% bound_Data_Districts]
  aa<-tribble(~var,
              paste0('selectInput(inputId="',var,'",'),
              "label = 'District',",      
              'choices=dTT,',
              'selected =dTT[1],',
              'selectize =T,',
              'multiple =F)',
              
  )
  aa$var
  paste(aa$var,collapse =' ')
}

#create_input_UI_district(var)
year_validation<-sliderInput(inputId = "new_model_Year_validation",
                             label = "Year",
                             min = 2008,
                             max=2030,
                             value =2013,
                             sep='',
                             step=1)

years.dat<-2012:2020

create_input_UI_year<-function(var){
  years.dat<-sort(unique(dat_slider$year))
  aa<-tribble(~var,
              paste0('sliderInput(inputId="',var,'",'),
              "label = 'Year',",      
              'min=max(years.dat)-1,',
              'max=max(years.dat),',
              'value =max(years.dat),',
              'sep="",',
              'step=1)',
              
  )
  aa$var
  paste(aa$var,collapse =' ')
}

create_input_UI_year2<-function(var){
  years.dat<-sort(unique(dat_slider$year))
  aa<-tribble(~var,
              paste0('sliderInput(inputId="',var,'",'),
              "label = 'Year',",      
              'min=min(years.dat),',
              'max=max(years.dat),',
              'value =min(years.dat),',
              'sep="",',
              'step=1)',
              
  )
  aa$var
  paste(aa$var,collapse =' ')
}

#output$dist_Input1<-renderUI(eval(parse(text=create_input_UI_district("district_new"))))
#output$dist_Input2<-renderUI(eval(parse(text=create_input_UI_district("output_dist_seas"))))
#output$dist_Input3<-renderUI(eval(parse(text=create_input_UI_district("output_dist_validation"))))
#output$Year_Input1<-renderUI(eval(parse(text=create_input_UI_district("new_model_Year_validation"))))
#output$Year_Input2<-renderUI(eval(parse(text=create_input_UI_year2("new_model_Year_plot"))))

validation_tab_Func<-function(){
  dist1<-sort(unique(dat_slider$district))
  dist<-dist1[which(dist1%in% bound_Data_Districts)]
  
  years.dat<-sort(unique(dat_slider$year))
  
  dist_in<<-eval(parse(text=create_input_UI_district("district_validation")))
  year_in<<-eval(parse(text=create_input_UI_year("new_model_Year_validation")))
  validation_Tab<-tribble(~var,
                          'fluidPage(inputPanel(column(12,offset=0,',
                          'dist_in),',
                          'column(12,offset=4,',
                          'year_in),',
                          'column(12,offset=4,z_outbreak_New)),',
                          'tabsetPanel(tabPanel("Runin period",',
                          'plotOutput("runin_ggplot_New_model"),',
                          'dygraphOutput("runin_interactive_New_model")),',
                          'tabPanel("Validation_period",',
                          'tabsetPanel(',
                          'tabPanel("Plots",',
                          'plotOutput("validation_ggplot_New_model"),',
                          'dygraphOutput("validation_interactive_New_model")),',
                          'tabPanel("Sensitivity/Specificity",',
                          'tableOutput("sen_spec_table_New_model")',
                          ')',
                          ')',
                          '))',
                          ')',
  )
  #validation_Tab$var
  paste(validation_Tab$var,collapse =' ')
}

#eval(parse(text=validation_tab_Func()))
