## Update dashboard II for the new model

observeEvent(input$dat_prospective,{
  
  req(input$dat_prospective)
  source("New_model_helper_Functions.R")
  
  population<-input$population_New_model
  pop.var.dat<-input$population_New_model
  alarm_indicators<-input$alarm_indicators_New_model
  other_alarm_indicators<-input$other_alarm_indicators_New_model
  number_of_cases<-input$number_of_cases_New_model
  base_vars<-c("district","year","week")
  
  boundary_file<-var_names_New_model()$SHP
  boundary_file$district<-as.numeric(boundary_file$district)
  
  if(!dir.exists(paste0(getwd(),"/INLA"))){
    untar("INLA_20.03.17.tar.gz")
    
  }
  
  pred_vals_all_promise %...>%  
    {
      pred_vals_all<<-.
    }
  
  
  pkgload::load_all(paste0(getwd(),"/INLA"))
  
  inla.dynload.workaround()

  ## read in uploaded data
  
  get_D_ata<-function(p){
    if(str_detect(p,".xlsx$")){
      data <- data.frame(read_xlsx(p,sheet=1),stringsAsFactors =F)
    }
    else if(str_detect(p,".xls$")){
      data <- data.frame(read_xls(p,sheet=1),stringsAsFactors =F)
    } 
    else if(str_detect(p,".csv$")){
      data <- read.csv(p,header =T,stringsAsFactors =F)
    } else{
      data <- data.frame(read_xlsx("Demo_Data.xlsx",sheet=1),stringsAsFactors =F)
    }
    data
  }
  
  inFile_a <- input$dat_new_Model
  inFile_b <- input$dat_prospective
  
  Survilance_data<-get_D_ata(inFile_a$datapath)
  prediction_Data<-get_D_ata(inFile_b$datapath)
  
  ## update the districts Tab on DBII
  
  #updateSelectInput(session,"district_prospective",
                    #choices=sort(unique(Survilance_data$district)),
                    #selected =20)
  
  dist<-Survilance_data$district
  output$dist_pros<-renderUI(eval(parse(text=create_input_UI_district("district_prospective"))))
  
  
  print("from vars_names")
  print(head(prediction_Data))
  
  alarm_ind.check<<-alarm_indicators
  
  dat_A<-Survilance_data %>% 
    dplyr::arrange(district,year,week) %>% 
    dplyr::filter(district %in% boundary_file$district &!week==53)
  
  alarm.indicators<<-alarm_indicators
  
  
  dat_dist<-expand.grid(week=1:52,
                        year=sort(unique(dat_A$year)),
                        district=sort(unique(dat_A$district))) %>% 
    dplyr::select(district,year,week)
  
  dat_dist_pred<-expand.grid(week=1:52,
                             year=sort(unique(prediction_Data$year)),
                             district=sort(unique(dat_A$district))) %>% 
    dplyr::select(district,year,week)
  
  data_augmented<<-merge(dat_dist,dat_A,by=c("district","year","week"),all.x=T,sort=T) %>% 
    dplyr::mutate(source="for_model") %>% 
    dplyr::arrange(district,year,week)
  
  prediction_Data_a<-merge(dat_dist_pred,prediction_Data,by=c("district","year","week"),all.x=T,sort=T)%>% 
    dplyr::mutate(source="predict")
  
  data_Combined<<-rbind(data_augmented,prediction_Data_a) %>% 
    dplyr::mutate(district_w=paste0(district,'_',week)) %>% 
    dplyr::arrange(district,year,week)
  
  nlag <<- input$nlags
  
  alarm_vars<-alarm_indicators
  
  all_basis_vars_all_pros<<-foreach(a=alarm_vars,.combine =c)%do% get_cross_basis(a,"data_Combined")
  
  ## remove the augmented data
  
  
  id.remove_pros<-which(!paste0(data_Combined$district,'_',data_Combined$year,'_',data_Combined$week) %in% 
                     paste0(prediction_Data$district,'_',prediction_Data$year,'_',prediction_Data$week) &
                     data_Combined$source=="predict")
  
  all_basis_vars_pros<<-lapply(all_basis_vars_all_pros, FUN=function(x)  x[-id.remove_pros,])
  cat("it computed...\n")

  data_Combined_model<<-data_Combined[-id.remove_pros,]
  
  print(dim(all_basis_vars_pros[[1]]))
  
  district_index<-data_Combined_model %>% 
    dplyr::select(district) %>% 
    unique() %>% 
    dplyr::arrange(district) 
  
  district_index$district_index<-1:nrow(district_index)
  
  data_Combined_model<<-merge(data_Combined_model,district_index,by="district",sort=F)
  
  min.year<-min(data_Combined_model$year)
  
  data_Combined_model$year_index <<- data_Combined_model$year - (min.year-1)
  
  
  ##create model data
  
  fixe_alarm_vars<<-input$other_alarm_indicators_New_model
  
  add.var<<-fixe_alarm_vars[which(!fixe_alarm_vars%in% alarm_vars)]
  
  
  if(length(add.var)>0){
    sel_mod.vars<<-c(number_of_cases,pop.var.dat,"week","year_index","district_index",alarm_vars,add.var)
  }else{
    sel_mod.vars<<-c(number_of_cases,pop.var.dat,"week","year_index","district_index",alarm_vars)
    
  }
  cat(paste(names(data_Combined_model),collapse =","),sep='\n')
  cat(paste(sel_mod.vars,collapse =","),sep='\n')
  
  df_pros<<-data_Combined_model[,sel_mod.vars]
  exists("df_pros")
  names(df_pros)[1:5]<<-c("Y","E","T1","T2","S1")
  df_pros$E<<-df_pros$E/1e5
  
  #all_basis_vars_check<<-all_basis_vars
  
  all_check<<-list(id.remove=id.remove_pros,
                   all_basis_vars=all_basis_vars_pros,
                   all_basis_vars_all=all_basis_vars_all_pros,
                   data_Combined=data_Combined,
                   data_Combined_model=data_Combined_model,
                   df=df_pros)
  
  names(df_pros)
  baseformula <<- Y ~ 1 + f(T1,replicate = S1, model = "rw1", cyclic = TRUE, constr = TRUE,
                           scale.model = TRUE,  hyper = precision.prior) +
    f(S1,replicate=T2, model = "iid") 
  
  #base_model <- mymodel(baseformula,df)
  
  basis_var_n<<-paste0('all_basis_vars_pros$',names(all_basis_vars_pros))
  
  ## get the variable not among spline and keep as linear
  
  #fixe_alarm_vars<-input$other_alarm_indicators_New_model
  
  
  
  formula0.1 <- eval(parse(text=paste0("update.formula(baseformula, ~. +",paste(alarm_vars,collapse ='+'),')')))
  
  if(length(add.var)>0){
    formula0.2 <- eval(parse(text=paste0("update.formula(baseformula, ~. +",paste(add.var,collapse ='+'),'+',paste(basis_var_n,collapse ='+'),')')))
  }else{
    formula0.2 <- eval(parse(text=paste0("update.formula(baseformula, ~. +",paste(basis_var_n,collapse ='+'),')')))
    
  }
  
  res<-mymodel(formula0.2,df_pros)
  
  ## get the predictions one week at a times
  
  pros_week<<-prediction_Data %>% 
    dplyr::select(year,week) %>% 
    unique()
  
  export_tables<-c("data_Combined_model","pros_week",
                   "pros_week_remove","all_basis_vars",
                   "all_basis_vars_pros",
                   "df_pros","baseformula","add.var")
  
  time_52_pros<<-system.time({
    forecast_dat<<-foreach(a=1:nrow(pros_week),
                          .combine =rbind,
                          .export =export_tables)%do% get_weekly_prop_pred(a)
    #saveRDS(forecast_dat,"forecast_dat_2022_04_13.rds")
  })
  time_52[3]/60
  
  #forecast_dat<<-readRDS("forecast_dat_2022_04_13.rds")
  
  ## read in validation data
  
  #pred_vals_all<-readRDS("pred_eval.rds") 
  
  sel_var_endemic<<-c(base_vars,number_of_cases,population)
  
  dat.4.endemic<<-data_augmented[,sel_var_endemic]
  names(dat.4.endemic)<-c(base_vars,"cases","pop")
  
  last_Yr_aug<-max(data_augmented$year,na.rm =T)
  
  
  ## now work with specific week
  observeEvent(c(input$district_prospective,
                 input$z_outbreak_new),{
        
                   if(is.null(input$district_prospective)){
                     district_prospective<-sort(unique(prediction_Data$district))[1]
                   }else{
                     district_prospective<-input$district_prospective
                   }
                     
                     if(is.null(input$z_outbreak_new)){
                       z_outbreak_new<-1.2
                     }else{
                       z_outbreak_new<-input$z_outbreak_new
                     } 
                   
                   for_endemic<-dat.4.endemic %>% 
                     dplyr::filter(!is.na(cases) & !year==last_Yr_aug) %>% 
                     dplyr::mutate(rate=(cases/pop)*1e5) %>% 
                     dplyr::group_by(district,week) %>% 
                     dplyr::summarise(.groups="drop",mean=mean(rate,na.rm =T),
                                      sd=sd(rate,na.rm =T)) %>% 
                     dplyr::mutate(threshold=mean+z_outbreak_new*(sd))
    
    data_use<-pred_vals_all %>% 
      dplyr::filter(district==district_prospective) %>% 
      dplyr::select(district,year,week,mu_mean,size_mean,observed,predicted,
                    p25,p975,index) 
    
    pros_forcasted<-forecast_dat %>% 
      dplyr::filter(district==district_prospective) %>% 
      dplyr::select(district,year,week,mu_mean,size_mean,observed,predicted,
                    p25,p975,pop,index) 
    
    data_use$pop<-dat.4.endemic$pop[data_use$index]
    
    data_use_<-data_use %>% 
      dplyr::mutate(mu_mean=(mu_mean/pop)*1e5,
                    observed=(observed/pop)*1e5,
                    predicted=(predicted/pop)*1e5,
                    p25=(p25/pop)*1e5,
                    p975=(p975/pop)*1e5) %>% 
      dplyr::left_join(for_endemic,by=c("district","week")) %>% 
      dplyr::mutate(outbreak=observed>threshold,
                    observed_alarm=case_when(outbreak==1~observed,
                                             TRUE~as.numeric(NA)))
    
    pros_forcasted_<-pros_forcasted %>% 
      dplyr::mutate(mu_mean=(mu_mean/pop)*1e5,
                    observed=(observed/pop)*1e5,
                    predicted=(predicted/pop)*1e5,
                    p25=(p25/pop)*1e5,
                    p975=(p975/pop)*1e5) %>% 
      dplyr::left_join(for_endemic,by=c("district","week")) %>% 
      dplyr::mutate(outbreak=observed>threshold,
                    observed_alarm=case_when(outbreak==1~observed,
                                             TRUE~as.numeric(NA)))
    
    probs<-pnbinom(data_use_$threshold, mu =data_use_$mu_mean, size = data_use_$size_mean,lower.tail =F)
    probs_forecast<-pnbinom(pros_forcasted_$threshold, mu =pros_forcasted_$mu_mean, size = pros_forcasted_$size_mean,lower.tail =F)
    
    print(probs)
    idx.comp<-which(!is.na(data_use_$outbreak))
    
    
    
    roc_try<-try(reportROC(gold=as.numeric(data_use_$outbreak)[idx.comp],
                               predictor=probs[idx.comp]),outFile =warning("please.."))
    
    roc_tab_names<-c("Cutoff","AUC","AUC.SE","AUC.low","AUC.up","P","ACC",
                     "ACC.low","ACC.up","SEN","SEN.low","SEN.up",
                     "SPE","SPE.low","SPE.up","PLR","PLR.low",
                     "PLR.up","NLR","NLR.low","NLR.up","PPV",
                     "PPV.low","PPV.up","NPV","NPV.low","NPV.up")
    
    kdd<-data.frame(t(rep(as.character(NA),length(roc_tab_names))))
    names(kdd)<-roc_tab_names
    
    if(class(roc_try) %in% c("NULL","try-error")){
      roc_report<-kdd
    }else{
      roc_report<-reportROC(gold=as.numeric(data_use_$outbreak)[idx.comp],
                            predictor=probs[idx.comp])
    }
    
    if(roc_report$Cutoff%in% c(NA,-Inf,NaN,Inf)){
      sens_ppv<-tribble(~var,~val,~CI_Lower,~CI_Upper,
                        "Cutoff probability",roc_report$Cutoff,NA,NA,
                        "Area under the Curve (AUC)",roc_report$AUC,roc_report$AUC.low,roc_report$AUC.up,
                        "Accuracy",roc_report$ACC ,roc_report$ACC.low,roc_report$ACC.up,
                        "Sensitivity",roc_report$SEN,roc_report$SEN.low,roc_report$SEN.up,
                        "Specificity",roc_report$SPE,roc_report$SPE.low,roc_report$SPE.up,
                        "Positive Predictive Value (PPV)",roc_report$PPV,roc_report$PPV.low,roc_report$PPV.up,
                        "Negative Predictive Value (NPV)",roc_report$NPV,roc_report$NPV.low,roc_report$NPV.up)
      
      data_use_a<-data_use_ %>% 
        dplyr::mutate(prob_exceed=probs,
                      cutoff=NA,
                      validation_alarm=as.numeric(NA))
      
    }else{
      sens_ppv<-tribble(~var,~val,~CI_Lower,~CI_Upper,
                        "Cutoff probability",roc_report$Cutoff,NA,NA,
                        "Area under the Curve (AUC)",roc_report$AUC,roc_report$AUC.low,roc_report$AUC.up,
                        "Accuracy",roc_report$ACC ,roc_report$ACC.low,roc_report$ACC.up,
                        "Sensitivity",roc_report$SEN,roc_report$SEN.low,roc_report$SEN.up,
                        "Specificity",roc_report$SPE,roc_report$SPE.low,roc_report$SPE.up,
                        "Positive Predictive Value (PPV)",roc_report$PPV,roc_report$PPV.low,roc_report$PPV.up,
                        "Negative Predictive Value (NPV)",roc_report$NPV,roc_report$NPV.low,roc_report$NPV.up)  
      
      data_use_a<-data_use_ %>% 
        dplyr::mutate(prob_exceed=probs,
                      cutoff=as.numeric(roc_report$Cutoff),
                      validation_alarm=case_when((prob_exceed>=cutoff)~prob_exceed,
                                                 TRUE~as.numeric(NA)))
    }
    
    
    pros_forcasted_a<-pros_forcasted_ %>% 
      dplyr::mutate(prob_exceed=probs_forecast,
                    cutoff=roc_report$Cutoff,
                    alarm=as.numeric((prob_exceed>=cutoff)))
    
    ##plots and Tables
    
    pros_forcasted_b<-pros_forcasted_a %>% 
      dplyr::mutate(alarm_threshold=roc_report$Cutoff,
                    outbreak=(predicted/pop)*1e5,
                    outbreak_probability=prob_exceed)
    
    
    dat_eval_merge<-for_endemic %>% 
      dplyr::mutate(outbreak_moving=round(mean,6),
                    outbreak_moving_sd=sd,
                    outbreak_moving_limit=round(threshold,6),
                    endemic_chanel=round(threshold,6)) %>% 
      dplyr::select(district,week,outbreak_moving,outbreak_moving_sd,outbreak_moving_limit,
                    endemic_chanel) %>% 
      dplyr::filter(district==district_prospective) %>% 
      dplyr::left_join(pros_forcasted_b,by=c("district","week")) %>% 
      dplyr::mutate(outbreak_period=case_when(outbreak>endemic_chanel~1,
                                              TRUE~0),
                    alarm_signal=case_when(outbreak_probability>alarm_threshold~1,
                                           is.na(outbreak_probability)~as.double(NA),
                                           TRUE~0))
    
    
    tem.d<-dat_eval_merge %>% mutate(lag0=dplyr::lag(alarm_signal,0),
                                     lag1=dplyr::lag(alarm_signal,1),
                                     lag2=dplyr::lag(alarm_signal,2),
                                     lag3=dplyr::lag(alarm_signal,3),
                                     lag4=dplyr::lag(alarm_signal,4)) %>% 
      mutate(response_cat=case_when(lag0==1 & lag1==1 & lag2 %in% c(0,NA) ~1,
                                    lag0==1 & lag1==1 & lag2==1 & lag3 %in% c(0,NA) ~1.5,
                                    lag0==1 & lag1==1 & lag2==1  & lag3==1 ~2,
                                    is.na(alarm_signal)~ as.double(NA),
                                    TRUE~0.5))
    
    
    
    dat_lab<-data.frame(response_cat=c("No response",
                                       "Initial response",
                                       "Early response",
                                       "Late/emergency response"),
                        x=-20,y=seq(0.65,2.5,0.5))
    
    plot1<-ggplot(aes(x=week,y=outbreak_moving_limit),data=tem.d)+
      geom_area(aes(fill="Endemic channel"))+
      geom_line(aes(y=outbreak,col="Confirmed cases"),lwd=0.3)+
      geom_point(aes(y=outbreak,col="Confirmed cases"),size=2.5,pch=15)+
      theme_bw()+
      scale_fill_manual(values =c("Endemic channel"=grey(0.7)))+
      scale_color_manual(values =c("Confirmed cases"='red1'))+
      scale_x_continuous(breaks=2:52,limits =c(2,52))+
      theme(panel.grid.major.x =element_blank(),
            panel.grid.minor.x =element_blank(),
            panel.grid.major.y =element_line(linetype=2),
            panel.grid.minor.y =element_blank(),
            axis.line.x.top =element_blank(),
            panel.border =element_blank(),
            axis.line.y =element_line(linetype=1,colour="grey",size=0.4,lineend="butt"),
            axis.line.x =element_line(linetype=1,colour="grey",size=0.4,lineend="butt"),
            legend.position ="top",
            axis.title.y =element_blank(),
            legend.text =element_text(size=14))+
      guides(fill=guide_legend(title =NULL),
             color=guide_legend(title =NULL))+
      xlab("Epidemiological week")
    
    str(tem.d)
    
    tem.d$alarm_threshold<-as.numeric(tem.d$alarm_threshold)
    ggplot_dat_DB2_test<<-tem.d
    plot2<-ggplot()+
      
      geom_line(aes(x=week,y=outbreak_probability,col="Outbreak probability"),lwd=0.3,data=tem.d)+
      geom_point(aes(x=week,y=outbreak_probability,col="Outbreak probability"),size=2.5,pch=15,data=tem.d)+
      geom_line(aes(x=week,y=alarm_threshold,col="Alarm threshold"),lwd=0.7,data=tem.d,lty=2)+
      
      theme_bw()+
      scale_color_manual(values =c("Outbreak probability"='dark blue',
                                   "Alarm threshold"="forest green"))+
      scale_x_continuous(breaks=2:52,limits =c(2,52))+
      theme(panel.grid.major.x =element_blank(),
            panel.grid.minor.x =element_blank(),
            panel.grid.major.y =element_line(linetype=2),
            panel.grid.minor.y =element_blank(),
            axis.line.x.top =element_blank(),
            panel.border =element_blank(),
            axis.line.y =element_line(linetype=1,colour="grey",size=0.4,lineend="butt"),
            axis.line.x =element_line(linetype=1,colour="grey",size=0.4,lineend="butt"),
            legend.position ="top",
            axis.title.y =element_blank(),
            legend.text =element_text(size=14)
      )+
      guides(fill=guide_legend(title =NULL),
             color=guide_legend(title =NULL))+
      xlab("Epidemiological week")
    
    ratio_DB2<-max(c(tem.d$outbreak_moving_limit,tem.d$outbreak,
                     tem.d$p975),na.rm =T)/
      max(c(tem.d$outbreak_probability,tem.d$alarm_threshold),na.rm =T)
    
    plot3<-ggplot(aes(x=week,y=outbreak_moving_limit),data=tem.d)+
      geom_area(aes(fill="Endemic channel"),alpha=0.6)+
      geom_ribbon(aes(ymin=p25,ymax=p975,
                      fill="Predicted 95 % CI"),alpha=0.2)+
      geom_line(aes(y=outbreak,col="Predicted"),lwd=0.3)+
      geom_point(aes(y=outbreak,col="Predicted"),size=2.5,pch=15)+
      geom_line(aes(x=week,y=outbreak_probability*ratio_DB2,col="Outbreak probability"),lwd=0.3,data=tem.d)+
      geom_point(aes(x=week,y=outbreak_probability*ratio_DB2,col="Outbreak probability"),size=2.5,pch=15,data=tem.d)+
      geom_line(aes(x=week,y=alarm_threshold*ratio_DB2,col="Alarm threshold"),lwd=1,data=tem.d)+
      theme_bw()+
      scale_fill_manual(values=c("Endemic channel"='wheat',
                                 "Predicted 95 % CI"=grey(0.3)))+
      scale_color_manual(values =c("Predicted"='red1',
                                   "Outbreak probability"='blue',
                                   "Alarm threshold"="forest green"))+
      scale_y_continuous(name = "DIR",sec.axis =sec_axis(~ . /ratio_DB2,name="Probability"))+
      scale_x_continuous(breaks=2:52,limits =c(2,52))+
      theme(panel.grid.major.x =element_blank(),
            panel.grid.minor.x =element_blank(),
            panel.grid.major.y =element_line(linetype=2),
            panel.grid.minor.y =element_blank(),
            axis.line.x.top =element_blank(),
            panel.border =element_blank(),
            axis.line.y =element_line(linetype=1,colour="grey",size=0.4,lineend="butt"),
            axis.line.x =element_line(linetype=1,colour="grey",size=0.4,lineend="butt"),
            legend.position ="top",
            ##axis.title.y =element_blank(),
            legend.text =element_text(size=14)
            
      )+
      guides(fill=guide_legend(title =NULL),
             color=guide_legend(title =NULL))+
      xlab("Epidemiological week")
    
    plot4<-ggplot(aes(x=week,y=response_cat),data=tem.d)+geom_point(pch=21,size=2.5)+
      geom_hline(yintercept =0.5,col="yellowgreen",lwd=0.8)+
      geom_hline(yintercept =1,col="orange",lwd=0.8)+
      geom_hline(yintercept =1.5,col="brown",lwd=0.8)+
      geom_hline(yintercept =2,col="red",lwd=0.8)+
      geom_text(aes(x=x,y=y,label=response_cat,col=response_cat),data=dat_lab,
                show.legend =F,hjust=0,nudge_x =0.2)+
      theme_bw()+
      scale_x_continuous(breaks=seq(2,52,2))+
      
      scale_color_manual(values=c("No response"='yellowgreen',
                                  "Initial response"='orange',
                                  "Early response"='brown',
                                  "Late/emergency response"='red'))+
      
      theme(panel.grid.minor.y =element_blank(),
            panel.grid.major.y =element_blank(),
            panel.grid.major.x =element_blank(),
            panel.grid.minor.x =element_blank(),
            panel.border =element_blank(),
            axis.line.x =element_line(linetype=1,
                                      colour="grey",
                                      size=0.4,
                                      lineend="butt"),
            axis.title.y =element_blank(),
            axis.text.y=element_blank(),
            axis.ticks.y =element_blank(),
            legend.text =element_text(size=14))+
      coord_fixed(6,ylim =c(0.3,3),xlim = c(-20,52))+
      xlab("Epidemiological week")
    
    #output$outbreak_plot_pros<-renderPlot(plot1)
    #output$prob_plot_pros<-renderPlot(plot2)
    output$out_break_prob_plot_pros<-renderPlot(plot3)
    output$response_plot_pros<-renderPlot(plot4)
    
    ## render the tables
    dat_pros_Dis<-prediction_Data %>% 
      dplyr::filter(district==district_prospective) %>% 
      dplyr::arrange(year,week)
    
    ## identify variables with decimals
    
    
    
    dat_pros_Dis_char<-dat_pros_Dis %>% 
      dplyr::mutate_all(.funs=as.character)
    
    ##get the columns to format
    
    real_columns<-as.numeric(apply(dat_pros_Dis_char,2,FUN = function(x) sum(str_detect(x,'[.]'))))
    
    which_round<-names(dat_pros_Dis_char)[which(real_columns>0)]
    
    
    dat_pros_Dis1<-dat_pros_Dis %>% 
      dplyr::mutate_at(.vars=which_round,.funs =function(x) round(x,2))
    
    output$uploaded_pros<-renderDataTable(data.table(dat_pros_Dis1))
    
    vars.keep<-c("district","year","week","pop","endemic_chanel","predicted",
                 "p25","p975","outbreak","alarm_threshold","outbreak_probability","alarm",
                 "alarm_signal","response_cat")
    tem.d1<-tem.d[,vars.keep]
    
    var.round<-c("endemic_chanel","predicted","p25","p975","outbreak","alarm_threshold",
                 "outbreak_probability")
    
    tem.d2<-tem.d1 %>% 
      dplyr::mutate_at(.vars=var.round,.funs =function(x) round(x,4))
    
    output$prediction_tab_pros<-renderDataTable(data.table(tem.d2))
    
    
  })
  
})


