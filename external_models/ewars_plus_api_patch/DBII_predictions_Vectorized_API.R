#DBII predictions API

library(rjson)
library(dplyr)
library(stringr)
library(flextable)
library(foreach)
library(ggplot2)
library(leaflet)
library(dlnm)
library(RColorBrewer)
library(lubridate)
library(xts)
library(dygraphs)
library(quarto)
library(khroma)




for (DD in 1:length(all_districts_pros)){
  
  District_Now<-all_districts_pros[DD]
  
  if(District_Now%in%Districts_DBI_surve_Dat){
    
  
  one_of_dist_str_pros<-paste0('(',DD,' of ',length(all_districts_pros),' districts)')
  
  
  
  Prospective_Data_Sub<-Prospective_Data |> 
    dplyr::filter(district==District_Now)
  
  dist_padded<-str_pad(District_Now,side="left",width=3,pad=0)
  shiny_obj_pth<-file.path(shiny_obj_Main_pth,paste0("District_",dist_padded))
  shinyDBII_obj_pth<-file.path(shinyDBII_obj_Main_pth,paste0("District_",dist_padded))
  pred_weights_pth<-file.path(pred_weights_Main_pth,paste0("District_",dist_padded))
  
  
    
  
  get_Ojs1<-function(){
    gc()
    readRDS(file.path(shiny_obj_pth,"Shiny_Objs.rds"))
    
    
  }
  
  shiny_obj1<-get_Ojs1()
  # reporT_pAth0<-shiny_obj1[["report_pth"]]
  # reporT_pAth<-stringr::str_replace(reporT_pAth0,'.',getwd())
  # 
  reporT_pAth0<-shiny_obj1[["report_pth"]]
  reporT_pAth<-stringr::str_replace(reporT_pAth0,'.',getwd())
  
  ## read in var splines
  
  Inlagrp_Vars_pros<-readRDS(file.path(shiny_obj_pth,"Shiny_Objs.rds"))$Inlagrp_Vars
  selected_zvalue_pros0<-readRDS(file.path(shiny_obj_pth,"Shiny_Objs.rds"))$selected_zvalue
  Selected_out_threshold0<-as.numeric(readRDS(file.path(shiny_obj_pth,"Shiny_Objs.rds"))$zvalue_sel_Ordered[1,"Cutoff"])
  
  selected_zvalue_pros<-ifelse(is.na(selected_zvalue_pros0),1.2,selected_zvalue_pros0)
  Selected_out_threshold<-ifelse(is.na(Selected_out_threshold0),0.1,Selected_out_threshold0)
  
  alarm_vars_pros<-readRDS(file.path(shiny_obj_pth,"Shiny_Objs.rds"))$alarm_vars
  
  Selected_lag_Vars_pros<-readRDS(file.path(shiny_obj_pth,"Shiny_Objs.rds"))$Selected_lag_Vars
  
  Computed_pred_Distance<-floor(mean(as.numeric(unlist(str_extract_all(Selected_lag_Vars_pros,"[:number:]+")))))
  
  
  Inla_grp_Nsize_pros<-readRDS(file.path(shiny_obj_pth,"Shiny_Objs.rds"))$Inla_grp_Nsize
  
  
  for_endemic_pros<-readRDS(file.path(all_files_Path,"Shiny_Objs_all.rds"))$all_endemic |> 
    dplyr::filter(district==District_Now) |> 
    dplyr::mutate(threshold_cases=mean_cases+selected_zvalue_pros*(sd_cases),
                  threshold_rate=mean_rate+selected_zvalue_pros*(sd_rate))
  
  work_CV_Weight_pros<-readRDS(file.path(pred_weights_pth,"Pred_weights_meta.rds"))$work_CV_Weight
  inla_grp_Names_pros<-readRDS(file.path(pred_weights_pth,"Pred_weights_meta.rds"))$inla_grp_Names
  
  last_Dat_Year<-unique(work_CV_Weight_pros$year)
  
  endemic_channel_Use<-for_endemic_pros |> 
    dplyr::filter(year==last_Dat_Year)
  
  cat(paste0('rows for_endemic_pros ..',nrow(for_endemic_pros)),sep='\n')
  ## get right prediction weight to load
  
  pred_week_Objs<-foreach(aa=1:nrow(work_CV_Weight_pros),.combine=rbind)%do%
    data.frame(seq=aa,week=work_CV_Weight_pros$beg_week[aa]:work_CV_Weight_pros$end_week[aa]) |> 
    dplyr::mutate(mn_pref=str_pad(seq,pad=0,side='left',width=2),
                  pred_obj=file.path(pred_weights_pth,paste0('Pred_weights_',last_Dat_Year,'_',mn_pref,'.rds')))

  
  #vars_For_inla_grp<-str_split_fixed(Selected_lag_Vars_pros,"_",2)[,1]
  vars_For_inla_grp<-str_remove(Selected_lag_Vars_pros,'_LAG[:number:]+')
  
  
  for (vv in 1:length(vars_For_inla_grp)){
    
    # Inlagrp_Int_dat<-Inlagrp_Vars_pros[[vv]] |> 
    #   dplyr::mutate(intval=str_remove_all(Interval,'[)()]|\\[|\\]'),
    #                 int_beg=as.numeric(str_split(intval,pattern=',',n=2,simplify=T)[,1]),
    #                 int_end=as.numeric(str_split(intval,pattern=',',n=2,simplify=T)[,2]))
    
    Inlagrp_Int_dat<-Inlagrp_Vars_pros[[vv]]
    
    grp_obj_Name_pros<-paste0('Var',vv,'_inla_grp')
    
    Var_grp_pref_pros<-paste0('Var',vv)
    
    ##find interval where the variables lie
    
    Vars_Find<-Prospective_Data_Sub[,vars_For_inla_grp[vv]]
    
    #findInterval(Vars_Find,Inlagrp_Int_dat$int_beg)
    Size.grp<-nrow(Inlagrp_Int_dat)
    
    #Matrix_wgts<-matrix(NA,length(Vars_Find),Inla_grp_Nsize_pros)
    Matrix_wgts<-matrix(NA,length(Vars_Find),Size.grp)
    
    
    for(ww in 1:length(Vars_Find)){
      int_found0<-findInterval(Vars_Find[ww],c(Inlagrp_Int_dat$int_beg))
      int_found<-ifelse(int_found0==0,1,int_found0)
      
      Matrix_wgts[ww,]<-as.numeric(1:Size.grp==int_found)
    }
    
    #padded_grp_N<-str_pad(1:Inla_grp_Nsize_pros,pad=0,side ='left',width =2)
    padded_grp_N<-str_pad(1:Size.grp,pad=0,side ='left',width =2)

    
    colnames(Matrix_wgts)<-paste0(Var_grp_pref_pros,'_Inla_group_',padded_grp_N)
    
    assign(grp_obj_Name_pros,
           Matrix_wgts)
    
  }
  
  
  ## Compute the predictions here
  
  rw_Names_string<-paste0(inla_grp_Names_pros,collapse=',')
  
  names_Select_pred<-c("a","coefspat","week",inla_grp_Names_pros,"pop_offset")
  
  #Var1_Spline
  
  Objects_Cbind<-c("Prospective_Data_Sub",paste0("Var",1:length(vars_For_inla_grp),"_inla_grp"))
  
  Prospective_Data_with_inla_grp<-foreach(aa=1:length(Objects_Cbind),.combine =cbind)%do% get(Objects_Cbind[aa])
  
  Pred_NSize<-nrow(Prospective_Data_with_inla_grp)
  
  all_Pros_Predictions_ls<-vector("list",Pred_NSize)
  
  cat("",sep='\n')
  cat("generating Predictions ..,",sep='\n')
  
  cov_matrix_Pred_all<-Prospective_Data_with_inla_grp|> 
    dplyr::mutate(a=1,coefspat=1,week=1) |>
    dplyr::select(all_of(names_Select_pred)) |> 
    as.matrix()
  
  
  ## create prediction matrix
  
  num_Features<-ncol(cov_matrix_Pred_all)
  num_Pred<-nrow(cov_matrix_Pred_all)
  
  
  matrix_Pred<-matrix(0,num_Pred,num_Pred*num_Features)
  
  
  beg_idx<-seq(1,num_Pred*num_Features,num_Features)
  end_idx<-seq(num_Features,num_Pred*num_Features,num_Features)
  
  #ii<-20
  
  for(ii in 1:num_Pred){
    matrix_Pred[ii,beg_idx[ii]:end_idx[ii]]<-1
  }
  
  
  
  
  cov_matrix_Mult<-do.call('cbind',replicate(num_Pred,cov_matrix_Pred_all,simplify =F))*matrix_Pred
  
  dim(cov_matrix_Mult)
  
  pred_week_Objs<-foreach(aa=1:nrow(work_CV_Weight_pros),.combine=rbind)%do%
    data.frame(seq=aa,week=work_CV_Weight_pros$beg_week[aa]:work_CV_Weight_pros$end_week[aa]) |> 
    dplyr::mutate(mn_pref=str_pad(seq,pad=0,side='left',width=2),
                  pred_obj=file.path(pred_weights_pth,paste0('Pred_weights_',last_Dat_Year,'_',mn_pref,'.rds')))
  
  get_Week_Weight_mat<-function(Pred_Week){
    
    pred_week_Objs_sub<-pred_week_Objs |> 
      dplyr::filter(week==Pred_Week)
    
    readRDS(pred_week_Objs_sub$pred_obj)$pred_weights[Pred_Week,,]
  }
  
  #Pred_Week<-26
  
  Size_cv_mat<-function(Pred_Week){
    
    pred_week_Objs_sub<-pred_week_Objs |> 
      dplyr::filter(week==Pred_Week)
    
    readRDS(pred_week_Objs_sub$pred_obj)$ns_size
  }
  
  #weight_Matrix_Mult_ls<-foreach(aa=Prospective_Data_with_inla_grp$week)%do% get_Week_Weight_mat(aa)
  
  weight_Matrix_Mult<-do.call('rbind',foreach(aa=Prospective_Data_with_inla_grp$week)%do% get_Week_Weight_mat(aa))
  
  All_pred_all<-exp(cov_matrix_Mult%*%weight_Matrix_Mult)
  
  dim(All_pred_all)
  
  #Size_cv_Mod<-readRDS(pred_week_Objs_sub$pred_obj)$ns_size
  
  #Size_cv_Mod<-do.call('cbind',foreach(aa=Prospective_Data_with_inla_grp$week)%do% Size_cv_mat(aa))
  Size_cv_Mod<-Size_cv_mat(Prospective_Data_with_inla_grp$week[2])
  
  #dim(Size_cv_Mod)
  

  ypred_NB_1000_all<-foreach(aa=1:1000,.combine =cbind)%do% rnbinom(num_Pred,mu=All_pred_all[,aa],size=Size_cv_Mod[aa])
  
  pop_1000<-replicate(1000,Prospective_Data_with_inla_grp$population,simplify ="matrix")
  
  
  ypred_NB_1000_rate_all<-(ypred_NB_1000_all/pop_1000)*1e5
  
  # CHAP patch (CLIM-617): the original idiom
  #   foreach(aa = Prospective_Data_with_inla_grp$week, .combine = c) %do%
  #     which(endemic_channel_Use$week == aa)
  # silently drops weeks that are not in the precomputed endemic-channel table
  # (which() returns integer(0); .combine=c skips it). idx.end_all then ends
  # up shorter than the prospective row count, week_rate_threshold_all0
  # inherits the shortened length, and the elementwise comparison
  #   ypred_NB_1000_rate_all > week_rate_threshold_all
  # crashes with "non-conformable arrays".
  # match() preserves length: unmatched weeks become NA, the threshold cell
  # becomes NA, and the comparison harmlessly produces NA in those rows.
  idx.end_all <- match(Prospective_Data_with_inla_grp$week, endemic_channel_Use$week)

  week_rate_threshold_all0<-endemic_channel_Use$threshold_rate[idx.end_all]
  
  week_rate_threshold_all<-replicate(1000,week_rate_threshold_all0,simplify ="matrix")
  
  prob_Exceed_mat<-ypred_NB_1000_rate_all>week_rate_threshold_all
  dim(prob_Exceed_mat)
  
  prob_exceed_threshold_all<-apply(prob_Exceed_mat,1,FUN=function(x) mean(x))
  
  vars_pro_Base_select<-c("district","year","week",alarm_vars_pros,"Cases","population","observed_rate")
  
  
  
  all_Pros_Predictions<-Prospective_Data_with_inla_grp |> 
    dplyr::select(all_of(vars_pro_Base_select)) |> 
    dplyr::mutate(
      # Cases
      predicted_cases = apply(ypred_NB_1000_all,1,FUN=function(x) mean(x,na.rm=T)), 
      predicted_cases_lci = apply(ypred_NB_1000_all,1,FUN=function(x) quantile(x,c(0.025),na.rm=T)), 
      predicted_cases_uci = apply(ypred_NB_1000_all,1,FUN=function(x) quantile(x,c(0.975),na.rm=T)), 
      # rates
      predicted_rate = apply(ypred_NB_1000_rate_all,1,FUN=function(x) mean(x,na.rm=T)), 
      predicted_rate_lci = apply(ypred_NB_1000_rate_all,1,FUN=function(x) quantile(x,c(0.025),na.rm=T)), 
      predicted_rate_uci = apply(ypred_NB_1000_rate_all,1,FUN=function(x) quantile(x,c(0.975),na.rm=T)), 
      
      endemic_threshold=week_rate_threshold_all0,
      outbreak=predicted_cases,
      outbreak_rate=predicted_rate,
      prob_exceed_threshold=prob_exceed_threshold_all,
      alarm_threshold=Selected_out_threshold,
      outbreak_probability=prob_exceed_threshold
    )
  
  #gc()
  
  
  year_pred<-unique(Prospective_Data_with_inla_grp$year)
  first_week_Pred<-min(Prospective_Data_with_inla_grp$week)
  last_week_Pred<-max(Prospective_Data_with_inla_grp$week)
  last_week_Pred_Plus_PredD<-last_week_Pred+Computed_pred_Distance+2
  Week_seQF<-max(c(last_week_Pred_Plus_PredD,54))
  
  alarm_thresh<-unique(all_Pros_Predictions$alarm_threshold)
  
  
  # year_week_Pred<-expand.grid(week=2:52,year=year_pred:(year_pred+1)) |> 
  #   dplyr::mutate(start_Plot=as.numeric(!(year==year_pred & week<(first_week_Pred-1)))) |> 
  #   dplyr::filter(start_Plot==1) |> 
  #   dplyr::mutate(week_seq=1:n(),
  #                 week_pad=str_pad(week,side="left",pad=0,width=2),
  #                 plot_lab=as.factor(paste0(year,'_',week_pad))) |> 
  #   dplyr::filter(week_seq<53)
  
  year_week_Pred<-expand.grid(week=1:52,year=year_pred:(year_pred+1)) |> 
    #dplyr::mutate(start_Plot=as.numeric(!(year==year_pred & week<(first_week_Pred-1)))) |> 
    dplyr::mutate(start_Plot=1) |> 
    dplyr::filter(start_Plot==1) |> 
    dplyr::mutate(week_seq=1:n(),
                  week_pad=str_pad(week,side="left",pad=0,width=2),
                  plot_lab=as.factor(paste0(year,'_',week_pad))) |> 
    dplyr::filter(week_seq<Week_seQF)
  
  
  endemic_channel_Use_a<-endemic_channel_Use |> 
    dplyr::select(-year)
  
  year_week_Pred_endmic<-year_week_Pred |> 
    dplyr::left_join(endemic_channel_Use_a,by="week") |> 
    dplyr::mutate(outbreak_moving=round(mean_rate,6),
                  outbreak_moving_sd=sd_rate,
                  outbreak_moving_limit=round(threshold_rate,6),
                  endemic_chanel=round(threshold_rate,6)) %>% 
    dplyr::select(district,year,week,plot_lab,week_seq,outbreak_moving,outbreak_moving_sd,outbreak_moving_limit,
                  endemic_chanel)
  
  all_Pros_Predictions_Week_a<-all_Pros_Predictions |> 
    dplyr::left_join(year_week_Pred,by=c("year","week"))
  
  #names(all_Pros_Predictions_Week_a)
  
  all_Pros_Predictions_Week<-all_Pros_Predictions_Week_a |> 
    dplyr::select(-week_pad,-start_Plot,-week_pad,-plot_lab) |> 
    dplyr::mutate(week_seq=week_seq+Computed_pred_Distance)
  
  names(all_Pros_Predictions_Week_a)
  
  Plot_obs_rate<-all_Pros_Predictions_Week_a |> 
    dplyr::select(district,year,week_seq,observed_rate)
  
  
  dat_eval_merge<-year_week_Pred_endmic |> 
    dplyr::left_join(all_Pros_Predictions_Week,by=c("district","week_seq")) |> 
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
                                  TRUE~0.5),
           pred_Distance=paste0("Prediction distance=",Computed_pred_Distance," Weeks"),
           alarm_threshold=alarm_thresh)
  
  
  
  dat_lab<-data.frame(response_cat=c("No response",
                                     "Initial response",
                                     "Early response",
                                     "Late/emergency response"),
                      x=-20,y=seq(0.65,2.5,0.5))
  
  iridescent<-khroma::color(palette="iridescent")
  bright<-khroma::color(palette="bright")
  
  
  iridescent(10)
  scales::show_col(iridescent(10))
  
  bright(7)
  scales::show_col(bright(7))
  
  plot1<-ggplot(aes(x=week_seq,y=outbreak_moving_limit),data=tem.d)+
    geom_area(aes(fill="Endemic channel"),alpha=0.4)+
    geom_ribbon(aes(ymin=predicted_rate_lci,
                    ymax=predicted_rate_uci,
                    fill="Prediction\n Interval"),alpha=0.6)+
    facet_wrap(~pred_Distance)+
    geom_line(aes(y=outbreak,col="Confirmed cases"),linewidth=0.6)+
    geom_point(aes(y=outbreak,col="Confirmed cases"),size=2.5,pch=15)+
    ylab("Incidence Rate\n per 100000")+
    theme_bw()+
    scale_fill_manual(values =c("Endemic channel"="#90CCE3",
                                "Prediction\n Interval"="#9C7BAF"))+
    scale_color_manual(values =c("Confirmed cases"='red1'))+
    scale_x_continuous(breaks=tem.d$week_seq,label=tem.d$plot_lab)+
    theme(panel.grid.major.x =element_blank(),
          panel.grid.minor.x =element_blank(),
          panel.grid.major.y =element_line(linetype=2),
          panel.grid.minor.y =element_blank(),
          axis.line.x.top =element_blank(),
          strip.text =element_text(size=16),
          axis.text.x.bottom =element_text(angle =90),
          panel.border =element_blank(),
          axis.line.y =element_line(linetype=1,colour="grey",linewidth=0.4,lineend="butt"),
          axis.line.x =element_line(linetype=1,colour="grey",linewidth=0.4,lineend="butt"),
          legend.position ="top",
          #axis.title.y =element_blank(),
          legend.text =element_text(size=14))+
    guides(fill=guide_legend(title =NULL),
           color=guide_legend(title =NULL))+
    xlab("Prediction \nYear_Week")
  
  tem.d$alarm_threshold<-as.numeric(tem.d$alarm_threshold)
  
  plot2<-ggplot()+
    
    geom_line(aes(x=week_seq,y=outbreak_probability,col="Outbreak probability"),linewidth=0.3,data=tem.d)+
    geom_point(aes(x=week_seq,y=outbreak_probability,col="Outbreak probability"),size=2.5,pch=15,data=tem.d)+
    geom_line(aes(x=week_seq,y=alarm_threshold,col="Alarm threshold"),linewidth=0.7,data=tem.d,lty=2)+
    facet_wrap(~pred_Distance)+
    
    theme_bw()+
    scale_color_manual(values =c("Outbreak probability"='dark blue',
                                 "Alarm threshold"="forest green"))+
    scale_x_continuous(breaks=tem.d$week_seq,label=tem.d$plot_lab)+
    theme(panel.grid.major.x =element_blank(),
          panel.grid.minor.x =element_blank(),
          panel.grid.major.y =element_line(linetype=2),
          panel.grid.minor.y =element_blank(),
          axis.line.x.top =element_blank(),
          strip.text =element_text(size=16),
          axis.text.x.bottom =element_text(angle =90),
          panel.border =element_blank(),
          axis.line.y =element_line(linetype=1,colour="grey",linewidth =0.4,lineend="butt"),
          axis.line.x =element_line(linetype=1,colour="grey",linewidth=0.4,lineend="butt"),
          legend.position ="top",
          axis.title.y =element_blank(),
          legend.text =element_text(size=14)
    )+
    guides(fill=guide_legend(title =NULL),
           color=guide_legend(title =NULL))+
    xlab("Prediction \nYear_Week")
  
  # ratio_DB2<-max(c(tem.d$outbreak_moving_limit,tem.d$outbreak,
  #                  tem.d$predicted_rate_uci),na.rm =T)/
  #   max(c(tem.d$outbreak_probability,tem.d$alarm_threshold),na.rm =T)
  
  ratio_DB2<-max(c(tem.d$outbreak_moving_limit,tem.d$outbreak_rate,
                   tem.d$predicted_rate_uci,
                   tem.d$observed_rate),na.rm =T)/
    max(c(tem.d$outbreak_probability,tem.d$alarm_threshold),na.rm =T)
  
  Plot_pros<-tem.d |> 
    dplyr::mutate(outbreak_probability_Signal=case_when(alarm_signal==1~outbreak_probability,
                                                        TRUE~as.numeric(NA)))
  #names(Plot_obs_rate)
  
  plot3<-ggplot(aes(x=week_seq,y=outbreak_moving_limit),data=Plot_pros)+
    geom_area(aes(fill="Endemic channel"),alpha=0.6)+
    geom_ribbon(aes(ymin=predicted_rate_lci,ymax=predicted_rate_uci,
                    fill="Prediction Interval"),alpha=0.5)+
    facet_wrap(~pred_Distance)+
    geom_line(aes(y=outbreak_rate,col="Predicted"),linewidth=0.3)+
    geom_point(aes(y=outbreak_rate,col="Predicted"),size=2.5,pch=15)+
    
    geom_line(aes(y=observed_rate,col="Observed rate"),linewidth=0.7,data=Plot_obs_rate)+
    
    geom_line(aes(x=week_seq,y=outbreak_probability*ratio_DB2,col="Outbreak probability"),linewidth=0.5,data=Plot_pros)+
    geom_point(aes(x=week_seq,y=outbreak_probability*ratio_DB2,col="Outbreak probability"),size=2.5,pch=15,data=Plot_pros)+
    geom_point(aes(x=week_seq,y=outbreak_probability_Signal*ratio_DB2,col="Alarm signal"),size=2.5,pch=8,data=Plot_pros)+
    geom_line(aes(x=week_seq,y=alarm_threshold*ratio_DB2,col="Alarm threshold"),linewidth=0.5,data=Plot_pros)+
    theme_bw()+
    scale_fill_manual(values=c("Endemic channel"='#90CCE3',
                               "Prediction Interval"="#9C7BAF"))+
    scale_color_manual(values =c("Predicted"='red1',
                                 "Observed rate"='#46353A',
                                 "Alarm signal"='#AA3377',
                                 "Outbreak probability"='blue',
                                 "Alarm threshold"="forest green"))+
    scale_y_continuous(name = "Incidence Rate\n per 100000",sec.axis =sec_axis(~ . /ratio_DB2,name="Probability"))+
    scale_x_continuous(breaks=Plot_pros$week_seq,label=Plot_pros$plot_lab)+
    theme(panel.grid.major.x =element_blank(),
          panel.grid.minor.x =element_blank(),
          panel.grid.major.y =element_line(linetype=2),
          panel.grid.minor.y =element_blank(),
          axis.line.x.top =element_blank(),
          strip.text =element_text(size=20,face='bold'),
          strip.background =element_rect(fill="#CCBB44"),
          axis.text.x.bottom =element_text(angle =90,size=14,vjust=0.5),
          panel.border =element_blank(),
          axis.line.y =element_line(linetype=1,colour="grey",linewidth=0.4,lineend="butt"),
          axis.line.x =element_line(linetype=1,colour="grey",linewidth=0.4,lineend="butt"),
          legend.position ="top",
          axis.title =element_text(size=18),
          axis.text.y =element_text(size=20,face='bold'),
          legend.text =element_text(size=14)
          
    )+
    guides(fill=guide_legend(title =NULL,nrow=2),
           color=guide_legend(title =NULL,nrow=2))+
    xlab("Prediction \nYear_Week")
  
  plot4<-ggplot(aes(x=week_seq,y=response_cat),data=Plot_pros)+geom_point(pch=21,size=2.5)+
    geom_hline(yintercept =0.5,col="yellowgreen",linewidth=0.8)+
    geom_hline(yintercept =1,col="orange",linewidth=0.8)+
    geom_hline(yintercept =1.5,col="brown",linewidth=0.8)+
    geom_hline(yintercept =2,col="red",linewidth=0.8)+
    geom_text(aes(x=x,y=y,label=response_cat,
                  size=8,
                  col=response_cat),data=dat_lab,
              show.legend =F,hjust=0,nudge_x =0.2)+
    theme_bw()+
    facet_wrap(~pred_Distance)+
    
    
    scale_x_continuous(breaks=Plot_pros$week_seq,label=Plot_pros$plot_lab)+
    
    scale_color_manual(values=c("No response"='yellowgreen',
                                "Initial response"='orange',
                                "Early response"='brown',
                                "Late/emergency response"='red'))+
    
    theme(panel.grid.minor.y =element_blank(),
          panel.grid.major.y =element_blank(),
          panel.grid.major.x =element_blank(),
          panel.grid.minor.x =element_blank(),
          panel.border =element_blank(),
          strip.background =element_rect(fill="#CCBB44"),
          axis.line.x =element_line(linetype=1,
                                    colour="grey",
                                    linewidth=0.4,
                                    lineend="butt"),
          axis.text.x.bottom =element_text(angle =90,size=14,vjust=0.5),
          axis.title.y =element_blank(),
          strip.text =element_text(size=20,face='bold'),
          axis.text.y=element_blank(),
          axis.ticks.y =element_blank(),
          legend.text =element_text(size=14))+
    coord_fixed(6,ylim =c(0.3,3),xlim = c(-20,Week_seQF))+
    xlab("Prediction \nYear_Week")
  
  meta_Prediction<-tribble(~var,~Value,
                           "Alarm indicators",paste0(alarm_vars_pros,collapse ='\n'),
                           "Selected lags",paste0(Selected_lag_Vars_pros,collapse ='\n'),
                           "Prediction distance",paste0(as.character(Computed_pred_Distance),' Weeks'),
                           "Z outbreak",as.character(selected_zvalue_pros),
                           "Alarm threshold",as.character(Selected_out_threshold),
                           
  )
  
  factor_vars<-c("Alarm indicators",
                 "Selected lags",
                 "Prediction distance",
                 "Z outbreak",
                 "Alarm threshold")
  
  meta_Prediction1<-meta_Prediction |> 
    dplyr::mutate(var=factor(var,levels=factor_vars),
                  id=1)
  
  meta_Prediction_wide<-meta_Prediction1 |> 
    dplyr::group_by(id) |> 
    tidyr::spread(var,'Value') |> 
    dplyr::ungroup() |> 
    dplyr::select(-id)
  
  border_prop<-officer::fp_border(width=0.5)
  
  meta_Prediction_wide |> 
    qflextable() |> 
    #flextable::autofit(add_w=0,add_h=10,unit='mm',part='body') |> 
    flextable::set_caption(as_paragraph(paste('District ',District_Now)),
                           fp_p=officer::fp_par(text.align = "left")) |> 
    flextable::fit_to_width(max_width =12,unit='in') |> 
    flextable::fontsize(part='header',size=14) |> 
    flextable::fontsize(part='body',size=12) |> 
    flextable::font(part='all',fontname ="Arial") |> 
    flextable::padding(part="body",padding = 2) |> 
    flextable::bold(i=1,part ="header") |> 
    flextable::border_inner_h(border =border_prop) |> 
    flextable::color(i=1,color="#875C7A",part="header") |> 
    flextable::color(i=1,j=5,color="#EE6677",part="body") |> 
    flextable::bg(i=1,bg="grey",part="header") |> 
    #flextable::border_remove() |> 
    flextable::align(part='all',align ="center") |> 
    flextable::hline() |> 
    flextable::vline() |> 
    flextable::vline_left() |> 
    flextable::hline_top() 
  
  
  pred_Metadata_tab<-meta_Prediction1[,-3] |> 
    qflextable() |> 
    flextable::set_caption(as_paragraph(list_values=paste0("\n District ",District_Now,'\n \n ')),
                           fp_p=officer::fp_par(text.align = "left")) |> 
    
    flextable::fit_to_width(max_width =12,unit='in') |> 
    flextable::fontsize(part='header',size=14) |> 
    flextable::fontsize(part='body',size=12) |> 
    flextable::font(part='all',fontname ="Arial") |> 
    flextable::padding(part="body",padding = 2) |> 
    flextable::bold(j=1,part ="body") |> 
    flextable::border_inner_h(border =border_prop) |> 
    flextable::color(j=1,color="#875C7A",part="body") |> 
    flextable::color(i=5,j=2,color="#EE6677",part="body") |> 
    flextable::bg(j=1,bg="grey",part="body") |> 
    #flextable::border_remove() |> 
    flextable::align(part='all',align ="left") |> 
    flextable::hline() |> 
    flextable::vline() |> 
    flextable::vline_left() |> 
    flextable::hline_top() |> 
    flextable::delete_part("header") |> 
    flextable::hline_top()
  
  pred_Metadata_tab
  

  vars_Tab_out<-c('district','year.x','week.x','week_seq','plot_lab','outbreak_moving',
                  'outbreak_moving_sd','outbreak_moving_limit','endemic_chanel',
                  'predicted_cases','predicted_cases_lci','predicted_cases_uci',
                  'predicted_rate','predicted_rate_lci','predicted_rate_uci',
                  'endemic_threshold','outbreak','outbreak_rate','prob_exceed_threshold',
                  'alarm_threshold','outbreak_probability','outbreak_period','alarm_signal')
  
  #paste0("'",names(Plot_pros),"'",collapse =',')
  
  table_Out_pros1<-Plot_pros |> 
    dplyr::select(all_of(vars_Tab_out)) |> 
    dplyr::rename(year=`year.x`,
                  week=`week.x`)
  
  ## link with user Input data
  
  #names(all_Pros_Predictions_Week_a)
  
  select_User_vars<-c("district","week_seq",alarm_vars_pros,"Cases","population","observed_rate")
  select_User_vars2<-c("district","year","week",alarm_vars_pros,"Cases","population","observed_rate")
  
  
  all_Pros_Predictions_Week_Link<-all_Pros_Predictions_Week_a |> 
    dplyr::select(all_of(select_User_vars))
  
  all_Pros_Predictions_Week_Link2<-all_Pros_Predictions_Week_a |> 
    dplyr::select(all_of(select_User_vars2))
  
  ## save DB2_output
  
  
  table_Out_pros2<-table_Out_pros1 |> 
    dplyr::left_join(all_Pros_Predictions_Week_Link,by=c("district","week_seq"))
  
  paste0("'",names(table_Out_pros2),"'",collapse =',')
  
  vars_Tab_all_Out<-c('district','year','week','week_seq','plot_lab',
                      alarm_vars_pros,'Cases','population','observed_rate',
                      'outbreak_moving','outbreak_moving_sd','outbreak_moving_limit','endemic_chanel',
                      'predicted_cases','predicted_cases_lci','predicted_cases_uci',
                      'predicted_rate','predicted_rate_lci','predicted_rate_uci','endemic_threshold',
                      'outbreak','outbreak_rate','prob_exceed_threshold','alarm_threshold',
                      'outbreak_probability','outbreak_period','alarm_signal'
  )
  
  table_Out_pros<-table_Out_pros2[,vars_Tab_all_Out]
  
  db2_Output<-list(Plot_pros=Plot_pros,
                   meta_Prediction_wide=meta_Prediction_wide,
                   meta_Prediction=meta_Prediction,
                   table_Out_pros=table_Out_pros,
                   Plot_obs_rate=Plot_obs_rate,
                   ratio_DB2=ratio_DB2,
                   dat_lab=dat_lab)
  
  ewars_json_Objects_pros<-list(Plot_pros=Plot_pros,
                                alarm_vars_pros=alarm_vars_pros,
                                select_User_vars2=select_User_vars2,
                                all_Pros_Predictions_Week_Link=all_Pros_Predictions_Week_Link2,
                                meta_Prediction_wide=meta_Prediction_wide,
                                meta_Prediction=meta_Prediction,
                                table_Out_pros=table_Out_pros,
                                Plot_obs_rate=Plot_obs_rate,
                                ratio_DB2=ratio_DB2,
                                dat_lab=dat_lab)
  
  
  
  ewars_json_Objects_pros_json<-toJSON(ewars_json_Objects_pros)
  
  #dbII_obj_name_save<-file.path(shinyDBII_obj_pth,"Shiny_DBII_Objs.rds")
 # saveRDS(db2_Output,dbII_obj_name_save,compress =T)
  
  #generate the report Document
  
  grob_Fun<-function(p_lot){
    if(class(p_lot)[1]=="gg"){
      ggplot2::ggplotGrob(p_lot)
    }else{
      p_lot
    }
  }
  
  plot3_Grobs<-list(grob_Fun(plot3))
  plot4_Grobs<-list(grob_Fun(plot4))
  
  # For_book_pros<-list(Pred_met_data=pred_Metadata_tab,
  #                #plot1=plot1,
  #                #plot2=plot2,
  #                Prediction_plot=plot3,
  #                Response_cat=plot4,
  #                ratio_DB2=ratio_DB2
  #                )
  
  For_book_pros<-list(Pred_met_data=pred_Metadata_tab,
                      #plot1=plot1,
                      #plot2=plot2,
                      Prediction_plot=plot3_Grobs,
                      Response_cat=plot4_Grobs,
                      ratio_DB2=ratio_DB2
  )
  
  
  path_mkDB2<-file.path(reporT_pAth,"Markdown_pros")
  

  #if(!dir.exists(report_pth)) dir.create(report_pth,recursive =T)
  if(!dir.exists(path_mkDB2)) dir.create(path_mkDB2,recursive =T)
  
  file.copy("_quarto_pros.yml",path_mkDB2,overwrite =T)
  file.rename(file.path(path_mkDB2,"_quarto_pros.yml"),file.path(path_mkDB2,"_quarto.yml"))

  suppressWarnings(saveRDS(For_book_pros,file.path(reporT_pAth,"report_obj_pros.rds")))
  write(ewars_json_Objects_pros_json,file.path(reporT_pAth,"ewars_json_Objects_prospective.json"))
  saveRDS(ewars_json_Objects_pros,file.path(reporT_pAth,"ewars_pros_Pred_Objects.rds"))
  
  
  #header_fl<-"book_header_File.qmd"
  header_fl<-"book_header_File.txt"
  
  
  heade_File<-read.csv(header_fl)
  names(heade_File)<-"var"
  
  
  

  n_pred_meta<-length(For_book_pros$Pred_met_data)
  n_Prediction_Plot<-1#length(For_book_pros$Prediction_plot)
  n_Response_plot<-1#length(For_book_pros$Response_cat)
  
  
  tribble_Meta_qmd<-tribble(~var,~type,~header_name,~N,
                            "Pred_met_data","Table","# Prediction meta data {#meta-dat}",1,
                            "Prediction_plot","Plot","# Prediction plot {#pred_plots}",n_Prediction_Plot,
                            "Response_cat","Plot","# Response {#response_plot}",n_Response_plot,
                            
  )
  
  mm<-2
  
  for (mm in 1:nrow(tribble_Meta_qmd)){
    
    tribble_Meta_qmd_Sub<-tribble_Meta_qmd[mm,]
    
    quarto_cmd0<-tribble(~var,
                         tribble_Meta_qmd_Sub$header_name,
                         " ",
                         "```{r}")
    
    quarto_cmd1<-tribble(~var,
                         paste0('report_pth<-"',reporT_pAth,'"'),
                         paste0('report_Obj<-readRDS(file.path(report_pth,"report_obj_pros.rds"))'),
                         #paste0('Descriptive_table1<-report_Obj$Descriptive_table1'),
                         paste0(tribble_Meta_qmd_Sub$var,'<-report_Obj$',tribble_Meta_qmd_Sub$var),
                         "ratio_DB2<-report_Obj$ratio_DB2",
                         "```"
    )
    
    quarto_cmd2<-tribble(~var,
                         " ",
                         " ",
                         "```{r}",
                         "#| echo: false",
                         "#| message: false",
                         "#| warning: false",
                         "#| comment: ' '" ,
    )
    
    plot_layout_cmd<-tribble(~var,
                             "#| layout-ncol: 2"
    )
    
    plot_fig_width<-tribble(~var,
                            "#| fig-width: 13" ,
                            "#| out-width: 13in" ,
    )
    
    # determine the output the tribble
    
    if(tribble_Meta_qmd_Sub$type=="Table"){
      quarto_cmd3<-tribble(~var,
                           tribble_Meta_qmd_Sub$var  
      )
    }else if(tribble_Meta_qmd_Sub$type=="Plot"){
      
     
        # quarto_cmd3_0<-tribble(~var,
        #                        tribble_Meta_qmd_Sub$var  
        # )
        plot_Con<-function(bb){
          
          var_Plot0<-paste0(tribble_Meta_qmd_Sub$var,"[[",bb,"]]")
          
          tribble(~var,
                  paste0("if(class(",var_Plot0,")[1]=='gtable'){"),
                  paste0("plot(",var_Plot0,")"),
                  "}else{",
                  var_Plot0,
                  "}"
          )
        }
        quarto_cmd3_0<-foreach(aa=1:tribble_Meta_qmd_Sub$N,.combine =rbind)%do% plot_Con(aa)
        quarto_cmd3<-rbind(plot_fig_width,quarto_cmd3_0)
     
    }
    quarto_Last<-tribble(~var,
                         "```")
    
    unlink(file.path(path_mkDB2,paste0( tribble_Meta_qmd_Sub$var,'.qmd')))
    
    cat(c(quarto_cmd0$var,
          heade_File$var,
          quarto_cmd1$var,
          quarto_cmd2$var,
          quarto_cmd3$var,
          quarto_Last$var),
        file =file.path(path_mkDB2,paste0( tribble_Meta_qmd_Sub$var,'.qmd')),
        sep="\n")
    
    
    
    
  }
  # organize the report into section
  
  tribble_Sections_pros<-tribble(~var,~type,~header_name,~N,
                            "index","text",paste0("# Prospective prediction district ",District_Now," {.unnumbered}"),1,
                            "Meta_data","text","# Prediction meta data {.unnumbered}",1,
                            "Prediction_plots","text","# Prediction plots  {.unnumbered}",1,
                            "Respose_category","text","# Response category{.unnumbered}",1,
                            
  )
  
  
  for (ss in 1:nrow(tribble_Sections_pros)){
    
    tribble_Sections_Sub<-tribble_Sections_pros[ss,]
    
    quarto_cmd0<-tribble(~var,
                         tribble_Sections_Sub$header_name,
                         " ")
    cat(quarto_cmd0$var,
        file =file.path(path_mkDB2,paste0( tribble_Sections_Sub$var,'.qmd')),
        sep="\n")
    
  }
  
  qmd_order<-c("index",
               "Meta_data",
               "Pred_met_data",
               "Prediction_plots",
               "Prediction_plot",
               "Respose_category",
               "Respose_cat"
  )
  
  cat(paste0('- ',qmd_order,'.qmd',sep='\n'))
  
  
  #?quarto_render
  if(generate_DBII_report_html==1){
    
  quarto_render(input=path_mkDB2,as_job=F)
    
  }
  
  }else{
    Message_pred<-paste0("Please generate prediction weights for district ..",District_Now," in DBI")
    cat(Message_pred,sep='\n')
  }
  
  
}

