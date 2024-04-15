r = getOption("repos")
r["CRAN"] = "http://cran.us.r-project.org"
options(repos = r)


#pkgs<-c('pse') # r version issue
#pkgs<-c('plotly')
pkgs<-c()
pkgs<-c('reportROC')
pkgs<-c('raster')  # dependencie issue
#pkgs<-c('rgeos')  # not available for  r version
pkgs<-c('doParallel')
pkgs<-c('profvis')
pkgs<-c('ROCit')

#, 'XLConnect','plyr','dplyr','car','stringr','zoo','foreign','ggplot2','splines',
#        'mgcv','Hmisc','xtable','foreach','xlsx','lattice','latticeExtra',"gridExtra","grid","shiny")

lapply(pkgs, install.packages, character.only = TRUE)


library(plyr)## This package should be loaded before dplyr
library(dplyr)
library(car)
library(stringr)
library(zoo)
library(foreign)
library(ggplot2)
library(splines)
library(mgcv)
library(Hmisc)
library(xtable)
library(foreach)
#library(xlsx)## Ensure java installed on system same as one for R, e.g 64 bit R with 64 bit Java
#library(XLConnect)
library(lattice)
library(latticeExtra)
library(gridExtra)
library(grid)
#library(googlesheets) # r version issue
library(reshape2)
library(plotly)
library(tidyr)
library(lubridate)
#library(pse)# for automatic calibration
library(reportROC)
library(caret)
library(e1071)
library(knitr)
#library(pkgload)
#library(rgeos)
#library(raster)
#library(leaflet)
#library(pak)

#options(repos=c(INLA="https://inla.r-inla-download.org/R/stable",
               # CRAN="https://cran.rstudio.com/"))
##load INLA package

#library(INLA)
library(doParallel)
library(profvis)

#library(future)
#library(promises)
#library(ipc)
library(readxl)
library(ROCit)
library(RSQLite)
library(DT)
## included after risk mapping integration

library(tmap)
library(tmaptools)
library(leaflet)
library(rgdal)
library(stringr)
library(xts)
library(dygraphs)
library(SpatialEpi)
library(spdep)
library(cleangeo)
library(dlnm)
library(ggthemes)
library(RColorBrewer)
library(tsModel)
library(kableExtra)
library(viridis)
#library(INLA)
library(data.table)
library(reportROC)
library(promises)
library(future)
plan(multisession,workers=2)


#if (!getDoParRegistered()){
  #cl <<- makeCluster(5)
  #registerDoParallel(cl)
#}


server<-function(input,output,session) {

  #ISO2<-"LKA"
  #Country_name<-"Sri Lanka"

  output$title_txt<-renderUI(tags$h3("Ewars Dashboard +",style="font:cambria"))
  con <- dbConnect(SQLite(),"users.sqlite")
  pb<-dbGetQuery(con, "SELECT user_name,password,role FROM users_db")
  #pb<-data.frame(user_name="demo",password="demo_2019",role="admin")
  pb_dis<-dbGetQuery(con, "SELECT user_name,district_code FROM users_districts")
  dbDisconnect(con)

  login = F
  USER <- reactiveValues(login = login)
  user_info<-reactiveValues(nm_pwd="demo",user="demo_2019")

  observeEvent(input$lg_in,{
    user_info$nm_pwd<-paste(str_trim(tolower(input$user_name)),str_trim(input$pwd))
    user_info$user<-str_trim(tolower(input$user_name))
    output$logged_in_user<-renderUI(input$user_name)

  })


  observe({

    role_d1<-pb %>% dplyr::filter(str_trim(tolower(user_name))==user_info$user)
    #role_d<-pb %>% dplyr::filter(str_trim(tolower(user_name))==user_info$user)
    if(nrow(role_d1)==1){
      role_d<-role_d1
    }else{
      role_d<-data.frame(user_name="xxx",role="xxx",stringsAsFactors =F)
    }
    #print(role_d)
    ##change this part for production 2020-11-26
    #-------------------------
    #-------------------------
    if (user_info$nm_pwd %in% paste(str_trim(tolower(pb$user_name)),str_trim(pb$password))){

      USER$login=T
    }
    #print(user_info$nm_pwd)
    #print(pb)
    output$log_list<-renderUI(
      if(USER$login==T & str_trim(tolower(role_d$role))=="admin"){
        ui_yes
      }

      else if(USER$login==T & str_trim(tolower(role_d$role))=="district manager"){
        ui_yes_Restricted
      }

      else if(USER$login==F & !user_info$nm_pwd ==""){
        tagList(login_screen,br(),tags$h4("Please enter a valid user name and password"))
      }else if(USER$login==F & user_info$nm_pwd ==""){
        login_screen
      }
    )
  })


  ## read in the user database


  output$logout <- renderUI({
    req(USER$login)
    tags$h5(a( "logout",
               href="javascript:window.location.reload(true)"))
    #class = "dropdown",
    #style = "background-color: #eee !important; border: 0;
    #font-weight: bold; margin:5px; padding: 10px;height:40px")
  })

  #paste0(getwd(),'/',"New_model_server_Script.R")
  #paste0(getwd(),'/',"New_model_Spatial_temporal_with_promises.R")
  #paste0(getwd(),'/',"Reactive_function_DBII_New_Model.R")
  #eval(parse(text=readLines("New_model_helper_Functions.R")))
  eval(parse(text=readLines("New_model_server_Script.R")))
  eval(parse(text=readLines("Spatial_temporal.R")))
  #eval(parse(text=readLines("New_model_Spatial_temporal_Compute_UI.R")))

  eval(parse(text=readLines("Reactive_function_DBII_New_Model.R")))
  output$spat_Display_new_Model<-renderUI({output_graphs_New_Model})

  ### New model



}
