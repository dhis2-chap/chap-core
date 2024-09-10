##objects

##login items  

login_screen<-fluidPage(tags$h1("EWARS Dashboard +"),
                        column(width=8,align="left",offset =0,
box(width =4,
wellPanel(tags$h3("Please login"),
                         textInput("user_name","user name",value="demo"),
                        passwordInput("pwd","Password",value="demo_2019"),
                        actionButton("lg_in","login",
                                     style="color: white; background-color:#3c8dbc;
                                 padding: 10px 15px; width: 150px; cursor: pointer;
                                 font-size: 18px; font-weight: 300;")
          
                        )

)
)
)

output_graphs<-tabsetPanel(
  tabPanel("Runnin",
           fluidRow(
             column(10,plotlyOutput("plot1",height ="20%",width="50%"))
           )),
  tabPanel("Evaluation",
           fluidRow(
             
             column(10,plotlyOutput("plot2",inline =F,height ="20%",width="50%"))
             
           )),
  tabPanel("Runnin +Evaluation",
           fluidRow(
             column(12,plotlyOutput("plot3",height ="20%",width="50%"))
           )),
  tabPanel("Sensitivity & PPV",
           fluidRow(
             column(5,
                    wellPanel(tableOutput("table1")))
                    
                    )
           ),
  tabPanel("Auto-calibration outputs",
           fluidRow(
             column(5,wellPanel(tableOutput("table2")))
           )),
  tabPanel("Output files",
                  uiOutput("output_files")                
  )  
  )

output_dis<-column(3,
  selectInput(inputId = "district",
            label = "District",
            choices = c(15,20,3),
            selected =4,
            multiple =F))

output_dis1<-"inputPanel(fluidRow(
             column(6,
                    selectInput(inputId = 'district_dB2',
                           label = 'District',
                   choices = 999,
                   selected =999,
                   multiple =F))))
                   "

output_dis2<-"inputPanel(
                  selectInput(inputId = 'district',
label = 'District',
choices = sel_districts(),
selected =sel_districts()[1],
multiple =F)
)"

dat_opts<- tabPanel("Data",
                    column(12,
                    shinydashboardPlus::box(width=12,
              
              fileInput('dat', 'choose file to upload', accept=c('.xls','.xlsx','.csv')),
              textInput("original_data_sheet_name", "Choice of sheet name for the original data", "Sheet1"),
              checkboxInput("graph_per_district", 
                            "Specify graph per district/municipality option",value =TRUE),
              
              
             #checkboxInput("generating_surveillance_workbook",
                            #"Generate surveillance workbooks", value=FALSE),
              checkboxInput("spline",
                            "Spline", value=FALSE),
              selectInput(inputId = "run_per_district",
                          label = "District codes",
                          choices = c(15,20,3,999),
                          selected =999,
                          multiple =T),
             checkboxInput("Run_all",
                           "Run all districts",
                           value =F))
))

sel_vars<-tabPanel("Variables & Runnin",
      
      column(12,
             shinydashboardPlus::box(width=12,
                     #selectInput('country_code', 'country code', 
                                 #c("XX"),
                                 #selected="XX", width="35%"),
                     #textInput("psswd", "8 digit password", "12345678"),
  selectInput("population",
              "Variable for annual total Population",
              "population",multiple =F),
  selectInput("number_of_cases", 
              "Variable for the weekly number of outbreak",
              "weekly_hospitalised_cases",multiple =F),
  selectInput("alarm_indicators", selected =c("rainsum","meantemperature"),
              "Alarm indicator(s)",
              choices=c("rainsum","meantemperature"),multiple =T),
  # when spline is selected show options to select varaibles
  
  conditionalPanel("input.spline",
  selectInput("spline_alarm_indicators", selected =c("rainsum","meantemperature"),
                               "Spline Alarm indicator(s)",
                               choices=c("rainsum","meantemperature"),multiple =T)),
  sliderInput("stop_runinYear", 
              "Specify the year the run-in period stops",
              value =2012,step =1,
              min=1990,max=2030,sep=''),
  sliderInput("stop_runinWeek", 
              "Specify the week the run-in period stops",
              value =52,step =1,
              min=52,max=52)
  )))


calib1<-                column(12,
                        shinydashboardPlus::box(
                          width = 12,
                          sliderInput(
                            "outbreak_week_length",
                            #"Outbreak weeks to declare outbreak period/stop outbreak",
                            "Outbreak period",
                            min = 1,
                            max = 24,
                            value = 3,
                            step = 1,
                            ticks = F,
                            dragRange = F
                          ),
                          sliderInput(
                            "alarm_window",
                            "Alarm window size",
                            value = 3,
                            min = 1,
                            max = 24,
                            step = 1
                          ),
                          sliderInput(
                            "alarm_threshold",
                            #"Alarm threshold for alarm signal",
                            "Alarm threshold",
                            value = .12,
                            min = 0.0005,
                            max = 1,
                            step = 0.0001
                          )
                          ,
                          selectInput(
                            "season_length",
                            "Seasons in a year",
                            choices = c(1,2,3,4,6,12),
                            selected=1,multiple =F
                             )
                        )
                 )



calib2<-                 column(12,
                        shinydashboardPlus::box(
                          width = 12,
                          sliderInput(
                            "z_outbreak",
                            #"Multiplier for standard deviation to vary  endemic channel",
                            "Z-outbreak",
                            value = 1.25,
                            min = 1,
                            max = 4,
                            step = 0.01
                          ),
                          sliderInput(
                            "outbreak_window",
                            "Outbreak window size",
                            value = 4,
                            min = 1,
                            max = 24,
                            step = 1
                          ),
                          sliderInput(
                            "prediction_distance",
                            #"Distance between current week and target week to predict an outbreak signal",
                            "Prediction Distance (time lag)",
                            value = 2,
                            min = 1,
                            max = 24
                          ),
                          sliderInput(
                            "outbreak_threshold",
                            #"Cut-off value to define the outbreak signal",
                            "Outbreak threshold",
                            
                            value = 0.75,
                            min = 0.00005,
                            max = 1,
                            step = 0.0001)
                        )
                )
             
            

calib1_auto<-           column(12,
                        shinydashboardPlus::box(
                          width = 12,
                          sliderInput(
                            "outbreak_week_length",
                            "Outbreak weeks to declare outbreak period/stop outbreak",
                            min = 1,
                            max = 24,
                            value = c(2,6),
                            step = 1,
                            ticks = F,
                            dragRange = F
                          ),
                          sliderInput(
                            "alarm_window",
                            "Alarm window size",
                            value = c(2,6),
                            min = 1,
                            max = 24,
                            step = 1
                          ),
                          sliderInput(
                            "alarm_threshold",
                            "Alarm threshold for alarm signal",
                            value = c(0.05,0.6),
                            min = 0.0005,
                            max = 1,
                            step = 0.0001
                          ),
                          sliderInput(
                            "season_length",
                            "Seasons in a year",
                            value = c(52,52),
                            min = 1,
                            max = 52,
                            step = 1 )
                        )
                 )



calib2_auto<-column(12,
                        shinydashboardPlus::box(
                          width = 12,
                          sliderInput(
                            "z_outbreak",
                            "Multiplier for standard deviation to vary  endemic channel",
                            value = c(0.05,3),
                            min = 0.0001,
                            max = 4,
                            step = 0.001
                          ),
                          sliderInput(
                            "outbreak_window",
                            "Outbreak window size",
                            value = c(2,10),
                            min = 1,
                            max = 24,
                            step = 1
                          ),
                          sliderInput(
                            "prediction_distance",
                            "Distance between current week and target week to predict an outbreak signal",
                            value = c(3,12),
                            min = 1,
                            max = 24
                          ),
                          sliderInput(
                            "outbreak_threshold",
                            "Cut-off value to define the outbreak signal",
                            value = c(0.05,0.6),
                            min = 0.00005,
                            max = 1,
                            step = 0.0001)
                        )
                 )



row_elements1.x<-fluidRow(column(3,
                               checkboxInput("automate",
                                             "Automatic Calibration",
                                             value =F)),
                               
                        )

row_elements1<-fluidRow(column(3,
                               checkboxInput("automate",
                                             "Automatic Calibration",
                                             value =F)),
                        
                        column(5,
                               selectInput(
                                 "iterations",
                                 "Auto iterations",
                                 choices = c(100,500,1000,5000,10000),
                                 selected=100,multiple =F
                               ))
                        
)


row_elements2<-fluidRow(     
  column(3,
    tabsetPanel(id="input_pan",
    dat_opts,
    sel_vars,
    tabPanel("calib1",calib1),
    tabPanel("Calib2",calib2)
    )),
  
  column(7,output_graphs,
            output_dis),
  column(1,
         actionButton('save_mode','Save Model',
                      style="color: forestgreen; background-color:grey(0.5);
                                 padding: 10px 15px; height: 80px; cursor: pointer;
                                 font-size: 20px; font-weight: 400;")
  )
  )

row_elements2_restricted<-fluidRow(     
  column(3,
         tabsetPanel(id="input_pan",
                     dat_opts,
                     sel_vars,
                     tabPanel("calib1",calib1),
                     tabPanel("Calib2",calib2)
         )),
  
  column(8,output_graphs,
         output_dis)
)
  
dashboad_elements_I<-tabPanel("Dashboard I",
         fluidPage(
           row_elements1,    
           row_elements2
                        
)
 )               #actionButton("manual",icon=icon("refresh"),
                #"Manual Calibration", value=T),
dashboad_elements_restricted_I<-tabPanel("Dashboard I",
                              fluidPage(
                                row_elements1,    
                                row_elements2_restricted
                                
                              )
)                    
         


#base_input<-"selectInput('District','District',choices =c(3,15,20,999),selected =999,multiple =F),

base_input<-"selectInput('year','Year',choices =2019:2030,selected =2019,multiple =F),
selectInput('week','Week',choices =1:53,selected =2,multiple =F),
numericInput('Cases','Weekly number of cases',value=NA),
numericInput('Population','Population',value=NA)"


## nest Risk mapping in dashboard II

## risk mapping elements

output_graphs_spat_Agg<-tabsetPanel(
  
  tabPanel("Spatial_Plots", 
           uiOutput("Error"), 
           uiOutput("Spat_Plot")  
  )
  ,
  tabPanel("Time_series",
           uiOutput("Time_series_Plots")
  ),
  tabPanel("Risk_maps",
           fluidRow(
             column(12,tmapOutput("plot3_spat"))
           ))
)

output_graphs_spat_point<-tabsetPanel(
  
  tabPanel("Spatial_Plots", 
           fluidRow(
           uiOutput("Error"), 
           column(12,tmapOutput("Spat_Plot"))
           )
  )
  ,
  tabPanel("Risk_maps",
           fluidRow(
             uiOutput("plot3_spat")
            
           ))
)

output_year_spat<-column(3,
                         sliderInput(inputId = "Year_plot_spat",
                                     label = "Year",
                                     min = 2008,
                                     max=2030,
                                     value =2012,
                                     sep='',
                                     step=1),
                         sliderInput(inputId = "Week_plot_spat",
                                     label = "Week",
                                     min = 1,
                                     max=52,
                                     step=1,
                                     sep='',
                                     value =35)
)


dat_opts_spat<- tabPanel("Spatial data Upload",
                         column(12,
                                shinydashboardPlus::box(width=12,
                                        ##choose point/ lat long data
                                        
                                        fileInput('shape_file', 'District sub-district boundary file (.shp file)',
                                                  accept=c('.shp','.dbf','.shx','.prj'),
                                                  multiple =T),
                                        ## choose aggregated or point data
                                        
                                        selectInput("spat_Input_Type", selected =c("sub_district"),
                                                    "Geographic data input (point or aggregated)",
                                                    choices=c("point (LatLon)","sub_district"),multiple =F),
                                        
                                        fileInput('dat_spat', 'Choose surveillance data with spatial inputs', accept=c('.xls','.xlsx','.csv')),
                                        uiOutput("Spat_error"),
                                        selectInput("population_spat",
                                                    "Variable for annual total Population",
                                                    "population",multiple =F),
                                        selectInput("number_of_cases_spat", 
                                                    "Variable for the weekly number of outbreak",
                                                    "weekly_hospitalised_cases",multiple =F),
                                        selectInput("alarm_indicators_spat", selected =c("rainsum","meantemperature"),
                                                    "Alarm indicator(s)",
                                                    choices=c("rainsum","meantemperature"),multiple =T),
                                        # when spline is selected show options to select variables
                                        
                                        sliderInput("stop_runinYear_spat", 
                                                    "Specify the year the run-in period stops",
                                                    value =2012,step =1,
                                                    min=1990,max=2030,sep=''),
                                        sliderInput("stop_runinWeek_spat", 
                                                    "Specify the week the run-in period stops",
                                                    value =52,step =1,
                                                    min=1,max=52)
                                        
                                        
                                )
                         ))



row_elements_spat<-fluidRow(     
  column(4,
         tabsetPanel(id="input_pan_spat",
                     dat_opts_spat
                     # sel_vars
         )),
  
  column(8,
         uiOutput("spat_Display")
         ,
         output_year_spat)
)



dashboad_elements_Risk_mapping<-tabPanel("Risk mapping",
                                         fluidPage(
                                           row_elements_spat    
                                           
                                         )
)              




dashboad_elements_II<-tabPanel("Dashboard II",
                             
                             column(3,
                                    tabsetPanel(  
                                      tabPanel("Parameters",
                                            wellPanel(tags$h3("Alarm Indicators:"),
                                              uiOutput('a_vars')),
                                              wellPanel(br(),
                                                
                                               tableOutput("alr_vars")),
                                            wellPanel(tags$h3("Spline Indicators:"),
                                                      uiOutput('s_vars')))
                                      
                                    )
                                    
                             ),
                             column(9,
                                    tabsetPanel(
                                      tabPanel("Input Data",
                                          fluidPage(fluidRow(uiOutput("input_dataset")),
                                                    
                                                      inputPanel(
                                                        fluidRow(
                                                      column(3,
                                                      actionButton('ins_dat','Update table'))),
                                                      column(3,offset =6,
                                                             actionButton('Refresh_DB','Refresh'))
                                                      ),
                                               DT::dataTableOutput("data_dis")
                                               )),
                                      
                                               tabPanel("Prediction tables",
                                                      # inputPanel(uiOutput("sel_diss")),
                                                      DT::dataTableOutput("pred_dis")
                                                        ),
                                      
                                               tabPanel("Outbreak",
                                                        #inputPanel(uiOutput("sel_diss")),
                                                        plotOutput("db2_plot1",width ="700px",height ="320px")
                                                        ),
                                      
                                    tabPanel("Probability",
                                             #inputPanel(uiOutput("sel_diss1")),
                                             plotOutput("db2_plot2",width ="700px",height ="320px")
                                    ),
                                    
                                    tabPanel("Outbreak and Probability",
                                             #inputPanel(uiOutput("sel_diss1")),
                                             plotOutput("db2_plot3",width ="700px",height ="320px")
                                             ),
                                    
                                    tabPanel("Response",
                                             #inputPanel(uiOutput("sel_diss1")),
                                             plotOutput("db2_plot4")
                                             ),
                                    dashboad_elements_Risk_mapping
                                               
                                    )          
                                    
                                    
                             )
                            
                  )                         

dashboad_elements_III<-tabPanel("Help",
                                tags$br(),
                                tags$br(),
                                tags$div(
                                  HTML(paste(tags$strong("Early Warning and Response System for Dengue Outbreaks: Operational Guide using the web-based Dashboard"), 
                                             tags$a(href="https://drive.google.com/file/d/1MJWocIyu3Ecdy950w0Z2d9i50hceEFA1/view?usp=sharing", target="_blank", tags$br(),tags$b(" Operational Guide")),
                                             sep = ""))
                                ),
                               # tags$div(
                                #  HTML(paste(tags$strong("Addendum: Automatised Early Warning And Respons System"), 
                                 #            tags$a(href="https://drive.google.com/open?id=1VCZbxO6Qfy7oyt_2O1T_XGa-VNhjaCsF", target="_blank", tags$br(),tags$b("Automatised Early Warning And Respons System")),
                                 #            sep = ""))
                                #),
                                tags$div(
                                  HTML(paste(tags$strong("Demo Excel workbook"), 
                                             tags$a(href="https://drive.google.com/file/d/1ujlq5oZVSF8dg7A3KN6Csw-5LsEWqYzF/view?usp=sharing", target="_blank", tags$br(),tags$b("Demo data")),
                                             sep = ""))),
                                  
                                  tags$div(
                                    HTML(paste(tags$strong("Risk mapping demo files"), 
                                               tags$a(href="https://drive.google.com/drive/folders/1GXZ6vwEONEqxvUjLB4QMG0aduKkyIGIF?usp=sharing", target="_blank", tags$br(),tags$b("Risk mapping demo files")),
                                               sep = ""))),
                                  
                              
                                tags$br(),
                                tags$br()
                                
                                
                                
)  


dashboad_elements_IV<-tabPanel("R scripts and Files",
                               tags$br(),
                               tags$br(),
                               tags$div(
                                 HTML(paste(tags$strong("R scripts and Files"), 
                                            tags$a(href="https://umeauniversity-my.sharepoint.com/:f:/g/personal/odse0001_ad_umu_se/EpwpHBzDg2pIr6pZPIMtEnoBvVJUHSL2NZ97RoOQi6sx7A?e=225F2c", target="_blank", tags$br(),tags$b("App R scripts and files")),
                                            sep = ""))
                               )
                               
                               
)  
## admin page

add_user<-tabPanel("Add users",
                   fluidPage(fluidRow(
                     inputPanel(textInput("First_name","First Name"),
                                textInput("Last_name","Last name"),
                                textInput("users_name","User name"),
                                passwordInput("pass_word","Password"),
                                textInput("email","email address"),
                                selectInput(inputId = 'role',
                                            label = 'Role',
                                            choices = c("Admin","District Manager"),
                                            selected ="District Manager",
                                            multiple =F),
                                conditionalPanel('input.role=="District Manager"',
                                                 selectInput(inputId = 'district_manage',
                                                             label = 'District access',
                                                             choices =c(9999,99999) ,
                                                             selected =9999,
                                                             multiple =T)),
                                actionButton("Enter_user","Add user",
                                             style="color: white; background-color:#3c8dbc;
                                 padding: 10px 15px; width: 150px; cursor: pointer;
                                 font-size: 18px; font-weight: 300;")
                     ))),
                   fluidRow(
                     column(8,
                            DT::dataTableOutput("users_dat")),
                     column(4,
                            DT::dataTableOutput("users_districts"))
                   ))


remove_user<-tabPanel("Delete users",
                   fluidPage(
                     inputPanel(fluidRow(column(10,
                                selectInput(inputId = 'user_de',
                                            label = 'Users to delete',
                                            choices = "john doe",
                                            selected ="john doe",
                                            multiple =T,width =unit(5,'in'))))),
                     
                     fluidRow(column(4,
                            actionButton("Delete_user","Delete",
                                         style="color: white; background-color:#3c8dbc;
                                 padding: 10px 15px; width: 150px; cursor: pointer;
                                 font-size: 18px; font-weight: 300;"))
                   )))
                   


admin_page<-tabPanel("Admin page",
                     tabsetPanel(
                        add_user,
                        remove_user)
)

source("new_model_UI_elements.R")

ui_yes<-fluidPage(fluidRow(
  column(3,offset =5,
         uiOutput("title_txt")),
  column(2,offset =11,
         uiOutput("logged_in_user"),
         uiOutput('logout'))
),
  fluidRow(navbarPage("",
           #dashboad_elements_I,
           
           #dashboad_elements_II,
          
          # admin_page,
           
           #dashboad_elements_Risk_mapping,
           dashboad_elements_New_model,
           dashboad_elements_New_model_pros,
           
           dashboad_elements_III,
           
           dashboad_elements_IV
           
           # admin page
           
)

)
)

ui_yes_Restricted<-fluidPage(fluidRow(
  column(3,offset =5,
         uiOutput("title_txt")),
  column(2,offset =11,
         uiOutput("logged_in_user"),
         uiOutput('logout'))
),
fluidRow(navbarPage("",
                    dashboad_elements_restricted_I,
                    
                    dashboad_elements_II,
                   # dashboad_elements_Risk_mapping,
                    
                    dashboad_elements_III,
                   
                   dashboad_elements_IV
                    
                    # admin page
                    
)

)
)

## UI nologin

UI_NO_login<-fluidPage(fluidRow(
  column(3,offset =5,
         uiOutput("title_txt")),
  #column(2,offset =11,
         #uiOutput("logged_in_user"),
         #uiOutput('logout'))
),
fluidRow(navbarPage("",
                    dashboad_elements_I,
                    
                    dashboad_elements_II,
                    
                    admin_page,
                    
                    #dashboad_elements_Risk_mapping,
                    dashboad_elements_New_model,
                    dashboad_elements_New_model_pros,
                    dashboad_elements_III,
                    
                    dashboad_elements_IV
                    
                    # admin page
                    
)

)
)
