
## risk mapping elements

dist_cesar<-c(20001,20011,20013,20060,20228,20710,20770)


output_dist_new_model<-selectInput(inputId = 'district_new',
            label = 'District',
            choices = dist_cesar,
            selected =20001,
            selectize =T,
            multiple =F)


output_dist_seas<-selectInput(inputId = 'district_seas',
                                   label = 'District',
                                   choices = dist_cesar,
                                   selected =20001,
                                   selectize =T,
                                   multiple =F)

output_dist_validation<-selectInput(inputId = 'district_validation',
                              label = 'District',
                              choices = dist_cesar,
                              selected =20001,
                              selectize =T,
                              multiple =F)

output_dist_pros<-selectInput(inputId = 'district_prospective',
                                    label = 'District',
                                    choices = dist_cesar,
                                    selected =20001,
                                    selectize =T,
                                    multiple =F)

year_validation<-sliderInput(inputId = "new_model_Year_validation",
            label = "Year",
            min = 2019,
            max=2020,
            value =2020,
            sep='',
            step=1)

N_lags<-sliderInput(inputId = "nlags",
                             label = "Lag Weeks",
                             min = 4,
                             max=12,
                             value =12,
                             sep='',
                             step=1)

z_outbreak_New<-sliderInput(inputId = "z_outbreak_new",
                    label = "Z outbreak",
                    min = 1,
                    max=4,
                    value =1.2,
                    sep='',
                    step=0.1)


output_year_New_model<-inputPanel(column(12,
                              sliderInput(inputId = "new_model_Year_plot",
                                          label = "Year",
                                          min = 2015,
                                          max=2020,
                                          value =2017,
                                          sep='',
                                          step=1)),
                              column(12,offset=6,sliderInput(inputId = "new_model_Week_plot_spat",
                                          label = "Week",
                                          min = 1,
                                          max=52,
                                          step=1,
                                          sep='',
                                          value =35))
)

surv.Input_prospective<-column(12,offset =2,
                               fileInput('dat_prospective', 'Upload prospective data', 
                                         accept=c('.xls','.xlsx','.csv')
                                         ))


header_Input_pros<-fluidRow(inputPanel(surv.Input_prospective,
                                       column(12,offset=7,
                                              uiOutput("dist_pros"))))

pros_new_out1<-tabPanel("Uploaded_data",
                        DT::dataTableOutput("uploaded_pros")
)

pros_new_out2<-tabPanel("Prediction table",
                        DT::dataTableOutput("prediction_tab_pros")
)

pros_new_out3<-tabPanel("Outbreak",
                        plotOutput("outbreak_plot_pros",width ="700px",height ="320px")
)

pros_new_out4<- tabPanel("Probability",
                         #inputPanel(uiOutput("sel_diss1")),
                         plotOutput("prob_plot_pros",width ="700px",height ="320px")
)

pros_new_out5<- tabPanel("Outbreak and Probability",
                         #inputPanel(uiOutput("sel_diss1")),
                         plotOutput("out_break_prob_plot_pros",width ="700px",height ="320px")
)

pros_new_out6<- tabPanel("Response",
                         #inputPanel(uiOutput("sel_diss1")),
                         plotOutput("response_plot_pros")
)

row_elements_New_Model_prospective<-fluidRow(     

        header_Input_pros,
  
  
         tabsetPanel(pros_new_out1,
                     pros_new_out2,
                     #pros_new_out3,
                     #pros_new_out4,
                     pros_new_out5,
                     pros_new_out6
         )
  
)



output_graphs_New_Model<-tabsetPanel(
  tabPanel("Descriptive analysis",
            fluidPage(inputPanel(column(12,offset=0,output_dist_new_model)),
             tabsetPanel(
  tabPanel("Tables",

  tableOutput("new_model_data_descriptives")
  
)
,

tabPanel("Plots",
      
     #     output_dist_new_model,
          uiOutput("new_model_data_descriptive_Plots")
         
)
)
)
)

,
tabPanel("Time_series",
         #   output_year_New_model,
         uiOutput("new_model_Time_series_Plots")
         
),
tabPanel("Spatial Plots",
         fluidPage(output_year_New_model,
                   tabsetPanel(
  tabPanel("Spatial_Covariate_Plots", 
           uiOutput("Error"),
           uiOutput("Spat_Covariate_Plots_new_Model")
            
  ),
  
  tabPanel("DIR",
           leafletOutput("spat_DIR_new_Model"),
  )
  )
  )
  ),
  tabPanel("Lag non linear plots",
           fluidPage(column(12,
                            N_lags),
                     tabsetPanel(
                       tabPanel("Lag Countour Plots",
           uiOutput("lag_contour_Plots")
                       ),
           tabPanel("Var Slices",
                    uiOutput("var_Slices_Plots")
           )
                     )
           
  )
  
),
tabPanel("Seasonality",
         fluidPage(inputPanel(column(12,offset=0,output_dist_seas)),
                   plotOutput("Seasonality_plot"))),
tabPanel("Model Validation",
         fluidPage(inputPanel(column(12,offset=0,output_dist_validation),
                              column(12,offset=4,year_validation),
                              column(12,offset=4,z_outbreak_New)),
                   tabsetPanel(tabPanel("Runin period",
                                        plotOutput("runin_ggplot_New_model"),
                                        dygraphOutput("runin_interactive_New_model")),
                               tabPanel("Validation_period",
                                        tabsetPanel(
                                 tabPanel("Plots",
                                        plotOutput("validation_ggplot_New_model"),
                                        dygraphOutput("validation_interactive_New_model")),
                                 tabPanel("Sensitivity/Specificity",
                                          tableOutput("sen_spec_table_New_model")
                                 )
                                 )
                                        ))
                   )
         )
)



boundary.Input<-column(12,offset =0,
                       fileInput('shape_file_new_Model', 'District sub-district boundary file (.shp file)',
                                 accept=c('.shp','.dbf','.shx','.prj'),
                                 multiple =T,
                                 width='70%'))



surv.Input<-column(12,offset =0,
                   fileInput('dat_new_Model', 'Choose surveillance data with spatial inputs', 
                             accept=c('.xls','.xlsx','.csv'),
                             width='70%'))



pop.Input<-column(12,offset =0,selectInput("population_New_model",
            "Variable for annual total Population",
            "population",multiple =F,
            width='70%'))


case.Input<-column(12,offset =0,selectInput("number_of_cases_New_model", 
            "Variable for the weekly number of outbreak",
            "weekly_hospitalised_cases",multiple =F,
            width='70%'))


alarm.Input<-column(12,offset =0,selectInput("alarm_indicators_New_model", selected =c("rainsum","meantemperature"),
            "Alarm indicator(s)",
            choices=c("rainsum","meantemperature"),multiple =T,
            width='70%'))

alarm_Spline.Input<-column(12,offset =0,selectInput("other_alarm_indicators_New_model", selected =c("rainsum","meantemperature"),
                                             "Other alarm indicator(s)",
                                             choices=c(" "),multiple =T,
                                             width='70%'))





header_Input<-fluidRow(boundary.Input,
                                  surv.Input,
                                  pop.Input,
                                  case.Input,
                                  alarm.Input,
                       alarm_Spline.Input)




dat_opts_new_Model<- tabPanel("Spatial data Upload",
                         column(12,
                                shinydashboardPlus::box(width=12,
                                                        ##choose point/ lat long data
                                                        
                                                        fileInput('shape_file_new_Model', 'District sub-district boundary file (.shp file)',
                                                                  accept=c('.shp','.dbf','.shx','.prj'),
                                                                  multiple =T),
                                                        ## choose aggregated or point data
                                                
                                                        
                                                        fileInput('dat_new_Model', 'Choose surveillance data with spatial inputs', 
                                                                  accept=c('.xls','.xlsx','.csv')),
                                                        uiOutput("Spat_error_new_Model"),
                                                        selectInput("population_New_model",
                                                                    "Variable for annual total Population",
                                                                    "population",multiple =F),
                                                        selectInput("number_of_cases_New_model", 
                                                                    "Variable for the weekly number of outbreak",
                                                                    "weekly_hospitalised_cases",multiple =F),
                                                        selectInput("alarm_indicators_New_model", selected =c("rainsum","meantemperature"),
                                                                    "Alarm indicator(s)",
                                                                    choices=c("rainsum","meantemperature"),multiple =T)
                                                        
                                                        
                                )
                         ))



row_elements_New_Model.prev<-fluidRow(     
  column(4,
         tabsetPanel(id="input_pan_new_Model",
                     dat_opts_new_Model
                     # sel_vars
         )),
  
  column(8,
         uiOutput("spat_Display_new_Model")
         )
)




row_elements_New_Model<-fluidRow(     
  column(3,
         header_Input),
  
  column(9,
         uiOutput("spat_Display_new_Model")
         
  )
)
dashboad_elements_New_model<-tabPanel("Dashboard I",
                                         fluidPage(
                                           row_elements_New_Model

                                         )
) 


dashboad_elements_New_model_pros<-tabPanel("Dashboard II",
         row_elements_New_Model_prospective
)