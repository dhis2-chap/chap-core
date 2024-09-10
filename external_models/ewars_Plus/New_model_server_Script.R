
var_names_New_model <- eventReactive(
  
  c(input$dat_new_Model,
    input$shape_file_new_Model),{           
      
      req(input$shape_file_new_Model,input$dat_new_Model)
      #req()
      #req(input$spat_Input_Type)
      
      
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
      
      
      inFile <- input$dat_new_Model
      data<-get_D_ata(inFile$datapath)
      print("from vars_names")
      print(head(data))
      print("from vars_names...")
      #dist<-sort(unique(data$district))
      #names(data)
      #data<-data %>% arrange(district,year,week) 
      #choose  temp , rainfall
      
      inFile_shp <- input$shape_file_new_Model
      print(inFile_shp)
      shp_pos1<-grep(".shp",inFile_shp$name)
      layer_nm<-str_remove(inFile_shp$name[shp_pos1],'.shp')
      #rename file
      
      #dist<-data$district
      #years.dat<-sort(unique(data$year))
      
      #output$valid_section<-renderUI(eval(parse(text=validation_tab_Func())))
      
      print("layer_name")
      print(layer_nm)
      ##get file with shp
      rename_fnm<-paste0(dirname(inFile_shp$datapath),'/',inFile_shp$name)
      file.rename(inFile_shp$datapath,rename_fnm)
      #ogrListLayers(inFile_shp$datapath[shp_pos])
      #list.files(dirname(inFile_shp$datapath))
      updateSelectInput(session,"district_new",choices=data$district,
                        selected =data$district[1])
      pp<-readOGR(rename_fnm[shp_pos1],layer_nm)
      
      list(vars=names(data),dat=data,dists=dist,SHP=pp)
      
      
      
    })




observe({
  
  
  #al_vs<-grep("rain|temp|rainfall|precipitation|prec|humid|rh|humid|ovi|eggs",var_names()$vars,ignore.case=T)
  al_vs<-grep("rain|temp|rainfall|precipitation|prec",var_names_New_model()$vars,ignore.case=T)
  
  hos_vs<-grep("hosp",var_names_New_model()$vars,ignore.case=T)
  pop_vs<-grep("population|poblacion",var_names_New_model()$vars,ignore.case=T)
  
  #freezeReactiveValue(input, "alarm_indicators")
  #freezeReactiveValue(input, "number_of_cases")
  #freezeReactiveValue(input, "population")

  
  #output$dist_Input1<-renderUI(eval(parse(text=create_input_UI_district("district_new"))))
  #output$dist_Input2<-renderUI(eval(parse(text=create_input_UI_district("output_dist_seas"))))
  
  updateSelectInput(session,"alarm_indicators_New_model",choices=var_names_New_model()$vars,
                    selected =var_names_New_model()$vars[al_vs] )
  
  updateSelectInput(session,"other_alarm_indicators_New_model",choices=var_names_New_model()$vars,
                    selected ="rhdailymean" )
 
  updateSelectInput(session,"number_of_cases_New_model",choices=var_names_New_model()$vars,
                      #selected=var_names()$vars[3])
                      selected=var_names_New_model()$vars[hos_vs])
  updateSelectInput(session,"population_New_model",choices=var_names_New_model()$vars,
                      selected =var_names_New_model()$vars[pop_vs])
  
  updateSelectInput(session,"district_new",choices=var_names_New_model()$dat$district,
                    selected =var_names_New_model()$dat$district[1])
  
  updateSelectInput(session,"district_seas",choices=var_names_New_model()$dat$district,
                    selected =var_names_New_model()$dat$district[1])
  
  updateSelectInput(session,"district_validation",choices=var_names_New_model()$dat$district,
                    selected =var_names_New_model()$dat$district[1])
  
  
  updateSliderInput(session,"new_model_Year_plot",
                    min=min(var_names_New_model()$dat$year),
                    max=max(var_names_New_model()$dat$year),
                    value =max(var_names_New_model()$dat$year))
  
  updateSliderInput(session,"new_model_Year_validation",
                    min=min(var_names_New_model()$dat$year),
                    max=max(var_names_New_model()$dat$year),
                    value =max(var_names_New_model()$dat$year))
  
  #value=max(var_names_spat()$dat$year)-1)
  
  updateSliderInput(session,"new_model_Week_plot_spat",
                    min=min(var_names_New_model()$dat$week),
                    max=max(var_names_New_model()$dat$week),
                    value =min(var_names_New_model()$dat$week))
  #value=max(var_names_spat()$dat$year)-1)
  
  output$Uploaded_data<-DT::renderDataTable(DT::datatable(var_names_New_model()$dat,
                                                          options = list(autoWidth = TRUE,
                                                                         searching = T)))
  
  ##plot the shapefile and render the data
  
  ##update the risk reactive values
  
}
) 

var_Update_Vnames_new_Model <- eventReactive(
  
  c(input$dat_new_Model),{           
    
    req(input$dat_new_Model)
    #req()
    #req(input$spat_Input_Type)
    
    
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
    
    
    inFile <- input$dat_new_Model
    data<-get_D_ata(inFile$datapath)
    print("from vars_names")
    print(head(data))
    print("from vars_names...")
    #dist<-sort(unique(data$district))
    #names(data)
    #data<-data %>% arrange(district,year,week) 
    #choose  temp , rainfall
    list(vars=names(data),dat=data)
    
    
    
  })


observe({
  
  
  #al_vs<-grep("rain|temp|rainfall|precipitation|prec|humid|rh|humid|ovi|eggs",var_names()$vars,ignore.case=T)
  al_vs<-grep("rain|temp|rainfall|precipitation|prec",var_Update_Vnames_new_Model()$vars,ignore.case=T)
  
  hos_vs<-grep("hosp",var_Update_Vnames_new_Model()$vars,ignore.case=T)
  pop_vs<-grep("population|poblacion",var_Update_Vnames_new_Model()$vars,ignore.case=T)
  
  updateSelectInput(session,"district_new",choices=var_names_New_model()$dat$district,
                    selected =var_names_New_model()$dat$district[1])
  
  updateSelectInput(session,"district_seas",choices=var_names_New_model()$dat$district,
                    selected =var_names_New_model()$dat$district[1])
  
  updateSelectInput(session,"district_validation",choices=var_names_New_model()$dat$district,
                    selected =var_names_New_model()$dat$district[1])
  
  
  updateSelectInput(session,"alarm_indicators_New_model",choices=var_Update_Vnames_new_Model()$vars,
                    selected =var_Update_Vnames_new_Model()$vars[al_vs] )
  
  updateSelectInput(session,"other_alarm_indicators_New_model",choices=var_Update_Vnames_new_Model()$vars,
                    selected ="rhdailymean" )
  
  updateSelectInput(session,"number_of_cases_New_model",choices=var_Update_Vnames_new_Model()$vars,
                      #selected=var_names()$vars[3])
                      selected=var_Update_Vnames_new_Model()$vars[hos_vs])
  updateSelectInput(session,"population_New_model",choices=var_Update_Vnames_new_Model()$vars,
                      selected =var_Update_Vnames_new_Model()$vars[pop_vs])

  
  updateSliderInput(session,"new_model_Year_plot",
                    min=min(var_Update_Vnames_new_Model()$dat$year),
                    max=max(var_Update_Vnames_new_Model()$dat$year),
                    value =max(var_Update_Vnames_new_Model()$dat$year))
  
  updateSliderInput(session,"new_model_Year_validation",
                    min=min(var_Update_Vnames_new_Model()$dat$year),
                    max=max(var_Update_Vnames_new_Model()$dat$year),
                    value =max(var_Update_Vnames_new_Model()$dat$year))
  
  #value=max(var_names_spat()$dat$year)-1)
  
  updateSliderInput(session,"new_model_Week_plot_spat",
                    min=min(var_Update_Vnames_new_Model()$dat$week),
                    max=max(var_Update_Vnames_new_Model()$dat$week),
                    value =min(var_Update_Vnames_new_Model()$dat$week))
  #value=max(var_names_spat()$dat$year)-1)
  
  
  ##plot the shapefile and render the data
  
  ##update the risk reactive values
  
  #alarm.indicators<-input$alarm_indicators_New_model
  #dat_slider<-var_Update_Vnames_new_Model()$dat
  
  
}

) 


