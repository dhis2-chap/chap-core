

library(shiny)
library(plotly)
library(shinydashboardPlus)
library(shinydashboard)
library(shinyjs)
library(tmap)
library(xts)
library(dygraphs)
library(leaflet)


source("objects in UI.R")
source("new_model_UI_elements_Compute.R")
shinyUI(
  
  #titlePanel(tags$h4("EWARS-Dashboard")),
 uiOutput('log_list') 
  

      )

        



