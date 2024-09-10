r = getOption("repos")
r["CRAN"] = "http://cran.us.r-project.org"
options(repos = r)
#install.packages("terra")
#install.packages("INLA", repos=c(getOption("repos"), INLA="https://inla.r-inla-download.org/R/testing"), dep=TRUE)

#file.create("my.csv")
# write to file
#write.csv(data.frame(a=1:10, b=letters[1:10]), "my.csv")
# print something
print("Hello world!")
