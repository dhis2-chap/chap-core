# An example image based on the inla image with one extra R package
FROM ivargr/r_inla
RUN R -e 'install.packages("ggplot2", dep=TRUE, repos="http://cran.us.r-project.org");'
