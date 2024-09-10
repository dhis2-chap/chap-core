#FROM ubuntu:22.04
FROM rstudio/r-base:4.3-jammy
RUN useradd -ms /bin/bash apprunner
RUN echo "apprunner:apprunner" | chpasswd
USER apprunner
