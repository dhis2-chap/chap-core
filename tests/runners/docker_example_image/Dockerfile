#FROM ubuntu:22.04
FROM rstudio/r-base:4.6-jammy@sha256:fee4e02ab0f4a847493e53ff7e10ede2622f8110fc907dac1655c0192d38a6f4
RUN useradd -ms /bin/bash apprunner
RUN echo "apprunner:apprunner" | chpasswd
USER apprunner
