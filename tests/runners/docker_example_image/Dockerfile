#FROM ubuntu:22.04
FROM rstudio/r-base:4.3-jammy@sha256:5c2537abd60b9b28282a7bd0f6743bba0d6c1e219a3bbc76b4656decf0c3e617
RUN useradd -ms /bin/bash apprunner
RUN echo "apprunner:apprunner" | chpasswd
USER apprunner
