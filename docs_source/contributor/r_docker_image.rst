
Creating a docker image for R models
====================================

When publishing or developing a model for CHAP that is used on R packages, it is very useful to have a docker image that can be used to run the model in a reproducible way. 

This page describes how to create a docker image for R models.

We have created a base Docker image that has INLA and some other commonly used packages installed. We recomment basing your image on this image to save time building your image.

`Please follow the guide here <https://github.com/dhis2-chap/docker_r_template/>`_ to create your own Docker image from this template.