.. _developing_custom_models:

Developing your own custom model with CHAP
============================================

CHAP is designed to allow model developers to easily develop their own models outside CHAP and use CHAP to benchmark/evaluate their models, or to import and use utility functions from CHAP in their own models.

We here provide guides for implementing custom models in *Python* and *R*. The recommended flow is slightly different for the two different languages, but the general idea is the same.

We have provided several example code base templates that contain minimal code and instructions on how to get started.

We recommend developing your model as a custom Python project, separate from the CHAP codebase, with command-line entry points for training 
(e.g., train.py) and prediction (e.g., predict.py). Your project should include an MLProject configuration file specifying these entry points. 
To get started, you can clone our barebone template repository, or follow tutorials/examples.

Specifically, we recommend going through a series of steps in order go from a minimal trivial model to a more sophisticated model:

- A good starting place is the `minimal Python example <https://github.com/dhis2-chap/minimalist_example>`_, that uses few variables without any lag and a standard machine learning model.
- As a next step, one can look at a `minimalist example that distinguishes multiple regions <https://github.com/dhis2-chap/minimalist_multiregion>`_.
- As a third step, one can look at an `example that introduces lagged features <https://github.com/dhis2-chap/minimalist_example_lag>`_.
- After these, you should have the necessary understanding to develop 

If you feel comfortable creating a model from scratch, you can start with with `a blank template <https://github.com/dhis2-chap/chap_model_template>`_.


Alternative examples in R
---------------------------
We also provide a `minimal example for creating an R model <https://github.com/dhis2-chap/minimalist_example_r>`_.
Note that creating R models might be more challenging due to requirement of R packages. CHAP does optionally support running models through docker containers,
which might be a good way to handle R models.

Alfo for R models, we provide `a blank model template <https://github.com/dhis2-chap/chap_model_template_r>`_.





