# Documentation for experienced modellers

The following are required to develop or integrate an existing model with Chap:

- Make chap-compatible [train and predict endpoints](train_and_predict.md) (functions) for you model that accepts the standard chap data formats
- [Describe your model](describe_model.md) in a simple yaml-based format
- Check that your model [runs through Chap](running_models_in_chap.md)

To get started, we recommend to follow our simple tutorial:

1. Clone or download our [minimalist example](https://github.com/dhis2-chap/minimalist_example) and make sure you can run the minimalist code used in the tutorial (as an isolated run and through Chap)
2. Replace the code in the train and predict functions with the code of your own model, doing any necessary data format conversion to make your model compatible with the Chap setup

If you are more comfortable with R than Python, you can alterantive clone/download our [R-based minimalist example](https://github.com/dhis2-chap/minimalist_example_r)

If you are unsure on some of the involved IT technologies (like version control, containerisation etc), please consult our [Introduction to Development Tools](../contributor/development_tools.md)
