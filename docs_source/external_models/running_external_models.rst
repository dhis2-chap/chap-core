.. _external_models:


Running models through CHAP
------------------------------

Running an external model on the command line
...............................................

Models that are compatible with CHAP can be used with the `chap evaluate` command:

.. code-block:: console

    $ chap evaluate --model-name /path/to/your/model/directory --dataset-name ISIMIP_dengue_harmonized --dataset-country brazil --report-filename report.pdf --ignore-environment  --debug

Note the `--ignore-environment` in the above command. 
This means that we don't ask CHAP to use Docker or a Python environment when running the model. 
This can be useful when developing and testing custom models before deploying them to a production environment.
Instead the model will be run directly using the current environment you are in. 
This usually works fine when developing a model, but requires you to have both chap-core and the dependencies of your model available. 

As an example, the following command runs the chap_auto_ewars model on public ISMIP data for Brazil (this does not use --ignore-environment and will set up
a docker container based on the specifications in the MLproject file of the model):

.. code-block:: bash

    chap evaluate --model-name https://github.com/dhis2-chap/chap_auto_ewars --dataset-name ISIMIP_dengue_harmonized --dataset-country brazil



If the above command runs without any error messages, you have successfully evaluated the model through CHAP, and a file `report.pdf` should have been generated with predictions for various regions.

A folder `runs/model_name/latest` should also have been generated that contains copy of your model directory along with data files used. This can be useful to inspect if something goes wrong.


External models can be run on the command line using the `chap evaluate` command. See `chap evaluate --help` for details.

This example runs an auto ewars R model on public ISMIP data for Brazil using a public docker image with the R inla package. After running, a report file `report.pdf` should be made.

Running an external model in Python
...................................

CHAP contains an API for loading models through Python. The following shows an example of loading and evaluating three different models by specifying paths/github urls, and evaluating those models:

.. literalinclude :: ../../scripts/external_model_example.py
   :language: python

