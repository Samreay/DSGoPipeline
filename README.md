# DSGoPipeline

Alrighty, let's get into it.

## Installation

For windows users, I'd recommend first running `conda install sqlalchemy==1.3.13` so you don't have issues with wheels building.

Then, everyone should be able to run:

`pip install -r requirements.txt`

## The starting point

Within the `starting_point` directory, I've tried to simulate the somewhat chaotic and unstructured way that some data science projects
begin. A combination of different people sharing parts of the workflow, without a consistent
structure or development route set in place.

```
1_starting_point
    - make_processed_data.ipynb
          an initial EDA notebook that took raw data, made plots and produced processed data
    - model_linreg.ipynb
          a quick and dirty linear regression model to the data
    - model_keras.ipynb
          an untuned and small neural net to do regression to the data
    - germany.csv
          The processed data generated from the first notebook
    - report
          Directory containing a weekly report (and its creation script) that should go out to clients
    - dashboard
        Directory containing a Dash dashboard to provide live insight into the data
```

The first goal will be to take the models and improve the workflow on training them, recording their hyperparameters,
logging their output artifacts and metrics.

Then once we have the models logged, it is easier to pick a model for production, so we'll cover staging models and 
deploying them locally. 

Finally, we'll turn the data processing and report into a multi stage workflow, and hook
the dashboard up to the production model.