# DSGoPipeline

Alrighty, let's get into it. The goal in this project is to go from the chaotic assortment of scripts and independent work
in the `1_starting_point` directory, into something that looks like a much nicer DS pipeline.

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

## Adding tracking

First thing we want to do is to start a tracking server. Instead of just pointing this at the file system
and hoping for the best, lets put in a tiny bit of effort now so that it better emulates a remote tracking
server like you might actually end up using better. 

So with the tracking, we need a place to put information, and a place to put artifacts. 

The latter is normally a google cloud, Azure blob storage, Amazon S3 bucket, etc. 
We'll emulate this with a super simple FTP server in the `server` folder. Let's start it with:

`python ftpserver.py`

which just launches a very small FTP server locally in the server folder, with a dummy user to make authentication
easier. Let us also utilise a small sqlite database to store all the information that isn't a collection of files. In the server folder, we also run

`mlflow server --backend-store-uri sqlite:///mlruns.db --default-artifact-root ftp://guest:12345@127.0.0.1::2121/artifacts`

Now one thing we want to do is make sure that this is the server we use for the entire workshop, so lets set a single environment variable.

`MLFLOW_TRACKING_URI=http://127.0.0.1:5000`

which is just the location of the server we launched. And that's it! Now we jump into the code!

We want to go through our notebooks and convert things over to make use of mlflow.

1. Starting with the linear regression, we see how simple it is to log metrics and the model.
2. With the RFRegressor, see how we can also log model parameters. And then see that you have multiple runs in a single execution, if you really want!
3. Then with Keras, we make use of autolog. And note that we could have done that with the other two as well!

This tracking raises a few questions:
* How can we compare multiple models ourselves, outside of the UI interface? (Note that this will probably be built into mlflow at some point, and atm tools like Neptune can do it for you, for now as a manual way, see the compare_runs notebook).
* How can someone reproduce the exact run? (mlflow projects)
* How can we pick a model to serve? (mlflow model repository)

## How do we promote a given model to production?

If we want hands on review, this is super easy. Simply go to the experiment, find the "best performing" algorithm that you want
to be staged, and then click on it. Under artifacts, click on model, and you'll see *Register Model*. So I made a new one, called BestRF, which is now registered.

I can then transition the model into either staging or production, by clicking on it again. Let's move it into production. But now, how can we use it?

For an example using it via the API, see the get_prod_model.ipynb.

We can also serve it locally (or deploy it to SageMaker, AzureML, Spark, etc). Here is the command to do it locally (note the source of the model)

`mlflow models serve -m ftp://user:12345@127.0.0.1:2121/artifacts/1/4fa0fc38c81d4bd8a2c74fe6467cf104/artifacts/model -p 8003`



## Turning our steps into independent projects

Lets bundle up our linear regression model into a simple MLFlow project. A project is defined by the MLproject file, its super simple and can have multiple stages.

```yaml
name: projectName
conda_env: conda.yaml # Or a docker container
entry_points:
  main:
    parameters:
      some_path: path
      some_param: {type: float, default: 1}
    command: "python your_file.py {some_param} {some_path}"
  validate:
    command: "python some_other_file.py"
```

First up, we can easily turn each notebook into an mlflow project. To make the `conda.yaml` step easy,
don't ask conda for your current environment, just go to your tracked mlflow runs and it'll have a minimal version! But note it only tracks
what is used in the actual experiment after you call `start_run`, so you might need to add extra stuff in.

We'll wrap up the linear regression example (and not worry about pulling the dataframe as a path argument), so we can run
`mlflow run lr --no-conda --experiment-name predicting_wind_solar`. Note again we specify the experiment name and URI via command line / environment variable,
rather than defining it in our code. `no-conda` simply because I already havea  working environment, and don't need a new one!

At this point, we could commit the `lr` directory (with the `conda.yaml`, `MLproject`, etc) as its own git repo, and then invoke 
`mlflow run git@ github.com/username/projectname.git --no-conda --experiment-name predicting_wind_solar`, so you can have each step version controlled.

So we could use the MLProject formalism to package up everything in here, but this still leaves us with a bit of an issue, in that its still rather manual. Now, DAGs are coming to mlflow at 
some point. They were teased during the 2019 keynote last year (https://www.youtube.com/watch?v=QJW_kkRWAUs, 28min)


