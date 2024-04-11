from azureml.core import Run, Dataset

# Get the current Azure ML run context
run = Run.get_context()

# Get workspace from the run context
workspace = run.experiment.workspace

try:
    # Access the registered dataset by its name
    dataset = Dataset.get_by_name(workspace, 
                                  name='microsoft_mlops_training_dev')

    df = dataset.to_pandas_dataframe()

    # Log that the dataset was accessed successfully
    run.log('DataAccessStatus', 'Success')
    run.log('DataFrameShape', str(df.shape))
except Exception as e:
    # Log failure to access the dataset along with the error message
    run.log('DataAccessStatus', 'Failure')
    run.log('DataAccessError', str(e))
