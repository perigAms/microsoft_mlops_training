from azureml.core import Run, Dataset

# Get the current Azure ML run context
run = Run.get_context()

# Get workspace from the run context
workspace = run.experiment.workspace

try:
    # Access the registered dataset by its name
    dataset = Dataset.get_by_name(workspace, name='microsoft_mlops_training_dev')
    
    # Assuming the dataset is a tabular dataset, convert it to a pandas DataFrame (or use the appropriate method if it's a different type of dataset)
    df = dataset.to_pandas_dataframe()

    # Log that the dataset was accessed successfully
    run.log('DataAccessStatus', 'Success')
    run.log('DataFrameShape', str(df.shape))  # Example of logging additional information like DataFrame shape
except Exception as e:
    # Log failure to access the dataset along with the error message
    run.log('DataAccessStatus', 'Failure')
    run.log('DataAccessError', str(e))
