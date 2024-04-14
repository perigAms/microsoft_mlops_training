# Import libraries

import argparse
import glob
import os

import pandas as pd
# import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from azureml.core import Run


# define functions
def main(args):
    # Start an Azure ML run in the current context
    run = Run.get_context()

    # read data
    df = get_csvs_df(args.training_data)

    # split data
    X_train, X_test, y_train, y_test = split_data(df)

    # train model
    model = train_model(args.reg_rate, X_train, X_test, y_train, y_test)

    # Log the regularization rate
    run.log('Regularization Rate', args.reg_rate)

    if args.dev_prod == '1':
        # Register the model
        model_path = "outputs/model.pkl"
        model.save(model_path)
        run.upload_file("outputs/model.pkl", model_path)
        run.register_model(
            model_name='logistic_regression_model',
            model_path='outputs/model.pkl',
            tags={'Training context': 'Script'})
        run.complete()


def split_data(df):
    X, y = df[['Pregnancies', 'PlasmaGlucose', 'DiastolicBloodPressure',
               'TricepsThickness', 'SerumInsulin', 'BMI',
               'DiabetesPedigree', 'Age']].values, df['Diabetic'].values

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.30, random_state=0)

    return X_train, X_test, y_train, y_test


def get_csvs_df(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")
    csv_files = glob.glob(f"{path}/*.csv")
    if not csv_files:
        raise RuntimeError(f"No CSV files found in provided data path: {path}")
    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)


# TO DO: add function to split data


def train_model(reg_rate, X_train, X_test, y_train, y_test):
    # train model
    model = LogisticRegression(C=1/reg_rate, solver="liblinear")
    model.fit(X_train, y_train)

    return model


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", dest='training_data',
                        type=str)
    parser.add_argument("--reg_rate", dest='reg_rate',
                        type=float, default=0.01)
    parser.add_argument("--dev_prod", dest='dev_prod',
                        type=str)

    # parse args
    args = parser.parse_args()

    # return args
    return args


# run script
if __name__ == "__main__":
    # add space in logs
    print("\n\n")
    print("*" * 60)

    # parse args
    args = parse_args()

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")
