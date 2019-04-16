# Adapted from:
# https://github.com/blue-yonder/tsfresh/blob/master/notebooks/basic_pipeline_example.ipynb

from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, \
    load_robot_execution_failures
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from tsfresh.examples.robot_execution_failures import download_robot_execution_failures
from tsfresh.examples import load_robot_execution_failures
from tsfresh.transformers import RelevantFeatureAugmenter
from tsfresh.utilities.dataframe_functions import impute

def main():

  download_robot_execution_failures()
  
  df_ts, y = load_robot_execution_failures()
  # We create an empty feature matrix that has the proper index
  X = pd.DataFrame(index=y.index)
  # Split data into train and test set
  X_train, X_test, y_train, y_test = train_test_split(X, y)
  print(df_ts)
  # We have a pipeline that consists of a feature extraction step with a subsequent Random Forest Classifier 
  ppl = Pipeline([('fresh', RelevantFeatureAugmenter(column_id='id', column_sort='time')),
                  ('clf', RandomForestClassifier())])
  # Here comes the tricky part, due to limitations of the sklearn pipeline API, we can not pass the dataframe
  # containing the time series dataframe but instead have to use the set_params method
  # In this case, df_ts contains the time series of both train and test set, if you have different dataframes for 
  # train and test set, you have to call set_params two times (see the notebook pipeline_with_two_datasets.ipynb)
  ppl.set_params(fresh__timeseries_container=df_ts)

  # We fit the pipeline
  ppl.fit(X_train, y_train)

  # Predicting works as well
  y_pred = ppl.predict(X_test)

  # So, finally we inspect the performance
  print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    main()
