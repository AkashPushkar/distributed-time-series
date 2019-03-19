
from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, \
    load_robot_execution_failures
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from tsfresh.transformers import RelevantFeatureAugmenter
from tsfresh.utilities.dataframe_functions import impute

from tsfresh import extract_features
from tsfresh.feature_selection.relevance import calculate_relevance_table

import csv

# read both the TEST and TRAIN files for a particular
# dataset into a single set, then partitions the data
# and label into X and y DataFrames
def get_combined_raw_dataset(root_path: str):
  name = root_path.split('/')[2]
  raw_train = pd.read_csv(root_path + name + '_TRAIN.tsv', delimiter='\t', header=None)
  raw_test = pd.read_csv(root_path + name + '_TEST.tsv', delimiter='\t', header=None)
  combined = raw_train.append(raw_test)

  v = combined.reset_index().drop(['index'], axis=1)  
  X = v.iloc[:,1:]
  y = v.iloc[:,:1]

  return (X, y)



# convert a raw dataframe into the vertically oriented
# format that tsfresh requires for feature extraction
def raw_to_tsfresh(X, y):
  ids = []
  values = []
  ys = []
  indices = []

  for id, row in X.iterrows():
    c = (y.loc[[id], :]).iloc[0][0]
    ys.append(int(c))
    indices.append(id)
    
    first = True 
    for v in row:
      if (not first):
        ids.append(id)
        values.append(float(v))
      first = False

  d = { 'id': ids, 'value': values }
  return (pd.DataFrame(data=d), pd.Series(data=ys, index=indices))


def filter_features(df, R):
  for id, row in R.iterrows():
    if (row['relevant'] == False):
      df = df.drop([row['feature']], axis=1)
  return df

def accuracy_rate(predicted, actual):
  correct = 0
  for p, a in zip(predicted, actual):
    if (p == a):
      correct += 1
  return correct / len(predicted)

# Process a single test/train fold
def process_fold(X_train, y_train, X_test, y_test):

  fresh_train_X, fresh_train_y = raw_to_tsfresh(X_train, y_train)
  fresh_test_X, fresh_test_y = raw_to_tsfresh(X_test, y_test)  
  
  # Run the feature extraction and relevance tests ONLY on the train
  # data set.  
  extracted_train = extract_features(fresh_train_X, column_id='id', column_value='value')
  R = calculate_relevance_table(extracted_train, y_train.squeeze())
  filtered_train = filter_features(extracted_train, R)
  
  # Extract features from the test set, but then apply the same relevant
  # features that we used from the train set
  extracted_test = extract_features(fresh_test_X, column_id='id', column_value='value')
  filtered_test = filter_features(extracted_test, R)
  
  # Train classifiers on the train set
  clf = RandomForestClassifier()
  trained_model = clf.fit(filtered_train, y_train.squeeze())
  predicted = list(map(lambda v: int(v), clf.predict(filtered_test)))
  actual = y_test.squeeze().tolist()
  
  return accuracy_rate(predicted, actual)



# Complete processing of one data set.  Does 10-fold cross-validation 
# extraction and classification
def process_data_set(root_path: str):

  combined_X, combined_y = get_combined_raw_dataset(root_path)
 
  skf = StratifiedKFold(n_splits=10)
  skf.get_n_splits(combined_X, combined_y)

  total_acc = 0

  for train_index, test_index in skf.split(combined_X, combined_y):
    X_train, X_test = combined_X.iloc[train_index], combined_X.iloc[test_index]
    y_train, y_test = combined_y.iloc[train_index], combined_y.iloc[test_index]

    total_acc += process_fold(X_train, y_train, X_test, y_test)
    
  accuracy = total_acc / 10
  print(accuracy)
    

def get_dataset_dirs():
  return glob("./data/*/")

def main():
  dataset_dirs = get_dataset_dirs()

  process_data_set(dataset_dirs[0])

  # Uncomment to run against all datasets:
  # for dataset in dataset_dirs:
  #   process_data_set(dataset)

if __name__ == '__main__':
    main()
