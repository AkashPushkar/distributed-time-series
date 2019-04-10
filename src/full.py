
from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, \
    load_robot_execution_failures
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from tsfresh.transformers import RelevantFeatureAugmenter
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters
settings = ComprehensiveFCParameters()
from tsfresh import extract_features
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from tsfresh.feature_selection.relevance import calculate_relevance_table
from pca import PCAForPandas
from dtwnn import KnnDtw
from boruta import BorutaPy
import copy 

import csv

num_folds = 10

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

def build_rfc():
  return RandomForestClassifier()

def build_ada():
  return AdaBoostClassifier()

def perform_fresh(X_train, y_train, X_test, y_test):

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
  clf = build_rfc()
  trained_model = clf.fit(filtered_train, y_train.squeeze())
  rfc_predicted = list(map(lambda v: int(v), clf.predict(filtered_test)))
  
  actual = y_test.squeeze().tolist()

  # Create and fit an AdaBoosted decision tree
  bdt = build_ada()
  trained_model = bdt.fit(filtered_train, y_train.squeeze())
  ada_predicted = list(map(lambda v: int(v), bdt.predict(filtered_test)))

  return { 
    'rfc':  accuracy_rate(rfc_predicted, actual), 
    'ada': accuracy_rate(ada_predicted, actual),
    'rfc_count': len(clf.feature_importances_),
    'ada_count': len(bdt.feature_importances_),
  }

  

def perform_fresh_pca_after(X_train, y_train, X_test, y_test):

  fresh_train_X, fresh_train_y = raw_to_tsfresh(X_train, y_train)
  fresh_test_X, fresh_test_y = raw_to_tsfresh(X_test, y_test)  

  # Run the feature extraction and relevance tests ONLY on the train
  # data set.  
  extracted_train = extract_features(fresh_train_X, column_id='id', column_value='value')
  
  R = calculate_relevance_table(extracted_train, y_train.squeeze())
  filtered_train = filter_features(extracted_train, R)

  # Perform PCA on the filtered set of features
  pca_train = PCAForPandas(n_components=0.95, svd_solver='full')
  filtered_train = pca_train.fit_transform(filtered_train)
    
  # Extract features from the test set, but then apply the same relevant
  # features that we used from the train set
  extracted_test = extract_features(fresh_test_X, column_id='id', column_value='value')
  filtered_test = filter_features(extracted_test, R)

  filtered_test = pca_train.transform(filtered_test)
  
  # Train classifiers on the train set
  clf = build_rfc()
  trained_model = clf.fit(filtered_train, y_train.squeeze())
  rfc_predicted = list(map(lambda v: int(v), clf.predict(filtered_test)))
  
  actual = y_test.squeeze().tolist()

  # Create and fit an AdaBoosted decision tree
  bdt = build_ada()
  trained_model = bdt.fit(filtered_train, y_train.squeeze())
  ada_predicted = list(map(lambda v: int(v), bdt.predict(filtered_test)))

  return { 
    'rfc':  accuracy_rate(rfc_predicted, actual), 
    'ada': accuracy_rate(ada_predicted, actual),
    'rfc_count': len(clf.feature_importances_),
    'ada_count': len(bdt.feature_importances_),
  }


def perform_fresh_pca_before(X_train, y_train, X_test, y_test):
  fresh_train_X, fresh_train_y = raw_to_tsfresh(X_train, y_train)
  fresh_test_X, fresh_test_y = raw_to_tsfresh(X_test, y_test)  

  # Run the feature extraction and relevance tests ONLY on the train
  # data set.  
  extracted_train = extract_features(fresh_train_X, column_id='id', column_value='value')
  
  # Perform PCA on the complete set of extracted features
  pca_train = PCAForPandas(n_components=0.95, svd_solver='full')
  extracted_train = pca_train.fit_transform(extracted_train)
  
  extracted_train = extracted_train.reset_index(drop=True)
  y_train = y_train.reset_index(drop=True)
  
  R = calculate_relevance_table(extracted_train, y_train.squeeze())
  filtered_train = filter_features(extracted_train, R)

  # Extract features from the test set, but then apply the same relevant
  # features that we used from the train set
  extracted_test = extract_features(fresh_test_X, column_id='id', column_value='value')
  extracted_test = pca_train.transform(extracted_test)

  filtered_test = filter_features(extracted_test, R)
  
  # Train classifiers on the train set
  clf = build_rfc()
  trained_model = clf.fit(filtered_train, y_train.squeeze())
  rfc_predicted = list(map(lambda v: int(v), clf.predict(filtered_test)))
  
  actual = y_test.squeeze().tolist()

  # Create and fit an AdaBoosted decision tree
  bdt = build_ada()
  trained_model = bdt.fit(filtered_train, y_train.squeeze())
  ada_predicted = list(map(lambda v: int(v), bdt.predict(filtered_test)))

  return { 
    'rfc':  accuracy_rate(rfc_predicted, actual), 
    'ada': accuracy_rate(ada_predicted, actual), 
    'rfc_count': len(clf.feature_importances_),
    'ada_count': len(bdt.feature_importances_),
  }

def perform_boruta(X_train, y_train, X_test, y_test):
  rf = build_rfc()
  feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=0)
  feat_selector.fit(X_train.values, y_train.values)

  X_filtered = feat_selector.transform(X_train.values)
  X_test_filtered = feat_selector.transform(X_test.values)

  trained_model = rf.fit(X_filtered, y_train.squeeze().values)
  rfc_predicted = list(map(lambda v: int(v), rf.predict(X_test_filtered)))
  actual = y_test.squeeze().tolist()

  bdt = build_ada()
  trained_model = bdt.fit(X_filtered, y_train.squeeze().values)
  ada_predicted = list(map(lambda v: int(v), bdt.predict(X_test_filtered)))
  
  return { 
    'rfc': accuracy_rate(rfc_predicted, actual), 
    'ada': accuracy_rate(ada_predicted, actual), 
    'rfc_count': len(rf.feature_importances_),
    'ada_count': len(bdt.feature_importances_),
  }

def perform_lda(X_train, y_train, X_test, y_test):

  X_train = X_train.values
  y_train = y_train.values
  X_test = X_test.values
  y_test = y_test.values

  sc = StandardScaler()  
  X_train = sc.fit_transform(X_train)  
  X_test = sc.transform(X_test)  

  lda = LDA()  
  X_train = lda.fit_transform(X_train, y_train)  
  X_test = lda.transform(X_test)  

  rf = build_rfc()
  trained_model = rf.fit(X_train, y_train.squeeze())
  rfc_predicted = list(map(lambda v: int(v), rf.predict(X_test)))
  actual = y_test.squeeze().tolist()

  bdt = build_ada()
  trained_model = bdt.fit(X_train, y_train.squeeze())
  ada_predicted = list(map(lambda v: int(v), bdt.predict(X_test)))
  
  return { 
    'rfc': accuracy_rate(rfc_predicted, actual), 
    'ada': accuracy_rate(ada_predicted, actual), 
    'rfc_count': len(rf.feature_importances_),
    'ada_count': len(bdt.feature_importances_),
  }

def perform_unfiltered(X_train, y_train, X_test, y_test):

  fresh_train_X, fresh_train_y = raw_to_tsfresh(X_train, y_train)
  fresh_test_X, fresh_test_y = raw_to_tsfresh(X_test, y_test)  

  # Run the feature extraction only
  extracted_train = extract_features(fresh_train_X, column_id='id', column_value='value')
  extracted_test = extract_features(fresh_test_X, column_id='id', column_value='value')
  
  # Train classifiers on the train set
  clf = build_rfc()
  trained_model = clf.fit(extracted_train, y_train.squeeze())
  rfc_predicted = list(map(lambda v: int(v), clf.predict(extracted_test)))
  
  actual = y_test.squeeze().tolist()

  # Create and fit an AdaBoosted decision tree
  bdt = build_ada()
  trained_model = bdt.fit(extracted_train, y_train.squeeze())
  ada_predicted = list(map(lambda v: int(v), bdt.predict(extracted_test)))

  return { 
    'rfc':  accuracy_rate(rfc_predicted, actual), 
    'ada': accuracy_rate(ada_predicted, actual),
    'rfc_count': len(clf.feature_importances_),
    'ada_count': len(bdt.feature_importances_),
  }

def perform_dtw_nn(X_train, y_train, X_test, y_test):

  m = KnnDtw(n_neighbors=1, max_warping_window=10)
  m.fit(X_train.values, y_train.values)
  predicted, proba = m.predict(X_test.values)

  actual = y_test.squeeze().tolist()

  return accuracy_rate(predicted, actual), 0 

# implements majority vote 
def perform_trivial(X_train, y_train, X_test, y_test):
  a = y_train.squeeze().values
  counts = np.bincount(a)
  majority = np.argmax(counts)
  predicted = np.full(len(y_test.squeeze().values), majority)
  actual = y_test.squeeze().tolist()
  return accuracy_rate(predicted, actual) 

# Process a single test/train fold
def process_fold(X_train, y_train, X_test, y_test):

  fresh_b = perform_fresh_pca_before(X_train, y_train, X_test, y_test)
  boruta = perform_boruta(X_train, y_train, X_test, y_test)
  trivial = perform_trivial(X_train, y_train, X_test, y_test)
  dtw = perform_dtw_nn(X_train, y_train, X_test, y_test)
  lda = perform_lda(X_train, y_train, X_test, y_test)  
  fresh = perform_fresh_pca_after(X_train, y_train, X_test, y_test)
  fresh_a = perform_fresh_pca_after(X_train, y_train, X_test, y_test)
  unfiltered = perform_unfiltered(X_train, y_train, X_test, y_test)
  
  return ({
    'Boruta_ada': boruta.get('ada'),
    'Boruta_rfc': boruta.get('rfc'),
    'DTW_NN': dtw[0],
    'FRESH_PCAa_ada': fresh_a.get('ada'),
    'FRESH_PCAa_rfc': fresh_a.get('rfc'),
    'FRESH_PCAb_ada': fresh_b.get('ada'),
    'FRESH_PCAb_rfc': fresh_b.get('rfc'),
    'FRESH_ada': fresh.get('ada'),
    'FRESH_rfc': fresh.get('rfc'),
    'LDA_ada': lda.get('ada'),
    'LDA_rfc': lda.get('rfc'),
    'ada': unfiltered.get('ada'),
    'rfc': unfiltered.get('rfc'),
    'trivial': trivial,
  }, {
    'Boruta_ada': boruta.get('ada_count'),
    'Boruta_rfc': boruta.get('rfc_count'),
    'DTW_NN': dtw[1],
    'FRESH_PCAa_ada': fresh_a.get('ada_count'),
    'FRESH_PCAa_rfc': fresh_a.get('rfc_count'),
    'FRESH_PCAb_ada': 0,
    'FRESH_PCAb_rfc': 0,
    'FRESH_ada': fresh.get('ada_count'),
    'FRESH_rfc': fresh.get('rfc_count'),
    'LDA_ada': lda.get('ada_count'),
    'LDA_rfc': lda.get('rfc_count'),
    'ada': unfiltered.get('ada_count'),
    'rfc': unfiltered.get('rfc_count'),
  })


# Complete processing of one data set.  Does 10-fold cross-validation 
# extraction and classification
def process_data_set(root_path: str):

  combined_X, combined_y = get_combined_raw_dataset(root_path)
 
  skf = StratifiedKFold(n_splits=num_folds)
  skf.get_n_splits(combined_X, combined_y)

  total_acc = 0

  results = []

  for train_index, test_index in skf.split(combined_X, combined_y):
    X_train, X_test = combined_X.iloc[train_index], combined_X.iloc[test_index]
    y_train, y_test = combined_y.iloc[train_index], combined_y.iloc[test_index]

    results.append(process_fold(X_train, y_train, X_test, y_test))

  # For this dataset, averages is a map from the name of the
  # pipeline (e.g. Boruta_rfc) to the average of all folds, 
  # similar for std_devs
  averages, std_devs, counts = calc_statistics(results)  

  return averages, std_devs, counts

def calc_statistics(results):
  # convert to numpy array then use 
  averages = {}
  std_devs = {}
  counts = {}
  
  for k in results[0][0]:
    values = []
    for r in results[0]:
      values.append(r.get(k))
    averages[k] = np.mean(values)
    std_devs[k] = np.std(values)
    
  for k in results[1][0]:
    values = []
    for r in results[1]:
      values.append(r.get(k))
    counts[k] = np.mean(values)
    
  return averages, std_devs, counts


def out_to_file(file: str, lines):
  f = open(file, 'w')
  for line in lines:
    f.write(line + '\n')
  f.close()

def output_results(results):
  
  header = 'dataset'
  first = results.get(next(iter(results)))[0]
  
  for k in first:
    header = header + '\t' + k

  # averages
  lines = [header]
  for r in results:
    line = r
    aves = results.get(r)[0]
    for k in aves:
      line = line + '\t' + str(aves.get(k))
    lines.append(line)
  out_to_file('./averages.tsv', lines)

  # std_devs
  lines = [header]
  for r in results:
    line = r
    aves = results.get(r)[1]
    for k in aves:
      line = line + '\t' + str(aves.get(k))
    lines.append(line)
  out_to_file('./std_devs.tsv', lines)

  # counts
  lines = [header]
  for r in results:
    line = r
    aves = results.get(r)[2]
    for k in aves:
      line = line + '\t' + str(aves.get(k))
    lines.append(line)
  out_to_file('./counts.tsv', lines)


def get_dataset_dirs():
  return glob("./data/*/")

def main():
  
  dataset_dirs = get_dataset_dirs()

  # map from the dataset name to a tuple of (averages, std_devs, counts)
  results = {}

  for dataset_path in dataset_dirs:
    name  = dataset_path.split('/')[2]
    results[name] = process_data_set(dataset_path)
    break

  output_results(results)

if __name__ == '__main__':
    main()
