
from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, \
    load_robot_execution_failures
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from glob import glob
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

import csv

features = ['maximum', 'minimum', 'mean', 'variance', 'standard_deviation', 'skewness', 'kurtosis', 'length', 'median', 'quantile',
  'abs_energy', 'augmented_dickey_fuller', 'binned_entropy',
  'has_variance_larger_than_std', 'is_symmetric_looking', 'mass_quantile', 'number_data_points_above_mean',
  'number_data_points_above_median', 'number_data_points_below_mean', 'number_data_points_below_median',
  'arima_model_coefficients', 'continuous_wavelet_transformation_coefficients', 'fast_fourier_transformation_coefficient',
  'first_index_max', 'first_index_min', 'lagged_autocorrelation', 'large_number_of_peaks', 'last_index_max',
  'last_index_min', 'longest_strike_above_mean', 'longest_strike_above_median', 'longest_strike_below_mean',
  'longest_strike_below_median', 'longest_strike_negative', 'longest_strike_positive', 'longest_strike_zero',
  'mean_absolute_change', 'mean_absolute_change_quantiles', 'mean_autocorrelation', 'mean_second_derivate_central',
  'number_continous_wavelet_transformation_peaks_of_size', 'number_peaks_of_size', 'spektral_welch_density',
  'time_reversal_asymmetry_statistic']

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
  clf = RandomForestClassifier()
  trained_model = clf.fit(filtered_train, y_train.squeeze())
  rfc_predicted = list(map(lambda v: int(v), clf.predict(filtered_test)))
  
  actual = y_test.squeeze().tolist()

  # Create and fit an AdaBoosted decision tree
  bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                          algorithm="SAMME",
                          n_estimators=200)
  trained_model = bdt.fit(filtered_train, y_train.squeeze())
  ada_predicted = list(map(lambda v: int(v), bdt.predict(filtered_test)))

  return { 
    'rfc':  accuracy_rate(rfc_predicted, actual), 
    'ada': accuracy_rate(ada_predicted, actual) 
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
  clf = RandomForestClassifier()
  trained_model = clf.fit(filtered_train, y_train.squeeze())
  rfc_predicted = list(map(lambda v: int(v), clf.predict(filtered_test)))
  
  actual = y_test.squeeze().tolist()

  # Create and fit an AdaBoosted decision tree
  bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                          algorithm="SAMME",
                          n_estimators=200)
  trained_model = bdt.fit(filtered_train, y_train.squeeze())
  ada_predicted = list(map(lambda v: int(v), bdt.predict(filtered_test)))

  return { 
    'rfc':  accuracy_rate(rfc_predicted, actual), 
    'ada': accuracy_rate(ada_predicted, actual) 
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

  R = calculate_relevance_table(extracted_train, y_train.squeeze())
  filtered_train = filter_features(extracted_train, R)

  
    
  # Extract features from the test set, but then apply the same relevant
  # features that we used from the train set
  extracted_test = extract_features(fresh_test_X, column_id='id', column_value='value')
  filtered_test = filter_features(extracted_test, R)

  filtered_test = pca_train.transform(filtered_test)
  
  # Train classifiers on the train set
  clf = RandomForestClassifier()
  trained_model = clf.fit(filtered_train, y_train.squeeze())
  rfc_predicted = list(map(lambda v: int(v), clf.predict(filtered_test)))
  
  actual = y_test.squeeze().tolist()

  # Create and fit an AdaBoosted decision tree
  bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                          algorithm="SAMME",
                          n_estimators=200)
  trained_model = bdt.fit(filtered_train, y_train.squeeze())
  ada_predicted = list(map(lambda v: int(v), bdt.predict(filtered_test)))

  return { 
    'rfc':  accuracy_rate(rfc_predicted, actual), 
    'ada': accuracy_rate(ada_predicted, actual) 
  }

def perform_boruta(X_train, y_train, X_test, y_test):
  rf = RandomForestClassifier()
  feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=0)
  feat_selector.fit(X_train.values, y_train.values)

  X_filtered = feat_selector.transform(X_train.values)
  X_test_filtered = feat_selector.transform(X_test.values)

  trained_model = rf.fit(X_filtered, y_train.squeeze().values)
  rfc_predicted = list(map(lambda v: int(v), rf.predict(X_test_filtered)))
  actual = y_test.squeeze().tolist()

  bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                          algorithm="SAMME",
                          n_estimators=200)
  trained_model = bdt.fit(X_filtered, y_train.squeeze().values)
  ada_predicted = list(map(lambda v: int(v), bdt.predict(X_test_filtered)))

  return { 
    'rfc': accuracy_rate(rfc_predicted, actual), 
    'ada': accuracy_rate(ada_predicted, actual), 
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

  rf = RandomForestClassifier()
  trained_model = rf.fit(X_train, y_train.squeeze())
  rfc_predicted = list(map(lambda v: int(v), rf.predict(X_test)))
  actual = y_test.squeeze().tolist()

  bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                          algorithm="SAMME",
                          n_estimators=200)
  trained_model = bdt.fit(X_train, y_train.squeeze())
  ada_predicted = list(map(lambda v: int(v), bdt.predict(X_test)))

  return { 
    'rfc': accuracy_rate(rfc_predicted, actual), 
    'ada': accuracy_rate(ada_predicted, actual), 
  }

def perform_dtw_nn(X_train, y_train, X_test, y_test):

  m = KnnDtw(n_neighbors=1, max_warping_window=10)
  m.fit(X_train.values, y_train.values)
  predicted, proba = m.predict(X_test.values)

  actual = y_test.squeeze().tolist()

  return accuracy_rate(predicted, actual) 


# Process a single test/train fold
def process_fold(X_train, y_train, X_test, y_test):

  dtw = perform_dtw_nn(X_train, y_train, X_test, y_test)
  boruta = perform_boruta(X_train, y_train, X_test, y_test)
  lda = perform_lda(X_train, y_train, X_test, y_test)
  fresh = perform_fresh_pca_after(X_train, y_train, X_test, y_test)
  fresh_a = perform_fresh_pca_after(X_train, y_train, X_test, y_test)
  
  return {
    'Boruta_ada': boruta.get('ada'),
    'Boruta_rfc': boruta.get('rfc'),
    'LDA_ada': lda.get('ada'),
    'LDA_rfc': lda.get('rfc'),
    'FRESH_PCAa_ada': fresh_a.get('ada'),
    'FRESH_PCAa_rfc': fresh_a.get('rfc'),
    'FRESH_ada': fresh.get('ada'),
    'FRESH_rfc': fresh.get('rfc'),
    'DTW_NN': dtw,
    'FRESH_PCAb_ada': 0,
    'FRESH_PCAb_rfc': 0,
    'ada': 0,
    'rfc': 0,
    'trivial': 0,
  }


# Complete processing of one data set.  Does 10-fold cross-validation 
# extraction and classification
def process_data_set(root_path: str):

  combined_X, combined_y = get_combined_raw_dataset(root_path)
 
  skf = StratifiedKFold(n_splits=10)
  skf.get_n_splits(combined_X, combined_y)

  total_acc = 0

  results = []

  for train_index, test_index in skf.split(combined_X, combined_y):
    X_train, X_test = combined_X.iloc[train_index], combined_X.iloc[test_index]
    y_train, y_test = combined_y.iloc[train_index], combined_y.iloc[test_index]

    results.append(process_fold(X_train, y_train, X_test, y_test))

  averages = results[0]
  for r in results[:1]:
    for k in r:
      averages[k] += r[k]
  for k in averages:
    averages[k] /= len(results)

  print(averages)
    

def get_dataset_dirs():
  return glob("./data/*/")

def main():
  
  for f in features:
    if f in settings:
      del settings[f]
  
  dataset_dirs = get_dataset_dirs()

  process_data_set(dataset_dirs[0])

  # Uncomment to run against all datasets:
  # for dataset in dataset_dirs:
  #   process_data_set(dataset)

if __name__ == '__main__':
    main()
