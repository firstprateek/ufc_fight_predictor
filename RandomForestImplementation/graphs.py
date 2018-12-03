# Time taken, Memory Usage, Accuracy
# Number of Trees, Number of attributes selected
import pandas as pd
import numpy as np
from RandomForest import RandomForest
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

def load_csv(filename):
  dataset = []
  with open(filename, 'r') as file:
    csv_reader = reader(file)
    for row in csv_reader:
      if not row: continue
      dataset.append(row)
  return dataset

def convert_column_to_float(dataset, column):
  for row in dataset:
    if row[column].replace('.','',1).isdigit():  
      row[column] = int(float(row[column].strip()))

def convert_str_column_to_int(dataset, *columns):
  for column in columns:    
    unique, unique_mapping = { row[column] for row in dataset }, {}
    for i, value in enumerate(unique): unique_mapping[value] = i
    for row in dataset: row[column] = unique_mapping[row[column]]

seed(55)
filename = 'data_out3.csv'
dataset = load_csv(filename)
headers = dataset[0]
dataset = dataset[1:]

# Convert the string columns into numbers
for col_idx in range(0, len(dataset[0])-1): 
  convert_column_to_float(dataset, col_idx)

convert_str_column_to_int(dataset, len(dataset[0]) - 4, len(dataset[0]) - 3, len(dataset[0])-1)

# Load the dataset into pandas for
# sklearn function expects
pd.DataFrame(dataset, columns=headers)
labels = np.array(features['winner'])
features= features.drop('winner', axis = 1)
feature_list = list(features.columns)
features = np.array(features)

# defaults for our scracth model
max_depth = 10
min_size = 1
sample_size = 1.0
n_features = int(sqrt(len(dataset[0])-1))
max_depth, min_size, sample_size, n_trees, n_features

def test_model(n_folds=5, n_trees):
  kf = KFold(n_splits=n_folds)
  kf.get_n_splits(features)
  model = RandomForest()
  model.n_features = n_features

  accuracies = []
  durations = []

  for train_index, test_index in kf.split(features):
    train_features, test_features = features[train_index], features[test_index]
    train_labels, test_labels = labels[train_index], labels[test_index]
    rf.fit(train_features, train_labels)
    model.train_set = train_features
    model.test_set = test_labels

    rf_predictions = rf.predict(test_features)
    model_prediction = model.run()
    
    errors = abs(predictions - test_labels)
    mape = 100 * (errors / test_labels)
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')

    rf_scratch = RandomForest()





# Test Scratch, RF and NB on number of trees being varied
# Calc all three, time, memory

algorithm(train_set, test_set, *args)
dataset, RandomForest, n_folds, max_depth, min_size, sample_size, n_trees, n_features
rf2 train, test, max_depth, min_size, sample_size, n_trees, n_features

for n_trees in range(1, 11):
  rf = RandomForestClassifier(n_estimatior = ntrees, radom_state = 55)


