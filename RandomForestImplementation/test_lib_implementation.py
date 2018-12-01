from csv import reader
from math import sqrt
from random import randrange
from random import seed
from RandomForest import RandomForest
from CrossValidation import cross_validate

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

# Test the random forest algorithm
seed(55)

# load and prepare data
filename = 'data_out3.csv'
dataset = load_csv(filename)
headers = dataset[0]
dataset = dataset[1:]
# convert string attributes to integers
for col_idx in range(0, len(dataset[0])-1): convert_column_to_float(dataset, col_idx)

# convert class column to integers
convert_str_column_to_int(dataset, len(dataset[0]) - 4, len(dataset[0]) - 3, len(dataset[0])-1)

dataset.insert(0, headers)

# Pandas is used for data manipulation
import pandas as pd
# Use numpy to convert to arrays
import numpy as np

# Read in data and display first 5 rows
#features = pd.read_csv('data_out3.csv')
pd.DataFrame(dataset, columns=headers)

# Labels are the values we want to predict
labels = np.array(features['winner'])

features= features.drop('winner', axis = 1)

feature_list = list(features.columns)

features = np.array(features)

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
kf = KFold(n_splits=5)

kf.get_n_splits(features)


rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)
for train_index, test_index in kf.split(features):
  features_train, features_test = features[train_index], features[test_index]
  labels_train, labels_test = labels[train_index], labels[test_index]
  rf.fit(features_train, labels_train)
  predictions = rf.predict(features_test)
  errors = abs(predictions - labels_test)
  mape = 100 * (errors / labels_test)
  accuracy = 100 - np.mean(mape)
  print('Accuracy:', round(accuracy, 2), '%.')
# Split the data into training and testing sets
#train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

gnb = GaussianNB()
for train_index, test_index in kf.split(features):
  features_train, features_test = features[train_index], features[test_index]
  labels_train, labels_test = labels[train_index], labels[test_index]
  gnb.fit(features_train, labels_train)
  predictions = gnb.predict(features_test)
  errors = abs(predictions - labels_test)
  mape = 100 * (errors / labels_test)
  accuracy = 100 - np.mean(mape)
  print('Accuracy:', round(accuracy, 2), '%.')



rf = RandomForestClassifier(n_estimators = 5, random_state = 42)

rf.fit(train_features, train_labels)

predictions = rf.predict(test_features)

# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')