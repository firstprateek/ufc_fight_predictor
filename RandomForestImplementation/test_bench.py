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
dataset = dataset[1:]
# convert string attributes to integers
for col_idx in range(0, len(dataset[0])-1): convert_column_to_float(dataset, col_idx)

# convert class column to integers
convert_str_column_to_int(dataset, len(dataset[0]) - 4, len(dataset[0]) - 3, len(dataset[0])-1)

# evaluate algorithm
n_folds = 5
max_depth = 10
min_size = 1
sample_size = 1.0
n_features = int(sqrt(len(dataset[0])-1))
for n_trees in [1, 5, 10]:
  scores = cross_validate(dataset, RandomForest, n_folds, max_depth, min_size, sample_size, n_trees, n_features)
  print('Trees: %d' % n_trees)
  print('Scores: %s' % scores)
  print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
