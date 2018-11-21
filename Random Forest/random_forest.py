from csv import reader
from math import sqrt
from random import randrange
from random import seed

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
    # print(unique_mapping)

# Divide data into k folds
def cross_validation_split(dataset, n_folds):
  dataset_split, dataset_copy, fold_size = [], list(dataset), int(len(dataset) / n_folds)
  for i in range(n_folds):
    fold = []
    while len(fold) < fold_size: fold.append(dataset_copy.pop(randrange(len(dataset_copy))))
    dataset_split.append(fold)
  return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
  correct = 0
  for i in range(len(actual)):
    if actual[i] == predicted[i]: correct += 1
  return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
  folds = cross_validation_split(dataset, n_folds)
  scores = []
  for fold in folds:
    train_set = list(folds)
    train_set.remove(fold)
    train_set = [data_value for fold in train_set for data_value in fold]
    test_set = []
    for row in fold:
      row_copy = list(row)
      test_set.append(row_copy)
      # row_copy[-1] = None
    predicted = algorithm(train_set, test_set, *args)
    actual = [row[-1] for row in fold]
    accuracy = accuracy_metric(actual, predicted)
    scores.append(accuracy)
  return scores

# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
  left, right = [], []
  for row in dataset:
    if row[index] < value: left.append(row)
    else: right.append(row)
  return left, right

# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
  # count all samples at split point
  n_instances = float(sum([len(group) for group in groups]))
  # sum weighted Gini index for each group
  gini = 0.0
  for group in groups:
    size = float(len(group))
    # avoid divide by zero
    if size == 0:
      continue
    score = 0.0
    # score the group based on the score for each class
    for class_val in classes:
      p = [row[-1] for row in group].count(class_val) / size
      score += p * p
    # weight the group score by its relative size
    gini += (1.0 - score) * (size / n_instances)
  return gini

# Select the best split point for a dataset
def get_split(dataset, n_features):
  class_values = list({ row[-1] for row in dataset })
  b_index, b_value, b_score, b_groups = 999, 999, 999, None
  features = []
  while len(features) < n_features:
    index = randrange(len(dataset[0])-1)
    if index not in features: features.append(index)
  for index in features:
    for row in dataset:
      groups = test_split(index, row[index], dataset)
      gini = gini_index(groups, class_values)
      if gini < b_score: b_index, b_value, b_score, b_groups = index, row[index], gini, groups
  return {'index':b_index, 'value':b_value, 'groups':b_groups}

# Create a terminal node value
def to_terminal(group):
  outcomes = [row[-1] for row in group]
  return max(set(outcomes), key=outcomes.count)

# Create child splits for a node or make terminal
def split(node, max_depth, min_size, n_features, depth):
  left, right = node['groups']
  del(node['groups'])
  # check for a no split
  if not left or not right:
    node['left'] = node['right'] = to_terminal(left + right)
    return
  # check for max depth
  if depth >= max_depth:
    node['left'], node['right'] = to_terminal(left), to_terminal(right)
    return
  # process left child
  if len(left) <= min_size:
    node['left'] = to_terminal(left)
  else:
    node['left'] = get_split(left, n_features)
    split(node['left'], max_depth, min_size, n_features, depth+1)
  # process right child
  if len(right) <= min_size:
    node['right'] = to_terminal(right)
  else:
    node['right'] = get_split(right, n_features)
    split(node['right'], max_depth, min_size, n_features, depth+1)

# Build a decision tree
def build_decision_tree(train, max_depth, min_size, n_features):
  root = get_split(train, n_features)
  split(root, max_depth, min_size, n_features, 1)
  return root

# Make a prediction with a decision tree
def predict(node, row):
  if row[node['index']] < node['value']:
    if isinstance(node['left'], dict):
      return predict(node['left'], row)
    else:
      return node['left']
  else:
    if isinstance(node['right'], dict):
      return predict(node['right'], row)
    else:
      return node['right']

# Create a random subsample from the dataset with replacement
def subsample(dataset, ratio):
  sample = []
  samples = round(len(dataset) * ratio)
  while len(sample) < samples: sample.append(dataset[randrange(len(dataset))])
  return sample

# Make a prediction with a list of bagged trees
def bagging_predict(trees, row):
  predictions = [predict(tree, row) for tree in trees]
  return max(set(predictions), key=predictions.count)

# Random Forest Algorithm
def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
  decision_trees = []
  for i in range(n_trees):
    sample = subsample(train, sample_size)
    decision_tree = build_decision_tree(sample, max_depth, min_size, n_features)
    decision_trees.append(decision_tree)
  predictions = [bagging_predict(decision_trees, row) for row in test]
  return(predictions)

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
  scores = evaluate_algorithm(dataset, random_forest, n_folds, max_depth, min_size, sample_size, n_trees, n_features)
  print('Trees: %d' % n_trees)
  print('Scores: %s' % scores)
  print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))