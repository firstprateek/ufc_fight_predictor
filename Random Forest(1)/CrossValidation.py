from random import randrange
from random import seed

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
def cross_validate_algorithm(dataset, algorithm, n_folds, *args):
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

    predicted = algorithm(train_set, test_set, *args).run()

    actual = [row[-1] for row in fold]
    accuracy = accuracy_metric(actual, predicted)
    scores.append(accuracy)
  return scores