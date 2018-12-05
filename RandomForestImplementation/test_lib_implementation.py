from csv import reader
from RandomForest import RandomForest
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
import time
from math import sqrt
from RandomForest import RandomForest

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


# load and prepare data
filename = 'data_out3.csv'
dataset = load_csv(filename)
headers = dataset[0]
dataset = dataset[1:]

for col_idx in range(0, len(dataset[0])-1): convert_column_to_float(dataset, col_idx)
convert_str_column_to_int(dataset, len(dataset[0]) - 4, len(dataset[0]) - 3, len(dataset[0])-1)

features = pd.DataFrame(dataset, columns=headers)
labels = np.array(features['winner'])
features= features.drop('winner', axis = 1)
feature_list = list(features.columns)
features = np.array(features)

kf = KFold(n_splits=5)
kf.get_n_splits(features)


accuracy_n_trres = []
time_n_trees = []
# for n_trees in [i for i in range(1, 101)]:
multiplier = (len(dataset[0]) - 1) / 10
m_array = [ multiplier * i for i in range(1, 11) ]
m_array.insert(0, int(sqrt(len(dataset[0])-1)))
n_features = int(sqrt(len(dataset[0])-1))
start = time.time()
# for n_trees in [i for i in range(1, 101)]:
for n_trees in [5]:
  # rf = RandomForestClassifier(max_features=n_features, n_estimators = n_trees, random_state = 42)
  
  fold_accuracy = []
  f_predicted = []
  f_actuals = []
  for train_index, test_index in kf.split(features):
    features_train, features_test = features[train_index], features[test_index]
    labels_train, labels_test = labels[train_index], labels[test_index]
    
    train_set = np.c_[features_train, labels_train]
    test_set = np.c_[features_test, labels_test]


    predictions = RandomForest(train_set.tolist(), test_set.tolist(), 10, 1, 1.0, n_trees, int(sqrt(len(dataset[0])-1))).run()

    # rf.fit(features_train, labels_train)
    # predictions = rf.predict(features_test)
    f_predicted.extend(predictions)
    f_actuals.extend(labels_test.tolist())
    errors = abs(np.array(predictions) - labels_test)
    mape = 100 * (errors / labels_test)
    accuracy = 100 - np.mean(mape)
    fold_accuracy.append(accuracy)


  print('trees: {}, accuracy {}, time: {}'.format(n_trees, accuracy, time.time() - start))
  accuracy_n_trres.append(sum(fold_accuracy) / float(len(fold_accuracy)))
  time_n_trees.append(time.time() - start)

# print(accuracy_n_trres)
# print('--------------')
# print(time_n_trees)
print(f_predicted)
print('--------------')
print(f_actuals)

# import matplotlib.pyplot as plt
# x_axis = [n_tree for n_tree in range(1, 101)] # 1 to 10 trees

# y1 = accuracy_n_trres
# # y2 = time_n_trees
# fig = plt.figure()
# ax  = fig.add_subplot(111)
# plt.xlabel('Number of trees in random forrest')
# plt.ylabel('Accuracy (out of 100%)')
# ax.plot(x_axis, y1, c='r', label='Model', linewidth=2.0)
# # ax.plot(x_axis, y2, c='b', label='Random Forest', linewidth=1.0)
# plt.title('Accuracy for sk learn for different number of trees')
# leg = plt.legend()
# plt.show()

  # print('n_trees: {}'.format(n_trees)) 
  # print('mean accuracy: {}'.format(sum(fold_accuracy) / float(len(fold_accuracy))))
  # print('time taken: {}'.format(time.time() - start))

  # gnb = GaussianNB()
  # for train_index, test_index in kf.split(features):
  #   features_train, features_test = features[train_index], features[test_index]
  #   labels_train, labels_test = labels[train_index], labels[test_index]
  #   gnb.fit(features_train, labels_train)
  #   predictions = gnb.predict(features_test)
  #   errors = abs(predictions - labels_test)
  #   mape = 100 * (errors / labels_test)
  #   accuracy = 100 - np.mean(mape)
  #   fold_accuracy.append(accuracy)

  # print('n_trees: {}'.format(n_trees))
  # print('mean accuracy: {}'.format(sum(fold_accuracy) / float(len(fold_accuracy))))
  # print('time taken: {}'.format(time.time() - start))


# rf = RandomForestClassifier(n_estimators = 5, random_state = 42)

# rf.fit(train_features, train_labels)

# predictions = rf.predict(test_features)

# # Calculate the absolute errors
# errors = abs(predictions - test_labels)
# # Print out the mean absolute error (mae)
# print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# # Calculate mean absolute percentage error (MAPE)
# mape = 100 * (errors / test_labels)
# # Calculate and display accuracy
# accuracy = 100 - np.mean(mape)
# print('Accuracy:', round(accuracy, 2), '%.')