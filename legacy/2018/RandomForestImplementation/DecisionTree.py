from random import randrange
from random import seed

class DecisionTree:
  def __init__(self, train, max_depth, min_size, n_features):
    seed(55)
    self.train_set = train
    self.max_depth = max_depth
    self.min_size = min_size
    self.n_features = n_features

  # Build a decision tree
  def build(self):
    root = self.get_split(self.train_set, self.n_features)
    self.split(root, self.max_depth, self.min_size, self.n_features, 1)
    return root  

  # Split a dataset based on an attribute and an attribute value
  def test_split(self, index, value, dataset):
    left, right = [], []
    for row in dataset:
      if row[index] < value: left.append(row)
      else: right.append(row)
    return left, right

  # Calculate the Gini index for a split dataset
  def gini_index(self, groups, classes):
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
  def get_split(self, dataset, n_features):
    class_values = list({ row[-1] for row in dataset })
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    features = []
    while len(features) < n_features:
      index = randrange(len(dataset[0])-1)
      if index not in features: features.append(index)
    for index in features:
      for row in dataset:
        groups = self.test_split(index, row[index], dataset)
        gini = self.gini_index(groups, class_values)
        if gini < b_score: b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index':b_index, 'value':b_value, 'groups':b_groups}

  # Create a terminal node value
  def to_terminal(self, group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

  # Create child splits for a node or make terminal
  def split(self, node, max_depth, min_size, n_features, depth):
    left, right = node['groups']
    del(node['groups'])
    # check for a no split
    if not left or not right:
      node['left'] = node['right'] = self.to_terminal(left + right)
      return
    # check for max depth
    if depth >= max_depth:
      node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
      return
    # process left child
    if len(left) <= min_size:
      node['left'] = self.to_terminal(left)
    else:
      node['left'] = self.get_split(left, n_features)
      self.split(node['left'], max_depth, min_size, n_features, depth+1)
    # process right child
    if len(right) <= min_size:
      node['right'] = self.to_terminal(right)
    else:
      node['right'] = self.get_split(right, n_features)
      self.split(node['right'], max_depth, min_size, n_features, depth+1)
