from random import randrange
from random import seed
from DecisionTree import DecisionTree

class RandomForest:
  def __init__(self, train, test, max_depth, min_size, sample_size, n_trees, n_features):
    seed(55)
    self.train_set = train
    self.test_set = test
    self.max_depth = max_depth
    self.min_size = min_size
    self.sample_size = sample_size
    self.n_trees = n_trees
    self.n_features = n_features

  def run(self):
    decision_trees = []
    for i in range(self.n_trees):
      sample = self.subsample(self.train_set, self.sample_size)
      
      decision_tree = DecisionTree(sample, self.max_depth, self.min_size, self.n_features).build()
      
      decision_trees.append(decision_tree)
    predictions = [self.bagging_predict(decision_trees, row) for row in self.test_set]
    return(predictions)

  def subsample(self, dataset, ratio):
    sample = []
    samples = round(len(dataset) * ratio)
    while len(sample) < samples: sample.append(dataset[randrange(len(dataset))])
    return sample

  # Make a prediction with a list of bagged trees
  def bagging_predict(self, trees, row):
    predictions = [self.predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)

  # Make a prediction with a decision tree
  def predict(self, node, row):
    if row[node['index']] < node['value']:
      if isinstance(node['left'], dict):
        return self.predict(node['left'], row)
      else:
        return node['left']
    else:
      if isinstance(node['right'], dict):
        return self.predict(node['right'], row)
      else:
        return node['right']
