import sys
import csv
import math
import copy
from operator import itemgetter

# reading command line arguements and creating lists with the provided data
features_file = sys.argv[1]
examples_file = sys.argv[2]
data_file = sys.argv[3]

features_list = []
examples_list = []
data_list = []

lines = open(features_file).read().splitlines()
for line in lines:
  features_list.append(line)

with open(examples_file) as csv_file:
  csv_reader = csv.reader(csv_file, delimiter=',')
  for lines in csv_reader:
    examples_list.append(lines)

with open(data_file) as csv_file:
  csv_reader = csv.reader(csv_file, delimiter=',')
  for lines in csv_reader:
    data_list.append(lines)

# feature = '0', '1', '2' if it is a leaf node to indicate decision
class Tree:
  def __init__(self, depth):
    self.left = None
    self.right = None
    self.feature = None
    self.threshold = None
    self.depth = depth

def equal_trees(t1, t2):
  # check if 2 trees are identical
  if ((not t1) and (not t2)):
    return True
  if (t1 and t2):
    return (t1.feature == t2.feature and
            t1.threshold == t2.threshold and 
            t1.depth == t2.depth and
            equal_trees(t1.left, t2.left) and 
            equal_trees(t1.right, t2.right))
  return False

def split_data(examples):
  # split dataset into 10 equal subsets
  length = len(examples)
  sub_length = length / 10
  return_list = []
  count = 0
  for iteration in range(10):
    temp_list = []
    for index in range(sub_length):
      temp_list.append(examples[count])
      count += 1
    return_list.append(temp_list)
  return return_list

def pretty_print(tree):
  if not tree:
    print "NONE"
  if (tree.left):
    pretty_print(tree.left)
  print tree.feature + " " + str(tree.threshold) + " " + str(tree.depth)
  if (tree.right):
    pretty_print(tree.right)

def count_decisions(examples):
  # count number of each decision in examples
  if not examples:
    return 0, 0, 0, 0
  entry_len = len(examples[0])
  total_count = count0 = count1 = count2 = 0
  for example in examples:
    total_count += 1
    if (float(example[entry_len-1]) == 0):
      count0 += 1
    elif (float(example[entry_len-1]) == 1):
      count1 += 1
    else:
      count2 += 1
  return total_count, count0, count1, count2

def calc_entropy(zero_count, one_count, two_count):
  # calculate entropy based on counts of all 3 decisions
  total = zero_count + one_count + two_count
  if (total == 0):
    return 0
  zero_frac = float(zero_count) / total
  one_frac = float(one_count) / total
  two_frac = float(two_count) / total
  if (zero_frac == 0):
    zero_final = 0
  else:
    zero_final = zero_frac * math.log(zero_frac, 2)
  if (one_frac == 0):
    one_final = 0
  else:
    one_final = one_frac * math.log(one_frac, 2)
  if (two_frac == 0):
    two_final = 0
  else:
    two_final = two_frac * math.log(two_frac, 2)
  return -1 * (zero_final + one_final + two_final)

def split_list(feature_id, threshold, examples):
  # split list into 2 based on threshold value of given feature
  lesser = []
  greater = []
  for example in examples:
    if (float(example[feature_id]) < threshold):
      lesser.append(example)
    else:
      greater.append(example)
  return lesser, greater

def order_vals(feature_id, examples):
  # order list based on given feature
  copy_examples = copy.deepcopy(examples)
  return sorted(copy_examples, key=itemgetter(feature_id))

def find_feature_thresh(examples):
  # find optimal feature/threshold pair to optimize information gain
  if not examples:
    return 0, 0, [], []
  entry_len = len(examples[0])
  examples_len = len(examples)
  entropy_min = float("inf")
  feature_min = 0
  thresh_min = 0
  less_min = []
  more_min = []
  # iterating through all features 
  for feature in range(entry_len-1):
    # ordering examples based on feature of interest
    new_examples = order_vals(feature, examples)
    for index in range(examples_len-1):
      # finding all possible threshold values to find threshold with max info gain
      val = (float(new_examples[index][feature]) + float(new_examples[index+1][feature])) / 2
      less, more = split_list(feature, val, examples)
      l_total, l_count0, l_count1, l_count2 = count_decisions(less)
      m_total, m_count0, m_count1, m_count2 = count_decisions(more)
      entropy = (float(l_total) / examples_len) * calc_entropy(l_count0, l_count1, l_count2) + (float(m_total) / examples_len) * calc_entropy(m_count0, m_count1, m_count2)
      # keeping track of feature/threshold combo with max gain
      if (entropy < entropy_min):
        entropy_min = entropy
        feature_min = feature
        thresh_min = val
        less_min = less
        more_min = more
  return feature_min, thresh_min, less_min, more_min

def id3(features, examples, iteration, max_it):
  return_tree = Tree(iteration)
  # no more examples
  if not examples:
    return_tree.feature = '-1'
    return return_tree
  entry_len = len(examples[0])

  total_count, count0, count1, count2 = count_decisions(examples)
  max_count = max(count0, count1, count2)

  # remaining examples all have same decision
  if (count0 == total_count):
    return_tree.feature = '0'
    return return_tree
  elif (count1 == total_count):
    return_tree.feature = '1'
    return return_tree
  elif (count2 == total_count):
    return_tree.feature = '2'
    return return_tree

  # no more features or max depth reached
  if (iteration == max_it):
    if (count0 == max_count):
      return_tree.feature = '0'
    elif(count1 == max_count):
      return_tree.feature = '1'
    else:
      return_tree.feature = '2'
    return return_tree
  
  # need to split tree further
  # find feature & threshold with max info gain, and split examples accordingly
  feature, thresh, less, more = find_feature_thresh(examples)
  return_tree.feature = features[feature]
  #return_tree.feature = str(feature)
  return_tree.threshold = thresh

  # recursion on left and right subtrees
  return_tree.left = id3(features, less, iteration+1, max_it)
  return_tree.right = id3(features, more, iteration+1, max_it)

  # giving leaf nodes a decision if leaf nodes have no examples left
  if (return_tree.left.feature == '-1'):
    if (count0 == max_count):
      return_tree.left.feature = '0'
    elif (count1 == max_count):
      return_tree.left.feature = '1'
    else:
      return_tree.left.feature = '2'
  if (return_tree.right.feature == '-1'):
    if (count0 == max_count):
      return_tree.right.feature = '0'
    elif (count1 == max_count):
      return_tree.right.feature = '1'
    else:
      return_tree.right.feature = '2'
  return return_tree

def deep_copy_tree(tree):
  # make deep copy of tree
  if not tree:
    return None
  new_tree = Tree(0)
  new_tree.feature = tree.feature
  new_tree.threshold = tree.threshold
  new_tree.depth = tree.depth
  new_tree.left = deep_copy_tree(tree.left)
  new_tree.right = deep_copy_tree(tree.right)
  return new_tree

def evaluate_tree(tree, examples):
  # evaluate decision tree based on given examples
  total = correct = 0
  for example in examples:
    temp = deep_copy_tree(tree) 
    entry_len = len(example)
    correct_decision = example[entry_len-1]
    total += 1
    while (True):
      cur_feature = temp.feature
      if (cur_feature == '0'):
        if (correct_decision == '0.0'):
          correct += 1
        break
      elif (cur_feature == '1'):
        if (correct_decision == '1.0'):
          correct += 1
        break
      elif (cur_feature == '2'):
        if (correct_decision == '2.0'):
          correct += 1
        break
      cur_index = features_list.index(cur_feature)
      threshold = temp.threshold
      if (float(example[cur_index]) < threshold):
        temp = temp.left
      else:
        temp = temp.right
  return float(correct) / total

def k_fold():
  # perform 10-fold cross validation
  new_data = split_data(examples_list)
  cur_depth = 0
  depth_accuracy = []
  quit = False
  while (not quit):
    total_valid = 0.0
    total_train = 0.0
    # 10 rounds of learning, each subset of examples is used as validation set,
    #   while the rest is used as the training set
    for iteration in range(10):
      validation_set = [] 
      training_set = []
      for i in range(10):
        if (i == iteration):
          for data in new_data[i]:
            validation_set.append(data)
        else:
          for data in new_data[i]:
            training_set.append(data)
      # creating a tree based on the current training_set at the current max depth
      cur_tree = id3(features_list, training_set, 0, cur_depth)
      accuracy_valid = evaluate_tree(cur_tree, validation_set)
      accuracy_train = evaluate_tree(cur_tree, training_set)
      total_valid += accuracy_valid
      total_train += accuracy_train
      
      if (equal_trees(id3(features_list, training_set, 0, 100), cur_tree)):
      #if (cur_depth == 11):
        quit = True
    
    total_valid /= 10
    total_train /= 10
    depth_accuracy.append((cur_depth, total_train, total_valid))
    cur_depth += 1 
  return depth_accuracy
  ### find max of depth_accuracy list

# find average prediction accuracies of different depths
avg_prediction_acc = k_fold()

# finding optimal depth based on dataset A
sorted_predictions = sorted(avg_prediction_acc, key=itemgetter(2))
optimal_depth = sorted_predictions[-1][0]

# creating decision tree based on examples as training set and optimal depth
decision_tree = id3(features_list, examples_list, 0, optimal_depth)

# print accuracy of decision tree on dataset B
print evaluate_tree(decision_tree, data_list)
