import pandas as pd
import numpy as np
from math import log2
import time
import matplotlib.pyplot as plt

def entropy(class0, class1):
    if class0==0 or class1==0:
        return 0
    return -(class0 * log2(class0) + class1 * log2(class1))

def information_gain(data, val, m):
    label = data[:,-1]
    n = len(label)
    no_pos_label = np.count_nonzero(label)
    no_neg_label = n - no_pos_label
    
    original_entropy = entropy(no_pos_label/n, no_neg_label/n)
    
    largest_mutual = -1
    index = -1
    value = val
    groups = None
    
#     features = []
#     while len(features) < m:
#         idx = np.random.randint(len(data[0])-1)
#         if idx not in features:
#             features.append(idx)

#     for i in features:
#     for i in random.sample(range(len(data[0])-1), m):
    for i in np.random.choice(len(data[0])-1, m, replace=False):
        left_idx = data[:, i] < val
        l = data[left_idx]
        n_l = len(l)
        n_l_pos = np.count_nonzero(l[:,-1])
        n_l_neg = n_l-n_l_pos
        
        right_idx = data[:, i] >= val
        r = data[right_idx]
        n_r = len(r)
        n_r_pos = np.count_nonzero(r[:,-1])
        n_r_neg = n_r-n_r_pos
        
        l_entropy = entropy(n_l_pos/n_l, n_l_neg/n_l) if n_l != 0 else 0
        r_entropy = entropy(n_r_pos/n_r, n_r_neg/n_r) if n_r != 0 else 0
        conditional_entropy = (n_l/n) * l_entropy + (n_r/n) * r_entropy
        mutual_information = original_entropy - conditional_entropy
        if mutual_information > largest_mutual:
            largest_mutual = mutual_information
            index = i
            groups = (l,r)
    return {'idx':index, 'val':value, 'groups':groups}

def select_best_split(data, m):
    return information_gain(data, 1, m)

def get_terminal_node_val(group):
    labels = [row[-1] for row in group]
    return max(set(labels), key=labels.count)

def split(node, max_depth, depth, m):
    left, right = node['groups']
    del(node['groups'])
    if len(left) == 0 or len(right) == 0:
        if len(left) == 0:
            node['l'] = node['r'] = get_terminal_node_val(right)
        else:
            node['l'] = node['r'] = get_terminal_node_val(left)
        return
    if depth >= max_depth:
        node['l'], node['r'] = get_terminal_node_val(left), get_terminal_node_val(right)
        return
    
    node['l'] = select_best_split(left, m)
    split(node['l'], max_depth, depth+1, m)
    
    node['r'] = select_best_split(right, m)
    split(node['r'], max_depth, depth+1, m)

def build_tree(t_data, max_depth, m):
    root_node = select_best_split(t_data, m)
    split(root_node, max_depth, 1, m)
    return root_node

def predict(node, sample_row):
    if sample_row[node['idx']] < node['val']:
        if isinstance(node['l'], dict):
            return predict(node['l'], sample_row)
        else:
#             terminal node
            return node['l']
    else:
        if isinstance(node['r'], dict):
            return predict(node['r'], sample_row)
        else:
#             terminal node
            return node['r']

def subsample(dataset):
    sample = []
    n_sample = len(dataset)
#     while len(sample) < n_sample:
#         idx = np.random.randint(len(dataset))
    for idx in np.random.choice(len(dataset), len(dataset), replace=True):
        sample.append(dataset[idx])
    return np.array(sample)

def predict_from_forest(trees, row):
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)

def generate_trees(t_data, max_depth, m, n_trees):
    trees = []
    for i in range(n_trees):
        t_sample = subsample(t_data)
        tree = build_tree(t_sample, max_depth, m)
        trees.append(tree)
    return trees

def random_forest(t_data, d_data, max_depth, m, t, fulltrees):
#     trees = np.random.choice(fulltrees, t, replace=False)
    trees = fulltrees[:t]
#     trees = []
#     for i in range(t):
#         t_sample = subsample(t_data)
#         tree = build_tree(t_sample, max_depth, m)
#         trees.append(tree)
        
    train_predictions = []
    t_size = len(t_data)
    for t_row in t_data:
        train_prediction = predict_from_forest(trees, t_row)
        train_predictions.append(train_prediction)

    t_label = t_data[:,-1].flatten()
    t_accuracy = t_size - np.count_nonzero(t_label-train_predictions)
    t_accuracy_rate = t_accuracy/t_size*100
    print("training set \tmax_depth:", max_depth, "m:", m, "t:", t, "\taccuracy:", t_accuracy_rate)

    dev_predictions = []
    d_size = len(d_data)
    for d_row in d_data:
        dev_prediction = predict_from_forest(trees, d_row)
        dev_predictions.append(dev_prediction)
        
    d_label = d_data[:,-1].flatten()
    d_accuracy = d_size - np.count_nonzero(d_label-dev_predictions)
    d_accuracy_rate = d_accuracy/d_size*100
    print("validation set \tmax_depth:", max_depth, "m:", m, "t:", t, "\taccuracy:", d_accuracy_rate)
    return t_accuracy_rate, d_accuracy_rate
    
def plot_accuracy(title, xticks, acc_list, y_lim):
    plt.figure(figsize=[6.4, 5.3])
    plt.title(title)
    plt.xlabel("Number of trees")
    plt.ylabel("Accuracy %")
    for i in range(4):
        plt.plot(acc_list[i])
    plt.legend(["m=5", "m=25", "m=50", "m=100"], loc ="upper right")
    plt.xticks(np.arange(len(xticks)), xticks)
#     plt.ylim(y_lim[0], y_lim[1])
    plt.ylim(67, 90)
    plt.show()
    
train_data = pd.read_csv("pa4_data/pa4_train_X.csv").values.astype(int)
traindata_label = pd.read_csv("pa4_data/pa4_train_y.csv", header=None).values.astype(int)
train_dataset = np.concatenate((train_data, traindata_label), axis=1);

dev_data = pd.read_csv("pa4_data/pa4_dev_X.csv").values.astype(int)
devdata_label = pd.read_csv("pa4_data/pa4_dev_y.csv", header=None).values.astype(int)
dev_dataset = np.concatenate((dev_data, devdata_label), axis=1);

np.random.seed(1)

dmaxs=[2, 10, 25]
ms = [5,25,50,100]
start = time.time()

# t_accuracies = []
# d_accuracies = []
y_lim = [(60,80),(72,82.5), (70,90)]
for i, d in enumerate(dmaxs):
    t_accuracies = []
    d_accuracies = []
    print("dmax:", d)
    for m in ms:
        t_accuracies.append([])
        d_accuracies.append([])
        fulltrees = generate_trees(train_dataset, d, m, 100)
        for n_trees in range(10,101,10):
            t_accuracy_rate, d_accuracy_rate = random_forest(train_dataset, dev_dataset, d, m, n_trees, fulltrees)
#             print("m:", m, "\tT:", n_trees, "\ttrain: ", t_accuracy_rate, "\tdev:", d_accuracy_rate)
            t_accuracies[-1].append(t_accuracy_rate)
            d_accuracies[-1].append(d_accuracy_rate)

    plot_accuracy("Training accuracy rate with max depth = " + str(d), range(10,101,10), t_accuracies, y_lim[i])
    plot_accuracy("Validation accuracy rate with max depth = " + str(d), range(10,101,10), d_accuracies, y_lim[i])

print("\nrunning time:", time.time()-start)
