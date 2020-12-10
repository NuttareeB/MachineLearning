import pandas as pd
import numpy as np
from math import log2
import time
import matplotlib.pyplot as plt

def entropy(class0, class1):
    if class0==0 or class1==0:
        return 0
    return -(class0 * log2(class0) + class1 * log2(class1))

def information_gain(data, val):
    label = data[:,-1]
    n = len(label)
    no_pos_label = np.count_nonzero(label)
    no_neg_label = n - no_pos_label
    
    original_entropy = entropy(no_pos_label/n, no_neg_label/n)
    
    largest_mutual = -1
    index = -1
    value = val
    groups = None
    for i in range(len(data[0])-1):
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
#     print("node:", index, "\tinformation gain:", largest_mutual)
    return {'idx':index, 'val':value, 'groups':groups}

def select_best_split(data):
    return information_gain(data, 1)

def get_terminal_node_val(group):
    labels = [row[-1] for row in group]
    return max(set(labels), key=labels.count)

def split(node, max_depth, depth):
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
    
    node['l'] = select_best_split(left)
    split(node['l'], max_depth, depth+1)
    
    node['r'] = select_best_split(right)
    split(node['r'], max_depth, depth+1)

def build_tree(t_data, max_depth):
    root_node = select_best_split(t_data)
    split(root_node, max_depth, 1)
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

def plot_accuracy(title, xticks, tas, das):
    plt.title(title)
    plt.xlabel("Tree depth")
    plt.ylabel("Accuracy %")
    plt.plot(tas)
    plt.plot(das)
    plt.legend(["train", "dev"], loc ="center right")
    plt.xticks(np.arange(len(xticks)), xticks)
    plt.grid(linestyle='-')
    plt.show()
    
t_accuracies = []
d_accuracies = []
def decision_tree(t_data, d_data, max_depth):
    tree = build_tree(t_data, max_depth)
    
    train_predictions = []
    t_size = len(t_data)
    for t_row in t_data:
        train_prediction = predict(tree, t_row)
        train_predictions.append(train_prediction)

    t_label = t_data[:,-1].flatten()
    t_accuracy = t_size - np.count_nonzero(t_label-train_predictions)
    t_accuracy_rate = t_accuracy/t_size*100
    t_accuracies.append(t_accuracy_rate)
    print("training set \tmax_depth:", max_depth, "\taccuracy:", t_accuracy_rate)

    dev_predictions = []
    d_size = len(d_data)
    for d_row in d_data:
        dev_prediction = predict(tree, d_row)
        dev_predictions.append(dev_prediction)
        
    d_label = d_data[:,-1].flatten()
    d_accuracy = d_size - np.count_nonzero(d_label-dev_predictions)
    d_accuracy_rate = d_accuracy/d_size*100
    d_accuracies.append(d_accuracy_rate)
    print("validation set \tmax_depth:", max_depth, "\taccuracy:", d_accuracy_rate)
        
    

train_data = pd.read_csv("pa4_data/pa4_train_X.csv").values.astype(int)
traindata_label = pd.read_csv("pa4_data/pa4_train_y.csv", header=None).values.astype(int)
train_dataset = np.concatenate((train_data, traindata_label), axis=1);

dev_data = pd.read_csv("pa4_data/pa4_dev_X.csv").values.astype(int)
devdata_label = pd.read_csv("pa4_data/pa4_dev_y.csv", header=None).values.astype(int)
dev_dataset = np.concatenate((dev_data, devdata_label), axis=1);
dmaxs=[2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
# dmaxs=[2]
start = time.time()
for dmax in dmaxs:
    decision_tree(train_dataset, dev_dataset, dmax)
print("\nrunning time:", time.time()-start)
plot_accuracy("Accuracy rate of each tree depth", dmaxs, t_accuracies, d_accuracies)

