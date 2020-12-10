import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import time

train_min = 0
train_max = 0
def preprocess(data, is_train):
    global train_min
    global train_max
    
    num_data = data[["Age", "Annual_Premium", "Vintage"]]
    bin_data = data.drop(["Age", "Annual_Premium", "Vintage"], axis=1)
    if is_train:
        train_min = num_data.min()
        train_max = num_data.max()
    normalized_data = (num_data - train_min)/(train_max - train_min)
    pre_data = pd.concat([normalized_data, bin_data], axis=1)
    return pre_data
    
def plot(train_costs, dev_costs):
    plt.title("Loss function")
    plt.xlabel("No. of iterations")
    plt.ylabel("Loss value")
    plt.plot(train_costs)
    plt.plot(dev_costs)
    plt.legend(["train", "dev"], loc ="lower right")
    plt.show()
    
def plot_gradient(gds):
    plt.title("gradient")
    plt.xlabel("No. of iterations")
    plt.ylabel("gradient")
    plt.plot(gds)
    plt.show()
    
def plot_accuracy(tas, das):
    plt.title("train accuracy")
    plt.xlabel("No. of iterations")
    plt.ylabel("train accuracy")
    plt.plot(tas)
    plt.plot(das)
    plt.legend(["train", "dev"], loc ="upper right")
    plt.show()
    
def sigmoid(x):
    return np.piecewise(x, [x > 0], [lambda i: 1 / (1 + np.exp(-i)), lambda i: np.exp(i) / (1 + np.exp(i))])

def loss_fn(w, x, y, regp=0.5):
    prob = sigmoid(np.dot(x, w))
    err = ((-1*y*np.log(prob))-((1-y)*np.log(1-prob)))
    n = len(y)
    return ((1/n) * np.sum(err)) + (regp*(np.linalg.norm(w[:-1], ord=1)))

def gradient(w, x, y, lr):
    predict = np.dot(x, w)
    sigmoid_val = sigmoid(predict)
    err = y - sigmoid_val
    
    gd = 1 / len(x) * np.dot(x.T, err)
    
    w += lr*gd

    return w, gd

def fit(t_x, t_y, d_x, d_y, lr=0.01, regp=0.5, epsilon=0.5):
    w = np.zeros(t_x.shape[1])
    gd = np.full((t_x.shape[1]), np.inf)
    no_iter = 0
    train_costs = []
    dev_costs = []
    
    prev_train_acc = 0
    best_rate = 0
    no_iter_best = 0
    best_dev_rate = 0
    no_iter_best_dev = 0
    smallest_gd = np.inf
    no_iter_smallest_gd = 0
    gds = []
    t_a = []
    d_a = []
    t_accuracy = 0
    d_accuracy = 0
    
    best_weight_dev = []
    while no_iter <= 2530:

        no_iter+=1
        w, gd = gradient(w, t_x, t_y, lr)
        gd_size = np.linalg.norm(gd,2)
        gds.append(gd_size)
        
        w[:-1] = np.sign(w[:-1]) * np.maximum((np.abs(w[:-1]) - lr * regp), 0)
        
        t_z = np.round(sigmoid(np.dot(t_x, w)))
        t_accuracy = np.count_nonzero(t_z == t_y)
        d_z = np.round(sigmoid(np.dot(d_x, w)))
        d_accuracy = np.count_nonzero(d_z == d_y)
        
        t_a.append(t_accuracy/len(t_y)*100)
        d_a.append(d_accuracy/len(d_y)*100)
        if best_rate < t_accuracy:
            best_rate = t_accuracy
            no_iter_best = no_iter
        
        if best_dev_rate < d_accuracy:
            best_dev_rate = d_accuracy
            no_iter_best_dev = no_iter
            best_weight_dev = w
            
        if smallest_gd > gd_size:
            smallest_gd = gd_size
            no_iter_smallest_gd = no_iter
    
    
    print("learning rate:" , lr)
    print("regualrization parameter:" , regp)
    print("number of iterations:", no_iter)
    total_w = len(w)
    w_zero_count = total_w - np.count_nonzero(w)
    print("total w:", total_w)
    print("w zero count:", w_zero_count)
    print("w sparsity:", w_zero_count/total_w*100)
    print("train acc:", t_accuracy/len(t_y))
    print("dev acc:", d_accuracy/len(d_y))
    print("best train acc:", best_rate/len(t_y))
    print("no of iterations of best train acc:", no_iter_best)
    print("best dev acc:", best_dev_rate/len(d_y))
    print("no of iterations of best dev acc:", no_iter_best_dev)
    print("smallest gds:", smallest_gd)
    print("no of iterations of smallest gd:", no_iter_smallest_gd)
    
    
    return train_costs, dev_costs
    
train_data = pd.read_csv("data/train_X.csv")
traindata = preprocess(train_data, True)
traindata_label = pd.read_csv("data/train_y.csv")

dev_data = pd.read_csv("data/dev_X.csv")
devdata = preprocess(dev_data, False)
devdata_label = pd.read_csv("data/dev_y.csv")

learning_rates = [0.1]
reg_params = [0.00001]
for i in learning_rates:
    for j in reg_params:
        start = time.time()
        t_costs, d_costs = fit(traindata, traindata_label["Response"], devdata, devdata_label["Response"], i, j, 0.00005)
        print("running time: ", time.time()-start)
