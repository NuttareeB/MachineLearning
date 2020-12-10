import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

def preprocess(data):
    num_data = data.drop(["dummy", "id", "date", "waterfront", "zipcode", "price"], axis=1)
    normalized_data = (num_data-num_data.min())/(num_data.max() - num_data.min())
    
    data[["month", "day", "year"]] = data["date"].str.split("/", expand=True)
    
    bin_data = pd.get_dummies(data[["month", "day", "year", "zipcode", "waterfront"]].astype(str))
    
# #     normalize
#     pre_data = pd.concat([data["dummy"], normalized_data, bin_data], axis=1)
#     not normalize
    pre_data = pd.concat([data["dummy"], num_data, bin_data], axis=1)
    return pre_data
    
def loss_fn(w, x, y):
    sq_err = ((np.dot(x, w.T))-y)**2
    n = len(y)
    return (1/n) * sq_err.sum()

def gradient(w, x, y, lr):
    predict = np.dot(x, w)
    err = predict - y
    
    gd = 2 / len(x) * np.dot(x.T, err)
    
    w -= lr * gd
    return w, gd

def plot(train_costs, dev_costs):
    plt.title("Loss function")
    plt.xlabel("No. of iterations")
    plt.ylabel("Loss value")
    plt.plot(train_costs)
    plt.plot(dev_costs)
    plt.legend(["train", "dev"], loc ="lower right")
    plt.show()

def fit(t_x, t_y, d_x, d_y, lr=0.01, epsilon=0.5):
    w = np.zeros(t_x.shape[1])
    
    gd = np.full((t_x.shape[1]), np.inf)
    no_iter = 0
    train_costs = []
    dev_costs = []
    
    train_costs.append(loss_fn(w, t_x, t_y))
    dev_costs.append(loss_fn(w, d_x, d_y))
    
#     while np.linalg.norm(gd,2) > epsilon and (not train_costs or (train_costs and not np.isinf(train_costs[-1]))):
    while no_iter < 10000 and (not train_costs or (train_costs and not np.isinf(train_costs[-1]))):
        no_iter+=1
        w, gd = gradient(w, t_x, t_y, lr)
        
        train_costs.append(loss_fn(w, t_x, t_y))
        dev_costs.append(loss_fn(w, d_x, d_y))
#     print("data value:", list(data.columns.values))
#     print("len(w):", len(w))
    
#     weight = [(c, w[i]) for i, c in enumerate(data.columns.values)]
#     print("weight: ")
#     print(weight)
#     print("max weight:", max(w))
#     print("min weight:", min(w))
    print("learning rate:" , lr)
    print("number of iteration:", no_iter)
    print("last value of loss function of the training set:", train_costs[-1])
    print("last value of loss function of the dev set:", dev_costs[-1])
    
    plot(train_costs, dev_costs)
    return train_costs, dev_costs

train_data = pd.read_csv("data/train.csv")
data = preprocess(train_data)

dev_data = pd.read_csv("data/dev.csv")
devdata = preprocess(dev_data)
    
# learning_rates = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
# learning_rates = [100, 10, 0.1, 0.01, 0.001, 0.0001]
learning_rates = [0.0000000001]
for r in learning_rates:
    t_costs, d_costs = fit(data, train_data["price"], devdata, dev_data["price"], r, 0.5)
