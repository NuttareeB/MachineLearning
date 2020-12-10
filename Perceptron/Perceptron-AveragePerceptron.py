import pandas as pd
import numpy as np
import time
import statistics
import matplotlib.pyplot as plt

def plot_accuracy(title, tas, das):
    plt.title(title)
    plt.xlabel("No. of iterations")
    plt.ylabel("Accuracy %")
    plt.plot(tas)
    plt.plot(das)
    plt.legend(["train", "dev"], loc ="upper right")
    plt.show()
    
def fit(t_x, t_y, d_x, d_y, maxiter):
    t_y = t_y.to_numpy().flatten()
    d_y = d_y.to_numpy().flatten()
    t_x = t_x.to_numpy()
    d_x = d_x.to_numpy()
    
    w = np.zeros(t_x.shape[1])
    w_bar = np.zeros(t_x.shape[1])
    
    s = 1
    no_iteration = 0
    
    online_train_acc_list = []
    online_dev_acc_list = []
    avg_train_acc_list = []
    avg_dev_acc_list = []
    avg_dev_no_correct_prediction = []
    
    while no_iteration < maxiter:
        for i, x in enumerate(t_x):
            y = t_y[i]
            if y * (w.dot(x)) <= 0:
                w += (y * x)
            w_bar = ((s * w_bar) + w)/(s+1)
            s += 1
        
        no_iteration += 1
        
        # online perceptron accuracy
        t_predict = t_y*np.dot(t_x, w)
        no_correct_prediction_t = sum(1 for i in t_predict if i > 0)
        acc_rate_t = no_correct_prediction_t/len(t_y)*100
        online_train_acc_list.append(acc_rate_t)
        
        d_predict = d_y*np.dot(d_x, w)
        no_correct_prediction_d = sum(1 for i in d_predict if i > 0)
        acc_rate_d = no_correct_prediction_d/len(d_y)*100
        online_dev_acc_list.append(acc_rate_d)
        
        # average perceptron accuracy
        a_t_predict = t_y*np.dot(t_x, w_bar)
        a_no_correct_prediction_t = sum(1 for i in a_t_predict if i > 0)
        a_acc_rate_t = a_no_correct_prediction_t/len(t_y)*100
        avg_train_acc_list.append(a_acc_rate_t)
        
        a_d_predict = d_y*np.dot(d_x, w_bar)
        a_no_correct_prediction_d = sum(1 for i in a_d_predict if i > 0)
        avg_dev_no_correct_prediction.append(a_no_correct_prediction_d)
        a_acc_rate_d = a_no_correct_prediction_d/len(d_y)*100
        avg_dev_acc_list.append(a_acc_rate_d)
        
    plot_accuracy("Online perceptron accuracy", online_train_acc_list, online_dev_acc_list)
    plot_accuracy("Average perceptron accuracy", avg_train_acc_list, avg_dev_acc_list)
    
    print("mean online train accuracy:", statistics.mean(online_train_acc_list))
    print("mean online dev accuracy:", statistics.mean(online_dev_acc_list))
    print("mean avg train accuracy:", statistics.mean(avg_train_acc_list))
    print("mean avg dev accuracy:", statistics.mean(avg_dev_acc_list))
    
    print("maxiter:", maxiter)
    highest_accuracy =  max(avg_dev_acc_list)
    best_iter_index = avg_dev_acc_list.index(highest_accuracy)
    print("Best iteration:", best_iter_index + 1)
    print("No. of accurated prediction:", avg_dev_no_correct_prediction[best_iter_index])
    print("Size of data set:", len(d_y))
    print("Highest accuracy rate", highest_accuracy)
    

train_data = pd.read_csv("data/train_X.csv")
traindata_label = pd.read_csv("data/train_y.csv")
dev_data = pd.read_csv("data/dev_X.csv")
devdata_label = pd.read_csv("data/dev_y.csv")

start = time.time()
fit(train_data, traindata_label, dev_data, devdata_label, 100)
print("running time:", time.time()-start, "s")
