import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

def plot_validation_accuracy_list(title, das, p):
    plt.title(title)
    plt.xlabel("No. of iterations")
    plt.ylabel("Accuracy %")
    for i in range(p):
        plt.plot(das[i], "o")
    plt.legend(["p=1", "p=2", "p=3", "p=4", "p=5"], loc ="lower left")
    plt.show()
    
def plot_validation_accuracy(title, tas, das):
    plt.title(title)
    plt.xlabel("No. of iterations")
    plt.ylabel("Accuracy %")
    plt.plot(tas)
    plt.plot(das)
    plt.legend(["train", "dev"], loc ="lower left")
    plt.show()
    
def plot_best_validation_accuracy_p(title, best_dev_acc):
    plt.title(title)
    plt.xlabel("p")
    plt.ylabel("Accuracy %")
    plt.plot(best_dev_acc)
    plt.xticks(np.arange(5), ['1', '2', '3', '4', '5'])
    plt.show()

def plot_time(title, run_time_p_1):
    plt.title(title)
    plt.xlabel("No. of iterations")
    plt.ylabel("Time(seconds)")
    plt.plot(run_time_p_1)
    plt.show()

def kernel(x_1, x_2, p):
    return (1 + np.dot(x_1, x_2.T))**p

online_train_acc_list = []
online_dev_acc_list = []
best_dev_acc = []
run_time_p_1 = []
cumulative_run_time_p_1 = []

def fit(t_x, t_y, d_x, d_y, maxiter, k, p):
    t_x = t_x.to_numpy()
    t_y = t_y.to_numpy()
    d_x = d_x.to_numpy()
    d_y = d_y.to_numpy()
    
    cumulative_run_time = 0
    
    datasize = len(t_x)
    alpha = np.zeros(datasize)
    
    K = kernel(t_x, t_x, p)
    K_d = kernel(t_x, d_x, p)
    no_iteration = 0
    
    while no_iteration < maxiter:
        i_start = time.time()
        
        no_iteration += 1
        for i, row in enumerate(K):
            u = np.sum(alpha * row * (t_y.T).flatten())
            if u * t_y[i] <= 0:
                alpha[i] += 1
        if p == 1:
            i_end = time.time() - i_start
            run_time_p_1.append(i_end)
            cumulative_run_time += i_end
            cumulative_run_time_p_1.append(cumulative_run_time)
            
        
        # online kernel perceptron accuracy
        t_predict = np.sign(np.dot((alpha * (t_y.T).flatten()), K))
        no_correct_prediction_t = np.sum(t_predict == (t_y.T).flatten())
        acc_rate_t = no_correct_prediction_t/len(t_y)*100
        if len(online_train_acc_list) < p:
            online_train_acc_list.append([])
        online_train_acc_list[p-1].append(acc_rate_t)
        
        d_predict = np.sign(np.dot((alpha * (t_y.T).flatten()), K_d))
        no_correct_prediction_d = np.sum(d_predict == (d_y.T).flatten())
        acc_rate_d = no_correct_prediction_d/len(d_y)*100
        if len(online_dev_acc_list) < p:
            online_dev_acc_list.append([])
        online_dev_acc_list[p-1].append(acc_rate_d)
        
    highest_accuracy = max(online_dev_acc_list[p-1])
    best_iter_index = online_dev_acc_list[p-1].index(highest_accuracy)
    best_dev_acc.append(highest_accuracy)
    print()
    print("p:", p)
    print("maxiter:", maxiter)
    print("Best iteration (validation accuracy):", best_iter_index + 1)
    print("Highest validation accuracy rate", highest_accuracy)
    

train_data = pd.read_csv("data/train_X.csv")
traindata_label = pd.read_csv("data/train_y.csv")
dev_data = pd.read_csv("data/dev_X.csv")
devdata_label = pd.read_csv("data/dev_y.csv")

p = [1,2,3,4,5]

for i in range(len(p)):
    start = time.time()
    fit(train_data, traindata_label, dev_data, devdata_label, 100, kernel, p[i])
    print("running time: p =", i+1 , time.time()-start, "s")

plot_best_validation_accuracy_p("Best validation accuracy", best_dev_acc)
for i in range(len(p)):
    print("p =", i+1, "Best validation accuracy:", best_dev_acc[i])

for i in range(len(p)):
    plot_validation_accuracy("Online perceptron accuracy for p=" + str(i+1), online_train_acc_list[i], online_dev_acc_list[i])
plot_validation_accuracy_list("Online perceptron train accuracy", online_train_acc_list, 5)
plot_validation_accuracy_list("Online perceptron validation accuracy", online_dev_acc_list, 5)
plot_time("The empirical runtime of p=1", run_time_p_1)
plot_time("The cumulative runtime of p=1", cumulative_run_time_p_1)
