import numpy as np

def load_data(filename):
    return open(filename).readlines()

def map_feature_train_data(new_row, feature, mapping):
    if feature not in mapping:
        mapping[feature] = len(mapping)
    new_row.append(mapping[feature])

def map_feature_dev_data(new_row, feature, mapping):
    if feature in mapping:
        new_row.append(mapping[feature])
        
mapping = {}

def binarize_data(datalist, func):
    data_list = list(map(lambda s: s.strip().split(', '), datalist))
    catag_data = [[value for idx, value in enumerate(line) if idx not in [0,7,9]] for line in data_list] # index 0, 7 are not categorical fields, 9 is target
    
    new_data = []
    for row in catag_data:
        new_row = []
        for j, x in enumerate(row):
            feature = (j, x)
            func(new_row, feature, mapping);
        new_data.append(new_row)

    num_data = len(datalist)
    num_catag_feature = len(mapping)
    bindata = np.zeros((num_data, num_catag_feature + 2)) # i = no of data, j = no of features + 2

    for i, row in enumerate(new_data):
        for j in row:
            bindata[i][j] = 1
        bindata[i][num_catag_feature] = int(data_list[i][0])/50 
        bindata[i][num_catag_feature + 1] = int(data_list[i][7])/50
    return data_list, bindata

def binarize_data_all(datalist, func):
    data_list = list(map(lambda s: s.strip().split(', '), datalist))
    raw_data = [[value for idx, value in enumerate(line) if idx not in [9]] for line in data_list] # index 9 is target
    
    new_data = []
    for row in raw_data:
        new_row = []
        for j, x in enumerate(row):
            feature = (j, x)
            func(new_row, feature, mapping);
        new_data.append(new_row)

    num_data = len(datalist)
    num_catag_feature = len(mapping)
    bindata = np.zeros((num_data, num_catag_feature)) # i = no of data, j = no of features

    for i, row in enumerate(new_data):
        for j in row:
            bindata[i][j] = 1
    return data_list, bindata
    
def get_nearest_neighbor_eucli(bin_train_list, bin_test, num_neighbors):
    eucli_dist = np.linalg.norm(bin_train_list - bin_test, axis=1)
    nearest_indices = np.argpartition(eucli_dist, range(num_neighbors))[:num_neighbors]
#     print("distance: ", [eucli_dist[i] for i in nearest_indices])
    return nearest_indices
    
def get_nearest_neighbor_manhat(bin_train_list, bin_test, num_neighbors):
    manhat_dist = np.sum(np.absolute(bin_train_list - bin_test), axis=1)
    nearest_indices = np.argpartition(manhat_dist, range(num_neighbors))[:num_neighbors]
    return nearest_indices
    
def knnClassifier(train_list, bin_train, testdata, num_neighbors, nearest_function):
    if num_neighbors > len(train_list):
        num_neighbors = len(train_list)
    neighbors = nearest_function(bin_train, testdata, num_neighbors)
    
#     print("neighbors: ", neighbors)
#     print("")
    
    target_values = [train_list[row][-1] for row in neighbors]
    predicted_value = max(set(target_values), key=target_values.count)
    return predicted_value

def exe_prediction(num_neighbors, nearest_function, binarized_function):
    train_data_from_file = load_data('data/income.train.txt.5k')
    dev_data_from_file = load_data('data/income.dev.txt')
    
    train_list, bin_train = binarized_function(train_data_from_file, map_feature_train_data)
    dev_list, bin_dev = binarized_function(dev_data_from_file, map_feature_dev_data)
    
#     Test on Dev data
    
    if num_neighbors > len(train_list):
        num_neighbors = len(train_list)
        
    dev_err_count = 0
    dev_positive = 0
    for i, testdata in enumerate(bin_dev):
        predicted_dev_value = knnClassifier(train_list, bin_train, testdata, num_neighbors, nearest_function)
        if predicted_dev_value != dev_list[i][-1]:
            dev_err_count += 1
        if predicted_dev_value == '>50K':
            dev_positive += 1

#     Test on training data
        
    train_err_count = 0
    train_positive = 0
    for i, testdata in enumerate(bin_train):
        predicted_train_value = knnClassifier(train_list, bin_train, testdata, num_neighbors, nearest_function)
        if predicted_train_value != train_list[i][-1]:
            train_err_count += 1
        if predicted_train_value == '>50K':
            train_positive += 1

    return (train_err_count*100)/len(train_list), (train_positive*100)/len(train_list), (dev_err_count*100)/len(dev_list), (dev_positive*100)/len(dev_list)

#     return (dev_err_count*100)/len(dev_list), (dev_positive*100)/len(dev_list)

k = [1,3,5,7,9,99,999,9999]

def get_rate(k):
    
#     best_model = (-1, 101)
    
#     print("Euclidean distance:")
#     for nn in range(1, 5000):
#         result = exe_prediction(nn, get_nearest_neighbor_eucli, binarize_data)
#         if result[0] < best_model[1]:
#             best_model = (k, result[0])
# #         print("k=", nn, "" if nn>999 else "\t", "train_err ", result[0],"%\t(+:", result[1], ",%)", "\t dev_err ", result[2],"% (+:", result[3], ",%)")
#         print("k=", nn, "" if nn>999 else "\t", "dev_err ", result[0],"% (+:", result[1], "%)")
#     print("The best result is where k = ", best_model[0], ": dev_err = ", best_model[1], "%")
        
    print("Euclidean distance:")
    for nn in k:
        result = exe_prediction(nn, get_nearest_neighbor_eucli, binarize_data)
        print("k=", nn, "" if nn>999 else "\t", "train_err ", result[0],"%\t(+:", result[1], "%)", "\t dev_err ", result[2],"% (+:", result[3], "%)")
    
    print("\nManhattan distance:")
    for nn in k:
        result = exe_prediction(nn, get_nearest_neighbor_manhat, binarize_data)
        print("k=", nn, "" if nn>999 else "\t", "train_err ", result[0],"%\t(+:", result[1], "%)", "\t dev_err ", result[2],"% (+:", result[3], "%)")
        
    print("\nEuclidean distance/ binarize all features:")
    for nn in k:
        result = exe_prediction(nn, get_nearest_neighbor_eucli, binarize_data_all)
        print("k=", nn, "" if nn>999 else "\t", "train_err ", result[0],"%\t(+:", result[1], "%)", "\t dev_err ", result[2],"% (+:", result[3], "%)")
    
    print("\nManhattan distance/ binarize all features:")
    for nn in k:
        result = exe_prediction(nn, get_nearest_neighbor_manhat, binarize_data_all)
        print("k=", nn, "" if nn>999 else "\t", "train_err ", result[0],"%\t(+:", result[1], "%)", "\t dev_err ", result[2],"% (+:", result[3], "%)")

def test_model():
    num_neighbors = 44
    
    train_data_from_file = load_data('data/income.train.txt.5k')
    test_data_from_file = load_data('data/income.test.blind')
    
    train_list, bin_train = binarize_data(train_data_from_file, map_feature_train_data)
    test_list, bin_test = binarize_data(test_data_from_file, map_feature_dev_data)
    
    predicted_results = []
    updated_data = []
    positive = 0
    for i, testdata in enumerate(bin_test):
        predicted_dev_value = knnClassifier(train_list, bin_train, testdata, num_neighbors, get_nearest_neighbor_eucli)
        predicted_results.append([predicted_dev_value])
        
        if predicted_dev_value == '>50K':
            positive+=1
    
    print("\npositive rate: ", (positive*100)/len(test_list), "%" )
    
    updated_data = np.concatenate((test_list, predicted_results), axis=1)

    np.savetxt('data/income.test.predicted', updated_data, fmt='%s', delimiter=', ')

def predict_dev():
    num_neighbors = 44
    
    train_data_from_file = load_data('data/income.train.txt.5k')
    dev_data_from_file = load_data('data/income.dev.txt')
    
    train_list, bin_train = binarize_data(train_data_from_file, map_feature_train_data)
    dev_list, bin_dev = binarize_data(dev_data_from_file, map_feature_dev_data)
    
#     Test on Dev data
    
    if num_neighbors > len(train_list):
        num_neighbors = len(train_list)
        
    dev_err_count = 0
    dev_positive = 0
    for i, testdata in enumerate(bin_dev):
        predicted_dev_value = knnClassifier(train_list, bin_train, testdata, num_neighbors, get_nearest_neighbor_eucli)
        if predicted_dev_value != dev_list[i][-1]:
            dev_list[i][-1] = predicted_dev_value
            dev_err_count += 1
        if predicted_dev_value == '>50K':
            dev_positive += 1
    
    np.savetxt('income.dev.predicted', dev_list, fmt='%s', delimiter=', ')

get_rate(k)

print("\nGenerate income.test.predicted file")
test_model()
