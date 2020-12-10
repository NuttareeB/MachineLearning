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

def binarize_data_all(datalist, func):
    data_list = list(map(lambda s: s.strip().split(', '), datalist))
    raw_data = [[value for idx, value in enumerate(line) if idx not in [9]] for line in data_list] # index 9 is target
    
    for line in raw_data:
        line.append(line[4]+line[8])
        line.append(line[4]+line[6]+line[8])
        
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
    
def exe_perceptron(train_list, bin_train, dev_list, bin_dev, perc_mode):
    num_feature = len(mapping) + 1 # add 1 for bias
    num_train_data = (len(bin_train))
    num_dev_data = (len(bin_dev))
    
    full_bin_train = np.concatenate((np.ones((num_train_data, 1), dtype=np.uint8), bin_train), axis=1)
    full_bin_dev = np.concatenate((np.ones((num_dev_data, 1), dtype=np.uint8), bin_dev), axis=1)
    
    w = np.zeros(num_feature)
    w_a = np.zeros(num_feature)
    c = 0
    w_use = np.zeros(num_feature)

    print("standard perceptron" if perc_mode == 1 else "average perceptron", end=" ")
    print("with combinations \n(occupation_country-of-origin, occupation_gender_country-of-origin)")
    
    print()
    
    best_err = 100.0
    best_err_epoch = 0
    
    
    for epoch in range(1):
        no_update = 0
        
        dev_err_count = 0
        dev_pos = 0
        
        if perc_mode == 1:
            # perceptron
            for i, bin_t in enumerate(full_bin_train):
                y = 1 if train_list[i][-1] == '>50K' else -1
                if y * (w.dot(bin_t)) <= 0:
                    no_update += 1
                    w += (y * bin_t)
            w_use = w
        else:
            # avg perceptron
            for i, bin_t in enumerate(full_bin_train):
                y = 1 if train_list[i][-1] == '>50K' else -1
                if y * (w.dot(bin_t)) <= 0:
                    no_update += 1
                    w += (y * bin_t)
                    w_a += c*y*bin_t
                c += 1
            w_use = (c*w) - w_a

        results = []
        # test test
        for i, bin_d in enumerate(full_bin_dev):
            predicted_result = 1 if w_use.dot(bin_d) > 0 else -1
            results.append(['>50K'] if predicted_result == 1 else ['<=50K'])
            dev_pos += 1 if predicted_result == 1 else 0
        
        dev_err_rate = round((dev_err_count*100)/num_dev_data, 1)
        print("epoch", epoch+1, "updates", no_update, "(" + str(round((no_update*100)/num_train_data, 1)) +            "%) dev_err cannot be determined" +            " (+:" + str(round((dev_pos*100)/num_dev_data, 1)) + "%)" )
        if dev_err_rate < best_err:
            best_err = dev_err_rate
            best_err_epoch = epoch
        
    return c, w_use, results
    
def exe_prediction():
    train_data_from_file = load_data('data2/income.train.txt.5k')
    dev_data_from_file = load_data('data2/income.test.blind')
    
    train_list, bin_train = binarize_data_all(train_data_from_file, map_feature_train_data)
    dev_list, bin_dev = binarize_data_all(dev_data_from_file, map_feature_dev_data)

    c, w_use, results = exe_perceptron(train_list, bin_train, dev_list, bin_dev, 2)

    updated_data = np.concatenate((dev_list, results), axis=1)
    np.savetxt('income.test.predicted', updated_data, fmt='%s', delimiter=', ')
    
exe_prediction()
