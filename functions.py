import numpy as np
"""
This module facilitate the working of machine learning models
"""


def train_validate_test_split (data, labels, testRatio = 0.3, valRatio = 0.3):
    """
    This function splits data into 3 subsets (train, validate, test) datasets

    Args:
        data(2d nparray): All the dataset
        labels(1d nparray): The target values
        testRatio(float): Ratio of splitted data
        valRatio(float): Ratio of splitted training data

    return:
        three data nparrays (training, validate, test)
        three labels nparrays (training, validate, test)
    """
    data_rows = data.shape[0]
    test_rows = int(testRatio * data_rows) #50
    training_rows = data_rows - test_rows #100
    validate_rows = int(valRatio * training_rows) #33
    training_rows -= validate_rows #67
    
    test_data = np.array(data[0 : test_rows, :]) # 0 : 50
    training_data = np.array(data[test_rows : test_rows + training_rows, :]) #50 : 117
    validate_data = np.array(data[training_rows + test_rows : data.shape[0], :]) # 117 : 150
    
    test_label = np.array(labels[0 : test_rows])
    training_label = np.array(labels[test_rows : test_rows + training_rows]) 
    validate_label = np.array(labels[training_rows + test_rows : labels.shape[0]])
    
    return (training_data, validate_data, test_data, training_label, validate_label, test_label)
    
    
