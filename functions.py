import numpy as np
"""
This module facilitate the working of machine learning models
"""


def calculate_accuracy(predicted_y, y):
    """
    a function to calculate the accuracy of a model
    Args:
        predicted_y(np array): the expected labels
        y(np array): the true labels

    Returns: accuracy of predicted y to the real y

    """
    true_expected = 0
    for i in range(len(y)):
        if y[i] == predicted_y[i]:
            true_expected += 1
    return true_expected / len(y)


def train_validate_test_split(data, labels, testRatio=0.3, valRatio=0.3):
    """
    This function splits data into 3 subsets (train, validate, test) datasets

    Args:
        data(2d nparray): All the dataset
        labels(1d nparray): The target values
        testRatio(float): Ratio to split data
        valRatio(float): Ratio to split training data

    return:
        three data np arrays (training, validate, test)
        three labels np arrays (training, validate, test)
    """
    data_rows = data.shape[0]
    test_rows = int(testRatio * data_rows)
    training_rows = data_rows - test_rows
    validate_rows = int(valRatio * training_rows)
    training_rows -= validate_rows
    
    test_data = np.array(data[0: test_rows, :])
    training_data = np.array(data[test_rows: test_rows + training_rows, :])
    validate_data = np.array(data[training_rows + test_rows: data.shape[0], :])
    
    test_label = np.array(labels[0: test_rows])
    training_label = np.array(labels[test_rows: test_rows + training_rows])
    validate_label = np.array(labels[training_rows + test_rows: labels.shape[0]])
    
    return training_data, validate_data, test_data, training_label, validate_label, test_label
    
    
