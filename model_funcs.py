"""
module contains functions to help with modelling processes in Machine Learning
functions here have capabilities such as
- benchmarking models
- evaluating model performance
- rectifying class distributions

Author: Oluwaseyi Ogunnowo
Date: May 2, 2022
"""
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score, confusion_matrix, recall_score
import matplotlib.pyplot as plt
import seaborn as sns



def plot_confusion_matrix(y_test, y_pred):
    """
    Function to plot confusion matrix
    ARGS:
        x_test: independent variables for testing
        y_test: dependent variables for testing
    RETURNS:
	confusion matrix plot
    """
    conf_matrix_df = pd.DataFrame(confusion_matrix(y_test, y_pred))
    plt.figure(figsize=(14, 12))
    sns.heatmap(conf_matrix_df, annot=True, fmt='g')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    plt.show()

def benchmark(x_train, x_test, y_train, y_test, algs, choice_metric = 'metric'):
    """
    Function to benchmark different algorithms and compare performance on test set

    ARGS:
        x_train: independent variables for training
        x_test: independent variables for testing
        y_train: dependent variables for training
        y_test: dependent variables for testing
        algs: (list) list containing names of scikit-learn algorithms
        choice_metric: (string) string to indicate metrics to consider as measue of performance.
                        options include: precision, recall, accuracy and f1_score
    RETURNS:
        returns choice metric score for each algorithm

    """
    train = []
    test = []
    name = []
    recall = []
    precision = []
    f1 = []
    for alg in algs:
        name.append(type(alg).__name__)
        print('\n')
        print('algorithm:', alg)
        training = alg.fit(x_train, y_train)
        training_score = alg.score(x_train, y_train)
        y_pred = alg.predict(x_test)
        test_score = alg.score(x_test, y_test)       
        f1.append(f1_score(y_test, y_pred))
        recall.append(recall_score(y_test, y_pred))
        precision.append(precision_score(y_test, y_pred))
        plot_confusion_matrix(y_test, y_pred)

        if (training_score - test_score) > 0.1:
            print('model likely overfitting, investigate further')

        train.append(training_score)
        test.append(test_score)

    benchmark = pd.DataFrame({'alg_name': name,
                             'train_score':train,
                             'test_score':test,
                             'precision':precision,
                             'recall':recall,
                             'f1':f1}, columns = ['alg_name', 'train_score', 
                             'test_score', 'precision', 'recall', 'f1'])

    return benchmark[[choice_metric]]

