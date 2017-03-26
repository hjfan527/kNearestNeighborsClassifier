"""
Created by Howard Fan
Machine Learning Problem Set 2
March 13, 2017

Classifies all instances in the testing dataset via the k-Nearest Neighbor (k-NN) classifier with L2
distance, comparing the testing dataset with an input training dataset. Returns a new testing dataset
file with an additional comma-separated field consisting of the predicted labels

INSTRUCTIONS TO USE (on command prompt):
python run.py <K> <training dataset filename> <testing dataset filename>
where K is the number of nearest neighbors to compare
Example:
    python run.py 3 crx.data.training.processed crx.data.testing.processed

Files should be in same folder as Python script. Otherwise, you must use full path of file location
"""


import sys
from collections import Counter
import numpy as np
import pandas as pd

def predictTest(k, train, test):
    """
    Implements the k-NN classifer to classify the testing dataset based on the training dataset
    :param k: number of nearest neighbors to consider
    :param train: training dataset
    :param test: testing dataset
    :return: list of predicted labels
    """

    pred_labels = []

    # for each instance in the testing dataset, calculate all L2 distance from all training instances
    for te in range(len(test)):
        all_D = np.zeros((len(train), 1))

        # calculate the L2 distance of the testing instance from each training instance
        for tr in range(len(train)):
            D = 0
            for var in range(len(train.columns)-1):
                # if feature is real-valued, add (testing value - training value)^2
                if train[var].dtype == np.float64 or train[var].dtype == np.int64:
                    D += (test[var][te] - train[var][tr])**2
                # if feature is nominal, add 1 if testing and training values are different
                else:
                    if test[var][te] != train[var][tr]:
                        D += 1
            all_D[tr] = D**(1/2)

        # sort all L2 distances, select K closest neighbors, and choose the most prevalent label
        all_D = np.column_stack((all_D, np.array(range(len(train)))))
        all_D = all_D[np.argsort(all_D[:, 0])]
        prob_labels = train[len(train.columns)-1][all_D[0:k, 1]].as_matrix()
        pred_labels.append(Counter(prob_labels).most_common(1)[0][0])

    return pred_labels

def main():
    # K nearest neighbors
    k = int(sys.argv[1])
    # training dataset
    train_fn = sys.argv[2]
    # testing dataset
    test_fn = sys.argv[3]

    # get datasets from training and testing files
    train = pd.read_csv(train_fn, header=None, sep=',')
    test = pd.read_csv(test_fn, header=None, sep=',')

    # determine and add predicted labels
    test = test.assign(pred=predictTest(k, train, test))

    # write testing dataset with predicted labels to new file labeled "(original testing filename).predicted"
    test.to_csv(test_fn + ".k"+ sys.argv[1] + '.predicted', sep=',', header=False, index=False)
if __name__ == '__main__':
    main()
