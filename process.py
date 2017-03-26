"""
Created by Howard Fan
Machine Learning Problem Set 2
March 13, 2017

Impute all missing values in the input datasets. For real-valued features, replaces missing values
with the labeled-conditioned mean and normalize. For nominal features, replaces missing values with
the most abundant value in the feature. Creates a new file called <original input filename>.processed

INSTRUCTIONS TO USE (on command prompt):
python process.py <dataset filename> x N
where N is number of desired datasets to process
Example:
    python process.py crx.data.training crx.data.testing

Files should be in same folder as Python script. Otherwise, you must use full path of file location
"""

import sys
from collections import Counter
import numpy as np
import pandas as pd

def processFile(filedata):
    """
    Impute all missing values in the input datasets. For real-valued features, replaces missing values
    with the labeled-conditioned mean and normalize. For nominal features, replaces missing values with
    the most abundant value in the feature.
    :param filedata: input dataset
    :return: dataset with all missing values replaced
    """

    # create copy of filedata
    processeddata = filedata.copy(deep=True)
    pd.options.mode.chained_assignment = None

    labels = filedata[len(filedata.columns)-1].as_matrix()
    positive = [i for i, e in enumerate(labels) if e == '+']
    negative = [i for i, e in enumerate(labels) if e == '-']

    # For each feature, impute all missing values
    for i in range(len(filedata.columns)-1):

        # if feature is real-valued
        if filedata[i].dtype == np.float64 or filedata[i].dtype == np.int64:
            # calculate label-conditioned means
            positive_mu = filedata[i][positive].mean()
            negative_mu = filedata[i][negative].mean()

            # replace missing values of instances with positive label with positive-label mean
            for pos in positive:
                if np.isnan(filedata[i][pos]):
                    processeddata[i][pos] = positive_mu
            # replace missing values of instances with negative label with negative-label mean
            for neg in negative:
                if np.isnan(filedata[i][neg]):
                    processeddata[i][neg] = negative_mu

            # normalize all instances in feature
            processeddata[i] = (processeddata[i] - processeddata[i].mean())/processeddata[i].std()
        # if feature is nominal
        else:
            # replace missing values with most abundant value
            processeddata[i].fillna(Counter(filedata[i]).most_common(1)[0][0], inplace=True)

    return processeddata

def main():
    # process each input file
    for i in range(1,len(sys.argv)):
        # get filename
        filename = sys.argv[i]

        # get raw dataset from file
        raw = pd.read_csv(filename, header=None, sep=',', na_values=["?"])

        # process raw dataset
        processed = processFile(raw)

        # write processed dataset to new file labeled "(original filename).processed"
        processed.to_csv(filename + '.processed', sep=',', header=False, index=False)

if __name__ == '__main__':
    main()
