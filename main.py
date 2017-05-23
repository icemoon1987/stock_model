#!/usr/bin/env python
# -*- coding: utf-8 -*-

######################################################
#
# File Name:  main.py
#
# Function:   
#
# Usage:  
#
# Input:  
#
# Output:	
#
# Author: panwenhai
#
# Create Time:    2017-05-02 15:08:49
#
######################################################

import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import os
import time
from datetime import datetime, timedelta

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict


def main():
    df = pd.read_csv('./data/data1.csv', header=None, index_col=False)

    features = df.ix[:, df.columns[:-1]]
    labels = df.ix[:, df.columns[-1]]

    # gbdt model
    gbdt = GradientBoostingClassifier()
    scores = cross_val_score(gbdt, features, labels, cv=5)

    print "cv result: " + str(scores)
    print "Accuracy: %2f" % (scores.mean())

    predicted = cross_val_predict(gbdt, features, labels, cv=5)

    compare_result = pd.DataFrame()
    compare_result["predict_label"] = predicted
    compare_result["true_label"] = labels
    compare_result["is_correct"] = False

    compare_result.loc[compare_result["predict_label"] == compare_result["true_label"], ["is_correct"]] = True

    print compare_result

    return


if __name__ == "__main__":
    main()


