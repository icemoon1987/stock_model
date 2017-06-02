#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""


File Name:  test.py

Function:   

Usage:  

Input:  

Output:	

Author: panwenhai

Create Time:    2017-05-31 15:50:27

"""

import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import os
import time
from datetime import datetime, timedelta

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession
from pyspark.sql.types import *


def main():
    spark = SparkSession.builder.appName("spark machine learning").getOrCreate()

    training = spark.read.csv("file:///home/panwenhai/stock_model/data/data1.csv", inferSchema=True)

    feature_col = training.columns[:-1]
    label_col = training.columns[-1]

    training = training.withColumnRenamed(label_col, "label")

    vecAssembler = VectorAssembler(inputCols=feature_col, outputCol="features")
    training = vecAssembler.transform(training)

    print "Training Data:"
    print training.show()
    print ""

    lr = LogisticRegression(maxIter=1000, regParam=0.01)
    print "LR params:"
    print lr.explainParams()
    print ""

    model1 = lr.fit(training)
    print model1
    print "model1 params:"
    print lr.extractParamMap()

    param_map = {}
    param_map[lr.maxIter] = 3000
    param_map.update(
            {
                lr.regParam: 0.1,
                lr.threshold: 0.55
            })

    param_map2 = {}
    param_map2[lr.probabilityCol] = "myProbability"

    params = param_map.copy()
    params.update(param_map2)

    model2 = lr.fit(training, params)
    print model2
    print "model2, params:"
    print lr.extractParamMap()

    test = training

    prediction = model2.transform(test)

    result = prediction.select("features", "label", "myProbability", "prediction").collect()

    for row in result:
        print "features=%s, label=%s    --> prob=%s, prediction=%s" % (row.features, row.label, row.myProbability, row.prediction)

    print "weights: "
    print model2.coefficients

    print "intercept: "
    print model2.intercept

    evaluate_result = model2.evaluate(test)

    print "auc: " + str(evaluate_result.areaUnderROC)

    print evaluate_result.fMeasureByThreshold.show()
    print evaluate_result.precisionByThreshold.show()
    print evaluate_result.recallByThreshold.show()
    print evaluate_result.pr.show()
    

    return

if __name__ == "__main__":
    main()




