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

from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer


def main():
    spark = SparkSession.builder.appName("spark machine learning").getOrCreate()

    data = []
    data.append((0, "a b c d e spark", 1.0))
    data.append((1, "b d", 0.0))
    data.append((2, "spark f g h", 1.0))
    data.append((3, "hadoop mapreduce", 0.0))

    training = spark.createDataFrame(data, ["id", "text", "label"])

    stages = []
    stages.append( Tokenizer(inputCol="text", outputCol="words") )
    stages.append( HashingTF(inputCol="words", outputCol="features") )
    stages.append( LogisticRegression(maxIter=500, regParam=0.001) )

    pipeline = Pipeline(stages = stages)

    model = pipeline.fit(training)
    
    test_data = []
    test_data.append((4, "spark i j k"))
    test_data.append((5, "l m n"))
    test_data.append((6, "spark hadoop spark"))
    test_data.append((7, "apache hadoop"))

    test = spark.createDataFrame(test_data, ["id", "text"])

    prediction = model.transform(test)

    result = prediction.select("id", "text", "probability", "prediction").collect()

    for row in result:
        rid, text, prob, prediction = row

        print "(%d, %s) --> prob=%s, prediction=%f" % (rid, text, str(prob), prediction)

    

    return

if __name__ == "__main__":
    main()




