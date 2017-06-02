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
from pyspark.ml.feature import HashingTF, Tokenizer, IDF, CountVectorizer


def main():
    spark = SparkSession.builder.appName("spark machine learning").getOrCreate()

    data = []
    data.append((0.0, "Hi I heard about Spark"))
    data.append((0.0, "I wish Java could user case classes"))
    data.append((1.0, "Logistic regression models are neat"))

    data_set = spark.createDataFrame(data, ["label", "sentence"])
    print data_set.show(truncate=False)

    tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
    data_set = tokenizer.transform(data_set)
    print data_set.show(truncate=False)

    hashingTF = HashingTF(inputCol="words", outputCol="hashing_tf", numFeatures=32)
    data_set = hashingTF.transform(data_set)
    print data_set.show(truncate=False)

    idf = IDF(inputCol="hashing_tf", outputCol="idf")
    idfModel = idf.fit(data_set)
    data_set = idfModel.transform(data_set)
    print data_set.show(truncate=False)

    return

if __name__ == "__main__":
    main()




