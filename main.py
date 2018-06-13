# -*- coding: utf-8 -*-
from __future__ import division
import findspark

findspark.init("/hdd/spark")
import pyspark
from pyspark.sql.functions import lit, col, udf
from pyspark.sql.types import StringType
from text_processing import detect_language, stem
from itertools import repeat
from operator import add

detect_language = udf(detect_language, StringType())
text_processing = udf(lambda text, lang: stem(text, lang), StringType())


def addMissing(vals, minPeriod, maxPeriod):
    vals = dict(vals)
    return [(period, vals[period]) if period in vals else (period, 0.0) for period in range(minPeriod, maxPeriod+1)]

def freqToSig(word, iterator):
    t = 6 * 3600 # ??????
    a = 1 - Math.exp(Math.log(0.5) / Math.log(t))
    b = 0.00000001
    global minPeriod
    global maxPeriod
    periodFreqs = addMissing(
        sorted(list(iterator).map(lambda a: (a[1], a[2])), key = lambda a: a[0]),
        minPeriod, maxPeriod
    )
    freqs = periodFreqs.map(lambda a: a[1])
    sig = []
    ewma = 0.0
    ewmv = 0.0
    for freq in freqs:
        sig.append((freq - Math.max(ewma, b)) / (Math.sqrt(ewmv) + b))
        delta = freq - ewma
        ewma = ewma + a * delta
        ewmv = (1 - a) * (ewmv + a * delta * delta)
    zip(repeat(word), periodFreqs.map(lambda a: a[0]), sig)




def get_data():
    sc = pyspark.SparkContext.getOrCreate()
    sqlContext = pyspark.SQLContext(sc)
    data = sqlContext.read.parquet("data")
    data = data.withColumn("lang", lit(detect_language(data["text"]))).limit(1000)

    words_count = data.rdd.flatMap(lambda row:
                                   zip(repeat(row["timestamp"] / 6 / 3600 / 1000),
                                       stem(row["text"], row["lang"]))
                                   ).map(lambda pair: (pair, 1)
                                         ).reduceByKey(add
                                                       ).map(lambda pair: (pair[0][0], pair[0][1], pair[1])
                                                             ).toDF(["period", "word", "count"])

    total_counts = words_count.rdd.map(lambda (period, word, counts): (period, counts)
                                       ).groupBy(lambda (period, counts): period
                                                 ).mapValues(list
                                                             ).map(lambda (period, counts):
                                                                   (period,
                                                                    sum([item[1] for item in counts]))
                                                                   ).toDF(["period", "total"])

    freqs = words_count.join(total_counts, words_count["period"] == total_counts["period"]).rdd.map(
        lambda (period, word, count, period2, counts): (period, word, (count / counts)))


    global minPeriod
    global maxPeriod
    minPeriod = total_counts.agg(min(col("total"))).collect()[0]
    maxPeriod = total_counts.agg(max(col("total"))).collect()[0]

    print("min: " + minPeriod)
    print("max: " + maxPeriod)

    smth = freqs.groupByKey(lambda a: a[0]).flatMapGroups(freqsToSig)
    smth = smth.filter(col("_2").notEqual(minPeriod))
    smth = smth.sort(col("_3").desc())
    smth.show(1000, truncate = false)


get_data()
