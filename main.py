# -*- coding: utf-8 -*-
from __future__ import division
import findspark

findspark.init("/usr/local/spark/spark-2.3.1-bin-hadoop2.7")
import pyspark
from pyspark.sql.functions import lit, col, udf
from pyspark.sql.types import StringType
from text_processing import detect_language, stem
from itertools import repeat
from operator import add

detect_language = udf(detect_language, StringType())
text_processing = udf(lambda text, lang: stem(text, lang), StringType())

sc = pyspark.SparkContext.getOrCreate()
sqlContext = pyspark.SQLContext(sc)
data = sqlContext.read.parquet("data")
data = data.withColumn("lang", lit(detect_language(data["text"]))).limit(10000)

words_count = data.rdd.flatMap(lambda row:
                               zip(repeat(int(row["timestamp"] / 6 / 3600 / 1000)),
                                   stem(row["text"], row["lang"]))
                               ).map(lambda pair: (pair, 1)
                                     ).reduceByKey(add
                                                   ).map(lambda pair: (pair[0][0], pair[0][1], pair[1])
                                                         ).toDF(["period", "word", "count"])

words_count.show()

total_counts = words_count.rdd.map(lambda (period, word, counts): (period, counts)
                                   ).groupBy(lambda (period, counts): period
                                             ).mapValues(list
                                                         ).map(lambda (period, counts):
                                                               (period,
                                                                sum([item[1] for item in counts]))
                                                               ).toDF(["period", "total"])

total_counts.show()

freqs = words_count.join(total_counts, words_count["period"] == total_counts["period"]).rdd.map(
    lambda (period, word, count, period2, counts): (period, word, (count / counts)
                                                    )).toDF(["period", "word", "freq"])

freqs = freqs.rdd.groupBy(lambda (period, word, freq): period
                          ).mapValues(list
                                      ).map(lambda (period, word_counts):
                                            (period, sorted(word_counts, key=lambda word_count: word_count[2])))


for item in freqs.collect():
    print ("Period %d TOP-5: " % item[0])
    for top in item[1][:5]:
        print("Word: %s" % (top[1]))
    print("=====================")