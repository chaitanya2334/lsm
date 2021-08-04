from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from typing import Callable
from pyspark.sql import Column
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType, IntegerType, ArrayType, DataType


def start_spark():
    c = SparkConf().set("spark.sql.execution.arrow.pyspark.enabled",
                        "true").set("spark.driver.maxResultSize", "10g")
    spark = SparkSession.builder.config(conf=c).getOrCreate()
    return spark


class py_or_udf:
    def __init__(self, returnType: DataType = StringType()):
        self.spark_udf_type = returnType

    def __call__(self, func: Callable):
        def wrapped_func(*args, **kwargs):
            if any([isinstance(arg, Column) for arg in args]) or \
                any([isinstance(vv, Column) for vv in kwargs.values()]):
                return udf(func, self.spark_udf_type)(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        return wrapped_func



