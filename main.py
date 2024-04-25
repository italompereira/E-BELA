import platform
import os

from Database.database_operation import DataBase

if platform.system() == 'Windows':
    print('Windows')
    # Environment variables used for Spark
    dict_os_environ = {
        "SPARK_HOME":"C:\Spark\spark-3.3.4-bin-hadoop3_new",
        "HADOOP_HOME":"C:\Spark\spark-3.3.4-bin-hadoop3_new\hadoop",
        "JAVA_HOME":"C:\Program Files\Java\jdk-17",
        "SPARK_LOCAL_IP":"127.0.0.1"
    }

    os.environ["SPARK_HOME"] = dict_os_environ["SPARK_HOME"]
    os.environ["HADOOP_HOME"] = dict_os_environ["HADOOP_HOME"]
    os.environ["JAVA_HOME"] = dict_os_environ["JAVA_HOME"]
    os.environ["SPARK_LOCAL_IP"] = dict_os_environ["SPARK_LOCAL_IP"]
else:
    print('Linux')
    dict_os_environ = {
        "SPARK_HOME": "/mnt/c/Spark/spark-3.5.0-bin-hadoop3/",
        "HADOOP_HOME": "/mnt/c/Spark/spark-3.5.0-bin-hadoop3/hadoop",
    }
    os.environ["SPARK_HOME"] = dict_os_environ["SPARK_HOME"]
    os.environ["HADOOP_HOME"] = dict_os_environ["HADOOP_HOME"]

#Disable GPU
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


from Dumps.kg_dump import KGDump
from Dumps.spark_data import SparkData
from Evaluate.evaluate import Evaluate
from Encoders.encoder import Encoder

import cProfile, pstats, io

# port 5432 postgres
# def profile(fnc):
#     """A decorator that uses cProfile to profile a function"""
#
#     def inner(*args, **kwargs):
#         pr = cProfile.Profile()
#         pr.enable()
#         retval = fnc(*args, **kwargs)
#         pr.disable()
#         s = io.StringIO()
#         sortby = 'cumulative'
#         ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
#         ps.print_stats()
#         print(s.getvalue())
#         return retval
#
#     return inner

if __name__ == "__main__":

    # Endpoint
    #path_endpoint = "https://dbpedia.org/sparql" #"http://200.131.10.200:8890/sparql" #

    # Download dump from dbpedia and convert to cvs files
    #KGDump()()

    # Load spark session and load dataframes
    spark_data = SparkData()

    # Get embeddings from data
    Encoder(spark_data)()

    # Save data on postgresql
    data_base = DataBase(spark_data)
    #data_base.save()

    # Evaluation
    evaluate = Evaluate(spark_data, data_base)
    evaluate()