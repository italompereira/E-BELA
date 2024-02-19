import platform
import os
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
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


from Parser.parse_files import Dump
from data import Data
from evaluate import Evaluate
from rdf2vec_dbpedia import NewRDF2VEC

import cProfile, pstats, io


def profile(fnc):
    """A decorator that uses cProfile to profile a function"""

    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner

if __name__ == "__main__":

    # Endpoint
    path_endpoint = "https://dbpedia.org/sparql" #"http://200.131.10.200:8890/sparql" #

    # Download dump from dbpedia and convert to cvs files
    dump = Dump()
    dump()

    # Build data object
    data = Data(path_endpoint, dict_os_environ)

    # Get embeddings for entities and literals
    newRDF2VEC = NewRDF2VEC(path_endpoint, data)
    newRDF2VEC()

    # if not os.path.isfile(embeddings_file):
    #     newRDF2VEC = NewRDF2VEC(path_endpoint, embeddings_file, literal_relations_file, data)
    #     newRDF2VEC()


    # files = [f for f in os.listdir(path_temp) if f.endswith('.csv')]
    # df = spark.read.options(delimiter="|", header=True).csv(path_temp + files[0])
    # for file in files[1:]:
    #     df = df.union(spark.read.options(delimiter="|", header=True).csv(path_temp + file))
    # #df.show()


    # kg = KG(endpoint, is_remote=True, skip_verify=True)
    # data = Data(kg)
    # entities_mapping, predicates_mapping = data.load_conll_grounth_truth()

    evaluate = Evaluate(path_endpoint, data, Evaluate.ALL)
    evaluate()