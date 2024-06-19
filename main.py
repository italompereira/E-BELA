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
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from Dumps.spark_data import SparkData
from Dumps.kg_dump import KGDump
from Encoders.encoder import Encoder
from Database.database_operation import DataBase
from Evaluate.evaluate import Evaluate

def run_encoder(spark_data, model, weighted, strategy, top_n, top_n_way):

    config_experiment = model + '_' + \
                        ('WGT' if weighted else 'NotWGT') + '_' + \
                        strategy + '_' + \
                        ('TopNone' if top_n is None else (f'Top{top_n}Grouped' if top_n_way == 'GROUPED' else f'Top{top_n}Simple'))
    print(f'Encoding: {config_experiment}')
    encoder_ = Encoder(spark_data, model=model, weighted=weighted, strategy=strategy, top_n=top_n, top_n_way=top_n_way, config_experiment=config_experiment)
    encoder_()

    data_base_ = DataBase(spark_data, model=model, weighted=weighted, top_n=top_n, top_n_way=top_n_way, config_experiment=config_experiment)
    data_base_.save()



def run_evaluate(spark_data, model, weighted, strategy, top_n, top_n_way, context_with_mention_avg, dataset):
    config_experiment = model + '_' + \
                        ('WGT' if weighted else 'NotWGT') + '_' + \
                        strategy + '_' + \
                        ('TopNone' if top_n is None else (f'Top{top_n}Grouped' if top_n_way == 'GROUPED' else f'Top{top_n}Simple')) + \
                        ('_ContWithMention' if context_with_mention_avg else '_ContWithoutMention') + '_' + \
                        dataset

    encoder_ = Encoder(spark_data, model=model, weighted=weighted, strategy=strategy, top_n=top_n, top_n_way=top_n_way, config_experiment=config_experiment)
    data_base_ = DataBase(spark_data, model=model, weighted=weighted, top_n=top_n, top_n_way=top_n_way, config_experiment='_'.join(config_experiment.split('_')[:-2]))
    evaluate_ = Evaluate(spark_data, encoder_, data_base_, context_with_mention_avg, dataset, config_experiment)
    evaluate_()

if __name__ == "__main__":

    # Endpoint
    #path_endpoint = "https://dbpedia.org/sparql" #"http://200.131.10.200:8890/sparql" #

    # Download dump from dbpedia and convert to cvs files
    KGDump()()

    # Load spark session and load dataframes
    spark_data = SparkData()

    tests_cases = {
        'model': [
            'ST',
            'USE'
        ],
        'weighted': [
            False,
            True
        ],
        'strategy': [
            'AVG',
            #'CONCAT'
        ],
        'top_n': [
            [(2, 'GROUPED')],#, (2, 'SIMPLE')],
            None
        ],
        'context_with_mention_avg': [
            True,
            False
        ],
        'dataset': [
            'lcquad',
            #'aidab',
            #'ace2004',
            #'aquaint',
            #'msnbc',
        ],
    }

    # Encoding
    for model in tests_cases['model']:
        for weighted in tests_cases['weighted']:
            for strategy in tests_cases['strategy']:
                for top_n in tests_cases['top_n']:
                    if top_n is not None:
                        for top_n_way in top_n:
                            run_encoder(spark_data, model=model, weighted=weighted, strategy=strategy, top_n=top_n_way[0], top_n_way=top_n_way[1])
                    else:
                        run_encoder(spark_data, model=model, weighted=weighted, strategy=strategy, top_n=top_n, top_n_way=[])

    # Evaluating
    for model in tests_cases['model']:
        for weighted in tests_cases['weighted']:
            for strategy in tests_cases['strategy']:
                for top_n in tests_cases['top_n']:
                    for dataset in tests_cases['dataset']:
                        for context_with_mention_avg in tests_cases['context_with_mention_avg']:
                            if top_n is not None:
                                for top_n_way in top_n:
                                    run_evaluate(spark_data, model=model, weighted=weighted, strategy=strategy, top_n=top_n_way[0], top_n_way=top_n_way[1], context_with_mention_avg= context_with_mention_avg, dataset=dataset)
                                    break
                            else:
                                run_evaluate(spark_data, model=model, weighted=weighted, strategy=strategy, top_n=top_n, top_n_way=[], context_with_mention_avg=context_with_mention_avg, dataset=dataset)
                                break