import json
import os
from concurrent.futures import ProcessPoolExecutor

import findspark

findspark.init()
findspark.find()
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import *
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, FloatType, IntegerType
#from pyspark.sql.window import Window as W
from pyspark.sql import functions as F

class SparkData:
    PATH_DUMP: str = './Dumps/KGDump/'
    PATH_EMBEDDINGS: str = 'D:/df_embs_avg_sent_trans/' #'D:/Embeddings/'
    PATH_AUX: str = './Embeddings/aux/'
    PATH_EMBEDDINGS_LITERALS = PATH_EMBEDDINGS + "embeddings_literals"
    PATH_EMBEDDINGS_ENTITIES_TEMP = PATH_EMBEDDINGS + "embeddings_entities_temp"
    PATH_EMBEDDINGS_ENTITIES = PATH_EMBEDDINGS + "embeddings_entities"
    PATH_TREES = PATH_EMBEDDINGS + "trees"

    def __init__(self):

        self.sparkSession = (SparkSession.builder.master('local[16]')
                             .config("spark.driver.memory", "20g")
                             .config('spark.executor.memory', '5G')
                             .config('spark.driver.maxResultSize', '3G')
                             .config('spark.driver.host', 'localhost')
                             .config("spark.local.dir", "d:/tmp")
                             .appName('dbpedia').getOrCreate())
        self.load_data_frames_from_dump()

        self.schema_df_embeddings = StructType([
            StructField('s', StringType(), True),
            StructField('o', StringType(), True),
            StructField('typeObject', StringType(), True),
            StructField("embedding", ArrayType(FloatType(), True), True)
        ])

        self.schema_df_embeddings_full = StructType([
            StructField('s', StringType(), True),
            StructField('p', StringType(), True),
            StructField('o', StringType(), True),
            StructField('typeObject', StringType(), True),
            StructField("embedding", ArrayType(FloatType(), True), True)
        ])

    def load_data_frames_from_dump(self):
        path_all = SparkData.PATH_DUMP + "all_entities"
        if not os.path.exists(path_all):

            files = [SparkData.PATH_DUMP + f for f in os.listdir(SparkData.PATH_DUMP) if f.endswith('.csv')]
            df = self.sparkSession.read.options(delimiter="|", header=True).csv(files[0])
            for file in files[1:]:
               df = df.union(self.sparkSession.read.options(delimiter="|", header=True).csv(file))
            self.df_main = df

            df_main = (df.select('s', 'p', 'o')
                  .distinct().withColumn('id', monotonically_increasing_id())
                  .select('id', 's', 'p', 'o'))
            df_main.write.options(header=True, delimiter='|').csv(path_all)
        self.df_main = self.sparkSession.read.options(delimiter="|", header=True).csv(path_all)


        path_e_l = SparkData.PATH_DUMP + "entities_literals"
        if not os.path.exists(path_e_l):
            df = (self.df_main.filter(F.col('o').rlike("(?i)^*@en .$"))
                  #.select('id', 's', 'p', 'o')
                  #.distinct().withColumn('id', monotonically_increasing_id())
                  #.select('id', 's', 'p', 'o')
                  )

            df.write.options(header=True, delimiter='|').csv(path_e_l)
        self.df_literals = self.sparkSession.read.options(delimiter="|", header=True).csv(path_e_l)

        path_e_e = SparkData.PATH_DUMP + "entities_entities"
        if not os.path.exists(path_e_e):
            df = (self.df_main.filter(F.col('o').rlike("^http")).sort('s'))

            df.write.options(header=True, delimiter='|').csv(path_e_e)
        self.df_entities = self.sparkSession.read.options(delimiter="|", header=True).csv(path_e_e)

    def load_Embeddings(self):
        df_emb_entities = self.sparkSession.read.schema(self.schema_df_embeddings).json(self.PATH_EMBEDDINGS_LITERALS)
        df_emb_literals = self.sparkSession.read.schema(self.schema_df_embeddings).json(self.PATH_EMBEDDINGS_ENTITIES_TEMP)
        df = df_emb_literals.union(df_emb_entities)
        return df

    def load_Embeddings_path(self, path):
        for i in range(len(path)):
            if i == 0:
                df_emb = self.sparkSession.read.schema(self.schema_df_embeddings_full).json(path[i])
            else:
                df_emb = df_emb.union(self.sparkSession.read.schema(self.schema_df_embeddings_full).json(path[i]))
        # if isinstance(path, tuple):
        #     df_emb_entities = self.sparkSession.read.schema(self.schema_df_embeddings).json(path[0])
        #     df_emb_literals = self.sparkSession.read.schema(self.schema_df_embeddings).json(path[1])
        #     return df_emb_literals.union(df_emb_entities)
        # else:
        #     return self.sparkSession.read.schema(self.schema_df_embeddings).json(path)
        return df_emb
